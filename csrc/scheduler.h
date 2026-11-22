#pragma once

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "rtc_ring.h"
#include "tsc_pool.h"

namespace tempo {

inline int64_t steady_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct SchedulerTelemetry {
  std::vector<double> fast_ttac_ms;
  std::vector<double> slow_latency_ms;
  uint64_t fast_batches_launched{0};
  uint64_t fast_requests_served{0};
  uint64_t slow_batches_completed{0};
  uint64_t slow_slices_launched{0};
  uint64_t fast_preemptions{0};
  uint64_t fast_coalesced_drops{0};
  uint64_t slow_coalesced_drops{0};
  uint64_t p2p_prepushes{0};
};

class TimeAwareScheduler {
 public:
  TimeAwareScheduler(
      int32_t num_streams,
      const std::vector<std::vector<int32_t>>& slow_devices,
      const std::vector<int32_t>& fast_devices,
      int32_t slow_batch_size,
      int32_t max_fast_batch_size,
      int32_t slow_frame_seq_len,
      double fast_deadline_ms,
      double max_slice_ms,
      double max_slow_aoi_ms);

  ~TimeAwareScheduler();

  TimeAwareScheduler(const TimeAwareScheduler&) = delete;
  TimeAwareScheduler& operator=(const TimeAwareScheduler&) = delete;

  int32_t num_fast_shards() const { return static_cast<int32_t>(fast_shards_.size()); }
  int32_t num_slow_replicas() const { return static_cast<int32_t>(slow_replicas_.size()); }
  int32_t num_slow_stages() const { return num_slow_stages_; }
  int32_t fast_shard_num_streams(int32_t shard_idx) const;
  int32_t slow_replica_num_streams(int32_t replica_idx) const;
  bool is_disaggregated() const { return disaggregated_; }
  bool p2p_direct_enabled() const { return p2p_direct_; }

  void init_fast_shard(
      int32_t shard_idx,
      std::shared_ptr<TemporalStateCache> tsc,
      std::shared_ptr<RTCHorizonRing> rtc,
      torch::Tensor fast_obs_static,
      torch::Tensor fast_prefix_static,
      torch::Tensor fast_prefix_valid_static,
      torch::Tensor fast_pt_static,
      torch::Tensor fast_seqlens_static,
      torch::Tensor fast_tnow_static,
      torch::Tensor fast_out_static);

  void init_slow_stage(
      int32_t replica_idx,
      int32_t stage_idx,
      std::shared_ptr<TemporalStateCache> slow_tsc,
      int32_t dst_layer_offset,
      int32_t layer_count,
      torch::Tensor slow_obs_static,
      std::vector<torch::Tensor> act_in_static,
      std::vector<torch::Tensor> act_out_static,
      torch::Tensor slow_pt_static,
      torch::Tensor slow_start_page_static,
      torch::Tensor slow_tobs_static);

  int64_t fast_exec_cuda_stream_raw(int32_t shard_idx) const;
  int64_t slow_exec_cuda_stream_raw(int32_t replica_idx, int32_t stage_idx) const;

  void begin_slow_slice_capture(int32_t replica_idx, int32_t stage_idx, int32_t slow_bucket);
  void end_slow_slice_capture(int32_t replica_idx, int32_t stage_idx, int32_t slow_bucket);
  std::vector<double> profile_slow_slices(int32_t replica_idx, int32_t stage_idx, int32_t slow_bucket);

  void begin_fast_bucket_capture(int32_t shard_idx, int32_t bucket_size);
  double end_fast_bucket_capture(int32_t shard_idx, int32_t bucket_size);
  void abort_active_captures() noexcept;

  void start();
  void stop();
  void reset_all_streams(c10::optional<torch::Tensor> init_pose = c10::nullopt);
  void reset_telemetry();
  void reset_rtc_stats();

  void submit_slow_obs(int32_t stream_id, int64_t t_obs_ns, const torch::Tensor& obs);
  void submit_fast_obs(int32_t stream_id, int64_t t_obs_ns, const torch::Tensor& fast_obs);
  bool submit_remote_hidden(int32_t stream_id, int64_t t_obs_ns, const torch::Tensor& hidden);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> sample_actions_at(
      int64_t t_query_ns, bool return_cpu = false);
  RTCStats rtc_stats() const;
  SchedulerTelemetry telemetry() const;
  int64_t now_ns() const { return steady_now_ns(); }

 private:
  static constexpr int32_t kHandoffRingSlots = 2;

  struct StreamIngress {
    std::mutex mu;
    bool slow_pending{false};
    bool slow_inflight{false};
    int64_t slow_t_obs_ns{0};
    bool fast_pending{false};
    int64_t fast_t_obs_ns{0};
    int64_t fast_wait_since_ns{0};
    int64_t min_valid_slow_t_obs_ns{0};
    int64_t last_pub_slow_t_obs_ns{0};
    bool has_published_slow{false};
  };

  struct FastShard {
    int32_t shard_idx{0};
    int32_t device_id{0};
    std::vector<int32_t> global_stream_ids;

    std::shared_ptr<TemporalStateCache> tsc;
    std::shared_ptr<RTCHorizonRing> rtc;

    c10::cuda::CUDAStream exec_stream;
    c10::cuda::CUDAStream ingest_stream;
    c10::cuda::CUDAStream remote_write_stream;
    std::mutex remote_write_mu;
    at::cuda::MempoolId_t fast_mempool;
    std::unique_ptr<at::cuda::CUDAGraph> capturing_graph;

    std::map<int32_t, std::unique_ptr<at::cuda::CUDAGraph>> fast_bucket_graphs;
    std::map<int32_t, int64_t> fast_bucket_exec_ns;
    std::map<int32_t, int64_t> runtime_bucket_exec_ns;
    std::vector<int32_t> sorted_buckets;

    torch::Tensor fast_obs_static;
    torch::Tensor fast_prefix_static;
    torch::Tensor fast_prefix_valid_static;
    torch::Tensor fast_pt_static;
    torch::Tensor fast_seqlens_static;
    torch::Tensor fast_tnow_static;
    torch::Tensor fast_out_static;
    torch::Tensor per_stream_fast_obs;

    std::vector<cudaEvent_t> fast_caller_events;
    std::vector<cudaEvent_t> fast_ingest_events;
    std::vector<cudaEvent_t> fast_consumed_events;
    cudaEvent_t fast_done_event{nullptr};

    int32_t* h_fast_local_ids{nullptr};
    int64_t* h_fast_tobs{nullptr};
    int64_t* h_fast_slow_tobs{nullptr};
    int32_t* d_fast_local_ids{nullptr};
    int64_t* d_fast_tobs{nullptr};
    int64_t* d_fast_slow_tobs{nullptr};

    std::thread worker_thread;

    FastShard(int32_t idx, int32_t dev)
        : shard_idx(idx),
          device_id(dev),
          exec_stream(c10::cuda::getStreamFromPool(/*isHighPriority=*/true, dev)),
          ingest_stream(c10::cuda::getStreamFromPool(/*isHighPriority=*/false, dev)),
          remote_write_stream(c10::cuda::getStreamFromPool(/*isHighPriority=*/false, dev)) {}
  };

  struct PipelinePacket {
    int32_t pkt_slot{-1};
    int32_t handoff_ring_slot{0};
    int32_t chosen_slow_bkt{1};
    int32_t real_slow_count{0};
    std::vector<int32_t> global_ids;
    std::vector<int32_t> local_slow_ids;
    std::vector<int64_t> tobs_ns;

    bool dst_slots_allocated{false};
    // Per fast_shard f: local_fast_ids, dst_global_slots, tobs_ns, and latest recorded p2p_event
    std::vector<std::vector<int32_t>> shard_local_fast_ids;
    std::vector<std::vector<int32_t>> shard_dst_slots;
    std::vector<std::vector<int64_t>> shard_tobs_ns;
    std::vector<cudaEvent_t> shard_prev_p2p_ev;
  };

  struct StageEdgeQueue {
    std::mutex mu;
    std::condition_variable not_empty_cv;
    std::condition_variable not_full_cv;
    std::deque<PipelinePacket> q;
    int32_t next_ring_slot{0};
    bool direct_p2p{true};

    // Allocated on downstream stage_{p+1}'s device: [kHandoffRingSlots][num_act_tensors]
    std::vector<torch::Tensor> ring_buffers[kHandoffRingSlots];
    // Allocated in pinned host memory when !direct_p2p: [kHandoffRingSlots][num_act_tensors]
    std::vector<torch::Tensor> h_bounce_buffers[kHandoffRingSlots];
    c10::cuda::CUDAStream dst_h2d_stream;
    cudaEvent_t src_d2h_ev[kHandoffRingSlots]{nullptr, nullptr};
    cudaEvent_t ready_ev[kHandoffRingSlots]{nullptr, nullptr};
    cudaEvent_t consumed_ev[kHandoffRingSlots]{nullptr, nullptr};

    explicit StageEdgeQueue(int32_t dst_dev)
        : dst_h2d_stream(c10::cuda::getStreamFromPool(/*isHighPriority=*/false, dst_dev)) {}
  };

  struct SlowStageExec {
    int32_t stage_idx{0};
    int32_t device_id{0};
    bool has_tsc_writes{false};
    int32_t dst_layer_offset{0};
    int32_t layer_count{0};

    std::shared_ptr<TemporalStateCache> slow_tsc;
    c10::cuda::CUDAStream exec_stream;
    c10::cuda::CUDAStream handoff_stream;
    at::cuda::MempoolId_t slow_mempool;
    std::unique_ptr<at::cuda::CUDAGraph> capturing_graph;

    std::map<int32_t, std::vector<std::unique_ptr<at::cuda::CUDAGraph>>> slow_bucket_slice_graphs;
    std::map<int32_t, int64_t> slow_bucket_slice_shadow_ns;
    std::vector<int32_t> sorted_slow_buckets;
    int64_t runtime_slice_shadow_ns{0};

    torch::Tensor slow_obs_static;
    std::vector<torch::Tensor> act_in_static;
    std::vector<torch::Tensor> act_out_static;
    torch::Tensor slow_pt_static;
    torch::Tensor slow_start_page_static;
    torch::Tensor slow_tobs_static;

    int64_t* h_slow_tobs{nullptr};
    cudaEvent_t tobs_h2d_done_event{nullptr};
    cudaEvent_t slice_start_event{nullptr};
    cudaEvent_t slice_done_event{nullptr};
    cudaEvent_t exec_done_event{nullptr};
    cudaEvent_t act_out_copied_event{nullptr};
    cudaEvent_t tsc_d2h_done_event{nullptr};

    // Per FastShard f: dedicated push stream + optional pinned host bounce buffer when !direct_p2p
    std::vector<c10::cuda::CUDAStream> p2p_push_streams;
    std::vector<void*> h_p2p_bounce;
    std::vector<cudaEvent_t> bounce_consumed_events;
    std::thread worker_thread;

    SlowStageExec(int32_t idx, int32_t dev, c10::cuda::CUDAStream exec_st)
        : stage_idx(idx),
          device_id(dev),
          exec_stream(exec_st),
          handoff_stream(c10::cuda::getStreamFromPool(/*isHighPriority=*/false, dev)) {}
  };

  struct SlowReplica {
    int32_t replica_idx{0};
    std::vector<int32_t> global_stream_ids;
    int32_t last_tsc_stage_idx{-1};

    c10::cuda::CUDAStream slow_ingest_stream;
    torch::Tensor per_stream_slow_obs;
    std::vector<cudaEvent_t> slow_caller_events;
    std::vector<cudaEvent_t> slow_ingest_events;
    std::vector<cudaEvent_t> slow_consumed_events;

    std::vector<std::unique_ptr<SlowStageExec>> stages;
    std::vector<std::unique_ptr<StageEdgeQueue>> edges;  // P - 1 edges

    // Pre-allocated packet slot pool with per-(pkt_slot, stage_idx, fast_shard) CUDA events
    std::mutex pkt_pool_mu;
    std::vector<int32_t> free_pkt_slots;
    // [pkt_pool_size][P][F] events created on fast_shards_[f]->device_id
    std::vector<std::vector<std::vector<cudaEvent_t>>> pkt_p2p_events;

    SlowReplica(int32_t idx, c10::cuda::CUDAStream ingest_st)
        : replica_idx(idx), slow_ingest_stream(ingest_st) {}
  };

  struct FastReq {
    int32_t local_id;
    int32_t global_id;
    int64_t t_obs_ns;
    int64_t wait_since_ns;
  };

  struct FastBatchPlan {
    int64_t remaining_budget_ns{0};
    int32_t upper_cap{1};
    int32_t take_fast_count{0};
    int32_t chosen_bucket{1};
  };

  FastBatchPlan plan_fast_batch(FastShard& fs, std::vector<FastReq>& pending_fast, int64_t now_ns);
  void execute_fast_batch(
      FastShard& fs,
      const std::vector<FastReq>& pending_fast,
      int32_t take_fast_count,
      int32_t chosen_bucket,
      int64_t now_ns,
      bool count_preemption);

  void colocated_shard_loop(int32_t shard_idx);
  void fast_disagg_shard_loop(int32_t shard_idx);
  void slow_disagg_stage_loop(int32_t replica_idx, int32_t stage_idx);

  int32_t pop_pkt_slot(SlowReplica& rep);
  void push_pkt_slot(SlowReplica& rep, int32_t slot);
  void abort_packet_writes(SlowReplica& rep, PipelinePacket& pkt);

  uint64_t mark_slow_published_locked(StreamIngress& st, int64_t slow_t_obs_ns, int64_t pub_now_ns) const {
    uint64_t dropped_stale_fast = 0;
    if (!st.has_published_slow && st.fast_pending) {
      if (pub_now_ns - st.fast_t_obs_ns > fast_deadline_ns_) {
        st.fast_pending = false;
        dropped_stale_fast = 1;
      } else {
        st.fast_wait_since_ns = pub_now_ns;
      }
    }
    st.last_pub_slow_t_obs_ns = slow_t_obs_ns;
    st.has_published_slow = true;
    return dropped_stale_fast;
  }

  bool has_p2p(int32_t dst_dev, int32_t src_dev) const {
    return dst_dev == src_dev || verified_p2p_pairs_.count({dst_dev, src_dev}) > 0;
  }

  int32_t num_streams_;
  int32_t num_slow_stages_;
  int32_t slow_batch_size_;
  int32_t max_fast_batch_size_;
  int32_t slow_frame_seq_len_;
  int64_t fast_deadline_ns_;
  int64_t init_slice_shadow_ns_;
  int64_t max_slow_aoi_ns_;
  bool disaggregated_{false};
  bool p2p_direct_{true};
  std::set<std::pair<int32_t, int32_t>> verified_p2p_pairs_;

  std::vector<int32_t> fast_shard_of_;
  std::vector<int32_t> local_fast_id_;
  std::vector<int32_t> slow_replica_of_;
  std::vector<int32_t> local_slow_id_;

  std::vector<std::unique_ptr<StreamIngress>> ingress_;
  std::vector<std::unique_ptr<FastShard>> fast_shards_;
  std::vector<std::unique_ptr<SlowReplica>> slow_replicas_;

  std::atomic<bool> running_{false};
  mutable std::mutex telem_mu_;
  SchedulerTelemetry telem_;
};

}  // namespace tempo
