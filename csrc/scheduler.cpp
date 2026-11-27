#include "scheduler.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#ifdef __linux__
#include <sys/prctl.h>
#endif

namespace tempo {

#define TS_SCHED_CUDA_CHECK(expr)                                              \
  do {                                                                         \
    cudaError_t err__ = (expr);                                                \
    if (err__ != cudaSuccess) {                                                \
      throw std::runtime_error(                                                \
          std::string("CUDA error at ") + __FILE__ + ":" +                     \
          std::to_string(__LINE__) + ": " + cudaGetErrorString(err__));        \
    }                                                                          \
  } while (0)

TimeAwareScheduler::TimeAwareScheduler(
    int32_t num_streams,
    const std::vector<std::vector<int32_t>>& slow_devices,
    const std::vector<int32_t>& fast_devices,
    int32_t slow_batch_size,
    int32_t max_fast_batch_size,
    int32_t slow_frame_seq_len,
    double fast_deadline_ms,
    double max_slice_ms,
    double max_slow_aoi_ms)
    : num_streams_(num_streams),
      num_slow_stages_(slow_devices.empty() ? 0 : static_cast<int32_t>(slow_devices[0].size())),
      slow_batch_size_(slow_batch_size),
      max_fast_batch_size_(max_fast_batch_size),
      slow_frame_seq_len_(slow_frame_seq_len),
      fast_deadline_ns_(static_cast<int64_t>(fast_deadline_ms * 1e6)),
      init_slice_shadow_ns_(static_cast<int64_t>(max_slice_ms * 1e6)),
      max_slow_aoi_ns_(static_cast<int64_t>(max_slow_aoi_ms * 1e6)) {
  if (num_streams_ <= 0) {
    throw std::invalid_argument("num_streams must be positive");
  }
  if (fast_devices.empty()) {
    throw std::invalid_argument("fast_devices must be non-empty");
  }
  if (!slow_devices.empty() && num_slow_stages_ <= 0) {
    throw std::invalid_argument("slow_devices pipeline stages must be non-empty");
  }
  if (static_cast<int32_t>(fast_devices.size()) > num_streams_ ||
      static_cast<int32_t>(slow_devices.size()) > num_streams_) {
    throw std::invalid_argument("Active replica/shard count cannot exceed num_streams");
  }
  for (const auto& pipe : slow_devices) {
    if (static_cast<int32_t>(pipe.size()) != num_slow_stages_) {
      throw std::invalid_argument("All slow replicas must have identical pipeline stage count");
    }
  }

  std::set<int32_t> fast_set(fast_devices.begin(), fast_devices.end());
  if (fast_set.size() != fast_devices.size()) {
    throw std::invalid_argument("fast_devices must not contain duplicate GPU IDs");
  }

  bool exact_colocated = (num_slow_stages_ == 1 && slow_devices.size() == fast_devices.size());
  if (exact_colocated) {
    for (size_t i = 0; i < fast_devices.size(); ++i) {
      if (slow_devices[i][0] != fast_devices[i]) {
        exact_colocated = false;
        break;
      }
    }
  }

  std::set<int32_t> slow_set;
  for (const auto& pipe : slow_devices) {
    for (int32_t d : pipe) {
      if (!slow_set.insert(d).second) {
        throw std::invalid_argument("slow_devices must not contain duplicate GPU IDs across replicas/stages");
      }
    }
  }

  if (!exact_colocated) {
    for (int32_t d : slow_set) {
      if (fast_set.count(d) > 0) {
        throw std::invalid_argument(
            "Slow and Fast device sets must either be identical 1-stage colocated GPUs or completely disjoint");
      }
    }
    disaggregated_ = true;
  } else {
    disaggregated_ = false;
    p2p_direct_ = false;
  }

  // Enable and hardware-verify P2P access across participating GPUs (guards against virtualized
  // dual-socket SYS/IOMMU topologies that report CanAccessPeer=1 yet drop BAR1 P2P TLPs to zero).
  std::set<int32_t> all_devs = fast_set;
  all_devs.insert(slow_set.begin(), slow_set.end());
  constexpr int32_t kProbeWords = 8;
  const uint64_t kSentinel = 0xA55A5AA5C33C3CC3ULL;
  std::vector<uint64_t> h_probe_src(kProbeWords, kSentinel);
  std::vector<uint64_t> h_probe_dst(kProbeWords, 0ULL);

  for (int32_t d_a : all_devs) {
    for (int32_t d_b : all_devs) {
      if (d_a == d_b) continue;
      int can_access = 0;
      TS_SCHED_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, d_a, d_b));
      if (!can_access) continue;

      {
        c10::cuda::CUDAGuard g_a(d_a);
        cudaError_t pe = cudaDeviceEnablePeerAccess(d_b, 0);
        if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) {
          TS_SCHED_CUDA_CHECK(pe);
        } else {
          cudaGetLastError();
        }
      }

      uint64_t* d_src = nullptr;
      uint64_t* d_dst = nullptr;
      bool p2p_valid = false;
      {
        c10::cuda::CUDAGuard g_b(d_b);
        TS_SCHED_CUDA_CHECK(cudaMalloc(&d_src, sizeof(uint64_t) * kProbeWords));
        TS_SCHED_CUDA_CHECK(
            cudaMemcpy(d_src, h_probe_src.data(), sizeof(uint64_t) * kProbeWords, cudaMemcpyHostToDevice));
        TS_SCHED_CUDA_CHECK(cudaDeviceSynchronize());
      }
      {
        c10::cuda::CUDAGuard g_a(d_a);
        cudaStream_t probe_st = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, d_a).stream();
        TS_SCHED_CUDA_CHECK(cudaMalloc(&d_dst, sizeof(uint64_t) * kProbeWords));
        TS_SCHED_CUDA_CHECK(cudaMemsetAsync(d_dst, 0, sizeof(uint64_t) * kProbeWords, probe_st));
        cudaError_t cp_err = cudaMemcpyPeerAsync(
            d_dst, d_a, d_src, d_b, sizeof(uint64_t) * kProbeWords, probe_st);
        if (cp_err == cudaSuccess && cudaStreamSynchronize(probe_st) == cudaSuccess) {
          std::fill(h_probe_dst.begin(), h_probe_dst.end(), 0ULL);
          TS_SCHED_CUDA_CHECK(
              cudaMemcpy(h_probe_dst.data(), d_dst, sizeof(uint64_t) * kProbeWords, cudaMemcpyDeviceToHost));
          p2p_valid = std::all_of(
              h_probe_dst.begin(), h_probe_dst.end(), [&](uint64_t v) { return v == kSentinel; });
        } else {
          cudaGetLastError();
        }
        TS_SCHED_CUDA_CHECK(cudaFree(d_dst));
        if (!p2p_valid) {
          cudaDeviceDisablePeerAccess(d_b);
          cudaGetLastError();
        }
      }
      {
        c10::cuda::CUDAGuard g_b(d_b);
        TS_SCHED_CUDA_CHECK(cudaFree(d_src));
      }
      if (p2p_valid) {
        verified_p2p_pairs_.insert({d_a, d_b});
      }
    }
  }

  if (disaggregated_) {
    p2p_direct_ = !slow_devices.empty();
    for (const auto& pipe : slow_devices) {
      for (int32_t p = 0; p + 1 < num_slow_stages_; ++p) {
        if (!has_p2p(pipe[p + 1], pipe[p])) p2p_direct_ = false;
      }
      for (int32_t s_dev : pipe) {
        for (int32_t f_dev : fast_devices) {
          if (!has_p2p(f_dev, s_dev)) p2p_direct_ = false;
        }
      }
    }
  }

  const int32_t F = static_cast<int32_t>(fast_devices.size());
  const int32_t R = static_cast<int32_t>(slow_devices.size());

  fast_shards_.reserve(F);
  for (int32_t f = 0; f < F; ++f) {
    fast_shards_.push_back(std::make_unique<FastShard>(f, fast_devices[f]));
  }

  slow_replicas_.reserve(R);
  for (int32_t r = 0; r < R; ++r) {
    const int32_t entry_dev = slow_devices[r][0];
    c10::cuda::CUDAStream ingest_st =
        disaggregated_
            ? c10::cuda::getStreamFromPool(/*isHighPriority=*/false, entry_dev)
            : fast_shards_[r]->ingest_stream;
    auto rep = std::make_unique<SlowReplica>(r, ingest_st);
    for (int32_t p = 0; p < num_slow_stages_; ++p) {
      const int32_t s_dev = slow_devices[r][p];
      c10::cuda::CUDAStream exec_st =
          disaggregated_
              ? c10::cuda::getStreamFromPool(/*isHighPriority=*/false, s_dev)
              : fast_shards_[r]->exec_stream;
      auto stg = std::make_unique<SlowStageExec>(p, s_dev, exec_st);
      stg->runtime_slice_shadow_ns = disaggregated_ ? 0 : init_slice_shadow_ns_;
      {
        c10::cuda::CUDAGuard sg(s_dev);
        stg->slow_mempool = at::cuda::graph_pool_handle();
        TS_SCHED_CUDA_CHECK(cudaEventCreate(&stg->slice_start_event));
        TS_SCHED_CUDA_CHECK(cudaEventCreate(&stg->slice_done_event));
        TS_SCHED_CUDA_CHECK(cudaEventCreateWithFlags(&stg->tobs_h2d_done_event, cudaEventDisableTiming));
        TS_SCHED_CUDA_CHECK(cudaEventCreateWithFlags(&stg->act_out_copied_event, cudaEventDisableTiming));
        TS_SCHED_CUDA_CHECK(cudaEventCreateWithFlags(&stg->tsc_d2h_done_event, cudaEventDisableTiming));
        TS_SCHED_CUDA_CHECK(cudaMallocHost(&stg->h_slow_tobs, sizeof(int64_t) * slow_batch_size_));
        TS_SCHED_CUDA_CHECK(cudaMallocHost(&stg->h_slow_stream_ids, sizeof(int32_t) * slow_batch_size_));
        TS_SCHED_CUDA_CHECK(cudaMallocHost(&stg->h_slow_seqlens, sizeof(int32_t) * slow_batch_size_));
      }
      if (disaggregated_) {
        stg->p2p_push_streams.reserve(F);
        stg->h_p2p_bounce.assign(F, nullptr);
        stg->bounce_consumed_events.assign(F, nullptr);
        for (int32_t f = 0; f < F; ++f) {
          c10::cuda::CUDAGuard fg(fast_devices[f]);
          stg->p2p_push_streams.push_back(
              c10::cuda::getStreamFromPool(/*isHighPriority=*/false, fast_devices[f]));
          TS_SCHED_CUDA_CHECK(
              cudaEventCreateWithFlags(&stg->bounce_consumed_events[f], cudaEventDisableTiming));
        }
      }
      rep->stages.push_back(std::move(stg));
    }

    for (int32_t p = 0; p + 1 < num_slow_stages_; ++p) {
      const int32_t src_dev = slow_devices[r][p];
      const int32_t dst_dev = slow_devices[r][p + 1];
      auto edge = std::make_unique<StageEdgeQueue>(dst_dev);
      edge->direct_p2p = has_p2p(dst_dev, src_dev);
      for (int32_t k = 0; k < kHandoffRingSlots; ++k) {
        {
          c10::cuda::CUDAGuard g_src(src_dev);
          TS_SCHED_CUDA_CHECK(cudaEventCreateWithFlags(&edge->src_d2h_ev[k], cudaEventDisableTiming));
          if (edge->direct_p2p) {
            TS_SCHED_CUDA_CHECK(cudaEventCreateWithFlags(&edge->ready_ev[k], cudaEventDisableTiming));
          }
        }
        {
          c10::cuda::CUDAGuard g_dst(dst_dev);
          if (!edge->direct_p2p) {
            TS_SCHED_CUDA_CHECK(cudaEventCreateWithFlags(&edge->ready_ev[k], cudaEventDisableTiming));
          }
          TS_SCHED_CUDA_CHECK(cudaEventCreateWithFlags(&edge->consumed_ev[k], cudaEventDisableTiming));
        }
      }
      rep->edges.push_back(std::move(edge));
    }

    if (disaggregated_) {
      // At most P microbatches per replica (1 per pipeline stage GPU) may be concurrently active.
      const int32_t pool_sz = num_slow_stages_;
      rep->pkt_p2p_events.resize(pool_sz);
      for (int32_t slot = 0; slot < pool_sz; ++slot) {
        rep->free_pkt_slots.push_back(pool_sz - 1 - slot);
        rep->pkt_p2p_events[slot].resize(num_slow_stages_);
        for (int32_t p = 0; p < num_slow_stages_; ++p) {
          rep->pkt_p2p_events[slot][p].resize(F, nullptr);
          for (int32_t f = 0; f < F; ++f) {
            c10::cuda::CUDAGuard fg(fast_devices[f]);
            TS_SCHED_CUDA_CHECK(
                cudaEventCreateWithFlags(&rep->pkt_p2p_events[slot][p][f], cudaEventDisableTiming));
          }
        }
      }
    }
    slow_replicas_.push_back(std::move(rep));
  }

  // Deterministic round-robin stream routing across Fast shards and Slow replicas
  fast_shard_of_.resize(num_streams_);
  local_fast_id_.resize(num_streams_);
  slow_replica_of_.resize(num_streams_);
  local_slow_id_.resize(num_streams_);
  ingress_.reserve(num_streams_);
  const int64_t init_ns = now_ns();

  for (int32_t s = 0; s < num_streams_; ++s) {
    auto ing = std::make_unique<StreamIngress>();
    ing->slow_seq_len = slow_frame_seq_len_;
    ing->last_pub_slow_t_obs_ns = init_ns;
    ingress_.push_back(std::move(ing));

    const int32_t f = s % F;
    fast_shard_of_[s] = f;
    local_fast_id_[s] = static_cast<int32_t>(fast_shards_[f]->global_stream_ids.size());
    fast_shards_[f]->global_stream_ids.push_back(s);

    if (R > 0) {
      const int32_t r = s % R;
      slow_replica_of_[s] = r;
      local_slow_id_[s] = static_cast<int32_t>(slow_replicas_[r]->global_stream_ids.size());
      slow_replicas_[r]->global_stream_ids.push_back(s);
    } else {
      slow_replica_of_[s] = -1;
      local_slow_id_[s] = -1;
    }
  }

  for (int32_t f = 0; f < F; ++f) {
    auto& fs = *fast_shards_[f];
    const int32_t n_f = static_cast<int32_t>(fs.global_stream_ids.size());
    c10::cuda::CUDAGuard fg(fs.device_id);
    fs.fast_mempool = at::cuda::graph_pool_handle();
    fs.fast_caller_events.resize(n_f);
    fs.fast_ingest_events.resize(n_f);
    fs.fast_consumed_events.resize(n_f);
    for (int32_t i = 0; i < n_f; ++i) {
      TS_SCHED_CUDA_CHECK(cudaEventCreateWithFlags(&fs.fast_caller_events[i], cudaEventDisableTiming));
      TS_SCHED_CUDA_CHECK(cudaEventCreateWithFlags(&fs.fast_ingest_events[i], cudaEventDisableTiming));
      TS_SCHED_CUDA_CHECK(cudaEventCreateWithFlags(&fs.fast_consumed_events[i], cudaEventDisableTiming));
    }
    TS_SCHED_CUDA_CHECK(cudaEventCreate(&fs.fast_start_event));
    TS_SCHED_CUDA_CHECK(cudaEventCreate(&fs.fast_done_event));
    TS_SCHED_CUDA_CHECK(cudaMallocHost(&fs.h_fast_local_ids, sizeof(int32_t) * max_fast_batch_size_));
    TS_SCHED_CUDA_CHECK(cudaMallocHost(&fs.h_fast_tobs, sizeof(int64_t) * max_fast_batch_size_));
    TS_SCHED_CUDA_CHECK(cudaMallocHost(&fs.h_fast_slow_tobs, sizeof(int64_t) * max_fast_batch_size_));
    TS_SCHED_CUDA_CHECK(cudaMalloc(&fs.d_fast_local_ids, sizeof(int32_t) * max_fast_batch_size_));
    TS_SCHED_CUDA_CHECK(cudaMalloc(&fs.d_fast_tobs, sizeof(int64_t) * max_fast_batch_size_));
    TS_SCHED_CUDA_CHECK(cudaMalloc(&fs.d_fast_slow_tobs, sizeof(int64_t) * max_fast_batch_size_));
  }

  for (int32_t r = 0; r < R; ++r) {
    auto& rep = *slow_replicas_[r];
    const int32_t n_r = static_cast<int32_t>(rep.global_stream_ids.size());
    c10::cuda::CUDAGuard sg(rep.stages[0]->device_id);
    rep.slow_caller_events.resize(n_r);
    rep.slow_ingest_events.resize(n_r);
    rep.slow_consumed_events.resize(n_r);
    for (int32_t i = 0; i < n_r; ++i) {
      TS_SCHED_CUDA_CHECK(cudaEventCreateWithFlags(&rep.slow_caller_events[i], cudaEventDisableTiming));
      TS_SCHED_CUDA_CHECK(cudaEventCreateWithFlags(&rep.slow_ingest_events[i], cudaEventDisableTiming));
      TS_SCHED_CUDA_CHECK(cudaEventCreateWithFlags(&rep.slow_consumed_events[i], cudaEventDisableTiming));
    }
  }
}

TimeAwareScheduler::~TimeAwareScheduler() {
  stop();
  abort_active_captures();

  for (auto& rep : slow_replicas_) {
    {
      c10::cuda::CUDAGuard sg(rep->stages[0]->device_id);
      for (auto ev : rep->slow_caller_events) cudaEventDestroy(ev);
      for (auto ev : rep->slow_ingest_events) cudaEventDestroy(ev);
      for (auto ev : rep->slow_consumed_events) cudaEventDestroy(ev);
    }
    for (int32_t p = 0; p + 1 < num_slow_stages_; ++p) {
      auto& edge = *rep->edges[p];
      for (int32_t k = 0; k < kHandoffRingSlots; ++k) {
        {
          c10::cuda::CUDAGuard g_src(rep->stages[p]->device_id);
          if (edge.src_d2h_ev[k]) cudaEventDestroy(edge.src_d2h_ev[k]);
          if (edge.direct_p2p && edge.ready_ev[k]) cudaEventDestroy(edge.ready_ev[k]);
        }
        {
          c10::cuda::CUDAGuard g_dst(rep->stages[p + 1]->device_id);
          if (!edge.direct_p2p && edge.ready_ev[k]) cudaEventDestroy(edge.ready_ev[k]);
          if (edge.consumed_ev[k]) cudaEventDestroy(edge.consumed_ev[k]);
        }
      }
    }
    for (auto& slot_vec : rep->pkt_p2p_events) {
      for (auto& stg_vec : slot_vec) {
        for (size_t f = 0; f < stg_vec.size(); ++f) {
          if (stg_vec[f]) {
            c10::cuda::CUDAGuard fg(fast_shards_[f]->device_id);
            cudaEventDestroy(stg_vec[f]);
          }
        }
      }
    }
    for (auto& stg : rep->stages) {
      {
        c10::cuda::CUDAGuard sg(stg->device_id);
        stg->slow_bucket_slice_graphs.clear();
        if (stg->slice_start_event) cudaEventDestroy(stg->slice_start_event);
        if (stg->slice_done_event) cudaEventDestroy(stg->slice_done_event);
        if (stg->tobs_h2d_done_event) cudaEventDestroy(stg->tobs_h2d_done_event);
        if (stg->act_out_copied_event) cudaEventDestroy(stg->act_out_copied_event);
        if (stg->tsc_d2h_done_event) cudaEventDestroy(stg->tsc_d2h_done_event);
        if (stg->h_slow_tobs) cudaFreeHost(stg->h_slow_tobs);
        if (stg->h_slow_stream_ids) cudaFreeHost(stg->h_slow_stream_ids);
        if (stg->h_slow_seqlens) cudaFreeHost(stg->h_slow_seqlens);
        for (void* ptr : stg->h_p2p_bounce) {
          if (ptr) cudaFreeHost(ptr);
        }
      }
      for (size_t f = 0; f < stg->bounce_consumed_events.size(); ++f) {
        if (stg->bounce_consumed_events[f]) {
          c10::cuda::CUDAGuard fg(fast_shards_[f]->device_id);
          cudaEventDestroy(stg->bounce_consumed_events[f]);
        }
      }
    }
  }

  for (auto& fs : fast_shards_) {
    c10::cuda::CUDAGuard fg(fs->device_id);
    fs->fast_bucket_graphs.clear();
    for (auto ev : fs->fast_caller_events) cudaEventDestroy(ev);
    for (auto ev : fs->fast_ingest_events) cudaEventDestroy(ev);
    for (auto ev : fs->fast_consumed_events) cudaEventDestroy(ev);
    if (fs->fast_start_event) cudaEventDestroy(fs->fast_start_event);
    if (fs->fast_done_event) cudaEventDestroy(fs->fast_done_event);
    if (fs->h_fast_local_ids) cudaFreeHost(fs->h_fast_local_ids);
    if (fs->h_fast_tobs) cudaFreeHost(fs->h_fast_tobs);
    if (fs->h_fast_slow_tobs) cudaFreeHost(fs->h_fast_slow_tobs);
    if (fs->d_fast_local_ids) cudaFree(fs->d_fast_local_ids);
    if (fs->d_fast_tobs) cudaFree(fs->d_fast_tobs);
    if (fs->d_fast_slow_tobs) cudaFree(fs->d_fast_slow_tobs);
  }
}

int32_t TimeAwareScheduler::fast_shard_num_streams(int32_t shard_idx) const {
  if (shard_idx < 0 || shard_idx >= static_cast<int32_t>(fast_shards_.size())) {
    throw std::out_of_range("shard_idx out of range");
  }
  return static_cast<int32_t>(fast_shards_[shard_idx]->global_stream_ids.size());
}

int32_t TimeAwareScheduler::slow_replica_num_streams(int32_t replica_idx) const {
  if (replica_idx < 0 || replica_idx >= static_cast<int32_t>(slow_replicas_.size())) {
    throw std::out_of_range("replica_idx out of range");
  }
  return static_cast<int32_t>(slow_replicas_[replica_idx]->global_stream_ids.size());
}

void TimeAwareScheduler::init_fast_shard(
    int32_t shard_idx,
    std::shared_ptr<TemporalStateCache> tsc,
    std::shared_ptr<RTCHorizonRing> rtc,
    torch::Tensor fast_obs_static,
    torch::Tensor fast_prefix_static,
    torch::Tensor fast_prefix_valid_static,
    torch::Tensor fast_pt_static,
    torch::Tensor fast_seqlens_static,
    torch::Tensor fast_tnow_static,
    torch::Tensor fast_out_static) {
  if (shard_idx < 0 || shard_idx >= static_cast<int32_t>(fast_shards_.size())) {
    throw std::out_of_range("shard_idx out of range");
  }
  auto& fs = *fast_shards_[shard_idx];
  if (!tsc || !rtc || tsc->device_id() != fs.device_id || rtc->device_id() != fs.device_id) {
    throw std::invalid_argument("init_fast_shard tsc/rtc device mismatch");
  }
  fs.tsc = std::move(tsc);
  fs.rtc = std::move(rtc);
  fs.fast_obs_static = std::move(fast_obs_static);
  fs.fast_prefix_static = std::move(fast_prefix_static);
  fs.fast_prefix_valid_static = std::move(fast_prefix_valid_static);
  fs.fast_pt_static = std::move(fast_pt_static);
  fs.fast_seqlens_static = std::move(fast_seqlens_static);
  fs.fast_tnow_static = std::move(fast_tnow_static);
  fs.fast_out_static = std::move(fast_out_static);

  c10::cuda::CUDAGuard fg(fs.device_id);
  std::vector<int64_t> fast_shape = fs.fast_obs_static.sizes().vec();
  fast_shape[0] = static_cast<int64_t>(fs.global_stream_ids.size());
  fs.per_stream_fast_obs = torch::zeros(fast_shape, fs.fast_obs_static.options());
}

void TimeAwareScheduler::init_slow_stage(
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
    torch::Tensor slow_tobs_static,
    torch::Tensor slow_stream_ids_static,
    torch::Tensor slow_seqlens_static) {
  if (replica_idx < 0 || replica_idx >= static_cast<int32_t>(slow_replicas_.size())) {
    throw std::out_of_range("replica_idx out of range");
  }
  if (stage_idx < 0 || stage_idx >= num_slow_stages_) {
    throw std::out_of_range("stage_idx out of range");
  }
  auto& rep = *slow_replicas_[replica_idx];
  auto& stg = *rep.stages[stage_idx];

  if (slow_tsc && slow_tsc->device_id() != stg.device_id) {
    throw std::invalid_argument("init_slow_stage slow_tsc device mismatch");
  }
  stg.slow_tsc = std::move(slow_tsc);
  stg.has_tsc_writes = (stg.slow_tsc != nullptr);
  stg.dst_layer_offset = dst_layer_offset;
  stg.layer_count = layer_count;
  if (stg.has_tsc_writes) {
    rep.last_tsc_stage_idx = std::max(rep.last_tsc_stage_idx, stage_idx);
  }

  stg.slow_obs_static = std::move(slow_obs_static);
  stg.act_in_static = std::move(act_in_static);
  stg.act_out_static = std::move(act_out_static);
  stg.slow_pt_static = std::move(slow_pt_static);
  stg.slow_start_page_static = std::move(slow_start_page_static);
  stg.slow_tobs_static = std::move(slow_tobs_static);
  stg.slow_stream_ids_static = std::move(slow_stream_ids_static);
  stg.slow_seqlens_static = std::move(slow_seqlens_static);

  if (stage_idx == 0) {
    c10::cuda::CUDAGuard sg(stg.device_id);
    std::vector<int64_t> slow_shape = stg.slow_obs_static.sizes().vec();
    slow_shape[0] = static_cast<int64_t>(rep.global_stream_ids.size());
    rep.per_stream_slow_obs = torch::zeros(slow_shape, stg.slow_obs_static.options());
  } else {
    auto& edge = *rep.edges[stage_idx - 1];
    c10::cuda::CUDAGuard sg(stg.device_id);
    for (int32_t k = 0; k < kHandoffRingSlots; ++k) {
      edge.ring_buffers[k].clear();
      edge.ring_buffers[k].reserve(stg.act_in_static.size());
      edge.h_bounce_buffers[k].clear();
      for (const auto& t : stg.act_in_static) {
        edge.ring_buffers[k].push_back(torch::zeros_like(t));
        if (!edge.direct_p2p) {
          edge.h_bounce_buffers[k].push_back(
              torch::empty(t.sizes(), torch::TensorOptions().dtype(t.scalar_type()).device(torch::kCPU).pinned_memory(true)));
        }
      }
    }
  }

  if (disaggregated_ && stg.has_tsc_writes) {
    c10::cuda::CUDAGuard sg(stg.device_id);
    const size_t bounce_bytes = stg.slow_tsc->stage_slice_bounce_bytes(slow_batch_size_, slow_frame_seq_len_);
    for (size_t f = 0; f < fast_shards_.size(); ++f) {
      if (!has_p2p(fast_shards_[f]->device_id, stg.device_id) && bounce_bytes > 0) {
        if (stg.h_p2p_bounce[f] != nullptr) {
          cudaFreeHost(stg.h_p2p_bounce[f]);
          stg.h_p2p_bounce[f] = nullptr;
        }
        TS_SCHED_CUDA_CHECK(cudaMallocHost(&stg.h_p2p_bounce[f], bounce_bytes));
      }
    }
  }
}

int64_t TimeAwareScheduler::fast_exec_cuda_stream_raw(int32_t shard_idx) const {
  if (shard_idx < 0 || shard_idx >= static_cast<int32_t>(fast_shards_.size())) {
    throw std::out_of_range("shard_idx out of range");
  }
  return reinterpret_cast<int64_t>(fast_shards_[shard_idx]->exec_stream.stream());
}

int64_t TimeAwareScheduler::slow_exec_cuda_stream_raw(int32_t replica_idx, int32_t stage_idx) const {
  if (replica_idx < 0 || replica_idx >= static_cast<int32_t>(slow_replicas_.size()) ||
      stage_idx < 0 || stage_idx >= num_slow_stages_) {
    throw std::out_of_range("replica_idx or stage_idx out of range");
  }
  return reinterpret_cast<int64_t>(slow_replicas_[replica_idx]->stages[stage_idx]->exec_stream.stream());
}

void TimeAwareScheduler::begin_slow_slice_capture(int32_t replica_idx, int32_t stage_idx, int32_t /*slow_bucket*/) {
  auto& stg = *slow_replicas_.at(replica_idx)->stages.at(stage_idx);
  c10::cuda::CUDAGuard sg(stg.device_id);
  c10::cuda::CUDAStreamGuard guard(stg.exec_stream);
  TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(stg.exec_stream.stream()));
  stg.capturing_graph = std::make_unique<at::cuda::CUDAGraph>();
  stg.capturing_graph->capture_begin(stg.slow_mempool, cudaStreamCaptureModeThreadLocal);
}

void TimeAwareScheduler::end_slow_slice_capture(int32_t replica_idx, int32_t stage_idx, int32_t slow_bucket) {
  auto& stg = *slow_replicas_.at(replica_idx)->stages.at(stage_idx);
  c10::cuda::CUDAGuard sg(stg.device_id);
  c10::cuda::CUDAStreamGuard guard(stg.exec_stream);
  stg.capturing_graph->capture_end();
  stg.slow_bucket_slice_graphs[slow_bucket].push_back(std::move(stg.capturing_graph));

  stg.sorted_slow_buckets.clear();
  for (const auto& kv : stg.slow_bucket_slice_graphs) {
    stg.sorted_slow_buckets.push_back(kv.first);
  }
}

std::vector<double> TimeAwareScheduler::profile_slow_slices(
    int32_t replica_idx, int32_t stage_idx, int32_t slow_bucket) {
  auto& stg = *slow_replicas_.at(replica_idx)->stages.at(stage_idx);
  c10::cuda::CUDAGuard sg(stg.device_id);
  c10::cuda::CUDAStreamGuard guard(stg.exec_stream);
  cudaStream_t st = stg.exec_stream.stream();
  auto it = stg.slow_bucket_slice_graphs.find(slow_bucket);
  if (it == stg.slow_bucket_slice_graphs.end()) {
    throw std::invalid_argument("profile_slow_slices unknown slow_bucket");
  }
  auto& graphs = it->second;
  std::vector<double> durations_ms;
  durations_ms.reserve(graphs.size());
  int64_t observed_max_ns = 0;

  for (auto& g : graphs) {
    g->replay();
  }
  TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(st));

  cudaEvent_t ev0, ev1;
  TS_SCHED_CUDA_CHECK(cudaEventCreate(&ev0));
  TS_SCHED_CUDA_CHECK(cudaEventCreate(&ev1));

  for (auto& g : graphs) {
    double best_ms = std::numeric_limits<double>::max();
    for (int r = 0; r < 4; ++r) {
      TS_SCHED_CUDA_CHECK(cudaEventRecord(ev0, st));
      g->replay();
      TS_SCHED_CUDA_CHECK(cudaEventRecord(ev1, st));
      TS_SCHED_CUDA_CHECK(cudaEventSynchronize(ev1));
      float ms = 0.0f;
      TS_SCHED_CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
      best_ms = std::min(best_ms, static_cast<double>(ms));
    }
    durations_ms.push_back(best_ms);
    observed_max_ns = std::max(observed_max_ns, static_cast<int64_t>(best_ms * 1e6));
  }

  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);

  if (!disaggregated_) {
    stg.slow_bucket_slice_shadow_ns[slow_bucket] = observed_max_ns;
    stg.runtime_slice_shadow_ns = stg.slow_bucket_slice_shadow_ns.begin()->second;
  } else {
    stg.slow_bucket_slice_shadow_ns[slow_bucket] = 0;
    stg.runtime_slice_shadow_ns = 0;
  }
  return durations_ms;
}

void TimeAwareScheduler::begin_fast_bucket_capture(int32_t shard_idx, int32_t /*bucket_size*/) {
  auto& fs = *fast_shards_.at(shard_idx);
  c10::cuda::CUDAGuard fg(fs.device_id);
  c10::cuda::CUDAStreamGuard guard(fs.exec_stream);
  TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(fs.exec_stream.stream()));
  fs.capturing_graph = std::make_unique<at::cuda::CUDAGraph>();
  fs.capturing_graph->capture_begin(fs.fast_mempool, cudaStreamCaptureModeThreadLocal);
}

double TimeAwareScheduler::end_fast_bucket_capture(int32_t shard_idx, int32_t bucket_size) {
  auto& fs = *fast_shards_.at(shard_idx);
  c10::cuda::CUDAGuard fg(fs.device_id);
  c10::cuda::CUDAStreamGuard guard(fs.exec_stream);
  cudaStream_t st = fs.exec_stream.stream();
  fs.capturing_graph->capture_end();
  for (int i = 0; i < 2; ++i) {
    fs.capturing_graph->replay();
  }
  TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(st));

  cudaEvent_t ev0, ev1;
  TS_SCHED_CUDA_CHECK(cudaEventCreate(&ev0));
  TS_SCHED_CUDA_CHECK(cudaEventCreate(&ev1));
  double best_ms = std::numeric_limits<double>::max();
  for (int i = 0; i < 6; ++i) {
    if (i > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(6));
    }
    TS_SCHED_CUDA_CHECK(cudaEventRecord(ev0, st));
    fs.capturing_graph->replay();
    TS_SCHED_CUDA_CHECK(cudaEventRecord(ev1, st));
    TS_SCHED_CUDA_CHECK(cudaEventSynchronize(ev1));
    float elapsed_ms = 0.0f;
    TS_SCHED_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev0, ev1));
    best_ms = std::min(best_ms, static_cast<double>(elapsed_ms));
  }
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);

  fs.fast_bucket_graphs[bucket_size] = std::move(fs.capturing_graph);
  const int64_t calib_ns = static_cast<int64_t>(best_ms * 1e6);
  fs.fast_bucket_exec_ns[bucket_size] = calib_ns;
  fs.runtime_bucket_exec_ns[bucket_size] = calib_ns + 500000;

  fs.sorted_buckets.clear();
  for (const auto& kv : fs.fast_bucket_graphs) {
    fs.sorted_buckets.push_back(kv.first);
  }
  return best_ms;
}

void TimeAwareScheduler::abort_active_captures() noexcept {
  auto abort_stream = [](int32_t dev_id,
                         c10::cuda::CUDAStream& st_obj,
                         at::cuda::MempoolId_t pool_id,
                         std::unique_ptr<at::cuda::CUDAGraph>& g) noexcept {
    if (!g) return;
    c10::cuda::CUDAGuard dg(dev_id);
    c10::cuda::CUDAStreamGuard sg(st_obj);
    cudaStream_t st = st_obj.stream();
    cudaGetLastError();
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(st, &status) == cudaSuccess && status == cudaStreamCaptureStatusInvalidated) {
      cudaGraph_t abandoned = nullptr;
      cudaStreamEndCapture(st, &abandoned);
      if (abandoned) cudaGraphDestroy(abandoned);
      cudaGetLastError();
      cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
    }
    try {
      g->capture_end();
    } catch (...) {
      c10::cuda::CUDACachingAllocator::endAllocateToPool(dev_id, pool_id);
    }
    cudaGetLastError();
    g.reset();
  };
  for (auto& rep : slow_replicas_) {
    for (auto& stg : rep->stages) {
      abort_stream(stg->device_id, stg->exec_stream, stg->slow_mempool, stg->capturing_graph);
    }
  }
  for (auto& fs : fast_shards_) {
    abort_stream(fs->device_id, fs->exec_stream, fs->fast_mempool, fs->capturing_graph);
  }
}

std::pair<int32_t, int32_t> TimeAwareScheduler::busy_slow_prompt_rows(int32_t stream_id) const {
  if (stream_id < 0 || stream_id >= num_streams_) {
    throw std::out_of_range("busy_slow_prompt_rows: stream_id out of range");
  }
  std::lock_guard<std::mutex> lk(ingress_[stream_id]->mu);
  const int32_t busy_pending =
      ingress_[stream_id]->slow_pending ? ingress_[stream_id]->pending_prompt_row : -1;
  const int32_t busy_inflight =
      ingress_[stream_id]->slow_inflight ? ingress_[stream_id]->inflight_prompt_row : -1;
  return {busy_pending, busy_inflight};
}

void TimeAwareScheduler::submit_slow_obs(
    int32_t stream_id,
    int64_t t_obs_ns,
    const torch::Tensor& obs,
    int32_t seq_len,
    int32_t prompt_row) {
  if (slow_replicas_.empty()) {
    throw std::logic_error("submit_slow_obs called on Fast-only scheduler without local Slow stage");
  }
  if (stream_id < 0 || stream_id >= num_streams_) {
    throw std::out_of_range("stream_id out of range");
  }
  if (!obs.is_contiguous() || obs.scalar_type() != torch::kBFloat16) {
    throw std::invalid_argument("submit_slow_obs requires contiguous bfloat16 tensor");
  }
  if (seq_len < 1 || seq_len > slow_frame_seq_len_) {
    throw std::invalid_argument("submit_slow_obs seq_len out of range [1, slow_frame_seq_len]");
  }
  const auto& f_tsc = fast_shards_[fast_shard_of_[stream_id]]->tsc;
  if (f_tsc && seq_len < slow_frame_seq_len_ && f_tsc->max_seq_len() > slow_frame_seq_len_) {
    throw std::invalid_argument(
        "submit_slow_obs: variable seq_len < slow_frame_seq_len is unsupported when window_frames > 1");
  }
  if (prompt_row < 0) {
    throw std::invalid_argument("submit_slow_obs prompt_row must be >= 0");
  }
  const int32_t r = slow_replica_of_[stream_id];
  const int32_t local_s = local_slow_id_[stream_id];
  auto& rep = *slow_replicas_[r];
  const int32_t dst_dev = rep.stages[0]->device_id;
  auto expected_slow_shape = rep.per_stream_slow_obs.sizes().slice(1);
  if (obs.sizes() != expected_slow_shape) {
    throw std::invalid_argument(
        "submit_slow_obs shape mismatch: expected " + c10::str(expected_slow_shape) + ", got " + c10::str(obs.sizes()));
  }
  const size_t bytes = rep.per_stream_slow_obs[local_s].nbytes();

  std::lock_guard<std::mutex> lk(ingress_[stream_id]->mu);
  if (obs.is_cuda() && obs.get_device() == dst_dev) {
    c10::cuda::CUDAGuard dev_guard(dst_dev);
    cudaStream_t caller_st = at::cuda::getCurrentCUDAStream(dst_dev).stream();
    TS_SCHED_CUDA_CHECK(cudaEventRecord(rep.slow_caller_events[local_s], caller_st));
    c10::cuda::CUDAStreamGuard guard(rep.slow_ingest_stream);
    TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(rep.slow_ingest_stream.stream(), rep.slow_caller_events[local_s], 0));
    c10::cuda::CUDACachingAllocator::recordStream(obs.storage().data_ptr(), rep.slow_ingest_stream);
    TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(rep.slow_ingest_stream.stream(), rep.slow_consumed_events[local_s], 0));
    TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
        rep.per_stream_slow_obs[local_s].data_ptr(),
        obs.data_ptr(),
        bytes,
        cudaMemcpyDeviceToDevice,
        rep.slow_ingest_stream.stream()));
    TS_SCHED_CUDA_CHECK(cudaEventRecord(rep.slow_ingest_events[local_s], rep.slow_ingest_stream.stream()));
    if (caller_st != rep.slow_ingest_stream.stream()) {
      TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(caller_st, rep.slow_ingest_events[local_s], 0));
    }
  } else if (obs.is_cuda() && has_p2p(dst_dev, obs.get_device())) {
    const int32_t src_dev = obs.get_device();
    {
      c10::cuda::CUDAGuard src_guard(src_dev);
      TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream(src_dev).stream()));
    }
    c10::cuda::CUDAGuard dst_guard(dst_dev);
    c10::cuda::CUDAStreamGuard guard(rep.slow_ingest_stream);
    TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(rep.slow_ingest_stream.stream(), rep.slow_consumed_events[local_s], 0));
    TS_SCHED_CUDA_CHECK(cudaMemcpyPeerAsync(
        rep.per_stream_slow_obs[local_s].data_ptr(),
        dst_dev,
        obs.data_ptr(),
        src_dev,
        bytes,
        rep.slow_ingest_stream.stream()));
    TS_SCHED_CUDA_CHECK(cudaEventRecord(rep.slow_ingest_events[local_s], rep.slow_ingest_stream.stream()));
    TS_SCHED_CUDA_CHECK(cudaEventSynchronize(rep.slow_ingest_events[local_s]));
  } else {
    torch::Tensor cpu_obs = obs.is_cuda() ? obs.cpu() : obs;
    c10::cuda::CUDAGuard dst_guard(dst_dev);
    c10::cuda::CUDAStreamGuard guard(rep.slow_ingest_stream);
    TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(rep.slow_ingest_stream.stream(), rep.slow_consumed_events[local_s], 0));
    TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
        rep.per_stream_slow_obs[local_s].data_ptr(),
        cpu_obs.data_ptr(),
        bytes,
        cudaMemcpyHostToDevice,
        rep.slow_ingest_stream.stream()));
    TS_SCHED_CUDA_CHECK(cudaEventRecord(rep.slow_ingest_events[local_s], rep.slow_ingest_stream.stream()));
    TS_SCHED_CUDA_CHECK(cudaEventSynchronize(rep.slow_ingest_events[local_s]));
  }

  if (ingress_[stream_id]->slow_pending) {
    std::lock_guard<std::mutex> tlk(telem_mu_);
    telem_.slow_coalesced_drops++;
  }
  ingress_[stream_id]->pending_prompt_row = prompt_row;
  ingress_[stream_id]->slow_pending = true;
  ingress_[stream_id]->slow_t_obs_ns = t_obs_ns;
  ingress_[stream_id]->slow_seq_len = seq_len;
}

void TimeAwareScheduler::ingest_fast_obs_locked(int32_t stream_id, const torch::Tensor& fast_obs) {
  const int32_t f = fast_shard_of_[stream_id];
  const int32_t local_s = local_fast_id_[stream_id];
  auto& fs = *fast_shards_[f];
  const int32_t dst_dev = fs.device_id;
  auto expected_fast_shape = fs.per_stream_fast_obs.sizes().slice(1);
  if (fast_obs.sizes() != expected_fast_shape) {
    throw std::invalid_argument(
        "submit_fast_obs shape mismatch: expected " + c10::str(expected_fast_shape) + ", got " +
        c10::str(fast_obs.sizes()));
  }
  const size_t bytes = fs.per_stream_fast_obs[local_s].nbytes();

  if (fast_obs.is_cuda() && fast_obs.get_device() == dst_dev) {
    c10::cuda::CUDAGuard dev_guard(dst_dev);
    cudaStream_t caller_st = at::cuda::getCurrentCUDAStream(dst_dev).stream();
    TS_SCHED_CUDA_CHECK(cudaEventRecord(fs.fast_caller_events[local_s], caller_st));
    c10::cuda::CUDAStreamGuard guard(fs.ingest_stream);
    TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(fs.ingest_stream.stream(), fs.fast_caller_events[local_s], 0));
    c10::cuda::CUDACachingAllocator::recordStream(fast_obs.storage().data_ptr(), fs.ingest_stream);
    TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(fs.ingest_stream.stream(), fs.fast_consumed_events[local_s], 0));
    TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
        fs.per_stream_fast_obs[local_s].data_ptr(),
        fast_obs.data_ptr(),
        bytes,
        cudaMemcpyDeviceToDevice,
        fs.ingest_stream.stream()));
    TS_SCHED_CUDA_CHECK(cudaEventRecord(fs.fast_ingest_events[local_s], fs.ingest_stream.stream()));
    if (caller_st != fs.ingest_stream.stream()) {
      TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(caller_st, fs.fast_ingest_events[local_s], 0));
    }
  } else if (fast_obs.is_cuda() && has_p2p(dst_dev, fast_obs.get_device())) {
    const int32_t src_dev = fast_obs.get_device();
    {
      c10::cuda::CUDAGuard src_guard(src_dev);
      TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream(src_dev).stream()));
    }
    c10::cuda::CUDAGuard dst_guard(dst_dev);
    c10::cuda::CUDAStreamGuard guard(fs.ingest_stream);
    TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(fs.ingest_stream.stream(), fs.fast_consumed_events[local_s], 0));
    TS_SCHED_CUDA_CHECK(cudaMemcpyPeerAsync(
        fs.per_stream_fast_obs[local_s].data_ptr(),
        dst_dev,
        fast_obs.data_ptr(),
        src_dev,
        bytes,
        fs.ingest_stream.stream()));
    TS_SCHED_CUDA_CHECK(cudaEventRecord(fs.fast_ingest_events[local_s], fs.ingest_stream.stream()));
    TS_SCHED_CUDA_CHECK(cudaEventSynchronize(fs.fast_ingest_events[local_s]));
  } else {
    torch::Tensor cpu_fobs = fast_obs.is_cuda() ? fast_obs.cpu() : fast_obs;
    c10::cuda::CUDAGuard dst_guard(dst_dev);
    c10::cuda::CUDAStreamGuard guard(fs.ingest_stream);
    TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(fs.ingest_stream.stream(), fs.fast_consumed_events[local_s], 0));
    TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
        fs.per_stream_fast_obs[local_s].data_ptr(),
        cpu_fobs.data_ptr(),
        bytes,
        cudaMemcpyHostToDevice,
        fs.ingest_stream.stream()));
    TS_SCHED_CUDA_CHECK(cudaEventRecord(fs.fast_ingest_events[local_s], fs.ingest_stream.stream()));
    TS_SCHED_CUDA_CHECK(cudaEventSynchronize(fs.fast_ingest_events[local_s]));
  }
}

void TimeAwareScheduler::submit_fast_obs(int32_t stream_id, int64_t t_obs_ns, const torch::Tensor& fast_obs) {
  if (stream_id < 0 || stream_id >= num_streams_) {
    throw std::out_of_range("stream_id out of range");
  }
  if (!fast_obs.is_contiguous() || fast_obs.scalar_type() != torch::kBFloat16) {
    throw std::invalid_argument("submit_fast_obs requires contiguous bfloat16 tensor");
  }
  std::lock_guard<std::mutex> lk(ingress_[stream_id]->mu);
  ingest_fast_obs_locked(stream_id, fast_obs);
  if (ingress_[stream_id]->fast_pending) {
    std::lock_guard<std::mutex> tlk(telem_mu_);
    telem_.fast_coalesced_drops++;
  } else {
    ingress_[stream_id]->fast_wait_since_ns = t_obs_ns;
  }
  ingress_[stream_id]->fast_pending = true;
  ingress_[stream_id]->fast_t_obs_ns = t_obs_ns;
}

void TimeAwareScheduler::submit_fast_obs_batch(
    const std::vector<int32_t>& stream_ids, int64_t t_obs_ns, const torch::Tensor& fast_obs_batch) {
  const int32_t B = static_cast<int32_t>(stream_ids.size());
  if (B <= 0) return;
  if (!fast_obs_batch.is_contiguous() || fast_obs_batch.scalar_type() != torch::kBFloat16 ||
      fast_obs_batch.dim() < 1 || fast_obs_batch.size(0) != B) {
    throw std::invalid_argument("submit_fast_obs_batch requires contiguous bfloat16 tensor with leading dim B");
  }
  std::vector<int32_t> sorted_unique = stream_ids;
  for (int32_t s : sorted_unique) {
    if (s < 0 || s >= num_streams_) {
      throw std::out_of_range("stream_id out of range in submit_fast_obs_batch");
    }
  }
  std::sort(sorted_unique.begin(), sorted_unique.end());
  sorted_unique.erase(std::unique(sorted_unique.begin(), sorted_unique.end()), sorted_unique.end());

  std::vector<std::unique_lock<std::mutex>> locks;
  locks.reserve(sorted_unique.size());
  for (int32_t s : sorted_unique) {
    locks.emplace_back(ingress_[s]->mu);
  }

  uint64_t drops = 0;
  for (int32_t i = 0; i < B; ++i) {
    const int32_t s = stream_ids[i];
    ingest_fast_obs_locked(s, fast_obs_batch[i]);
    if (ingress_[s]->fast_pending) {
      drops++;
    } else {
      ingress_[s]->fast_wait_since_ns = t_obs_ns;
    }
    ingress_[s]->fast_pending = true;
    ingress_[s]->fast_t_obs_ns = t_obs_ns;
  }
  locks.clear();

  if (drops > 0) {
    std::lock_guard<std::mutex> tlk(telem_mu_);
    telem_.fast_coalesced_drops += drops;
  }
}

bool TimeAwareScheduler::submit_remote_hidden(int32_t stream_id, int64_t t_obs_ns, const torch::Tensor& hidden) {
  if (!slow_replicas_.empty()) {
    throw std::logic_error("submit_remote_hidden requires a Fast-only scheduler without local Slow replicas");
  }
  if (stream_id < 0 || stream_id >= num_streams_) {
    throw std::out_of_range("stream_id out of range");
  }
  if (!hidden.is_contiguous() || hidden.scalar_type() != torch::kBFloat16) {
    throw std::invalid_argument("submit_remote_hidden requires contiguous bfloat16 tensor");
  }

  const int32_t f = fast_shard_of_[stream_id];
  const int32_t local_s = local_fast_id_[stream_id];
  auto& fs = *fast_shards_[f];

  std::lock_guard<std::mutex> wlk(fs.remote_write_mu);
  if (!fs.tsc || fs.tsc->kv_mode()) {
    throw std::logic_error("submit_remote_hidden requires an initialized hidden-mode TemporalStateCache");
  }
  const int32_t feat_dim = fs.tsc->num_kv_heads() * fs.tsc->head_dim();
  const bool valid_2d = (hidden.dim() == 2 && hidden.size(0) == slow_frame_seq_len_ && hidden.size(1) == feat_dim);
  const bool valid_3d =
      (hidden.dim() == 3 && hidden.size(0) == 1 && hidden.size(1) == slow_frame_seq_len_ && hidden.size(2) == feat_dim);
  if (!valid_2d && !valid_3d) {
    throw std::invalid_argument("submit_remote_hidden tensor shape mismatch with (frame_seq_len, feat_dim)");
  }

  {
    std::lock_guard<std::mutex> ilk(ingress_[stream_id]->mu);
    if (t_obs_ns <= ingress_[stream_id]->min_valid_slow_t_obs_ns ||
        (ingress_[stream_id]->has_published_slow && t_obs_ns <= ingress_[stream_id]->last_pub_slow_t_obs_ns)) {
      std::lock_guard<std::mutex> tlk(telem_mu_);
      telem_.slow_coalesced_drops++;
      return false;
    }
  }

  c10::cuda::CUDAGuard dev_guard(fs.device_id);
  c10::cuda::CUDAStreamGuard st_guard(fs.remote_write_stream);
  cudaStream_t raw_w = fs.remote_write_stream.stream();

  torch::Tensor gpu_h;
  if (hidden.is_cuda() && hidden.get_device() == fs.device_id) {
    cudaStream_t caller_st = at::cuda::getCurrentCUDAStream(fs.device_id).stream();
    if (caller_st != raw_w) {
      TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(caller_st));
    }
    gpu_h = hidden.view({1, slow_frame_seq_len_, feat_dim});
  } else {
    torch::Tensor cpu_h = hidden.is_cuda() ? hidden.cpu() : hidden;
    gpu_h = torch::empty(
        {1, slow_frame_seq_len_, feat_dim},
        torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, fs.device_id));
    TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
        gpu_h.data_ptr(), cpu_h.data_ptr(), cpu_h.nbytes(), cudaMemcpyHostToDevice, raw_w));
  }

  const int32_t gs = fs.tsc->begin_write(local_s, slow_frame_seq_len_, raw_w);
  try {
    fs.tsc->write_hidden_state(gs, gpu_h, t_obs_ns, raw_w);
    fs.tsc->publish_write(local_s, gs, t_obs_ns, slow_frame_seq_len_, raw_w);
    TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(raw_w));
  } catch (...) {
    fs.tsc->abort_write_batch({gs}, raw_w);
    throw;
  }

  const int64_t pub_now = now_ns();
  uint64_t dropped_fast = 0;
  {
    std::lock_guard<std::mutex> ilk(ingress_[stream_id]->mu);
    dropped_fast = mark_slow_published_locked(*ingress_[stream_id], t_obs_ns, pub_now);
  }
  {
    std::lock_guard<std::mutex> tlk(telem_mu_);
    telem_.slow_batches_completed++;
    telem_.fast_coalesced_drops += dropped_fast;
    telem_.slow_latency_ms.push_back(static_cast<double>(pub_now - t_obs_ns) * 1e-6);
  }
  return true;
}

int32_t TimeAwareScheduler::pop_pkt_slot(SlowReplica& rep) {
  while (running_.load(std::memory_order_relaxed)) {
    {
      std::lock_guard<std::mutex> lk(rep.pkt_pool_mu);
      if (!rep.free_pkt_slots.empty()) {
        int32_t s = rep.free_pkt_slots.back();
        rep.free_pkt_slots.pop_back();
        return s;
      }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(15));
  }
  return -1;
}

void TimeAwareScheduler::push_pkt_slot(SlowReplica& rep, int32_t slot) {
  if (slot < 0) return;
  std::lock_guard<std::mutex> lk(rep.pkt_pool_mu);
  rep.free_pkt_slots.push_back(slot);
}

void TimeAwareScheduler::abort_packet_writes(SlowReplica& /*rep*/, PipelinePacket& pkt) {
  if (pkt.dst_slots_allocated) {
    for (size_t f = 0; f < fast_shards_.size(); ++f) {
      if (!pkt.shard_dst_slots[f].empty()) {
        c10::cuda::CUDAGuard fg(fast_shards_[f]->device_id);
        if (pkt.shard_prev_p2p_ev[f] != nullptr) {
          cudaEventSynchronize(pkt.shard_prev_p2p_ev[f]);
        }
        fast_shards_[f]->tsc->abort_write_batch(pkt.shard_dst_slots[f], nullptr);
      }
    }
    pkt.dst_slots_allocated = false;
  }
  for (int32_t g_s : pkt.global_ids) {
    std::lock_guard<std::mutex> lk(ingress_[g_s]->mu);
    ingress_[g_s]->slow_inflight = false;
    ingress_[g_s]->inflight_prompt_row = -1;
  }
}

void TimeAwareScheduler::start() {
  if (running_.exchange(true)) return;
  for (const auto& fs : fast_shards_) {
    if (!fs->tsc || !fs->rtc || fs->fast_bucket_graphs.empty()) {
      running_.store(false);
      throw std::runtime_error("Cannot start scheduler before initializing and capturing all Fast shards");
    }
  }
  for (const auto& rep : slow_replicas_) {
    if (rep->last_tsc_stage_idx < 0) {
      running_.store(false);
      throw std::runtime_error("At least one Slow pipeline stage must write to TemporalStateCache");
    }
    for (const auto& stg : rep->stages) {
      if (stg->slow_bucket_slice_graphs.empty()) {
        running_.store(false);
        throw std::runtime_error("Cannot start scheduler before capturing all Slow stage CUDA graphs");
      }
    }
  }

  if (disaggregated_) {
    for (size_t f = 0; f < fast_shards_.size(); ++f) {
      fast_shards_[f]->worker_thread =
          std::thread(&TimeAwareScheduler::fast_disagg_shard_loop, this, static_cast<int32_t>(f));
    }
    for (size_t r = 0; r < slow_replicas_.size(); ++r) {
      for (int32_t p = 0; p < num_slow_stages_; ++p) {
        slow_replicas_[r]->stages[p]->worker_thread = std::thread(
            &TimeAwareScheduler::slow_disagg_stage_loop, this, static_cast<int32_t>(r), p);
      }
    }
  } else {
    for (size_t k = 0; k < fast_shards_.size(); ++k) {
      fast_shards_[k]->worker_thread =
          std::thread(&TimeAwareScheduler::colocated_shard_loop, this, static_cast<int32_t>(k));
    }
  }
}

void TimeAwareScheduler::stop() {
  if (!running_.exchange(false, std::memory_order_seq_cst)) return;
  {
    std::lock_guard<std::mutex> clk(clock_mu_);
  }
  clock_cv_.notify_all();

  for (auto& rep : slow_replicas_) {
    for (auto& edge : rep->edges) {
      {
        std::lock_guard<std::mutex> lk(edge->mu);
      }
      edge->not_empty_cv.notify_all();
      edge->not_full_cv.notify_all();
    }
  }

  for (auto& fs : fast_shards_) {
    if (fs->worker_thread.joinable()) {
      fs->worker_thread.join();
    }
  }
  for (auto& rep : slow_replicas_) {
    for (auto& stg : rep->stages) {
      if (stg->worker_thread.joinable()) {
        stg->worker_thread.join();
      }
    }
  }

  // Drain any in-flight packets remaining in pipeline queues and abort unpublished write slots
  for (auto& rep : slow_replicas_) {
    for (auto& stg : rep->stages) {
      c10::cuda::CUDAGuard sg(stg->device_id);
      TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(stg->exec_stream.stream()));
      TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(stg->handoff_stream.stream()));
      for (size_t f = 0; f < stg->p2p_push_streams.size(); ++f) {
        c10::cuda::CUDAGuard fg(fast_shards_[f]->device_id);
        TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(stg->p2p_push_streams[f].stream()));
      }
    }
    for (auto& edge : rep->edges) {
      std::lock_guard<std::mutex> lk(edge->mu);
      while (!edge->q.empty()) {
        auto pkt = std::move(edge->q.front());
        edge->q.pop_front();
        abort_packet_writes(*rep, pkt);
        push_pkt_slot(*rep, pkt.pkt_slot);
      }
    }
    {
      c10::cuda::CUDAGuard sg(rep->stages[0]->device_id);
      TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(rep->slow_ingest_stream.stream()));
    }
  }

  for (auto& fs : fast_shards_) {
    std::lock_guard<std::mutex> wlk(fs->remote_write_mu);
    c10::cuda::CUDAGuard fg(fs->device_id);
    TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(fs->ingest_stream.stream()));
    TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(fs->remote_write_stream.stream()));
  }
}

void TimeAwareScheduler::reset_all_streams(c10::optional<torch::Tensor> init_pose) {
  if (running_.load(std::memory_order_seq_cst)) {
    throw std::logic_error("Stop the engine before calling reset()");
  }
  const int32_t A = fast_shards_[0]->rtc->action_dim();
  torch::Tensor pose_cpu;
  bool per_stream_pose = false;
  if (init_pose.has_value() && init_pose->defined()) {
    pose_cpu = init_pose->detach().to(torch::kCPU, torch::kFloat32).contiguous();
    if (pose_cpu.dim() == 1 && pose_cpu.size(0) == A) {
      per_stream_pose = false;
    } else if (pose_cpu.dim() == 2 && pose_cpu.size(0) == num_streams_ && pose_cpu.size(1) == A) {
      per_stream_pose = true;
    } else {
      throw std::invalid_argument(
          "reset(q_init) expected shape [" + std::to_string(A) + "] or [" + std::to_string(num_streams_) + ", " +
          std::to_string(A) + "]");
    }
  }

  resume_clock();
  const int64_t now = now_ns();
  for (int32_t s = 0; s < num_streams_; ++s) {
    std::lock_guard<std::mutex> lk(ingress_[s]->mu);
    ingress_[s]->slow_pending = false;
    ingress_[s]->slow_inflight = false;
    ingress_[s]->inflight_prompt_row = -1;
    ingress_[s]->pending_prompt_row = -1;
    ingress_[s]->slow_t_obs_ns = 0;
    ingress_[s]->expected_slow_t_obs_ns = 0;
    ingress_[s]->fast_pending = false;
    ingress_[s]->fast_t_obs_ns = 0;
    ingress_[s]->fast_wait_since_ns = 0;
    ingress_[s]->last_committed_fast_t_obs_ns = 0;
    ingress_[s]->min_valid_slow_t_obs_ns = now;
    ingress_[s]->last_pub_slow_t_obs_ns = now;
    ingress_[s]->has_published_slow = false;
  }

  const size_t F = fast_shards_.size();
  for (size_t f = 0; f < F; ++f) {
    auto& fs = *fast_shards_[f];
    const int32_t n_f = static_cast<int32_t>(fs.global_stream_ids.size());
    for (int32_t ls = 0; ls < n_f; ++ls) {
      fs.tsc->reset_stream(ls);
    }
    if (!pose_cpu.defined()) {
      fs.rtc->reset_all(c10::nullopt);
    } else if (!per_stream_pose) {
      fs.rtc->reset_all(pose_cpu);
    } else {
      auto shard_pose = torch::empty({n_f, A}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
      float* dst = shard_pose.data_ptr<float>();
      const float* src = pose_cpu.data_ptr<float>();
      for (int32_t ls = 0; ls < n_f; ++ls) {
        std::memcpy(dst + static_cast<size_t>(ls) * A, src + static_cast<size_t>(fs.global_stream_ids[ls]) * A, sizeof(float) * A);
      }
      fs.rtc->reset_all(shard_pose);
    }
  }

  if (disaggregated_) {
    for (auto& rep : slow_replicas_) {
      const int32_t n_r = static_cast<int32_t>(rep->global_stream_ids.size());
      for (auto& stg : rep->stages) {
        if (stg->slow_tsc) {
          for (int32_t lr = 0; lr < n_r; ++lr) {
            stg->slow_tsc->reset_stream(lr);
          }
        }
      }
    }
  }

  reset_telemetry();
}

void TimeAwareScheduler::reset_telemetry() {
  std::lock_guard<std::mutex> lk(telem_mu_);
  telem_ = SchedulerTelemetry{};
  telem_.fast_ttac_ms.reserve(32768);
  telem_.slow_latency_ms.reserve(4096);
}

void TimeAwareScheduler::reset_rtc_stats() {
  for (auto& fs : fast_shards_) {
    if (fs->rtc) {
      fs->rtc->reset_stats();
    }
  }
}

RTCStats TimeAwareScheduler::rtc_stats() const {
  RTCStats agg{};
  for (const auto& fs : fast_shards_) {
    if (fs->rtc) {
      RTCStats s = fs->rtc->stats();
      agg.total_commits += s.total_commits;
      agg.soft_overruns += s.soft_overruns;
      agg.hard_underruns += s.hard_underruns;
    }
  }
  return agg;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> TimeAwareScheduler::sample_actions_at(
    int64_t t_query_ns, bool return_cpu) {
  const size_t F = fast_shards_.size();
  if (F == 1) {
    auto [act, fa, sa, u] = fast_shards_[0]->rtc->sample_at(t_query_ns);
    if (return_cpu) {
      return {act.cpu(), fa.cpu(), sa.cpu(), u};
    }
    return {std::move(act), std::move(fa), std::move(sa), u};
  }

  std::vector<PendingRTCSample> pending;
  pending.reserve(F);
  for (size_t f = 0; f < F; ++f) {
    pending.push_back(fast_shards_[f]->rtc->launch_sample_async(t_query_ns));
  }

  const int32_t A = fast_shards_[0]->rtc->action_dim();
  auto cpu_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto full_act_cpu = torch::empty({num_streams_, A}, cpu_f32);
  auto full_fa_cpu = torch::empty({num_streams_}, cpu_f32);
  auto full_sa_cpu = torch::empty({num_streams_}, cpu_f32);

  float* dst_act = full_act_cpu.data_ptr<float>();
  float* dst_fa = full_fa_cpu.data_ptr<float>();
  float* dst_sa = full_sa_cpu.data_ptr<float>();
  int64_t total_underruns = 0;

  for (size_t f = 0; f < F; ++f) {
    auto [act_gpu, fa_gpu, sa_gpu, u_f] =
        fast_shards_[f]->rtc->finish_sample_async(std::move(pending[f]));
    total_underruns += u_f;

    auto act_c = act_gpu.cpu();
    auto fa_c = fa_gpu.cpu();
    auto sa_c = sa_gpu.cpu();
    const float* src_act = act_c.data_ptr<float>();
    const float* src_fa = fa_c.data_ptr<float>();
    const float* src_sa = sa_c.data_ptr<float>();

    const auto& g_ids = fast_shards_[f]->global_stream_ids;
    for (size_t i = 0; i < g_ids.size(); ++i) {
      const int32_t g_s = g_ids[i];
      std::memcpy(dst_act + static_cast<size_t>(g_s) * A, src_act + i * A, sizeof(float) * A);
      dst_fa[g_s] = src_fa[i];
      dst_sa[g_s] = src_sa[i];
    }
  }

  if (return_cpu) {
    return {std::move(full_act_cpu), std::move(full_fa_cpu), std::move(full_sa_cpu), total_underruns};
  }
  const int32_t primary_dev = fast_shards_[0]->device_id;
  c10::cuda::CUDAGuard fg(primary_dev);
  auto gpu_dev = torch::Device(torch::kCUDA, primary_dev);
  return {full_act_cpu.to(gpu_dev), full_fa_cpu.to(gpu_dev), full_sa_cpu.to(gpu_dev), total_underruns};
}

SchedulerTelemetry TimeAwareScheduler::telemetry() const {
  std::lock_guard<std::mutex> lk(telem_mu_);
  return telem_;
}

void TimeAwareScheduler::pause_clock() {
  std::lock_guard<std::mutex> lk(clock_mu_);
  if (!clock_paused_.load(std::memory_order_relaxed)) {
    clock_pause_start_ns_ = steady_now_ns();
    clock_ever_paused_.store(true, std::memory_order_release);
    clock_paused_.store(true, std::memory_order_release);
  }
}

void TimeAwareScheduler::resume_clock() {
  {
    std::lock_guard<std::mutex> lk(clock_mu_);
    if (clock_paused_.load(std::memory_order_relaxed)) {
      clock_paused_accum_ns_ += std::max<int64_t>(0, steady_now_ns() - clock_pause_start_ns_);
      clock_paused_.store(false, std::memory_order_release);
    }
  }
  clock_cv_.notify_all();
}

bool TimeAwareScheduler::is_clock_paused() const {
  return clock_paused_.load(std::memory_order_acquire);
}

int64_t TimeAwareScheduler::now_ns() const {
  if (!clock_ever_paused_.load(std::memory_order_acquire)) {
    return steady_now_ns();
  }
  std::lock_guard<std::mutex> lk(clock_mu_);
  const int64_t raw = clock_paused_.load(std::memory_order_relaxed) ? clock_pause_start_ns_ : steady_now_ns();
  return raw - clock_paused_accum_ns_;
}

void TimeAwareScheduler::wait_if_clock_paused() {
  if (!clock_paused_.load(std::memory_order_acquire)) return;
  std::unique_lock<std::mutex> lk(clock_mu_);
  clock_cv_.wait(lk, [&]() {
    return !clock_paused_.load(std::memory_order_relaxed) || !running_.load(std::memory_order_relaxed);
  });
}

void TimeAwareScheduler::wait_until_logical_ns(int64_t target_ns) {
  if (!clock_ever_paused_.load(std::memory_order_acquire)) return;
  while (running_.load(std::memory_order_relaxed)) {
    int64_t rem_ns = 0;
    {
      std::unique_lock<std::mutex> lk(clock_mu_);
      if (!clock_paused_.load(std::memory_order_relaxed)) {
        rem_ns = target_ns - (steady_now_ns() - clock_paused_accum_ns_);
        if (rem_ns <= 0) return;
      } else {
        clock_cv_.wait(lk, [&]() {
          return !clock_paused_.load(std::memory_order_relaxed) || !running_.load(std::memory_order_relaxed);
        });
        continue;
      }
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(std::min<int64_t>(rem_ns, 200000)));
  }
}

TimeAwareScheduler::FastBatchPlan TimeAwareScheduler::plan_fast_batch(
    FastShard& fs, std::vector<FastReq>& pending_fast, int64_t now_ns) {
  const int32_t local_n = static_cast<int32_t>(fs.global_stream_ids.size());
  std::sort(pending_fast.begin(), pending_fast.end(), [](const FastReq& a, const FastReq& b) {
    return a.wait_since_ns < b.wait_since_ns;
  });

  const int64_t earliest_deadline_ns = pending_fast[0].wait_since_ns + fast_deadline_ns_;
  const int64_t remaining_budget_ns = earliest_deadline_ns - now_ns;

  int32_t max_deadline_bucket = 0;
  for (int32_t bkt : fs.sorted_buckets) {
    if (bkt <= max_fast_batch_size_ && fs.fast_bucket_exec_ns[bkt] <= fast_deadline_ns_) {
      max_deadline_bucket = bkt;
    }
  }
  if (max_deadline_bucket == 0) {
    double best_ns_per_req = std::numeric_limits<double>::max();
    max_deadline_bucket = fs.sorted_buckets.front();
    for (int32_t bkt : fs.sorted_buckets) {
      if (bkt > max_fast_batch_size_) continue;
      const int32_t eff_n = std::min(bkt, local_n);
      const double rate = static_cast<double>(fs.runtime_bucket_exec_ns[bkt]) / std::max(1, eff_n);
      if (rate < best_ns_per_req) {
        best_ns_per_req = rate;
        max_deadline_bucket = bkt;
      }
    }
  }

  const int32_t upper_cap = std::min(local_n, max_deadline_bucket);
  int32_t rem_feasible = 0;
  for (int32_t bkt : fs.sorted_buckets) {
    if (bkt <= max_fast_batch_size_ && bkt <= max_deadline_bucket &&
        fs.runtime_bucket_exec_ns[bkt] <= remaining_budget_ns) {
      rem_feasible = std::min(bkt, upper_cap);
    }
  }

  int32_t batch_cap = upper_cap;
  if (rem_feasible > 0 && rem_feasible < static_cast<int32_t>(pending_fast.size()) &&
      rem_feasible < upper_cap) {
    const int32_t rem_count =
        std::min(static_cast<int32_t>(pending_fast.size()) - rem_feasible, upper_cap);
    int32_t rem_bkt = fs.sorted_buckets.back();
    for (int32_t bkt : fs.sorted_buckets) {
      if (bkt >= rem_count) {
        rem_bkt = bkt;
        break;
      }
    }
    const int64_t next_batch_deadline_ns = pending_fast[rem_feasible].wait_since_ns + fast_deadline_ns_;
    if (now_ns + fs.runtime_bucket_exec_ns[rem_feasible] + fs.runtime_bucket_exec_ns[rem_bkt] <=
        next_batch_deadline_ns) {
      batch_cap = rem_feasible;
    }
  }

  const int32_t take_fast_count = std::min(static_cast<int32_t>(pending_fast.size()), batch_cap);
  int32_t chosen_bucket = fs.sorted_buckets.back();
  for (int32_t bkt : fs.sorted_buckets) {
    if (bkt >= take_fast_count) {
      chosen_bucket = bkt;
      break;
    }
  }
  return FastBatchPlan{remaining_budget_ns, upper_cap, take_fast_count, chosen_bucket};
}

void TimeAwareScheduler::execute_fast_batch(
    FastShard& fs,
    const std::vector<FastReq>& pending_fast,
    int32_t take_fast_count,
    int32_t chosen_bucket,
    int64_t now_ns,
    bool count_preemption) {
  cudaStream_t raw_exec = fs.exec_stream.stream();
  const size_t fast_elem_bytes = fs.per_stream_fast_obs[0].nbytes();
  const int64_t t_step_start_ns = steady_now_ns();
  std::vector<int32_t> batch_local_streams(chosen_bucket, pending_fast[0].local_id);
  std::vector<int64_t> batch_tnow(chosen_bucket, now_ns);
  TS_SCHED_CUDA_CHECK(cudaEventRecord(fs.fast_start_event, raw_exec));

  for (int32_t i = 0; i < take_fast_count; ++i) {
    const int32_t l_s = pending_fast[i].local_id;
    const int32_t g_s = pending_fast[i].global_id;
    batch_local_streams[i] = l_s;
    fs.h_fast_local_ids[i] = l_s;
    {
      std::lock_guard<std::mutex> lk(ingress_[g_s]->mu);
      fs.h_fast_tobs[i] = ingress_[g_s]->fast_t_obs_ns;
      ingress_[g_s]->fast_pending = false;
      TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(raw_exec, fs.fast_ingest_events[l_s], 0));
      TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
          static_cast<char*>(fs.fast_obs_static.data_ptr()) + i * fast_elem_bytes,
          fs.per_stream_fast_obs[l_s].data_ptr(),
          fast_elem_bytes,
          cudaMemcpyDeviceToDevice,
          raw_exec));
      TS_SCHED_CUDA_CHECK(cudaEventRecord(fs.fast_consumed_events[l_s], raw_exec));
    }
  }
  for (int32_t i = take_fast_count; i < chosen_bucket; ++i) {
    TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
        static_cast<char*>(fs.fast_obs_static.data_ptr()) + i * fast_elem_bytes,
        fs.fast_obs_static.data_ptr(),
        fast_elem_bytes,
        cudaMemcpyDeviceToDevice,
        raw_exec));
  }

  std::vector<int64_t> slow_tobs_out;
  std::vector<int32_t> pinned_slots;
  fs.tsc->acquire_read_batch(
      /*reader_id=*/0,
      batch_local_streams,
      batch_tnow,
      fs.fast_pt_static.data_ptr<int32_t>(),
      fs.fast_seqlens_static.data_ptr<int32_t>(),
      fs.fast_tnow_static.data_ptr<int64_t>(),
      slow_tobs_out,
      pinned_slots,
      raw_exec);

  for (int32_t i = 0; i < take_fast_count; ++i) {
    fs.h_fast_slow_tobs[i] = slow_tobs_out[i];
  }

  TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
      fs.d_fast_local_ids,
      fs.h_fast_local_ids,
      sizeof(int32_t) * take_fast_count,
      cudaMemcpyHostToDevice,
      raw_exec));
  TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
      fs.d_fast_tobs, fs.h_fast_tobs, sizeof(int64_t) * take_fast_count, cudaMemcpyHostToDevice, raw_exec));
  TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
      fs.d_fast_slow_tobs,
      fs.h_fast_slow_tobs,
      sizeof(int64_t) * take_fast_count,
      cudaMemcpyHostToDevice,
      raw_exec));

  auto prefix_slice = fs.fast_prefix_static.slice(0, 0, chosen_bucket);
  auto prefix_valid_slice = fs.fast_prefix_valid_static.slice(0, 0, chosen_bucket);
  fs.rtc->extract_frozen_prefix(
      fs.d_fast_local_ids,
      fs.d_fast_tobs,
      fs.d_fast_slow_tobs,
      take_fast_count,
      prefix_slice,
      prefix_valid_slice,
      raw_exec);

  fs.fast_bucket_graphs[chosen_bucket]->replay();
  fs.tsc->release_read_batch(/*reader_id=*/0, pinned_slots, raw_exec);
  TS_SCHED_CUDA_CHECK(cudaEventRecord(fs.fast_done_event, raw_exec));

  while (true) {
    cudaError_t st = cudaEventQuery(fs.fast_done_event);
    if (st == cudaSuccess) break;
    if (st != cudaErrorNotReady) TS_SCHED_CUDA_CHECK(st);
    std::this_thread::sleep_for(std::chrono::microseconds(15));
  }
  fs.tsc->reclaim_retired_reads();
  const int64_t t_pre_wait_ns = steady_now_ns();

  float fast_ms = 0.0f;
  TS_SCHED_CUDA_CHECK(cudaEventElapsedTime(&fast_ms, fs.fast_start_event, fs.fast_done_event));
  const int64_t gpu_exec_ns = static_cast<int64_t>(fast_ms * 1e6);
  wait_until_logical_ns(now_ns + gpu_exec_ns);
  if (!running_.load(std::memory_order_relaxed)) return;
  const int64_t t_post_wait_ns = steady_now_ns();
  const int64_t t_done_ns = this->now_ns();
  auto out_slice = fs.fast_out_static.slice(0, 0, take_fast_count);
  fs.rtc->commit_chunks(
      fs.d_fast_local_ids,
      fs.d_fast_tobs,
      fs.d_fast_slow_tobs,
      t_done_ns,
      fast_deadline_ns_,
      out_slice,
      take_fast_count,
      raw_exec);
  for (int32_t i = 0; i < take_fast_count; ++i) {
    const int32_t g_s = pending_fast[i].global_id;
    std::lock_guard<std::mutex> lk(ingress_[g_s]->mu);
    ingress_[g_s]->last_committed_fast_t_obs_ns =
        std::max(ingress_[g_s]->last_committed_fast_t_obs_ns, fs.h_fast_tobs[i]);
  }

  const int64_t obs_fast_ns = (t_pre_wait_ns - t_step_start_ns) + (steady_now_ns() - t_post_wait_ns);
  int64_t& est_ref = fs.runtime_bucket_exec_ns[chosen_bucket];
  if (obs_fast_ns > est_ref) {
    est_ref = (est_ref * 2 + obs_fast_ns) / 3;
  } else {
    est_ref = (est_ref * 15 + obs_fast_ns) / 16;
  }

  {
    std::lock_guard<std::mutex> tlk(telem_mu_);
    telem_.fast_batches_launched++;
    telem_.fast_requests_served += static_cast<uint64_t>(take_fast_count);
    if (count_preemption) {
      telem_.fast_preemptions++;
    }
    for (int32_t i = 0; i < take_fast_count; ++i) {
      telem_.fast_ttac_ms.push_back(static_cast<double>(t_done_ns - fs.h_fast_tobs[i]) * 1e-6);
    }
  }
}

void TimeAwareScheduler::colocated_shard_loop(int32_t shard_idx) {
#ifdef __linux__
  prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
#endif
  auto& fs = *fast_shards_[shard_idx];
  auto& rep = *slow_replicas_[shard_idx];
  auto& stg = *rep.stages[0];

  c10::cuda::CUDAGuard fg(fs.device_id);
  c10::cuda::CUDAStreamGuard guard(fs.exec_stream);
  cudaStream_t raw_exec = fs.exec_stream.stream();

  const int32_t local_n = static_cast<int32_t>(fs.global_stream_ids.size());
  const size_t slow_elem_bytes = rep.per_stream_slow_obs[0].nbytes();

  bool slow_batch_active = false;
  bool slice_gpu_inflight = false;
  int64_t slice_launch_logical_ns = 0;
  int32_t consec_slow_slices = 0;
  int32_t active_slow_bucket = stg.sorted_slow_buckets.front();
  size_t cur_slice_idx = 0;
  std::vector<int32_t> active_slow_global_ids;
  std::vector<int32_t> active_slow_local_ids;
  std::vector<int64_t> active_slow_tobs;
  std::vector<int32_t> active_slow_seqlens;
  std::vector<int32_t> active_slow_global_slots;
  int32_t real_slow_count = 0;

  while (running_.load(std::memory_order_relaxed)) {
    if (slice_gpu_inflight) {
      cudaError_t st = cudaEventQuery(stg.slice_done_event);
      if (st == cudaErrorNotReady) {
        std::this_thread::sleep_for(std::chrono::microseconds(20));
        continue;
      }
      TS_SCHED_CUDA_CHECK(st);
      float slice_ms = 0.0f;
      TS_SCHED_CUDA_CHECK(cudaEventElapsedTime(&slice_ms, stg.slice_start_event, stg.slice_done_event));
      const int64_t obs_slice_ns = static_cast<int64_t>(slice_ms * 1e6);
      const int64_t bkt_floor_ns = stg.slow_bucket_slice_shadow_ns[active_slow_bucket];
      if (obs_slice_ns > stg.runtime_slice_shadow_ns) {
        stg.runtime_slice_shadow_ns = (stg.runtime_slice_shadow_ns * 3 + obs_slice_ns) / 4;
      } else {
        stg.runtime_slice_shadow_ns =
            std::max(bkt_floor_ns, (stg.runtime_slice_shadow_ns * 15 + obs_slice_ns) / 16);
      }
      slice_gpu_inflight = false;
      cur_slice_idx++;
      wait_until_logical_ns(slice_launch_logical_ns + obs_slice_ns);
      if (!running_.load(std::memory_order_relaxed)) break;

      const auto& cur_slices = stg.slow_bucket_slice_graphs[active_slow_bucket];
      if (cur_slice_idx == cur_slices.size()) {
        const int64_t pub_now = this->now_ns();
        uint64_t dropped_fast = 0;
        for (int32_t i = 0; i < real_slow_count; ++i) {
          const int32_t l_s = active_slow_local_ids[i];
          const int32_t g_s = active_slow_global_ids[i];
          fs.tsc->publish_write(
              l_s, active_slow_global_slots[i], active_slow_tobs[i], active_slow_seqlens[i], raw_exec);
          {
            std::lock_guard<std::mutex> lk(ingress_[g_s]->mu);
            dropped_fast += mark_slow_published_locked(*ingress_[g_s], active_slow_tobs[i], pub_now);
            ingress_[g_s]->slow_inflight = false;
            ingress_[g_s]->inflight_prompt_row = -1;
          }
        }
        {
          std::lock_guard<std::mutex> tlk(telem_mu_);
          telem_.slow_batches_completed++;
          telem_.fast_coalesced_drops += dropped_fast;
          for (int32_t i = 0; i < real_slow_count; ++i) {
            telem_.slow_latency_ms.push_back(static_cast<double>(pub_now - active_slow_tobs[i]) * 1e-6);
          }
        }
        slow_batch_active = false;
      }
    }

    fs.tsc->reclaim_retired_reads();
    wait_if_clock_paused();
    if (!running_.load(std::memory_order_relaxed)) break;
    const int64_t now_ns = this->now_ns();

    std::vector<FastReq> pending_fast;
    pending_fast.reserve(local_n);
    int64_t worst_slow_aoi_ns = 0;
    bool any_slow_pending = false;

    for (int32_t i = 0; i < local_n; ++i) {
      const int32_t g_s = fs.global_stream_ids[i];
      std::lock_guard<std::mutex> lk(ingress_[g_s]->mu);
      if (ingress_[g_s]->fast_pending && ingress_[g_s]->has_published_slow) {
        pending_fast.push_back(
            FastReq{i, g_s, ingress_[g_s]->fast_t_obs_ns, ingress_[g_s]->fast_wait_since_ns});
      }
      if (ingress_[g_s]->slow_pending) {
        any_slow_pending = true;
        worst_slow_aoi_ns = std::max(worst_slow_aoi_ns, now_ns - ingress_[g_s]->last_pub_slow_t_obs_ns);
      }
    }
    if (slow_batch_active) {
      for (int32_t i = 0; i < real_slow_count; ++i) {
        const int32_t g_s = active_slow_global_ids[i];
        std::lock_guard<std::mutex> lk(ingress_[g_s]->mu);
        worst_slow_aoi_ns = std::max(worst_slow_aoi_ns, now_ns - ingress_[g_s]->last_pub_slow_t_obs_ns);
      }
    }

    const auto select_slow_bucket = [&](int32_t pending_cnt, bool has_cold) -> int32_t {
      const int32_t capped = std::min(pending_cnt, slow_batch_size_);
      int32_t bkt = stg.sorted_slow_buckets.front();
      if (has_cold || pending_cnt >= slow_batch_size_) {
        for (int32_t b_s : stg.sorted_slow_buckets) {
          if (b_s <= capped) {
            bkt = b_s;
          }
        }
      }
      return bkt;
    };

    const bool slow_work_present = (slow_batch_active || any_slow_pending);
    bool launch_fast_now = false;
    bool allow_slow_slice = slow_work_present;
    int32_t chosen_bucket = 1;
    int32_t take_fast_count = 0;

    if (pending_fast.empty()) {
      consec_slow_slices = 0;
    } else {
      constexpr int64_t kPreemptGuardNs = 800000;
      constexpr int64_t kMaxCoalesceWindowNs = 1800000;

      const auto plan = plan_fast_batch(fs, pending_fast, now_ns);
      take_fast_count = plan.take_fast_count;
      chosen_bucket = plan.chosen_bucket;

      const int64_t cur_exec_ns = fs.runtime_bucket_exec_ns[chosen_bucket];
      const int64_t slice_guard_ns = stg.runtime_slice_shadow_ns + kPreemptGuardNs;
      const int64_t coalesce_window_ns = std::min<int64_t>(kMaxCoalesceWindowNs, fast_deadline_ns_ / 12);
      const bool want_more_fast_batch = (local_n > 1 && take_fast_count < plan.upper_cap);

      if (slow_work_present && want_more_fast_batch &&
          cur_exec_ns + slice_guard_ns <= plan.remaining_budget_ns) {
        launch_fast_now = false;
        allow_slow_slice = true;
      } else if (slow_work_present && worst_slow_aoi_ns > max_slow_aoi_ns_ && consec_slow_slices < 1 &&
                 cur_exec_ns + slice_guard_ns <= plan.remaining_budget_ns) {
        launch_fast_now = false;
        allow_slow_slice = true;
      } else if (!slow_work_present && want_more_fast_batch &&
                 (now_ns - pending_fast[0].wait_since_ns) < coalesce_window_ns &&
                 cur_exec_ns + 2 * kPreemptGuardNs <= plan.remaining_budget_ns) {
        launch_fast_now = false;
        allow_slow_slice = false;
      } else {
        launch_fast_now = true;
        allow_slow_slice = false;
      }
    }

    if (launch_fast_now) {
      execute_fast_batch(
          fs, pending_fast, take_fast_count, chosen_bucket, now_ns, /*count_preemption=*/slow_batch_active);
      consec_slow_slices = 0;
      continue;
    }

    if (allow_slow_slice && !slow_batch_active) {
      struct SlowCand {
        int32_t local_id;
        int32_t global_id;
        int64_t aoi_ns;
        int64_t t_obs_ns;
        bool cold_start;
      };
      std::vector<SlowCand> cands;
      for (int32_t i = 0; i < local_n; ++i) {
        const int32_t g_s = rep.global_stream_ids[i];
        std::lock_guard<std::mutex> lk(ingress_[g_s]->mu);
        if (ingress_[g_s]->slow_pending) {
          cands.push_back(SlowCand{
              i,
              g_s,
              now_ns - ingress_[g_s]->last_pub_slow_t_obs_ns,
              ingress_[g_s]->slow_t_obs_ns,
              !ingress_[g_s]->has_published_slow});
        }
      }

      if (!cands.empty()) {
        std::sort(cands.begin(), cands.end(), [](const SlowCand& a, const SlowCand& b) {
          if (a.cold_start != b.cold_start) return a.cold_start;
          if (a.aoi_ns != b.aoi_ns) return a.aoi_ns > b.aoi_ns;
          return a.t_obs_ns < b.t_obs_ns;
        });
        const int32_t capped_slow = std::min(static_cast<int32_t>(cands.size()), slow_batch_size_);
        active_slow_bucket = select_slow_bucket(static_cast<int32_t>(cands.size()), cands[0].cold_start);
        real_slow_count = std::min(capped_slow, active_slow_bucket);

        active_slow_global_ids.assign(active_slow_bucket, cands[0].global_id);
        active_slow_local_ids.assign(active_slow_bucket, cands[0].local_id);
        active_slow_tobs.assign(active_slow_bucket, cands[0].t_obs_ns);
        active_slow_seqlens.assign(active_slow_bucket, slow_frame_seq_len_);
        active_slow_global_slots.assign(active_slow_bucket, -1);

        while (true) {
          cudaError_t st = cudaEventQuery(stg.tobs_h2d_done_event);
          if (st == cudaSuccess) break;
          if (st != cudaErrorNotReady) TS_SCHED_CUDA_CHECK(st);
          std::this_thread::sleep_for(std::chrono::microseconds(15));
        }

        std::vector<int32_t> real_local_ids(real_slow_count);
        for (int32_t i = 0; i < real_slow_count; ++i) {
          const int32_t l_s = cands[i].local_id;
          const int32_t g_s = cands[i].global_id;
          real_local_ids[i] = l_s;
          active_slow_local_ids[i] = l_s;
          active_slow_global_ids[i] = g_s;

          std::lock_guard<std::mutex> lk(ingress_[g_s]->mu);
          active_slow_tobs[i] = ingress_[g_s]->slow_t_obs_ns;
          active_slow_seqlens[i] = ingress_[g_s]->slow_seq_len;
          const int32_t picked_row = ingress_[g_s]->pending_prompt_row;
          ingress_[g_s]->inflight_prompt_row = picked_row;
          stg.h_slow_tobs[i] = active_slow_tobs[i];
          stg.h_slow_stream_ids[i] = picked_row;
          stg.h_slow_seqlens[i] = active_slow_seqlens[i];
          ingress_[g_s]->slow_pending = false;
          ingress_[g_s]->slow_inflight = true;
          TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(raw_exec, rep.slow_ingest_events[l_s], 0));
          TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
              static_cast<char*>(stg.slow_obs_static.data_ptr()) + i * slow_elem_bytes,
              rep.per_stream_slow_obs[l_s].data_ptr(),
              slow_elem_bytes,
              cudaMemcpyDeviceToDevice,
              raw_exec));
          TS_SCHED_CUDA_CHECK(cudaEventRecord(rep.slow_consumed_events[l_s], raw_exec));
        }
        for (int32_t i = real_slow_count; i < active_slow_bucket; ++i) {
          TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
              static_cast<char*>(stg.slow_obs_static.data_ptr()) + i * slow_elem_bytes,
              stg.slow_obs_static.data_ptr(),
              slow_elem_bytes,
              cudaMemcpyDeviceToDevice,
              raw_exec));
        }

        std::vector<int32_t> real_slots;
        auto tmp_pt_gpu = stg.slow_pt_static.slice(0, 0, real_slow_count);
        auto tmp_start_gpu = stg.slow_start_page_static.slice(0, 0, real_slow_count);
        fs.tsc->prepare_write_batch(
            real_local_ids, slow_frame_seq_len_, tmp_pt_gpu, tmp_start_gpu, real_slots, raw_exec);

        for (int32_t i = 0; i < real_slow_count; ++i) {
          active_slow_global_slots[i] = real_slots[i];
        }
        if (real_slow_count < active_slow_bucket) {
          stg.slow_pt_static.slice(0, real_slow_count, active_slow_bucket).fill_(-1);
          stg.slow_start_page_static.slice(0, real_slow_count, active_slow_bucket).zero_();
          for (int32_t i = real_slow_count; i < active_slow_bucket; ++i) {
            stg.h_slow_tobs[i] = stg.h_slow_tobs[0];
            stg.h_slow_stream_ids[i] = stg.h_slow_stream_ids[0];
            stg.h_slow_seqlens[i] = stg.h_slow_seqlens[0];
          }
        }
        TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
            stg.slow_tobs_static.data_ptr<int64_t>(),
            stg.h_slow_tobs,
            sizeof(int64_t) * active_slow_bucket,
            cudaMemcpyHostToDevice,
            raw_exec));
        TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
            stg.slow_stream_ids_static.data_ptr<int32_t>(),
            stg.h_slow_stream_ids,
            sizeof(int32_t) * active_slow_bucket,
            cudaMemcpyHostToDevice,
            raw_exec));
        TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
            stg.slow_seqlens_static.data_ptr<int32_t>(),
            stg.h_slow_seqlens,
            sizeof(int32_t) * active_slow_bucket,
            cudaMemcpyHostToDevice,
            raw_exec));
        TS_SCHED_CUDA_CHECK(cudaEventRecord(stg.tobs_h2d_done_event, raw_exec));

        cur_slice_idx = 0;
        slow_batch_active = true;
      }
    }

    if (allow_slow_slice && slow_batch_active && !slice_gpu_inflight) {
      slice_launch_logical_ns = this->now_ns();
      TS_SCHED_CUDA_CHECK(cudaEventRecord(stg.slice_start_event, raw_exec));
      stg.slow_bucket_slice_graphs[active_slow_bucket][cur_slice_idx]->replay();
      TS_SCHED_CUDA_CHECK(cudaEventRecord(stg.slice_done_event, raw_exec));
      slice_gpu_inflight = true;
      if (!pending_fast.empty()) {
        consec_slow_slices++;
      }
      {
        std::lock_guard<std::mutex> tlk(telem_mu_);
        telem_.slow_slices_launched++;
      }
      continue;
    }

    std::this_thread::sleep_for(std::chrono::microseconds(25));
  }

  TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(raw_exec));
  if (slow_batch_active) {
    fs.tsc->abort_write_batch(active_slow_global_slots, nullptr);
    for (int32_t i = 0; i < real_slow_count; ++i) {
      const int32_t g_s = active_slow_global_ids[i];
      std::lock_guard<std::mutex> lk(ingress_[g_s]->mu);
      ingress_[g_s]->slow_inflight = false;
      ingress_[g_s]->inflight_prompt_row = -1;
    }
    slow_batch_active = false;
  }
}

void TimeAwareScheduler::fast_disagg_shard_loop(int32_t shard_idx) {
#ifdef __linux__
  prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
#endif
  auto& fs = *fast_shards_[shard_idx];
  c10::cuda::CUDAGuard fg(fs.device_id);
  c10::cuda::CUDAStreamGuard guard(fs.exec_stream);
  const int32_t local_n = static_cast<int32_t>(fs.global_stream_ids.size());

  std::vector<FastReq> pending_fast;
  pending_fast.reserve(local_n);

  while (running_.load(std::memory_order_relaxed)) {
    fs.tsc->reclaim_retired_reads();
    wait_if_clock_paused();
    if (!running_.load(std::memory_order_relaxed)) break;
    const int64_t now_ns = this->now_ns();

    pending_fast.clear();
    for (int32_t i = 0; i < local_n; ++i) {
      const int32_t g_s = fs.global_stream_ids[i];
      std::lock_guard<std::mutex> lk(ingress_[g_s]->mu);
      if (ingress_[g_s]->fast_pending && ingress_[g_s]->has_published_slow) {
        pending_fast.push_back(
            FastReq{i, g_s, ingress_[g_s]->fast_t_obs_ns, ingress_[g_s]->fast_wait_since_ns});
      }
    }

    if (pending_fast.empty()) {
      std::this_thread::sleep_for(std::chrono::microseconds(20));
      continue;
    }

    const auto plan = plan_fast_batch(fs, pending_fast, now_ns);
    const int64_t cur_exec_ns = fs.runtime_bucket_exec_ns[plan.chosen_bucket];
    const int64_t coalesce_window_ns = std::min<int64_t>(1800000, fast_deadline_ns_ / 12);
    if (local_n > 1 && plan.take_fast_count < plan.upper_cap &&
        (now_ns - pending_fast[0].wait_since_ns) < coalesce_window_ns &&
        cur_exec_ns + 1500000 <= plan.remaining_budget_ns) {
      std::this_thread::sleep_for(std::chrono::microseconds(20));
      continue;
    }

    execute_fast_batch(
        fs, pending_fast, plan.take_fast_count, plan.chosen_bucket, now_ns, /*count_preemption=*/false);
  }

  TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(fs.exec_stream.stream()));
}

void TimeAwareScheduler::slow_disagg_stage_loop(int32_t replica_idx, int32_t stage_idx) {
#ifdef __linux__
  prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
#endif
  auto& rep = *slow_replicas_[replica_idx];
  auto& stg = *rep.stages[stage_idx];
  const int32_t F = static_cast<int32_t>(fast_shards_.size());
  const int32_t local_r_n = static_cast<int32_t>(rep.global_stream_ids.size());
  const size_t slow_elem_bytes = (stage_idx == 0) ? rep.per_stream_slow_obs[0].nbytes() : 0;
  cudaStream_t raw_exec = stg.exec_stream.stream();
  cudaStream_t raw_handoff = stg.handoff_stream.stream();

  struct SlowCand {
    int32_t local_slow_id;
    int32_t global_id;
    int64_t aoi_ns;
    int64_t t_obs_ns;
    bool cold_start;
  };
  std::vector<SlowCand> cands;
  if (stage_idx == 0) {
    cands.reserve(local_r_n);
  }

  while (running_.load(std::memory_order_relaxed)) {
    wait_if_clock_paused();
    if (!running_.load(std::memory_order_relaxed)) break;
    PipelinePacket pkt;

    // 1. Acquire microbatch packet (entry AoI queue on stage 0, or upstream handoff edge on stage > 0)
    if (stage_idx == 0) {
      const int64_t now_ns = this->now_ns();
      cands.clear();
      for (int32_t i = 0; i < local_r_n; ++i) {
        const int32_t g_s = rep.global_stream_ids[i];
        std::lock_guard<std::mutex> lk(ingress_[g_s]->mu);
        if (ingress_[g_s]->slow_pending && !ingress_[g_s]->slow_inflight) {
          cands.push_back(SlowCand{
              i,
              g_s,
              now_ns - ingress_[g_s]->last_pub_slow_t_obs_ns,
              ingress_[g_s]->slow_t_obs_ns,
              !ingress_[g_s]->has_published_slow});
        }
      }
      if (cands.empty()) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        continue;
      }

      std::sort(cands.begin(), cands.end(), [](const SlowCand& a, const SlowCand& b) {
        if (a.cold_start != b.cold_start) return a.cold_start;
        if (a.aoi_ns != b.aoi_ns) return a.aoi_ns > b.aoi_ns;
        return a.t_obs_ns < b.t_obs_ns;
      });

      const int32_t real_cnt = std::min(static_cast<int32_t>(cands.size()), slow_batch_size_);
      int32_t chosen_bkt = stg.sorted_slow_buckets.back();
      for (int32_t b_s : stg.sorted_slow_buckets) {
        if (b_s >= real_cnt) {
          chosen_bkt = b_s;
          break;
        }
      }

      const int32_t p_slot = pop_pkt_slot(rep);
      if (p_slot < 0) break;

      pkt.pkt_slot = p_slot;
      pkt.chosen_slow_bkt = chosen_bkt;
      pkt.real_slow_count = real_cnt;
      pkt.global_ids.resize(real_cnt);
      pkt.local_slow_ids.resize(real_cnt);
      pkt.tobs_ns.resize(real_cnt);
      pkt.seqlens.resize(real_cnt);
      pkt.prompt_slot_ids.resize(real_cnt);
      pkt.shard_local_fast_ids.resize(F);
      pkt.shard_dst_slots.resize(F);
      pkt.shard_tobs_ns.resize(F);
      pkt.shard_seqlens.resize(F);
      pkt.shard_prev_p2p_ev.assign(F, nullptr);

      c10::cuda::CUDAGuard sg(stg.device_id);
      c10::cuda::CUDAStreamGuard s_guard(stg.exec_stream);
      for (int32_t i = 0; i < real_cnt; ++i) {
        const int32_t l_s = cands[i].local_slow_id;
        const int32_t g_s = cands[i].global_id;
        pkt.local_slow_ids[i] = l_s;
        pkt.global_ids[i] = g_s;

        const int32_t f = fast_shard_of_[g_s];
        pkt.shard_local_fast_ids[f].push_back(local_fast_id_[g_s]);

        std::lock_guard<std::mutex> lk(ingress_[g_s]->mu);
        const int64_t picked_tobs_ns = ingress_[g_s]->slow_t_obs_ns;
        const int32_t picked_seq_len = ingress_[g_s]->slow_seq_len;
        const int32_t picked_row = ingress_[g_s]->pending_prompt_row;
        ingress_[g_s]->inflight_prompt_row = picked_row;
        pkt.tobs_ns[i] = picked_tobs_ns;
        pkt.seqlens[i] = picked_seq_len;
        pkt.prompt_slot_ids[i] = picked_row;
        pkt.shard_tobs_ns[f].push_back(picked_tobs_ns);
        pkt.shard_seqlens[f].push_back(picked_seq_len);
        ingress_[g_s]->slow_pending = false;
        ingress_[g_s]->slow_inflight = true;
        TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(raw_exec, rep.slow_ingest_events[l_s], 0));
        TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
            static_cast<char*>(stg.slow_obs_static.data_ptr()) + i * slow_elem_bytes,
            rep.per_stream_slow_obs[l_s].data_ptr(),
            slow_elem_bytes,
            cudaMemcpyDeviceToDevice,
            raw_exec));
        TS_SCHED_CUDA_CHECK(cudaEventRecord(rep.slow_consumed_events[l_s], raw_exec));
      }
      for (int32_t i = real_cnt; i < chosen_bkt; ++i) {
        TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
            static_cast<char*>(stg.slow_obs_static.data_ptr()) + i * slow_elem_bytes,
            stg.slow_obs_static.data_ptr(),
            slow_elem_bytes,
            cudaMemcpyDeviceToDevice,
            raw_exec));
      }
    } else {
      auto& edge_in = *rep.edges[stage_idx - 1];
      {
        std::unique_lock<std::mutex> lk(edge_in.mu);
        edge_in.not_empty_cv.wait(
            lk, [&]() { return !edge_in.q.empty() || !running_.load(std::memory_order_relaxed); });
        if (edge_in.q.empty()) {
          break;
        }
        pkt = std::move(edge_in.q.front());
        edge_in.q.pop_front();
      }
      edge_in.not_full_cv.notify_one();

      c10::cuda::CUDAGuard sg(stg.device_id);
      c10::cuda::CUDAStreamGuard s_guard(stg.exec_stream);
      TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(raw_exec, edge_in.ready_ev[pkt.handoff_ring_slot], 0));
      for (size_t t_idx = 0; t_idx < stg.act_in_static.size(); ++t_idx) {
        const auto& src_t = edge_in.ring_buffers[pkt.handoff_ring_slot][t_idx];
        auto& dst_t = stg.act_in_static[t_idx];
        const size_t per_item_bytes = static_cast<size_t>(dst_t.stride(0)) * dst_t.element_size();
        TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
            dst_t.data_ptr(),
            src_t.data_ptr(),
            static_cast<size_t>(pkt.chosen_slow_bkt) * per_item_bytes,
            cudaMemcpyDeviceToDevice,
            raw_exec));
      }
      TS_SCHED_CUDA_CHECK(cudaEventRecord(edge_in.consumed_ev[pkt.handoff_ring_slot], raw_exec));
    }

    // 2. Execute Stage CUDA Graph on stg.device_id
    std::vector<int32_t> src_slots;
    {
      c10::cuda::CUDAGuard sg(stg.device_id);
      c10::cuda::CUDAStreamGuard s_guard(stg.exec_stream);

      if (stage_idx + 1 < num_slow_stages_) {
        TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(raw_exec, stg.act_out_copied_event, 0));
      }

      if (stg.has_tsc_writes) {
        auto tmp_pt_gpu = stg.slow_pt_static.slice(0, 0, pkt.real_slow_count);
        auto tmp_start_gpu = stg.slow_start_page_static.slice(0, 0, pkt.real_slow_count);
        stg.slow_tsc->prepare_write_batch(
            pkt.local_slow_ids, slow_frame_seq_len_, tmp_pt_gpu, tmp_start_gpu, src_slots, raw_exec);
        if (pkt.real_slow_count < pkt.chosen_slow_bkt) {
          stg.slow_pt_static.slice(0, pkt.real_slow_count, pkt.chosen_slow_bkt).fill_(-1);
          stg.slow_start_page_static.slice(0, pkt.real_slow_count, pkt.chosen_slow_bkt).zero_();
        }
      }

      while (true) {
        cudaError_t st = cudaEventQuery(stg.tobs_h2d_done_event);
        if (st == cudaSuccess) break;
        if (st != cudaErrorNotReady) TS_SCHED_CUDA_CHECK(st);
        std::this_thread::sleep_for(std::chrono::microseconds(15));
      }
      for (int32_t i = 0; i < pkt.real_slow_count; ++i) {
        stg.h_slow_tobs[i] = pkt.tobs_ns[i];
        stg.h_slow_stream_ids[i] = pkt.prompt_slot_ids[i];
        stg.h_slow_seqlens[i] = pkt.seqlens[i];
      }
      for (int32_t i = pkt.real_slow_count; i < pkt.chosen_slow_bkt; ++i) {
        stg.h_slow_tobs[i] = stg.h_slow_tobs[0];
        stg.h_slow_stream_ids[i] = stg.h_slow_stream_ids[0];
        stg.h_slow_seqlens[i] = stg.h_slow_seqlens[0];
      }
      if (stg.has_tsc_writes) {
        TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
            stg.slow_tobs_static.data_ptr<int64_t>(),
            stg.h_slow_tobs,
            sizeof(int64_t) * pkt.chosen_slow_bkt,
            cudaMemcpyHostToDevice,
            raw_exec));
      }
      TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
          stg.slow_stream_ids_static.data_ptr<int32_t>(),
          stg.h_slow_stream_ids,
          sizeof(int32_t) * pkt.chosen_slow_bkt,
          cudaMemcpyHostToDevice,
          raw_exec));
      TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
          stg.slow_seqlens_static.data_ptr<int32_t>(),
          stg.h_slow_seqlens,
          sizeof(int32_t) * pkt.chosen_slow_bkt,
          cudaMemcpyHostToDevice,
          raw_exec));
      TS_SCHED_CUDA_CHECK(cudaEventRecord(stg.tobs_h2d_done_event, raw_exec));

      const int64_t stg_start_logical_ns = this->now_ns();
      TS_SCHED_CUDA_CHECK(cudaEventRecord(stg.slice_start_event, raw_exec));
      const auto& graphs = stg.slow_bucket_slice_graphs[pkt.chosen_slow_bkt];
      for (const auto& g : graphs) {
        g->replay();
      }

      if (stg.has_tsc_writes) {
        for (int32_t i = 0; i < pkt.real_slow_count; ++i) {
          stg.slow_tsc->publish_write(
              pkt.local_slow_ids[i], src_slots[i], pkt.tobs_ns[i], pkt.seqlens[i], raw_exec);
        }
      }
      TS_SCHED_CUDA_CHECK(cudaEventRecord(stg.slice_done_event, raw_exec));
      while (true) {
        cudaError_t st = cudaEventQuery(stg.slice_done_event);
        if (st == cudaSuccess) break;
        if (st != cudaErrorNotReady) TS_SCHED_CUDA_CHECK(st);
        std::this_thread::sleep_for(std::chrono::microseconds(20));
      }
      float stg_ms = 0.0f;
      TS_SCHED_CUDA_CHECK(cudaEventElapsedTime(&stg_ms, stg.slice_start_event, stg.slice_done_event));
      wait_until_logical_ns(stg_start_logical_ns + static_cast<int64_t>(stg_ms * 1e6));
      if (!running_.load(std::memory_order_relaxed)) {
        abort_packet_writes(rep, pkt);
        push_pkt_slot(rep, pkt.pkt_slot);
        break;
      }
    }

    // 3. Enqueue asynchronous TSC pre-push (direct P2P or non-blocking pinned DMA) across touched Fast shards
    try {
      if (stg.has_tsc_writes) {
        std::vector<std::vector<int32_t>> shard_src_slots(F);
        for (int32_t i = 0; i < pkt.real_slow_count; ++i) {
          const int32_t f = fast_shard_of_[pkt.global_ids[i]];
          shard_src_slots[f].push_back(src_slots[i]);
        }

        if (!pkt.dst_slots_allocated) {
          for (int32_t f = 0; f < F; ++f) {
            if (!pkt.shard_local_fast_ids[f].empty()) {
              fast_shards_[f]->tsc->allocate_write_slots(
                  pkt.shard_local_fast_ids[f],
                  slow_frame_seq_len_,
                  pkt.shard_dst_slots[f],
                  stg.p2p_push_streams[f].stream());
            }
          }
          pkt.dst_slots_allocated = true;
        }

        const bool copy_ts = (stage_idx == rep.last_tsc_stage_idx);
        for (int32_t f = 0; f < F; ++f) {
          if (pkt.shard_local_fast_ids[f].empty()) continue;
          const int32_t f_dev = fast_shards_[f]->device_id;
          cudaStream_t raw_p2p = stg.p2p_push_streams[f].stream();
          cudaEvent_t ev = rep.pkt_p2p_events[pkt.pkt_slot][stage_idx][f];

          if (has_p2p(f_dev, stg.device_id)) {
            c10::cuda::CUDAGuard fg(f_dev);
            if (pkt.shard_prev_p2p_ev[f] != nullptr) {
              TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(raw_p2p, pkt.shard_prev_p2p_ev[f], 0));
            }
            fast_shards_[f]->tsc->enqueue_peer_copy_batch(
                *stg.slow_tsc,
                shard_src_slots[f],
                pkt.shard_dst_slots[f],
                stg.dst_layer_offset,
                stg.layer_count,
                slow_frame_seq_len_,
                copy_ts,
                raw_p2p,
                ev);
          } else {
            {
              c10::cuda::CUDAGuard fg(f_dev);
              while (true) {
                cudaError_t bst = cudaEventQuery(stg.bounce_consumed_events[f]);
                if (bst == cudaSuccess) break;
                if (bst != cudaErrorNotReady) TS_SCHED_CUDA_CHECK(bst);
                std::this_thread::sleep_for(std::chrono::microseconds(15));
              }
            }
            {
              c10::cuda::CUDAGuard sg(stg.device_id);
              stg.slow_tsc->stage_slice_to_host_pinned(
                  shard_src_slots[f], slow_frame_seq_len_, copy_ts, stg.h_p2p_bounce[f], raw_handoff);
              TS_SCHED_CUDA_CHECK(cudaEventRecord(stg.tsc_d2h_done_event, raw_handoff));
              while (true) {
                cudaError_t dst = cudaEventQuery(stg.tsc_d2h_done_event);
                if (dst == cudaSuccess) break;
                if (dst != cudaErrorNotReady) TS_SCHED_CUDA_CHECK(dst);
                std::this_thread::sleep_for(std::chrono::microseconds(15));
              }
            }
            {
              c10::cuda::CUDAGuard fg(f_dev);
              if (pkt.shard_prev_p2p_ev[f] != nullptr) {
                TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(raw_p2p, pkt.shard_prev_p2p_ev[f], 0));
              }
              fast_shards_[f]->tsc->enqueue_host_pinned_copy_batch(
                  stg.h_p2p_bounce[f],
                  pkt.shard_dst_slots[f],
                  stg.dst_layer_offset,
                  stg.layer_count,
                  slow_frame_seq_len_,
                  copy_ts,
                  raw_p2p,
                  ev);
              TS_SCHED_CUDA_CHECK(cudaEventRecord(stg.bounce_consumed_events[f], raw_p2p));
            }
          }
          pkt.shard_prev_p2p_ev[f] = ev;
        }
      }

      // 4. Handoff activations to Stage p + 1 OR finalize atomic TSC publish on final stage
      if (stage_idx + 1 < num_slow_stages_) {
        auto& edge_out = *rep.edges[stage_idx];
        const int32_t next_dev = rep.stages[stage_idx + 1]->device_id;
        std::unique_lock<std::mutex> lk(edge_out.mu);
        edge_out.not_full_cv.wait(
            lk, [&]() { return edge_out.q.empty() || !running_.load(std::memory_order_relaxed); });
        if (!running_.load(std::memory_order_relaxed)) {
          lk.unlock();
          abort_packet_writes(rep, pkt);
          push_pkt_slot(rep, pkt.pkt_slot);
          break;
        }

        const int32_t ring_slot = edge_out.next_ring_slot;
        edge_out.next_ring_slot = (ring_slot + 1) % kHandoffRingSlots;

        if (edge_out.direct_p2p) {
          c10::cuda::CUDAGuard sg(stg.device_id);
          c10::cuda::CUDAStreamGuard h_guard(stg.handoff_stream);
          TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(raw_handoff, edge_out.consumed_ev[ring_slot], 0));
          for (size_t t_idx = 0; t_idx < stg.act_out_static.size(); ++t_idx) {
            const auto& src_t = stg.act_out_static[t_idx];
            auto& dst_t = edge_out.ring_buffers[ring_slot][t_idx];
            const size_t per_item_bytes = static_cast<size_t>(src_t.stride(0)) * src_t.element_size();
            TS_SCHED_CUDA_CHECK(cudaMemcpyPeerAsync(
                dst_t.data_ptr(),
                next_dev,
                src_t.data_ptr(),
                stg.device_id,
                static_cast<size_t>(pkt.chosen_slow_bkt) * per_item_bytes,
                raw_handoff));
          }
          TS_SCHED_CUDA_CHECK(cudaEventRecord(stg.act_out_copied_event, raw_handoff));
          TS_SCHED_CUDA_CHECK(cudaEventRecord(edge_out.ready_ev[ring_slot], raw_handoff));
        } else {
          {
            c10::cuda::CUDAGuard sg(stg.device_id);
            c10::cuda::CUDAStreamGuard h_guard(stg.handoff_stream);
            for (size_t t_idx = 0; t_idx < stg.act_out_static.size(); ++t_idx) {
              const auto& src_t = stg.act_out_static[t_idx];
              auto& h_buf = edge_out.h_bounce_buffers[ring_slot][t_idx];
              const size_t per_item_bytes = static_cast<size_t>(src_t.stride(0)) * src_t.element_size();
              TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
                  h_buf.data_ptr(),
                  src_t.data_ptr(),
                  static_cast<size_t>(pkt.chosen_slow_bkt) * per_item_bytes,
                  cudaMemcpyDeviceToHost,
                  raw_handoff));
            }
            TS_SCHED_CUDA_CHECK(cudaEventRecord(stg.act_out_copied_event, raw_handoff));
            TS_SCHED_CUDA_CHECK(cudaEventRecord(edge_out.src_d2h_ev[ring_slot], raw_handoff));
            while (true) {
              cudaError_t dst = cudaEventQuery(edge_out.src_d2h_ev[ring_slot]);
              if (dst == cudaSuccess) break;
              if (dst != cudaErrorNotReady) TS_SCHED_CUDA_CHECK(dst);
              std::this_thread::sleep_for(std::chrono::microseconds(15));
            }
          }
          {
            c10::cuda::CUDAGuard dg(next_dev);
            cudaStream_t raw_dst_h2d = edge_out.dst_h2d_stream.stream();
            TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(raw_dst_h2d, edge_out.consumed_ev[ring_slot], 0));
            for (size_t t_idx = 0; t_idx < stg.act_out_static.size(); ++t_idx) {
              const auto& h_buf = edge_out.h_bounce_buffers[ring_slot][t_idx];
              auto& dst_t = edge_out.ring_buffers[ring_slot][t_idx];
              const size_t per_item_bytes = static_cast<size_t>(dst_t.stride(0)) * dst_t.element_size();
              TS_SCHED_CUDA_CHECK(cudaMemcpyAsync(
                  dst_t.data_ptr(),
                  h_buf.data_ptr(),
                  static_cast<size_t>(pkt.chosen_slow_bkt) * per_item_bytes,
                  cudaMemcpyHostToDevice,
                  raw_dst_h2d));
            }
            TS_SCHED_CUDA_CHECK(cudaEventRecord(edge_out.ready_ev[ring_slot], raw_dst_h2d));
          }
        }

        pkt.handoff_ring_slot = ring_slot;
        const size_t n_slices = stg.slow_bucket_slice_graphs[pkt.chosen_slow_bkt].size();
        edge_out.q.push_back(std::move(pkt));
        lk.unlock();
        edge_out.not_empty_cv.notify_one();

        {
          std::lock_guard<std::mutex> tlk(telem_mu_);
          telem_.slow_slices_launched += n_slices;
        }
      } else {
        for (int32_t f = 0; f < F; ++f) {
          if (pkt.shard_local_fast_ids[f].empty()) continue;
          cudaStream_t raw_p2p = stg.p2p_push_streams[f].stream();
          cudaEvent_t done_ev = pkt.shard_prev_p2p_ev[f];
          if (!stg.has_tsc_writes && done_ev != nullptr) {
            c10::cuda::CUDAGuard fg(fast_shards_[f]->device_id);
            TS_SCHED_CUDA_CHECK(cudaStreamWaitEvent(raw_p2p, done_ev, 0));
            done_ev = rep.pkt_p2p_events[pkt.pkt_slot][stage_idx][f];
            TS_SCHED_CUDA_CHECK(cudaEventRecord(done_ev, raw_p2p));
          }
          fast_shards_[f]->tsc->finalize_peer_publish_batch(
              pkt.shard_local_fast_ids[f],
              pkt.shard_dst_slots[f],
              pkt.shard_tobs_ns[f],
              pkt.shard_seqlens[f],
              raw_p2p,
              done_ev);
        }
        pkt.dst_slots_allocated = false;

        const int64_t pub_now = this->now_ns();
        uint64_t dropped_fast = 0;
        for (int32_t i = 0; i < pkt.real_slow_count; ++i) {
          const int32_t g_s = pkt.global_ids[i];
          std::lock_guard<std::mutex> lk(ingress_[g_s]->mu);
          dropped_fast += mark_slow_published_locked(*ingress_[g_s], pkt.tobs_ns[i], pub_now);
          ingress_[g_s]->slow_inflight = false;
          ingress_[g_s]->inflight_prompt_row = -1;
        }
        {
          std::lock_guard<std::mutex> tlk(telem_mu_);
          telem_.slow_batches_completed++;
          telem_.fast_coalesced_drops += dropped_fast;
          telem_.slow_slices_launched += stg.slow_bucket_slice_graphs[pkt.chosen_slow_bkt].size();
          telem_.p2p_prepushes += static_cast<uint64_t>(pkt.real_slow_count);
          for (int32_t i = 0; i < pkt.real_slow_count; ++i) {
            telem_.slow_latency_ms.push_back(static_cast<double>(pub_now - pkt.tobs_ns[i]) * 1e-6);
          }
        }
        push_pkt_slot(rep, pkt.pkt_slot);
      }
    } catch (...) {
      abort_packet_writes(rep, pkt);
      push_pkt_slot(rep, pkt.pkt_slot);
      throw;
    }
  }

  {
    c10::cuda::CUDAGuard sg(stg.device_id);
    TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(raw_exec));
    TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(raw_handoff));
  }
  if (stage_idx + 1 < num_slow_stages_ && !rep.edges[stage_idx]->direct_p2p) {
    c10::cuda::CUDAGuard dg(rep.stages[stage_idx + 1]->device_id);
    TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(rep.edges[stage_idx]->dst_h2d_stream.stream()));
  }
  for (int32_t f = 0; f < F; ++f) {
    c10::cuda::CUDAGuard fg(fast_shards_[f]->device_id);
    TS_SCHED_CUDA_CHECK(cudaStreamSynchronize(stg.p2p_push_streams[f].stream()));
  }
}

}  // namespace tempo
