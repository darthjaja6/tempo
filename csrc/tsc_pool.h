#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tempo {

enum SlotState : uint32_t {
  SLOT_FREE = 0,
  SLOT_WRITING = 1,
  SLOT_READY = 2,
  SLOT_RECLAIMING = 3,
};

struct alignas(64) SlotHeader {
  std::atomic<uint32_t> reader_mask{0};
  std::atomic<uint32_t> state{SLOT_FREE};
  int64_t t_obs_ns{0};
  uint64_t version{0};
  int32_t seq_len{0};
  int32_t num_pages{0};
  int32_t page_table_offset{0};
  uint32_t reserved[7]{0, 0, 0, 0, 0, 0, 0};
};
static_assert(sizeof(SlotHeader) == 64, "SlotHeader must be exactly one 64-byte cache line");

class TemporalStateCache {
 public:
  TemporalStateCache(
      int32_t num_streams,
      int32_t num_readers,
      int32_t max_pages,
      int32_t num_layers,
      int32_t page_size,
      int32_t num_kv_heads,
      int32_t head_dim,
      int32_t max_seq_len,
      bool kv_mode = true,
      int32_t max_q_len = 64,
      int32_t max_q_heads = 16,
      int32_t device_id = 0);

  ~TemporalStateCache();

  TemporalStateCache(const TemporalStateCache&) = delete;
  TemporalStateCache& operator=(const TemporalStateCache&) = delete;

  void reclaim_retired_reads();

  int32_t begin_write(int32_t stream_id, int32_t frame_seq_len, cudaStream_t write_stream);

  void write_layer_kv(
      int32_t global_slot,
      int32_t layer_idx,
      const torch::Tensor& k_in,
      const torch::Tensor& v_in,
      int64_t t_obs_ns,
      bool bake_spatial_rope,
      cudaStream_t write_stream);

  void write_hidden_state(
      int32_t global_slot,
      const torch::Tensor& hidden_in,
      int64_t t_obs_ns,
      cudaStream_t write_stream);

  void prepare_write_batch(
      const std::vector<int32_t>& stream_ids,
      int32_t frame_seq_len,
      torch::Tensor& out_slot_page_table_gpu,
      torch::Tensor& out_start_page_idx_gpu,
      std::vector<int32_t>& out_global_slots,
      cudaStream_t copy_stream);

  void publish_write(
      int32_t stream_id,
      int32_t global_slot,
      int64_t t_obs_ns,
      int32_t frame_seq_len,
      cudaStream_t write_stream);

  void allocate_write_slots(
      const std::vector<int32_t>& stream_ids,
      int32_t frame_seq_len,
      std::vector<int32_t>& out_dst_slots,
      cudaStream_t init_stream);

  void enqueue_peer_copy_batch(
      const TemporalStateCache& src_tsc,
      const std::vector<int32_t>& src_global_slots,
      const std::vector<int32_t>& dst_global_slots,
      int32_t dst_layer_offset,
      int32_t layer_count,
      int32_t frame_seq_len,
      bool copy_timestamps,
      cudaStream_t push_stream,
      cudaEvent_t completion_event = nullptr);

  size_t stage_slice_bounce_bytes(int32_t batch_size, int32_t frame_seq_len) const;

  void stage_slice_to_host_pinned(
      const std::vector<int32_t>& src_global_slots,
      int32_t frame_seq_len,
      bool copy_timestamps,
      void* h_pinned_dst,
      cudaStream_t d2h_stream) const;

  void enqueue_host_pinned_copy_batch(
      const void* h_pinned_src,
      const std::vector<int32_t>& dst_global_slots,
      int32_t dst_layer_offset,
      int32_t layer_count,
      int32_t frame_seq_len,
      bool copy_timestamps,
      cudaStream_t push_stream,
      cudaEvent_t completion_event = nullptr);

  void finalize_peer_publish_batch(
      const std::vector<int32_t>& stream_ids,
      const std::vector<int32_t>& dst_global_slots,
      const std::vector<int64_t>& t_obs_ns,
      int32_t frame_seq_len,
      cudaStream_t push_stream,
      cudaEvent_t push_done_event);

  void abort_write_batch(const std::vector<int32_t>& global_slots, cudaStream_t write_stream);

  void reset_stream(int32_t stream_id);

  void acquire_read_batch(
      int32_t reader_id,
      const std::vector<int32_t>& stream_ids,
      const std::vector<int64_t>& t_now_ns,
      int32_t* d_batch_page_table,
      int32_t* d_batch_seq_lens,
      int64_t* d_batch_t_now_ns,
      std::vector<int64_t>& out_slow_t_obs_ns,
      std::vector<int32_t>& out_pinned_global_slots,
      cudaStream_t fast_stream);

  void release_read_batch(
      int32_t reader_id,
      const std::vector<int32_t>& pinned_global_slots,
      cudaStream_t fast_stream);

  void abort_read_batch(
      int32_t reader_id,
      const std::vector<int32_t>& pinned_global_slots);

  int32_t num_streams() const { return num_streams_; }
  int32_t slots_per_stream() const { return slots_per_stream_; }
  int32_t max_pages_per_slot() const { return max_pages_per_slot_; }
  int32_t page_size() const { return page_size_; }
  int32_t num_layers() const { return num_layers_; }
  int32_t num_kv_heads() const { return num_kv_heads_; }
  int32_t head_dim() const { return head_dim_; }
  int32_t max_seq_len() const { return max_seq_len_; }
  bool kv_mode() const { return kv_mode_; }
  int32_t device_id() const { return device_id_; }
  int32_t free_page_count() const { return free_count_.load(std::memory_order_relaxed); }

  torch::Tensor k_pool() const { return k_pool_; }
  torch::Tensor v_pool() const { return v_pool_; }
  torch::Tensor page_timestamps() const { return page_t_obs_ns_; }
  torch::Tensor splitk_partial_out(int32_t reader_id = 0) const;
  torch::Tensor splitk_partial_lse(int32_t reader_id = 0) const;

 private:
  enum ReaderLifecycle : uint32_t {
    READER_IDLE = 0,
    READER_ACTIVE = 1,
    READER_RETIRING = 2,
  };

  struct ReaderState {
    std::atomic<uint32_t> state{READER_IDLE};
    cudaEvent_t done_event{nullptr};
    std::vector<int32_t> pinned_slots;
  };

  int32_t pop_free_page();
  void push_free_page(int32_t page_id);
  void release_page_ref(int32_t page_id);
  void try_reclaim_slot(int32_t stream_id, int32_t global_slot, bool wait_gpu_write = false);
  void retire_reader(int32_t reader_id, bool sync_if_busy);

  int32_t num_streams_;
  int32_t num_readers_;
  int32_t slots_per_stream_;
  int32_t total_slots_;
  int32_t max_pages_;
  int32_t num_layers_;
  int32_t page_size_;
  int32_t num_kv_heads_;
  int32_t head_dim_;
  int32_t max_seq_len_;
  int32_t max_pages_per_slot_;
  bool kv_mode_;
  int32_t device_id_;

  SlotHeader* headers_{nullptr};
  std::atomic<int32_t>* latest_slot_{nullptr};
  std::unique_ptr<std::atomic<int32_t>[]> publishing_slot_;
  std::unique_ptr<std::atomic<uint64_t>[]> stream_version_gen_;

  std::vector<int32_t> host_page_table_;
  std::vector<int32_t> slot_write_start_page_;

  // Lock-free tagged Treiber page stack + atomic page refcounts for zero-copy sliding-window co-pooling
  std::atomic<uint64_t> free_head_{0};
  std::atomic<int32_t> free_count_{0};
  std::unique_ptr<std::atomic<int32_t>[]> next_free_page_;
  std::unique_ptr<std::atomic<int32_t>[]> page_ref_count_;

  torch::Tensor k_pool_;
  torch::Tensor v_pool_;
  torch::Tensor page_t_obs_ns_;

  torch::Tensor splitk_partial_out_;
  torch::Tensor splitk_partial_lse_;

  std::vector<cudaEvent_t> write_ready_events_;
  std::vector<std::unique_ptr<ReaderState>> reader_states_;

  int32_t pinned_capacity_{0};
  cudaEvent_t h2d_write_staging_done_ev_{nullptr};
  int32_t* h_pinned_write_pt_{nullptr};
  int32_t* h_pinned_write_start_{nullptr};
  int32_t* h_pinned_page_table_{nullptr};
  int32_t* h_pinned_seq_lens_{nullptr};
  int64_t* h_pinned_t_now_{nullptr};
};

void launch_write_paged_kv(
    const torch::Tensor& k_in,
    const torch::Tensor& v_in,
    torch::Tensor& k_pool,
    torch::Tensor& v_pool,
    torch::Tensor& page_t_obs_ns,
    const torch::Tensor& page_table,
    const torch::Tensor& start_page_idx,
    const torch::Tensor& t_obs_ns,
    int32_t layer_idx,
    bool bake_spatial_rope,
    int32_t spatial_grid_w,
    cudaStream_t stream);

void launch_write_paged_hidden(
    const torch::Tensor& hidden_in,
    torch::Tensor& v_pool,
    torch::Tensor& page_t_obs_ns,
    const torch::Tensor& page_table,
    const torch::Tensor& start_page_idx,
    const torch::Tensor& t_obs_ns,
    cudaStream_t stream);

void launch_gather_paged_hidden(
    const torch::Tensor& v_pool,
    const torch::Tensor& page_t_obs_ns,
    const int32_t* d_page_table,
    const int32_t* d_seq_lens,
    const int64_t* d_t_now_ns,
    torch::Tensor& out,
    torch::Tensor& out_delta_sec,
    int32_t batch_size,
    int32_t max_seq_len,
    int32_t max_pages_per_slot,
    cudaStream_t stream);

void launch_gather_paged_kv_history(
    const torch::Tensor& k_pool,
    const torch::Tensor& v_pool,
    const torch::Tensor& page_t_obs_ns,
    const torch::Tensor& page_table,
    const torch::Tensor& start_page_idx,
    const torch::Tensor& t_obs_ns,
    torch::Tensor& k_out,
    torch::Tensor& v_out,
    int32_t layer_idx,
    bool apply_rel_time_rope,
    cudaStream_t stream);

void launch_paged_rel_rope_cross_attn(
    const torch::Tensor& q,
    const torch::Tensor& k_pool,
    const torch::Tensor& v_pool,
    const torch::Tensor& page_t_obs,
    torch::Tensor& partial_out_workspace,
    torch::Tensor& partial_lse_workspace,
    const int32_t* d_page_table,
    const int32_t* d_seq_lens,
    const int64_t* d_t_now_ns,
    torch::Tensor& out,
    int32_t batch_size,
    int32_t max_pages_per_slot,
    int32_t layer_idx,
    bool apply_rel_time_rope,
    cudaStream_t stream,
    float* d_out_lse = nullptr);

}  // namespace tempo
