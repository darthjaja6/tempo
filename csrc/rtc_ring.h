#pragma once

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <mutex>
#include <tuple>

namespace tempo {

struct RTCStats {
  uint64_t total_commits{0};
  uint64_t soft_overruns{0};
  uint64_t hard_underruns{0};
};

struct PendingRTCSample {
  std::unique_lock<std::mutex> lk;
  c10::cuda::CUDAStream caller_stream;
  torch::Tensor out_act;
  torch::Tensor out_fast_aoi;
  torch::Tensor out_slow_aoi;
  torch::Tensor d_underrun;
};

class RTCHorizonRing {
 public:
  RTCHorizonRing(
      int32_t num_streams,
      int32_t horizon_steps,
      int32_t prefix_steps,
      int32_t action_dim,
      int64_t step_dt_ns,
      int32_t device_id = 0);

  ~RTCHorizonRing();

  RTCHorizonRing(const RTCHorizonRing&) = delete;
  RTCHorizonRing& operator=(const RTCHorizonRing&) = delete;

  void reset_all(c10::optional<torch::Tensor> init_pose = c10::nullopt);
  void reset_stats();

  // Extract [B, prefix_steps, action_dim] bf16 frozen prefix and [B] bool validity mask on GPU.
  void extract_frozen_prefix(
      const int32_t* d_stream_ids,
      const int64_t* d_t_obs_ns,
      int32_t batch_size,
      torch::Tensor& out_prefix_bf16,
      torch::Tensor& out_prefix_valid_bool,
      cudaStream_t stream);

  // Commit newly generated [B, horizon_steps, action_dim] bf16 chunks at completion time t_commit_ns.
  void commit_chunks(
      const int32_t* d_stream_ids,
      const int64_t* d_t_obs_ns,
      const int64_t* d_slow_t_obs_ns,
      int64_t t_commit_ns,
      int64_t deadline_ns,
      const torch::Tensor& new_chunks_bf16,
      int32_t batch_size,
      cudaStream_t stream);

  // Sample continuous interpolated actions [N, action_dim] float32 and AoI metrics at t_query_ns.
  PendingRTCSample launch_sample_async(int64_t t_query_ns);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> finish_sample_async(PendingRTCSample pending);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> sample_at(int64_t t_query_ns);

  RTCStats stats() const;
  int64_t max_staged_splice_ns() const;

  int32_t num_streams() const { return num_streams_; }
  int32_t device_id() const { return device_id_; }
  int32_t horizon_steps() const { return horizon_steps_; }
  int32_t prefix_steps() const { return prefix_steps_; }
  int32_t action_dim() const { return action_dim_; }
  int64_t step_dt_ns() const { return step_dt_ns_; }

 private:
  int32_t num_streams_;
  int32_t horizon_steps_;
  int32_t prefix_steps_;
  int32_t action_dim_;
  int64_t step_dt_ns_;
  int32_t device_id_;
  bool has_init_pose_{false};

  mutable std::mutex mu_;
  c10::cuda::CUDAStream rtc_stream_;
  cudaEvent_t exec_done_ev_{nullptr};
  cudaEvent_t sample_done_ev_{nullptr};

  // Active trajectory state on GPU
  torch::Tensor active_chunks_;        // [N, H, A] float32
  torch::Tensor active_t_obs_ns_;      // [N] int64
  torch::Tensor active_slow_t_obs_ns_; // [N] int64
  torch::Tensor active_valid_;         // [N] int32

  // Staged trajectory awaiting t_splice_ns
  torch::Tensor staged_chunks_;        // [N, H, A] float32
  torch::Tensor staged_t_obs_ns_;      // [N] int64
  torch::Tensor staged_t_splice_ns_;   // [N] int64
  torch::Tensor staged_slow_t_obs_ns_; // [N] int64
  torch::Tensor staged_valid_;         // [N] int32

  // Atomic device counters [total_commits, soft_overruns, hard_underruns]
  torch::Tensor d_counters_;
};

}  // namespace tempo
