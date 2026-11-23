#include "rtc_ring.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tempo {

#define TS_RTC_CUDA_CHECK(expr)                                                \
  do {                                                                         \
    cudaError_t err__ = (expr);                                                \
    if (err__ != cudaSuccess) {                                                \
      throw std::runtime_error(                                                \
          std::string("CUDA error at ") + __FILE__ + ":" +                     \
          std::to_string(__LINE__) + ": " + cudaGetErrorString(err__));        \
    }                                                                          \
  } while (0)

namespace {

__device__ __forceinline__ float eval_chunk_interp(
    const float* chunk_base,
    int32_t H,
    int32_t A,
    int32_t a,
    int64_t t_ns,
    int64_t chunk_t_obs_ns,
    int64_t step_dt_ns) {
  float f_step = static_cast<float>(t_ns - chunk_t_obs_ns) / static_cast<float>(step_dt_ns);
  f_step = fmaxf(0.0f, fminf(f_step, static_cast<float>(H - 1)));
  int32_t idx0 = static_cast<int32_t>(floorf(f_step));
  int32_t idx1 = min(idx0 + 1, H - 1);
  float frac = f_step - static_cast<float>(idx0);
  float v0 = chunk_base[idx0 * A + a];
  float v1 = chunk_base[idx1 * A + a];
  return v0 + frac * (v1 - v0);
}

__global__ void extract_frozen_prefix_kernel(
    const int32_t* __restrict__ stream_ids,
    const int64_t* __restrict__ t_obs_ns,
    const float* __restrict__ active_chunks,
    const int64_t* __restrict__ active_t_obs_ns,
    const int32_t* __restrict__ active_valid,
    const float* __restrict__ staged_chunks,
    const int64_t* __restrict__ staged_t_obs_ns,
    const int64_t* __restrict__ staged_t_splice_ns,
    const int32_t* __restrict__ staged_valid,
    bool has_init_pose,
    __nv_bfloat16* __restrict__ out_prefix,
    bool* __restrict__ out_prefix_valid,
    int32_t B,
    int32_t max_B,
    int32_t H,
    int32_t P_steps,
    int32_t A,
    int64_t step_dt_ns) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t total = max_B * P_steps * A;
  if (idx >= total) return;

  const int32_t a = idx % A;
  int32_t rem = idx / A;
  const int32_t p = rem % P_steps;
  const int32_t b = rem / P_steps;

  if (b >= B) {
    if (p == 0 && a == 0) {
      out_prefix_valid[b] = false;
    }
    out_prefix[idx] = __float2bfloat16(0.0f);
    return;
  }

  const int32_t s = stream_ids[b];
  if (p == 0 && a == 0) {
    out_prefix_valid[b] = has_init_pose || (active_valid[s] != 0) || (staged_valid[s] != 0);
  }
  const int64_t t_step = t_obs_ns[b] + static_cast<int64_t>(p) * step_dt_ns;

  float val;
  if (staged_valid[s] && t_step >= staged_t_splice_ns[s]) {
    val = eval_chunk_interp(
        staged_chunks + static_cast<int64_t>(s) * H * A, H, A, a, t_step, staged_t_obs_ns[s], step_dt_ns);
  } else if (active_valid[s]) {
    val = eval_chunk_interp(
        active_chunks + static_cast<int64_t>(s) * H * A, H, A, a, t_step, active_t_obs_ns[s], step_dt_ns);
  } else {
    val = active_chunks[static_cast<int64_t>(s) * H * A + a];
  }
  out_prefix[idx] = __float2bfloat16(val);
}

__global__ void commit_chunks_kernel(
    const int32_t* __restrict__ stream_ids,
    const int64_t* __restrict__ t_obs_ns,
    const int64_t* __restrict__ slow_t_obs_ns,
    int64_t t_commit_ns,
    int64_t deadline_ns,
    bool has_init_pose,
    const __nv_bfloat16* __restrict__ new_chunks,
    float* __restrict__ active_chunks,
    int64_t* __restrict__ active_t_obs_ns,
    int64_t* __restrict__ active_slow_t_obs_ns,
    int32_t* __restrict__ active_valid,
    float* __restrict__ staged_chunks,
    int64_t* __restrict__ staged_t_obs_ns,
    int64_t* __restrict__ staged_t_splice_ns,
    int64_t* __restrict__ staged_slow_t_obs_ns,
    int32_t* __restrict__ staged_valid,
    unsigned long long* __restrict__ counters,
    int32_t B,
    int32_t H,
    int32_t A,
    int64_t step_dt_ns) {
  const int32_t b = blockIdx.x;
  if (b >= B) return;
  const int32_t s = stream_ids[b];
  const int64_t obs_ns = t_obs_ns[b];
  const int64_t splice_ns = obs_ns + deadline_ns;
  const int64_t horizon_ns = static_cast<int64_t>(H - 1) * step_dt_ns;

  if (staged_valid[s] && t_commit_ns >= staged_t_splice_ns[s]) {
    for (int32_t idx = threadIdx.x; idx < H * A; idx += blockDim.x) {
      active_chunks[static_cast<int64_t>(s) * H * A + idx] =
          staged_chunks[static_cast<int64_t>(s) * H * A + idx];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      active_t_obs_ns[s] = staged_t_obs_ns[s];
      active_slow_t_obs_ns[s] = staged_slow_t_obs_ns[s];
      active_valid[s] = 1;
      staged_valid[s] = 0;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(&counters[0], 1ULL);
  }

  const bool was_valid = (active_valid[s] != 0);
  if (!was_valid && (!has_init_pose || t_commit_ns <= splice_ns)) {
    for (int32_t idx = threadIdx.x; idx < H * A; idx += blockDim.x) {
      active_chunks[static_cast<int64_t>(s) * H * A + idx] =
          __bfloat162float(new_chunks[static_cast<int64_t>(b) * H * A + idx]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      active_t_obs_ns[s] = obs_ns;
      active_slow_t_obs_ns[s] = slow_t_obs_ns[b];
      active_valid[s] = 1;
      staged_valid[s] = 0;
    }
    return;
  }

  const int64_t active_end_ns = active_t_obs_ns[s] + horizon_ns;
  if (was_valid && t_commit_ns <= splice_ns && splice_ns <= active_end_ns) {
    for (int32_t idx = threadIdx.x; idx < H * A; idx += blockDim.x) {
      staged_chunks[static_cast<int64_t>(s) * H * A + idx] =
          __bfloat162float(new_chunks[static_cast<int64_t>(b) * H * A + idx]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      staged_t_obs_ns[s] = obs_ns;
      staged_t_splice_ns[s] = splice_ns;
      staged_slow_t_obs_ns[s] = slow_t_obs_ns[b];
      staged_valid[s] = 1;
    }
  } else {
    const bool prev_covered = !was_valid || (t_commit_ns <= active_end_ns);
    if (threadIdx.x == 0) {
      if (prev_covered) {
        atomicAdd(&counters[1], 1ULL);
      } else {
        atomicAdd(&counters[2], 1ULL);
      }
    }

    // Smoothly blend over up to 5 steps starting at t_commit_ns from active/held endpoint into new trajectory,
    // clamping blend window to H - start_step so the horizon endpoint H - 1 always reaches 100% new trajectory.
    const int32_t start_step =
        max(0, min(H - 1, static_cast<int32_t>((t_commit_ns - obs_ns) / step_dt_ns)));
    const int32_t eff_blend = min(5, H - start_step);
    float* act_ptr = active_chunks + static_cast<int64_t>(s) * H * A;
    const __nv_bfloat16* new_ptr = new_chunks + static_cast<int64_t>(b) * H * A;
    const int64_t old_obs_ns = was_valid ? active_t_obs_ns[s] : obs_ns;

    for (int32_t idx = threadIdx.x; idx < H * A; idx += blockDim.x) {
      const int32_t h = idx / A;
      const int32_t a = idx % A;
      float target_v = __bfloat162float(new_ptr[idx]);
      if (!was_valid && h < start_step) {
        target_v = act_ptr[idx];
      } else if (eff_blend >= 2 && h >= start_step && h < start_step + eff_blend) {
        int64_t t_h = obs_ns + static_cast<int64_t>(h) * step_dt_ns;
        float old_v = eval_chunk_interp(act_ptr, H, A, a, t_h, old_obs_ns, step_dt_ns);
        float w = static_cast<float>(h - start_step) / static_cast<float>(eff_blend - 1);
        float alpha = w * w * (3.0f - 2.0f * w);
        target_v = old_v + alpha * (target_v - old_v);
      }
      staged_chunks[static_cast<int64_t>(s) * H * A + idx] = target_v;
    }
    __syncthreads();
    for (int32_t idx = threadIdx.x; idx < H * A; idx += blockDim.x) {
      act_ptr[idx] = staged_chunks[static_cast<int64_t>(s) * H * A + idx];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      active_t_obs_ns[s] = obs_ns;
      active_slow_t_obs_ns[s] = slow_t_obs_ns[b];
      active_valid[s] = 1;
      staged_valid[s] = 0;
    }
  }
}

__global__ void sample_at_kernel(
    int64_t t_query_ns,
    float* __restrict__ active_chunks,
    int64_t* __restrict__ active_t_obs_ns,
    int64_t* __restrict__ active_slow_t_obs_ns,
    int32_t* __restrict__ active_valid,
    const float* __restrict__ staged_chunks,
    const int64_t* __restrict__ staged_t_obs_ns,
    const int64_t* __restrict__ staged_t_splice_ns,
    const int64_t* __restrict__ staged_slow_t_obs_ns,
    int32_t* __restrict__ staged_valid,
    float* __restrict__ out_actions,
    float* __restrict__ out_fast_aoi_ms,
    float* __restrict__ out_slow_aoi_ms,
    unsigned long long* __restrict__ underrun_this_tick,
    int32_t N,
    int32_t H,
    int32_t A,
    int64_t step_dt_ns) {
  const int32_t s = blockIdx.x;
  if (s >= N) return;

  if (staged_valid[s] && t_query_ns >= staged_t_splice_ns[s]) {
    for (int32_t idx = threadIdx.x; idx < H * A; idx += blockDim.x) {
      active_chunks[static_cast<int64_t>(s) * H * A + idx] =
          staged_chunks[static_cast<int64_t>(s) * H * A + idx];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      active_t_obs_ns[s] = staged_t_obs_ns[s];
      active_slow_t_obs_ns[s] = staged_slow_t_obs_ns[s];
      active_valid[s] = 1;
      staged_valid[s] = 0;
    }
    __syncthreads();
  }

  if (!active_valid[s]) {
    for (int32_t a = threadIdx.x; a < A; a += blockDim.x) {
      out_actions[s * A + a] = active_chunks[static_cast<int64_t>(s) * H * A + a];
    }
    if (threadIdx.x == 0) {
      out_fast_aoi_ms[s] = -1.0f;
      out_slow_aoi_ms[s] = -1.0f;
    }
    return;
  }

  const int64_t horizon_ns = static_cast<int64_t>(H - 1) * step_dt_ns;
  const int64_t fast_age_ns = t_query_ns - active_t_obs_ns[s];
  const int64_t slow_age_ns = t_query_ns - active_slow_t_obs_ns[s];

  if (threadIdx.x == 0) {
    out_fast_aoi_ms[s] = static_cast<float>(fast_age_ns) * 1e-6f;
    out_slow_aoi_ms[s] = (active_slow_t_obs_ns[s] > 0) ? (static_cast<float>(slow_age_ns) * 1e-6f) : -1.0f;
    if (fast_age_ns > horizon_ns) {
      atomicAdd(underrun_this_tick, 1ULL);
    }
  }

  const float* act_ptr = active_chunks + static_cast<int64_t>(s) * H * A;
  for (int32_t a = threadIdx.x; a < A; a += blockDim.x) {
    out_actions[s * A + a] =
        eval_chunk_interp(act_ptr, H, A, a, t_query_ns, active_t_obs_ns[s], step_dt_ns);
  }
}

}  // namespace

RTCHorizonRing::RTCHorizonRing(
    int32_t num_streams,
    int32_t horizon_steps,
    int32_t prefix_steps,
    int32_t action_dim,
    int64_t step_dt_ns,
    int32_t device_id)
    : num_streams_(num_streams),
      horizon_steps_(horizon_steps),
      prefix_steps_(prefix_steps),
      action_dim_(action_dim),
      step_dt_ns_(step_dt_ns),
      device_id_(device_id),
      rtc_stream_(c10::cuda::getStreamFromPool(/*isHighPriority=*/true, device_id)) {
  c10::cuda::CUDAGuard guard(device_id_);
  TS_RTC_CUDA_CHECK(cudaEventCreateWithFlags(&exec_done_ev_, cudaEventDisableTiming));
  TS_RTC_CUDA_CHECK(cudaEventCreateWithFlags(&sample_done_ev_, cudaEventDisableTiming));

  auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device_id_);
  auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, device_id_);
  auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_id_);

  active_chunks_ = torch::zeros({num_streams_, horizon_steps_, action_dim_}, f32);
  active_t_obs_ns_ = torch::zeros({num_streams_}, i64);
  active_slow_t_obs_ns_ = torch::zeros({num_streams_}, i64);
  active_valid_ = torch::zeros({num_streams_}, i32);

  staged_chunks_ = torch::zeros({num_streams_, horizon_steps_, action_dim_}, f32);
  staged_t_obs_ns_ = torch::zeros({num_streams_}, i64);
  staged_t_splice_ns_ = torch::zeros({num_streams_}, i64);
  staged_slow_t_obs_ns_ = torch::zeros({num_streams_}, i64);
  staged_valid_ = torch::zeros({num_streams_}, i32);

  d_counters_ = torch::zeros({3}, i64);
}

RTCHorizonRing::~RTCHorizonRing() {
  c10::cuda::CUDAGuard guard(device_id_);
  if (exec_done_ev_) cudaEventDestroy(exec_done_ev_);
  if (sample_done_ev_) cudaEventDestroy(sample_done_ev_);
}

void RTCHorizonRing::reset_all(c10::optional<torch::Tensor> init_pose) {
  std::lock_guard<std::mutex> lk(mu_);
  c10::cuda::CUDAGuard dev_guard(device_id_);
  c10::cuda::CUDAStreamGuard guard(rtc_stream_);
  TS_RTC_CUDA_CHECK(cudaStreamWaitEvent(rtc_stream_.stream(), exec_done_ev_, 0));
  if (init_pose.has_value() && init_pose->defined()) {
    torch::Tensor pose = init_pose->to(torch::Device(torch::kCUDA, device_id_), torch::kFloat32);
    if (pose.dim() == 1 && pose.size(0) == action_dim_) {
      active_chunks_.copy_(pose.view({1, 1, action_dim_}).expand_as(active_chunks_));
    } else if (pose.dim() == 2 && pose.size(0) == num_streams_ && pose.size(1) == action_dim_) {
      active_chunks_.copy_(pose.view({num_streams_, 1, action_dim_}).expand_as(active_chunks_));
    } else {
      throw std::invalid_argument("RTCHorizonRing::reset_all init_pose must be [action_dim] or [num_streams, action_dim]");
    }
    has_init_pose_ = true;
  } else {
    active_chunks_.zero_();
    has_init_pose_ = false;
  }
  active_t_obs_ns_.zero_();
  active_slow_t_obs_ns_.zero_();
  active_valid_.zero_();
  staged_valid_.zero_();
  d_counters_.zero_();
  TS_RTC_CUDA_CHECK(cudaEventRecord(sample_done_ev_, rtc_stream_.stream()));
  TS_RTC_CUDA_CHECK(cudaStreamSynchronize(rtc_stream_.stream()));
}

void RTCHorizonRing::reset_stats() {
  std::lock_guard<std::mutex> lk(mu_);
  c10::cuda::CUDAGuard dev_guard(device_id_);
  c10::cuda::CUDAStreamGuard guard(rtc_stream_);
  TS_RTC_CUDA_CHECK(cudaStreamWaitEvent(rtc_stream_.stream(), exec_done_ev_, 0));
  d_counters_.zero_();
  TS_RTC_CUDA_CHECK(cudaEventRecord(sample_done_ev_, rtc_stream_.stream()));
  TS_RTC_CUDA_CHECK(cudaStreamSynchronize(rtc_stream_.stream()));
}

void RTCHorizonRing::extract_frozen_prefix(
    const int32_t* d_stream_ids,
    const int64_t* d_t_obs_ns,
    int32_t batch_size,
    torch::Tensor& out_prefix_bf16,
    torch::Tensor& out_prefix_valid_bool,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lk(mu_);
  c10::cuda::CUDAGuard dev_guard(device_id_);
  TS_RTC_CUDA_CHECK(cudaStreamWaitEvent(stream, sample_done_ev_, 0));
  const int32_t max_B = static_cast<int32_t>(out_prefix_bf16.size(0));
  const int32_t total = max_B * prefix_steps_ * action_dim_;
  const int32_t threads = 128;
  const int32_t blocks = (total + threads - 1) / threads;
  extract_frozen_prefix_kernel<<<blocks, threads, 0, stream>>>(
      d_stream_ids,
      d_t_obs_ns,
      active_chunks_.data_ptr<float>(),
      active_t_obs_ns_.data_ptr<int64_t>(),
      active_valid_.data_ptr<int32_t>(),
      staged_chunks_.data_ptr<float>(),
      staged_t_obs_ns_.data_ptr<int64_t>(),
      staged_t_splice_ns_.data_ptr<int64_t>(),
      staged_valid_.data_ptr<int32_t>(),
      has_init_pose_,
      reinterpret_cast<__nv_bfloat16*>(out_prefix_bf16.data_ptr<at::BFloat16>()),
      out_prefix_valid_bool.data_ptr<bool>(),
      batch_size,
      max_B,
      horizon_steps_,
      prefix_steps_,
      action_dim_,
      step_dt_ns_);
  TS_RTC_CUDA_CHECK(cudaGetLastError());
  TS_RTC_CUDA_CHECK(cudaEventRecord(exec_done_ev_, stream));
}

void RTCHorizonRing::commit_chunks(
    const int32_t* d_stream_ids,
    const int64_t* d_t_obs_ns,
    const int64_t* d_slow_t_obs_ns,
    int64_t t_commit_ns,
    int64_t deadline_ns,
    const torch::Tensor& new_chunks_bf16,
    int32_t batch_size,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lk(mu_);
  c10::cuda::CUDAGuard dev_guard(device_id_);
  TS_RTC_CUDA_CHECK(cudaStreamWaitEvent(stream, sample_done_ev_, 0));
  commit_chunks_kernel<<<batch_size, 128, 0, stream>>>(
      d_stream_ids,
      d_t_obs_ns,
      d_slow_t_obs_ns,
      t_commit_ns,
      deadline_ns,
      has_init_pose_,
      reinterpret_cast<const __nv_bfloat16*>(new_chunks_bf16.data_ptr<at::BFloat16>()),
      active_chunks_.data_ptr<float>(),
      active_t_obs_ns_.data_ptr<int64_t>(),
      active_slow_t_obs_ns_.data_ptr<int64_t>(),
      active_valid_.data_ptr<int32_t>(),
      staged_chunks_.data_ptr<float>(),
      staged_t_obs_ns_.data_ptr<int64_t>(),
      staged_t_splice_ns_.data_ptr<int64_t>(),
      staged_slow_t_obs_ns_.data_ptr<int64_t>(),
      staged_valid_.data_ptr<int32_t>(),
      reinterpret_cast<unsigned long long*>(d_counters_.data_ptr<int64_t>()),
      batch_size,
      horizon_steps_,
      action_dim_,
      step_dt_ns_);
  TS_RTC_CUDA_CHECK(cudaGetLastError());
  TS_RTC_CUDA_CHECK(cudaEventRecord(exec_done_ev_, stream));
}

PendingRTCSample RTCHorizonRing::launch_sample_async(int64_t t_query_ns) {
  c10::cuda::CUDAGuard dev_guard(device_id_);
  c10::cuda::CUDAStream caller_stream = at::cuda::getCurrentCUDAStream(device_id_);
  std::unique_lock<std::mutex> lk(mu_);
  c10::cuda::CUDAStreamGuard guard(rtc_stream_);
  cudaStream_t st = rtc_stream_.stream();
  TS_RTC_CUDA_CHECK(cudaStreamWaitEvent(st, exec_done_ev_, 0));

  auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device_id_);
  auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, device_id_);

  auto out_act = torch::empty({num_streams_, action_dim_}, f32);
  auto out_fast_aoi = torch::empty({num_streams_}, f32);
  auto out_slow_aoi = torch::empty({num_streams_}, f32);
  auto d_underrun = torch::zeros({1}, i64);

  sample_at_kernel<<<num_streams_, 64, 0, st>>>(
      t_query_ns,
      active_chunks_.data_ptr<float>(),
      active_t_obs_ns_.data_ptr<int64_t>(),
      active_slow_t_obs_ns_.data_ptr<int64_t>(),
      active_valid_.data_ptr<int32_t>(),
      staged_chunks_.data_ptr<float>(),
      staged_t_obs_ns_.data_ptr<int64_t>(),
      staged_t_splice_ns_.data_ptr<int64_t>(),
      staged_slow_t_obs_ns_.data_ptr<int64_t>(),
      staged_valid_.data_ptr<int32_t>(),
      out_act.data_ptr<float>(),
      out_fast_aoi.data_ptr<float>(),
      out_slow_aoi.data_ptr<float>(),
      reinterpret_cast<unsigned long long*>(d_underrun.data_ptr<int64_t>()),
      num_streams_,
      horizon_steps_,
      action_dim_,
      step_dt_ns_);
  TS_RTC_CUDA_CHECK(cudaGetLastError());
  TS_RTC_CUDA_CHECK(cudaEventRecord(sample_done_ev_, st));

  return PendingRTCSample{
      std::move(lk),
      caller_stream,
      std::move(out_act),
      std::move(out_fast_aoi),
      std::move(out_slow_aoi),
      std::move(d_underrun)};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> RTCHorizonRing::finish_sample_async(
    PendingRTCSample pending) {
  c10::cuda::CUDAGuard dev_guard(device_id_);
  TS_RTC_CUDA_CHECK(cudaEventSynchronize(sample_done_ev_));

  if (pending.caller_stream != rtc_stream_) {
    c10::cuda::CUDACachingAllocator::recordStream(pending.out_act.storage().data_ptr(), pending.caller_stream);
    c10::cuda::CUDACachingAllocator::recordStream(pending.out_fast_aoi.storage().data_ptr(), pending.caller_stream);
    c10::cuda::CUDACachingAllocator::recordStream(pending.out_slow_aoi.storage().data_ptr(), pending.caller_stream);
  }

  int64_t underruns = pending.d_underrun.cpu().item<int64_t>();
  return {
      std::move(pending.out_act),
      std::move(pending.out_fast_aoi),
      std::move(pending.out_slow_aoi),
      underruns};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> RTCHorizonRing::sample_at(int64_t t_query_ns) {
  return finish_sample_async(launch_sample_async(t_query_ns));
}

RTCStats RTCHorizonRing::stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  c10::cuda::CUDAGuard guard(device_id_);
  TS_RTC_CUDA_CHECK(cudaEventSynchronize(exec_done_ev_));
  auto c = d_counters_.cpu();
  const int64_t* ptr = c.data_ptr<int64_t>();
  return RTCStats{
      static_cast<uint64_t>(ptr[0]),
      static_cast<uint64_t>(ptr[1]),
      static_cast<uint64_t>(ptr[2])};
}

int64_t RTCHorizonRing::max_staged_splice_ns() const {
  std::lock_guard<std::mutex> lk(mu_);
  c10::cuda::CUDAGuard guard(device_id_);
  TS_RTC_CUDA_CHECK(cudaEventSynchronize(exec_done_ev_));
  auto valid = staged_valid_.cpu();
  auto splice = staged_t_splice_ns_.cpu();
  const int32_t* v_ptr = valid.data_ptr<int32_t>();
  const int64_t* s_ptr = splice.data_ptr<int64_t>();
  int64_t max_ns = 0;
  for (int32_t i = 0; i < num_streams_; ++i) {
    if (v_ptr[i] != 0 && s_ptr[i] > max_ns) max_ns = s_ptr[i];
  }
  return max_ns;
}

}  // namespace tempo
