#include "tsc_pool.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace tempo {

#define TS_CUDA_CHECK(expr)                                                    \
  do {                                                                         \
    cudaError_t err__ = (expr);                                                \
    if (err__ != cudaSuccess) {                                                \
      throw std::runtime_error(                                                \
          std::string("CUDA error at ") + __FILE__ + ":" +                     \
          std::to_string(__LINE__) + ": " + cudaGetErrorString(err__));        \
    }                                                                          \
  } while (0)

namespace {

constexpr uint32_t kInvalidPageIdx = 0xFFFFFFFFu;

__device__ __forceinline__ float rope_inv_freq(int pair_idx, int pair_count) {
  return exp2f(-13.28771238f * static_cast<float>(pair_idx) / static_cast<float>(max(pair_count, 1)));
}

__global__ void write_paged_kv_kernel(
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,
    __nv_bfloat16* __restrict__ k_pool,
    __nv_bfloat16* __restrict__ v_pool,
    int64_t* __restrict__ page_t_obs_ns,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ start_page_idx,
    const int64_t* __restrict__ t_obs_ns,
    int32_t B,
    int32_t S,
    int32_t L,
    int32_t P,
    int32_t H_kv,
    int32_t D,
    int32_t max_pages_per_slot,
    int32_t layer_idx,
    bool bake_spatial_rope,
    int32_t spatial_grid_w) {
  const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t half_D = D / 2;
  const int32_t total_pairs = B * S * H_kv * half_D;
  if (idx >= total_pairs) return;

  const int32_t pair_d = idx % half_D;
  int32_t rem = idx / half_D;
  const int32_t h_kv = rem % H_kv;
  rem /= H_kv;
  const int32_t s = rem % S;
  const int32_t b = rem / S;

  const int32_t logical_page = start_page_idx[b] + (s / P);
  if (logical_page < 0 || logical_page >= max_pages_per_slot) return;
  const int32_t page_offset = s % P;
  const int32_t phys_page = page_table[b * max_pages_per_slot + logical_page];
  if (phys_page < 0) return;

  if (layer_idx == 0 && page_offset == 0 && h_kv == 0 && pair_d == 0) {
    page_t_obs_ns[phys_page] = t_obs_ns[b];
  }

  const int64_t in_base = ((static_cast<int64_t>(b) * S + s) * H_kv + h_kv) * D;
  const int64_t out_base =
      ((((static_cast<int64_t>(phys_page) * L + layer_idx) * P + page_offset) * H_kv + h_kv) * D);

  const int32_t d0 = 2 * pair_d;
  const int32_t d1 = d0 + 1;

  float k0 = __bfloat162float(k_in[in_base + d0]);
  float k1 = __bfloat162float(k_in[in_base + d1]);
  const __nv_bfloat16 v0 = v_in[in_base + d0];
  const __nv_bfloat16 v1 = v_in[in_base + d1];

  const int32_t spatial_pairs = half_D / 2;
  if (bake_spatial_rope && pair_d < spatial_pairs) {
    const int32_t grid_w = max(spatial_grid_w, 1);
    const int32_t row = s / grid_w;
    const int32_t col = s % grid_w;
    const int32_t axis_pairs = max(spatial_pairs / 2, 1);
    float pos = (pair_d < axis_pairs) ? static_cast<float>(row) : static_cast<float>(col);
    int32_t local_p = (pair_d < axis_pairs) ? pair_d : (pair_d - axis_pairs);
    float angle = pos * rope_inv_freq(local_p, axis_pairs);
    float c, sn;
    __sincosf(angle, &sn, &c);
    float nk0 = k0 * c - k1 * sn;
    float nk1 = k0 * sn + k1 * c;
    k0 = nk0;
    k1 = nk1;
  }

  k_pool[out_base + d0] = __float2bfloat16(k0);
  k_pool[out_base + d1] = __float2bfloat16(k1);
  v_pool[out_base + d0] = v0;
  v_pool[out_base + d1] = v1;
}

__global__ void gather_paged_kv_history_kernel(
    const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool,
    const int64_t* __restrict__ page_t_obs_ns,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ start_page_idx,
    const int64_t* __restrict__ t_obs_ns,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int32_t B,
    int32_t hist_S,
    int32_t L,
    int32_t P,
    int32_t H_kv,
    int32_t D,
    int32_t max_pages_per_slot,
    int32_t layer_idx,
    bool apply_rel_time_rope) {
  const int32_t half_D = D >> 1;
  const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_pairs = static_cast<int64_t>(B) * hist_S * H_kv * half_D;
  if (pair_idx >= total_pairs) return;

  const int32_t pair_d = static_cast<int32_t>(pair_idx % half_D);
  int64_t rem = pair_idx / half_D;
  const int32_t h_kv = static_cast<int32_t>(rem % H_kv);
  rem /= H_kv;
  const int32_t s = static_cast<int32_t>(rem % hist_S);
  const int32_t b = static_cast<int32_t>(rem / hist_S);

  const int64_t out_base = (((static_cast<int64_t>(b) * hist_S + s) * H_kv + h_kv) * D) + 2 * pair_d;
  const int32_t hist_pages = start_page_idx[b];
  const int32_t lp = s / P;
  if (lp >= hist_pages) {
    k_out[out_base] = __float2bfloat16(0.0f);
    k_out[out_base + 1] = __float2bfloat16(0.0f);
    v_out[out_base] = __float2bfloat16(0.0f);
    v_out[out_base + 1] = __float2bfloat16(0.0f);
    return;
  }
  const int32_t phys_page = page_table[b * max_pages_per_slot + lp];
  if (phys_page < 0) {
    k_out[out_base] = __float2bfloat16(0.0f);
    k_out[out_base + 1] = __float2bfloat16(0.0f);
    v_out[out_base] = __float2bfloat16(0.0f);
    v_out[out_base + 1] = __float2bfloat16(0.0f);
    return;
  }

  const int64_t pool_base =
      ((((static_cast<int64_t>(phys_page) * L + layer_idx) * P + (s % P)) * H_kv + h_kv) * D) + 2 * pair_d;
  __nv_bfloat16 k0 = k_pool[pool_base];
  __nv_bfloat16 k1 = k_pool[pool_base + 1];

  const int32_t spatial_pairs = half_D >> 1;
  if (apply_rel_time_rope && pair_d >= spatial_pairs) {
    const int32_t temporal_pairs = half_D - spatial_pairs;
    const int64_t age_ns = t_obs_ns[b] - page_t_obs_ns[phys_page];
    const float age_sec = fmaxf(0.0f, fminf(static_cast<float>(age_ns) * 1e-9f, 60.0f));
    const float angle = -(age_sec * 100.0f) * rope_inv_freq(pair_d - spatial_pairs, temporal_pairs);
    float s_rot = 0.0f, c_rot = 1.0f;
    __sincosf(angle, &s_rot, &c_rot);
    const float fk0 = __bfloat162float(k0);
    const float fk1 = __bfloat162float(k1);
    k0 = __float2bfloat16(fk0 * c_rot - fk1 * s_rot);
    k1 = __float2bfloat16(fk0 * s_rot + fk1 * c_rot);
  }

  k_out[out_base] = k0;
  k_out[out_base + 1] = k1;
  v_out[out_base] = v_pool[pool_base];
  v_out[out_base + 1] = v_pool[pool_base + 1];
}

__global__ void write_paged_hidden_kernel(
    const __nv_bfloat16* __restrict__ hidden_in,
    __nv_bfloat16* __restrict__ v_pool,
    int64_t* __restrict__ page_t_obs_ns,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ start_page_idx,
    const int64_t* __restrict__ t_obs_ns,
    int32_t B,
    int32_t S,
    int32_t P,
    int32_t feat_dim,
    int32_t max_pages_per_slot) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(B) * S * feat_dim;
  if (idx >= total) return;

  const int32_t f = static_cast<int32_t>(idx % feat_dim);
  const int64_t rem = idx / feat_dim;
  const int32_t s = static_cast<int32_t>(rem % S);
  const int32_t b = static_cast<int32_t>(rem / S);

  const int32_t logical_page = start_page_idx[b] + (s / P);
  if (logical_page < 0 || logical_page >= max_pages_per_slot) return;
  const int32_t page_offset = s % P;
  const int32_t phys_page = page_table[b * max_pages_per_slot + logical_page];
  if (phys_page < 0) return;

  if (page_offset == 0 && f == 0) {
    page_t_obs_ns[phys_page] = t_obs_ns[b];
  }
  v_pool[(static_cast<int64_t>(phys_page) * P + page_offset) * feat_dim + f] = hidden_in[idx];
}

__global__ void gather_paged_hidden_kernel(
    const __nv_bfloat16* __restrict__ v_pool,
    const int64_t* __restrict__ page_t_obs_ns,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ seq_lens,
    const int64_t* __restrict__ t_now_ns,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ out_delta_sec,
    int32_t B,
    int32_t S,
    int32_t P,
    int32_t feat_dim,
    int32_t max_pages_per_slot) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(B) * S * feat_dim;
  if (idx >= total) return;

  const int32_t f = static_cast<int32_t>(idx % feat_dim);
  const int64_t rem = idx / feat_dim;
  const int32_t s = static_cast<int32_t>(rem % S);
  const int32_t b = static_cast<int32_t>(rem / S);

  if (s >= seq_lens[b]) {
    out[idx] = __float2bfloat16(0.0f);
    if (f == 0 && out_delta_sec != nullptr) {
      out_delta_sec[b * S + s] = -1.0f;
    }
    return;
  }
  const int32_t phys_page = page_table[b * max_pages_per_slot + (s / P)];
  if (phys_page < 0) {
    out[idx] = __float2bfloat16(0.0f);
    if (f == 0 && out_delta_sec != nullptr) {
      out_delta_sec[b * S + s] = -1.0f;
    }
    return;
  }
  out[idx] = v_pool[(static_cast<int64_t>(phys_page) * P + (s % P)) * feat_dim + f];
  if (f == 0 && out_delta_sec != nullptr) {
    const int64_t dt_ns = t_now_ns[b] - page_t_obs_ns[phys_page];
    out_delta_sec[b * S + s] = (dt_ns > 0) ? static_cast<float>(static_cast<double>(dt_ns) * 1e-9) : 0.0f;
  }
}

// High-Occupancy Register-Accumulator FlashDecoding Split-KV Tensor-Core (WMMA m16n16k16 BF16)
// Paged Cross-Attention templated over head_dim D in {64, 128, 256} (page_size P=16).
template <int SPLIT_K, int D>
__global__ void paged_rel_rope_wmma_splitk_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool,
    const int64_t* __restrict__ page_t_obs_ns,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ seq_lens,
    const int64_t* __restrict__ t_now_ns,
    float* __restrict__ partial_out,
    float* __restrict__ partial_lse,
    __nv_bfloat16* __restrict__ direct_out,
    float* __restrict__ out_lse,
    int32_t B,
    int32_t Q_len,
    int32_t H_q,
    int32_t H_kv,
    int32_t L,
    int32_t max_pages_per_slot,
    int32_t layer_idx,
    float sm_scale,
    bool apply_rel_time_rope) {
  namespace wmma = nvcuda::wmma;
  constexpr int WARPS_PER_BLOCK = 2;
  constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
  constexpr int P = 16;
  constexpr int V_SPLITS = (D == 256) ? 2 : 1;
  constexpr int OUT_D = D / V_SPLITS;
  constexpr int D_TILES = D / 16;
  constexpr int OUT_TILES = OUT_D / 16;
  constexpr int VEC_COLS = D / 8;
  constexpr int OUT_VEC_COLS = OUT_D / 8;
  constexpr int HALF_VEC_COLS = VEC_COLS / 2;
  constexpr int TEMPORAL_PAIRS = D / 4;
  constexpr int LOAD_STEPS = (P * VEC_COLS) / THREADS_PER_BLOCK;
  constexpr int OUT_LOAD_STEPS = (P * OUT_VEC_COLS) / THREADS_PER_BLOCK;
  constexpr int D_STRIDE = D + 8;
  constexpr int V_STRIDE = OUT_D + 8;
  constexpr int S_STRIDE = 24;

  union WarpTileScratch {
    alignas(16) float scores[16 * S_STRIDE];
    alignas(16) __nv_bfloat16 probs[16 * S_STRIDE];
  };
  struct SharedStorage {
    WarpTileScratch sp[WARPS_PER_BLOCK];
    alignas(16) float alpha[WARPS_PER_BLOCK][16];
    alignas(16) __nv_bfloat16 q[WARPS_PER_BLOCK][16 * D_STRIDE];
    alignas(16) __nv_bfloat16 k[P * D_STRIDE];
    alignas(16) __nv_bfloat16 v[P * V_STRIDE];
  };
  __shared__ SharedStorage smem;

  const int32_t split_id = blockIdx.x % SPLIT_K;
  const int32_t b_hkv = blockIdx.x / SPLIT_K;
  if (b_hkv >= B * H_kv) return;
  const int32_t v_split = static_cast<int32_t>(blockIdx.z);
  const int32_t v_col_off = v_split * OUT_D;
  const int32_t b = b_hkv / H_kv;
  const int32_t h_kv = b_hkv % H_kv;
  const int32_t gqa_group = H_q / H_kv;
  const int32_t total_q_per_kv = Q_len * gqa_group;

  const int32_t warp_id = threadIdx.x >> 5;
  const int32_t lane = threadIdx.x & 31;
  const int32_t warp_q_base = (static_cast<int32_t>(blockIdx.y) * WARPS_PER_BLOCK + warp_id) * 16;
  const int32_t S_len = seq_lens[b];
  const int32_t num_logical_pages = (S_len > 0) ? ((S_len + P - 1) / P) : 0;
  const int32_t pages_per_split = (num_logical_pages + SPLIT_K - 1) / SPLIT_K;
  const int32_t lp_start = split_id * pages_per_split;
  const int32_t lp_end = min(num_logical_pages, lp_start + pages_per_split);

  const int32_t row = lane >> 1;
  const int32_t c_base = (lane & 1) << 3;

  if (S_len <= 0 || lp_start >= lp_end) {
    if constexpr (SPLIT_K == 1) {
      if (warp_q_base < total_q_per_kv) {
        const int32_t q_flat = warp_q_base + row;
        if (q_flat < total_q_per_kv) {
          const int32_t q_pos = q_flat / gqa_group;
          const int32_t h_q = h_kv * gqa_group + (q_flat % gqa_group);
          const int64_t qh_idx = (static_cast<int64_t>(b) * Q_len + q_pos) * H_q + h_q;
          __nv_bfloat16* dst = &direct_out[qh_idx * D + v_col_off];
          const uint4 z = make_uint4(0, 0, 0, 0);
          #pragma unroll
          for (int dt = 0; dt < OUT_TILES; ++dt) {
            *reinterpret_cast<uint4*>(&dst[dt * 16 + c_base]) = z;
          }
          if (v_split == 0 && out_lse != nullptr && (lane & 1) == 0) {
            out_lse[qh_idx] = -1e30f;
          }
        }
      }
    } else {
      if (v_split == 0 && warp_q_base < total_q_per_kv && (lane & 1) == 0) {
        const int32_t q_flat = warp_q_base + row;
        if (q_flat < total_q_per_kv) {
          const int32_t q_pos = q_flat / gqa_group;
          const int32_t h_q = h_kv * gqa_group + (q_flat % gqa_group);
          partial_lse[(((static_cast<int64_t>(b) * SPLIT_K + split_id) * Q_len + q_pos) * H_q + h_q)] = -1e30f;
        }
      }
    }
    return;
  }

  float* smem_scores_w = smem.sp[warp_id].scores;
  __nv_bfloat16* smem_probs_w = smem.sp[warp_id].probs;
  float* smem_alpha_w = smem.alpha[warp_id];

  // Discover hardware accumulator fragment element -> matrix row mapping once in registers
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> row_idx_frag;
  constexpr int ACC_ELEMS = decltype(row_idx_frag)::num_elements;
  #pragma unroll
  for (int c = 0; c < 8; ++c) {
    smem_scores_w[row * S_STRIDE + c_base + c] = static_cast<float>(row);
  }
  __syncwarp();
  wmma::load_matrix_sync(row_idx_frag, smem_scores_w, S_STRIDE, wmma::mem_row_major);
  __syncwarp();
  int frag_row[ACC_ELEMS];
  #pragma unroll
  for (int e = 0; e < ACC_ELEMS; ++e) {
    frag_row[e] = __float2int_rn(row_idx_frag.x[e]);
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[OUT_TILES];
  __nv_bfloat16* smem_q_w = smem.q[warp_id];
  const int32_t q_flat_lane = warp_q_base + row;
  const int64_t q_row_off = (q_flat_lane < total_q_per_kv)
      ? (((static_cast<int64_t>(b) * Q_len + (q_flat_lane / gqa_group)) * H_q +
          (h_kv * gqa_group + (q_flat_lane % gqa_group))) * D)
      : -1;
  #pragma unroll
  for (int dt = 0; dt < OUT_TILES; ++dt) {
    wmma::fill_fragment(acc_frag[dt], 0.0f);
  }
  #pragma unroll
  for (int kt = 0; kt < D_TILES; ++kt) {
    uint4 q_vec = (q_row_off >= 0)
        ? *reinterpret_cast<const uint4*>(&q[q_row_off + kt * 16 + c_base])
        : make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(&smem_q_w[row * D_STRIDE + kt * 16 + c_base]) = q_vec;
  }

  float m_row = -1e30f;
  float l_row = 0.0f;
  bool has_accum = false;
  const int64_t now_ns = t_now_ns[b];

  const int32_t vec_col = threadIdx.x & (VEC_COLS - 1);
  const int32_t vec_row_grp = threadIdx.x / VEC_COLS;
  constexpr int ROWS_PER_STEP = THREADS_PER_BLOCK / VEC_COLS;
  const int32_t out_vec_col = threadIdx.x & (OUT_VEC_COLS - 1);
  const int32_t out_row_grp = threadIdx.x / OUT_VEC_COLS;
  constexpr int OUT_ROWS_PER_STEP = THREADS_PER_BLOCK / OUT_VEC_COLS;
  float inv_f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (vec_col >= HALF_VEC_COLS) {
    const int32_t bp = (vec_col - HALF_VEC_COLS) * 4;
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
      inv_f[p] = rope_inv_freq(bp + p, TEMPORAL_PAIRS);
    }
  }
  __syncthreads();

  for (int32_t lp = lp_start; lp < lp_end; ++lp) {
    const int32_t phys_page = page_table[b * max_pages_per_slot + lp];
    const int32_t valid_tokens = min(P, S_len - lp * P);

    float c_rot[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float s_rot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (phys_page >= 0 && apply_rel_time_rope && vec_col >= HALF_VEC_COLS) {
      const int64_t age_ns = now_ns - page_t_obs_ns[phys_page];
      const float age_scaled = -(fmaxf(0.0f, fminf(static_cast<float>(age_ns) * 1e-9f, 60.0f)) * 100.0f);
      #pragma unroll
      for (int p = 0; p < 4; ++p) {
        __sincosf(age_scaled * inv_f[p], &s_rot[p], &c_rot[p]);
      }
    }

    #pragma unroll
    for (int step = 0; step < LOAD_STEPS; ++step) {
      const int po = step * ROWS_PER_STEP + vec_row_grp;
      uint4 k_out = make_uint4(0, 0, 0, 0);
      if (phys_page >= 0 && po < valid_tokens) {
        const int64_t kv_off =
            ((((static_cast<int64_t>(phys_page) * L + layer_idx) * P + po) * H_kv + h_kv) * D) + 8 * vec_col;
        const uint4 k_vec = *reinterpret_cast<const uint4*>(&k_pool[kv_off]);
        if (apply_rel_time_rope && vec_col >= HALF_VEC_COLS) {
          const __nv_bfloat162* kp = reinterpret_cast<const __nv_bfloat162*>(&k_vec);
          __nv_bfloat162* ko = reinterpret_cast<__nv_bfloat162*>(&k_out);
          #pragma unroll
          for (int p = 0; p < 4; ++p) {
            const float2 kf = __bfloat1622float2(kp[p]);
            ko[p] = __floats2bfloat162_rn(
                kf.x * c_rot[p] - kf.y * s_rot[p],
                kf.x * s_rot[p] + kf.y * c_rot[p]);
          }
        } else {
          k_out = k_vec;
        }
      }
      *reinterpret_cast<uint4*>(&smem.k[po * D_STRIDE + 8 * vec_col]) = k_out;
    }

    #pragma unroll
    for (int step = 0; step < OUT_LOAD_STEPS; ++step) {
      const int po = step * OUT_ROWS_PER_STEP + out_row_grp;
      uint4 v_out = make_uint4(0, 0, 0, 0);
      if (phys_page >= 0 && po < valid_tokens) {
        const int64_t v_off =
            ((((static_cast<int64_t>(phys_page) * L + layer_idx) * P + po) * H_kv + h_kv) * D) +
            v_col_off + 8 * out_vec_col;
        v_out = *reinterpret_cast<const uint4*>(&v_pool[v_off]);
      }
      *reinterpret_cast<uint4*>(&smem.v[po * V_STRIDE + 8 * out_vec_col]) = v_out;
    }
    __syncthreads();

    if (warp_q_base < total_q_per_kv && phys_page >= 0 && valid_tokens > 0) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> score_frag;
      wmma::fill_fragment(score_frag, 0.0f);
      #pragma unroll
      for (int kt = 0; kt < D_TILES; ++kt) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;
        wmma::load_matrix_sync(q_frag, smem_q_w + kt * 16, D_STRIDE);
        wmma::load_matrix_sync(k_frag, smem.k + kt * 16, D_STRIDE);
        wmma::mma_sync(score_frag, q_frag, k_frag, score_frag);
      }
      wmma::store_matrix_sync(smem_scores_w, score_frag, S_STRIDE, wmma::mem_row_major);
      __syncwarp();

      float s_vals[8];
      float r_max = -1e30f;
      #pragma unroll
      for (int c = 0; c < 8; ++c) {
        const int col = c_base + c;
        const float sv = (col < valid_tokens) ? (smem_scores_w[row * S_STRIDE + col] * sm_scale) : -1e30f;
        s_vals[c] = sv;
        r_max = fmaxf(r_max, sv);
      }
      __syncwarp();

      r_max = fmaxf(r_max, __shfl_xor_sync(0xffffffffu, r_max, 1));
      const float m_new = fmaxf(m_row, r_max);
      const float alpha = __expf(m_row - m_new);

      float r_sum = 0.0f;
      #pragma unroll
      for (int c = 0; c < 8; ++c) {
        const int col = c_base + c;
        const float p = (col < valid_tokens) ? __expf(s_vals[c] - m_new) : 0.0f;
        r_sum += p;
        smem_probs_w[row * S_STRIDE + col] = __float2bfloat16(p);
      }
      r_sum += __shfl_xor_sync(0xffffffffu, r_sum, 1);
      l_row = l_row * alpha + r_sum;
      m_row = m_new;
      if ((lane & 1) == 0) {
        smem_alpha_w[row] = alpha;
      }
      __syncwarp();

      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> prob_frag;
      wmma::load_matrix_sync(prob_frag, smem_probs_w, S_STRIDE);
      __syncwarp();

      const bool need_rescale = has_accum && (__any_sync(0xffffffffu, alpha < 0.9999999f) != 0);
      if (need_rescale) {
        float e_alpha[ACC_ELEMS];
        #pragma unroll
        for (int e = 0; e < ACC_ELEMS; ++e) {
          e_alpha[e] = smem_alpha_w[frag_row[e]];
        }
        #pragma unroll
        for (int dt = 0; dt < OUT_TILES; ++dt) {
          #pragma unroll
          for (int e = 0; e < ACC_ELEMS; ++e) {
            acc_frag[dt].x[e] *= e_alpha[e];
          }
        }
      }

      #pragma unroll
      for (int dt = 0; dt < OUT_TILES; ++dt) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v_f;
        wmma::load_matrix_sync(v_f, smem.v + dt * 16, V_STRIDE);
        wmma::mma_sync(acc_frag[dt], prob_frag, v_f, acc_frag[dt]);
      }
      has_accum = true;
    }
    __syncthreads();
  }

  if (warp_q_base < total_q_per_kv) {
    if ((lane & 1) == 0) {
      smem_alpha_w[row] = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
      if (v_split == 0) {
        const float final_lse = (l_row > 0.0f) ? (m_row + __logf(l_row)) : -1e30f;
        if constexpr (SPLIT_K > 1) {
          if (q_flat_lane < total_q_per_kv) {
            const int32_t q_pos = q_flat_lane / gqa_group;
            const int32_t h_q = h_kv * gqa_group + (q_flat_lane % gqa_group);
            partial_lse[(((static_cast<int64_t>(b) * SPLIT_K + split_id) * Q_len + q_pos) * H_q + h_q)] = final_lse;
          }
        } else if (out_lse != nullptr && q_flat_lane < total_q_per_kv) {
          const int32_t q_pos = q_flat_lane / gqa_group;
          const int32_t h_q = h_kv * gqa_group + (q_flat_lane % gqa_group);
          out_lse[(static_cast<int64_t>(b) * Q_len + q_pos) * H_q + h_q] = final_lse;
        }
      }
    }
    __syncwarp();

    float e_inv[ACC_ELEMS];
    #pragma unroll
    for (int e = 0; e < ACC_ELEMS; ++e) {
      e_inv[e] = smem_alpha_w[frag_row[e]];
    }
    #pragma unroll
    for (int dt = 0; dt < OUT_TILES; ++dt) {
      #pragma unroll
      for (int e = 0; e < ACC_ELEMS; ++e) {
        acc_frag[dt].x[e] *= e_inv[e];
      }
      wmma::store_matrix_sync(smem_scores_w, acc_frag[dt], S_STRIDE, wmma::mem_row_major);
      __syncwarp();
      if (q_row_off >= 0) {
        if constexpr (SPLIT_K == 1) {
          uint4 packed;
          __nv_bfloat162* p2 = reinterpret_cast<__nv_bfloat162*>(&packed);
          #pragma unroll
          for (int p = 0; p < 4; ++p) {
            p2[p] = __floats2bfloat162_rn(
                smem_scores_w[row * S_STRIDE + c_base + 2 * p],
                smem_scores_w[row * S_STRIDE + c_base + 2 * p + 1]);
          }
          *reinterpret_cast<uint4*>(&direct_out[q_row_off + v_col_off + dt * 16 + c_base]) = packed;
        } else {
          const int32_t q_pos = q_flat_lane / gqa_group;
          const int32_t h_q = h_kv * gqa_group + (q_flat_lane % gqa_group);
          const int64_t out_base =
              ((((static_cast<int64_t>(b) * SPLIT_K + split_id) * Q_len + q_pos) * H_q + h_q) * D) +
              v_col_off + dt * 16 + c_base;
          #pragma unroll
          for (int c = 0; c < 8; ++c) {
            partial_out[out_base + c] = smem_scores_w[row * S_STRIDE + c_base + c];
          }
        }
      }
      __syncwarp();
    }
  }
}

template <int SPLIT_K, int D>
__global__ void combine_splitk_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_lse,
    const int32_t* __restrict__ seq_lens,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ out_lse,
    int32_t B,
    int32_t Q_len,
    int32_t H_q) {
  const int32_t bqh = blockIdx.x;
  if (bqh >= B * Q_len * H_q) return;
  const int32_t d = threadIdx.x;

  const int32_t h_q = bqh % H_q;
  const int32_t rem = bqh / H_q;
  const int32_t q_pos = rem % Q_len;
  const int32_t b = rem / Q_len;

  if (seq_lens[b] <= 0) {
    out[static_cast<int64_t>(bqh) * D + d] = __float2bfloat16(0.0f);
    if (d == 0 && out_lse != nullptr) {
      out_lse[bqh] = -1e30f;
    }
    return;
  }

  float lses[SPLIT_K];
  float max_lse = -1e30f;
  const int64_t stride_split = static_cast<int64_t>(Q_len) * H_q;
  const int64_t base_qh = static_cast<int64_t>(q_pos) * H_q + h_q;
  const int64_t b_base = static_cast<int64_t>(b) * SPLIT_K * stride_split + base_qh;

  #pragma unroll
  for (int k = 0; k < SPLIT_K; ++k) {
    const float v = partial_lse[b_base + k * stride_split];
    lses[k] = v;
    max_lse = fmaxf(max_lse, v);
  }

  if (max_lse <= -1e29f) {
    out[static_cast<int64_t>(bqh) * D + d] = __float2bfloat16(0.0f);
    if (d == 0 && out_lse != nullptr) {
      out_lse[bqh] = -1e30f;
    }
    return;
  }

  float w_sum = 0.0f;
  float acc = 0.0f;
  #pragma unroll
  for (int k = 0; k < SPLIT_K; ++k) {
    if (lses[k] > -1e29f) {
      const float w = __expf(lses[k] - max_lse);
      w_sum += w;
      acc += w * partial_out[(b_base + k * stride_split) * D + d];
    }
  }
  out[static_cast<int64_t>(bqh) * D + d] = __float2bfloat16(acc / fmaxf(w_sum, 1e-20f));
  if (d == 0 && out_lse != nullptr) {
    out_lse[bqh] = max_lse + __logf(fmaxf(w_sum, 1e-20f));
  }
}

template <int SK, int D>
void launch_wmma_splitk_impl(
    const torch::Tensor& q,
    const torch::Tensor& k_pool,
    const torch::Tensor& v_pool,
    const torch::Tensor& page_t_obs,
    float* d_partial_out,
    float* d_partial_lse,
    const int32_t* d_page_table,
    const int32_t* d_seq_lens,
    const int64_t* d_t_now_ns,
    torch::Tensor& out,
    float* d_out_lse,
    int32_t batch_size,
    int32_t Q_len,
    int32_t H_q,
    int32_t H_kv,
    int32_t L,
    int32_t max_pages_per_slot,
    int32_t layer_idx,
    float sm_scale,
    bool apply_rel_time_rope,
    uint32_t q_tiles,
    cudaStream_t stream) {
  const dim3 wmma_blocks(static_cast<uint32_t>(batch_size * H_kv * SK), q_tiles, (D == 256) ? 2u : 1u);
  __nv_bfloat16* d_out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  paged_rel_rope_wmma_splitk_kernel<SK, D><<<wmma_blocks, 64, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k_pool.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(v_pool.data_ptr<at::BFloat16>()),
      page_t_obs.data_ptr<int64_t>(),
      d_page_table,
      d_seq_lens,
      d_t_now_ns,
      d_partial_out,
      d_partial_lse,
      d_out,
      d_out_lse,
      batch_size,
      Q_len,
      H_q,
      H_kv,
      L,
      max_pages_per_slot,
      layer_idx,
      sm_scale,
      apply_rel_time_rope);
  if constexpr (SK > 1) {
    combine_splitk_kernel<SK, D><<<batch_size * Q_len * H_q, D, 0, stream>>>(
        d_partial_out,
        d_partial_lse,
        d_seq_lens,
        d_out,
        d_out_lse,
        batch_size,
        Q_len,
        H_q);
  }
}

}  // namespace

TemporalStateCache::TemporalStateCache(
    int32_t num_streams,
    int32_t num_readers,
    int32_t max_pages,
    int32_t num_layers,
    int32_t page_size,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t max_seq_len,
    bool kv_mode,
    int32_t max_q_len,
    int32_t max_q_heads,
    int32_t device_id)
    : num_streams_(num_streams),
      num_readers_(num_readers),
      slots_per_stream_(num_readers + 2),
      total_slots_(num_streams * (num_readers + 2)),
      max_pages_(max_pages),
      num_layers_(num_layers),
      page_size_(page_size),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      max_seq_len_(max_seq_len),
      max_pages_per_slot_((max_seq_len + page_size - 1) / page_size),
      kv_mode_(kv_mode),
      device_id_(device_id) {
  if (num_streams <= 0 || num_readers <= 0 || num_readers > 30) {
    throw std::invalid_argument("Invalid num_streams or num_readers (must be 1..30)");
  }
  const bool valid_kv_dim = (head_dim == 64 || head_dim == 128 || head_dim == 256);
  if (kv_mode && (page_size != 16 || !valid_kv_dim)) {
    throw std::invalid_argument(
        "TemporalStateCache(kv_mode=true) requires page_size=16 and head_dim in {64, 128, 256}");
  }
  if (!kv_mode && (page_size <= 0 || head_dim <= 0 || (head_dim % 4) != 0)) {
    throw std::invalid_argument("TemporalStateCache(kv_mode=false) requires positive page_size and 4-aligned head_dim");
  }
  if (max_pages <= 0) {
    throw std::invalid_argument("max_pages must be positive");
  }

  c10::cuda::CUDAGuard guard(device_id_);

  void* raw_headers = nullptr;
  if (posix_memalign(&raw_headers, 64, sizeof(SlotHeader) * total_slots_) != 0) {
    throw std::bad_alloc();
  }
  headers_ = new (raw_headers) SlotHeader[total_slots_];
  for (int32_t i = 0; i < total_slots_; ++i) {
    headers_[i].page_table_offset = i * max_pages_per_slot_;
  }

  latest_slot_ = new std::atomic<int32_t>[num_streams_];
  publishing_slot_ = std::make_unique<std::atomic<int32_t>[]>(num_streams_);
  stream_version_gen_ = std::make_unique<std::atomic<uint64_t>[]>(num_streams_);
  for (int32_t s = 0; s < num_streams_; ++s) {
    latest_slot_[s].store(-1, std::memory_order_relaxed);
    publishing_slot_[s].store(-1, std::memory_order_relaxed);
    stream_version_gen_[s].store(0, std::memory_order_relaxed);
  }

  host_page_table_.assign(total_slots_ * max_pages_per_slot_, -1);
  slot_write_start_page_.assign(total_slots_, 0);

  next_free_page_ = std::make_unique<std::atomic<int32_t>[]>(max_pages_);
  page_ref_count_ = std::make_unique<std::atomic<int32_t>[]>(max_pages_);
  for (int32_t p = 0; p < max_pages_; ++p) {
    const int32_t nxt = (p + 1 < max_pages_) ? (p + 1) : static_cast<int32_t>(kInvalidPageIdx);
    next_free_page_[p].store(nxt, std::memory_order_relaxed);
    page_ref_count_[p].store(0, std::memory_order_relaxed);
  }
  free_head_.store(static_cast<uint64_t>(0u), std::memory_order_relaxed);
  free_count_.store(max_pages_, std::memory_order_relaxed);

  auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, device_id_);
  auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device_id_);
  auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, device_id_);

  if (kv_mode_) {
    k_pool_ = torch::zeros({max_pages_, num_layers_, page_size_, num_kv_heads_, head_dim_}, opts_bf16);
    v_pool_ = torch::zeros({max_pages_, num_layers_, page_size_, num_kv_heads_, head_dim_}, opts_bf16);
  } else {
    k_pool_ = torch::empty({0}, opts_bf16);
    v_pool_ = torch::zeros({max_pages_, 1, page_size_, num_kv_heads_, head_dim_}, opts_bf16);
  }
  page_t_obs_ns_ = torch::zeros({max_pages_}, opts_i64);

  const int32_t max_bsk = std::max(num_streams_ * 16, 256);
  const int32_t ws_q_len = std::max(max_q_len, 16);
  const int32_t ws_q_heads = std::max(max_q_heads, num_kv_heads_);
  if (kv_mode_ && page_size_ == 16 && (head_dim_ == 64 || head_dim_ == 128 || head_dim_ == 256)) {
    splitk_partial_out_ = torch::empty({num_readers_, max_bsk, ws_q_len, ws_q_heads, head_dim_}, opts_f32);
    splitk_partial_lse_ = torch::empty({num_readers_, max_bsk, ws_q_len, ws_q_heads}, opts_f32);
  }

  write_ready_events_.resize(total_slots_);
  for (int32_t i = 0; i < total_slots_; ++i) {
    TS_CUDA_CHECK(cudaEventCreateWithFlags(&write_ready_events_[i], cudaEventDisableTiming));
  }

  reader_states_.reserve(num_readers_);
  for (int32_t r = 0; r < num_readers_; ++r) {
    auto rs = std::make_unique<ReaderState>();
    TS_CUDA_CHECK(cudaEventCreateWithFlags(&rs->done_event, cudaEventDisableTiming));
    reader_states_.push_back(std::move(rs));
  }

  pinned_capacity_ = std::max(num_streams_, 64);
  TS_CUDA_CHECK(cudaEventCreateWithFlags(&h2d_write_staging_done_ev_, cudaEventDisableTiming));
  TS_CUDA_CHECK(cudaMallocHost(&h_pinned_write_pt_, sizeof(int32_t) * pinned_capacity_ * max_pages_per_slot_));
  TS_CUDA_CHECK(cudaMallocHost(&h_pinned_write_start_, sizeof(int32_t) * pinned_capacity_));
  TS_CUDA_CHECK(
      cudaMallocHost(&h_pinned_page_table_, sizeof(int32_t) * num_readers_ * pinned_capacity_ * max_pages_per_slot_));
  TS_CUDA_CHECK(cudaMallocHost(&h_pinned_seq_lens_, sizeof(int32_t) * num_readers_ * pinned_capacity_));
  TS_CUDA_CHECK(cudaMallocHost(&h_pinned_t_now_, sizeof(int64_t) * num_readers_ * pinned_capacity_));
}

TemporalStateCache::~TemporalStateCache() {
  c10::cuda::CUDAGuard guard(device_id_);
  for (int32_t r = 0; r < num_readers_; ++r) {
    retire_reader(r, /*sync_if_busy=*/true);
    if (reader_states_[r]->done_event) {
      cudaEventDestroy(reader_states_[r]->done_event);
    }
  }
  for (auto& ev : write_ready_events_) {
    cudaEventDestroy(ev);
  }
  if (h2d_write_staging_done_ev_) cudaEventDestroy(h2d_write_staging_done_ev_);
  if (h_pinned_write_pt_) cudaFreeHost(h_pinned_write_pt_);
  if (h_pinned_write_start_) cudaFreeHost(h_pinned_write_start_);
  if (h_pinned_page_table_) cudaFreeHost(h_pinned_page_table_);
  if (h_pinned_seq_lens_) cudaFreeHost(h_pinned_seq_lens_);
  if (h_pinned_t_now_) cudaFreeHost(h_pinned_t_now_);
  delete[] latest_slot_;
  if (headers_) {
    for (int32_t i = 0; i < total_slots_; ++i) {
      headers_[i].~SlotHeader();
    }
    std::free(headers_);
  }
}

torch::Tensor TemporalStateCache::splitk_partial_out(int32_t reader_id) const {
  if (reader_id < 0 || reader_id >= num_readers_) {
    throw std::out_of_range("reader_id out of range");
  }
  return splitk_partial_out_.defined() ? splitk_partial_out_[reader_id] : splitk_partial_out_;
}

torch::Tensor TemporalStateCache::splitk_partial_lse(int32_t reader_id) const {
  if (reader_id < 0 || reader_id >= num_readers_) {
    throw std::out_of_range("reader_id out of range");
  }
  return splitk_partial_lse_.defined() ? splitk_partial_lse_[reader_id] : splitk_partial_lse_;
}

int32_t TemporalStateCache::pop_free_page() {
  uint64_t cur = free_head_.load(std::memory_order_acquire);
  while (true) {
    const uint32_t head_idx = static_cast<uint32_t>(cur & 0xFFFFFFFFull);
    if (head_idx == kInvalidPageIdx) {
      return -1;
    }
    const uint32_t tag = static_cast<uint32_t>(cur >> 32);
    const uint32_t nxt = static_cast<uint32_t>(next_free_page_[head_idx].load(std::memory_order_relaxed));
    const uint64_t desired = (static_cast<uint64_t>(tag + 1u) << 32) | static_cast<uint64_t>(nxt);
    if (free_head_.compare_exchange_weak(cur, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
      free_count_.fetch_sub(1, std::memory_order_relaxed);
      page_ref_count_[head_idx].store(1, std::memory_order_release);
      return static_cast<int32_t>(head_idx);
    }
  }
}

void TemporalStateCache::push_free_page(int32_t page_id) {
  uint64_t cur = free_head_.load(std::memory_order_relaxed);
  while (true) {
    const uint32_t head_idx = static_cast<uint32_t>(cur & 0xFFFFFFFFull);
    const uint32_t tag = static_cast<uint32_t>(cur >> 32);
    next_free_page_[page_id].store(static_cast<int32_t>(head_idx), std::memory_order_relaxed);
    const uint64_t desired =
        (static_cast<uint64_t>(tag + 1u) << 32) | static_cast<uint64_t>(static_cast<uint32_t>(page_id));
    if (free_head_.compare_exchange_weak(cur, desired, std::memory_order_release, std::memory_order_relaxed)) {
      free_count_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  }
}

void TemporalStateCache::release_page_ref(int32_t page_id) {
  if (page_id < 0) return;
  const int32_t prev = page_ref_count_[page_id].fetch_sub(1, std::memory_order_acq_rel);
  if (prev == 1) {
    push_free_page(page_id);
  }
}

void TemporalStateCache::try_reclaim_slot(int32_t stream_id, int32_t global_slot, bool wait_gpu_write) {
  if (latest_slot_[stream_id].load(std::memory_order_seq_cst) == global_slot) return;
  if (publishing_slot_[stream_id].load(std::memory_order_seq_cst) == global_slot) return;
  if (headers_[global_slot].reader_mask.load(std::memory_order_seq_cst) != 0u) return;

  uint32_t expected = SLOT_READY;
  if (!wait_gpu_write && cudaEventQuery(write_ready_events_[global_slot]) != cudaSuccess) {
    return;
  }
  if (!headers_[global_slot].state.compare_exchange_strong(
          expected, SLOT_RECLAIMING, std::memory_order_seq_cst, std::memory_order_seq_cst)) {
    return;
  }

  // Re-verify after transitioning out of SLOT_READY in case a reader pinned or publish_write raced just before CAS
  if (latest_slot_[stream_id].load(std::memory_order_seq_cst) == global_slot ||
      publishing_slot_[stream_id].load(std::memory_order_seq_cst) == global_slot ||
      headers_[global_slot].reader_mask.load(std::memory_order_seq_cst) != 0u) {
    headers_[global_slot].state.store(SLOT_READY, std::memory_order_seq_cst);
    return;
  }

  if (wait_gpu_write) {
    TS_CUDA_CHECK(cudaEventSynchronize(write_ready_events_[global_slot]));
  }

  SlotHeader& hdr = headers_[global_slot];
  for (int32_t i = 0; i < hdr.num_pages; ++i) {
    int32_t& p = host_page_table_[hdr.page_table_offset + i];
    if (p >= 0) {
      release_page_ref(p);
      p = -1;
    }
  }
  hdr.num_pages = 0;
  hdr.seq_len = 0;
  slot_write_start_page_[global_slot] = 0;
  headers_[global_slot].state.store(SLOT_FREE, std::memory_order_seq_cst);
}

void TemporalStateCache::retire_reader(int32_t reader_id, bool sync_if_busy) {
  ReaderState& rs = *reader_states_[reader_id];
  while (true) {
    uint32_t st_val = rs.state.load(std::memory_order_acquire);
    if (st_val == READER_IDLE) {
      return;
    }
    if (st_val == READER_RETIRING) {
      if (!sync_if_busy) return;
      std::this_thread::yield();
      continue;
    }
    if (rs.state.compare_exchange_strong(
            st_val, READER_RETIRING, std::memory_order_acq_rel, std::memory_order_acquire)) {
      break;
    }
  }

  cudaError_t st = cudaEventQuery(rs.done_event);
  if (st == cudaErrorNotReady) {
    if (!sync_if_busy) {
      rs.state.store(READER_ACTIVE, std::memory_order_release);
      return;
    }
    TS_CUDA_CHECK(cudaEventSynchronize(rs.done_event));
  } else {
    TS_CUDA_CHECK(st);
  }

  const uint32_t clr_mask = ~(1u << static_cast<uint32_t>(reader_id));
  for (int32_t gs : rs.pinned_slots) {
    if (gs >= 0) {
      const uint32_t prev_mask = headers_[gs].reader_mask.fetch_and(clr_mask, std::memory_order_seq_cst);
      if ((prev_mask & clr_mask) == 0u) {
        try_reclaim_slot(gs / slots_per_stream_, gs, /*wait_gpu_write=*/false);
      }
    }
  }
  rs.pinned_slots.clear();
  rs.state.store(READER_IDLE, std::memory_order_release);
}

void TemporalStateCache::reclaim_retired_reads() {
  c10::cuda::CUDAGuard guard(device_id_);
  for (int32_t r = 0; r < num_readers_; ++r) {
    retire_reader(r, /*sync_if_busy=*/false);
  }
}

int32_t TemporalStateCache::begin_write(int32_t stream_id, int32_t frame_seq_len, cudaStream_t write_stream) {
  if (stream_id < 0 || stream_id >= num_streams_) {
    throw std::out_of_range("stream_id out of range");
  }
  if (frame_seq_len <= 0 || frame_seq_len > max_seq_len_) {
    throw std::invalid_argument("frame_seq_len exceeds max_seq_len");
  }
  const int32_t frame_pages = (frame_seq_len + page_size_ - 1) / page_size_;
  if (max_pages_per_slot_ > frame_pages && (frame_seq_len % page_size_) != 0) {
    throw std::invalid_argument(
        "Multi-frame sliding-window co-pooling requires frame_seq_len divisible by page_size");
  }

  c10::cuda::CUDAGuard guard(device_id_);
  reclaim_retired_reads();

  const int32_t base = stream_id * slots_per_stream_;
  int32_t chosen = -1;

  // Sweep retired slots of this stream first; retry up to 4 passes against transient reader_mask handoffs
  for (int pass = 0; pass < 4 && chosen < 0; ++pass) {
    if (pass > 0) {
      std::this_thread::yield();
    }
    const int32_t cur_latest = latest_slot_[stream_id].load(std::memory_order_seq_cst);
    for (int32_t offset = 0; offset < slots_per_stream_; ++offset) {
      const int32_t gs = base + offset;
      if (gs == cur_latest) continue;
      try_reclaim_slot(stream_id, gs, /*wait_gpu_write=*/(pass >= 2));
      uint32_t expected = SLOT_FREE;
      if (headers_[gs].reader_mask.load(std::memory_order_seq_cst) == 0u &&
          headers_[gs].state.compare_exchange_strong(
              expected, SLOT_WRITING, std::memory_order_seq_cst, std::memory_order_seq_cst)) {
        chosen = gs;
        break;
      }
    }
  }
  if (chosen < 0) {
    throw std::runtime_error("No free TSC write slot found: active readers exceeded (M+2) contract");
  }

  int32_t keep_pages = 0;
  const int32_t cur_latest = latest_slot_[stream_id].load(std::memory_order_seq_cst);
  if (cur_latest >= 0 && cur_latest != chosen && max_pages_per_slot_ > frame_pages &&
      headers_[cur_latest].state.load(std::memory_order_seq_cst) == SLOT_READY) {
    const SlotHeader& prev_hdr = headers_[cur_latest];
    keep_pages = std::min(prev_hdr.num_pages, max_pages_per_slot_ - frame_pages);
    const int32_t drop_pages = prev_hdr.num_pages - keep_pages;
    for (int32_t i = 0; i < keep_pages; ++i) {
      const int32_t p = host_page_table_[prev_hdr.page_table_offset + drop_pages + i];
      page_ref_count_[p].fetch_add(1, std::memory_order_acq_rel);
      host_page_table_[headers_[chosen].page_table_offset + i] = p;
    }
    TS_CUDA_CHECK(cudaStreamWaitEvent(write_stream, write_ready_events_[cur_latest], 0));
  }

  SlotHeader& hdr = headers_[chosen];
  for (int32_t i = 0; i < frame_pages; ++i) {
    int32_t p = pop_free_page();
    if (p < 0) {
      reclaim_retired_reads();
      for (int32_t s = 0; s < num_streams_ && p < 0; ++s) {
        const int32_t s_base = s * slots_per_stream_;
        for (int32_t off = 0; off < slots_per_stream_; ++off) {
          const int32_t gs = s_base + off;
          if (gs != chosen) {
            try_reclaim_slot(s, gs, /*wait_gpu_write=*/false);
          }
        }
        p = pop_free_page();
      }
    }
    if (p < 0) {
      for (int32_t s = 0; s < num_streams_ && p < 0; ++s) {
        const int32_t s_base = s * slots_per_stream_;
        for (int32_t off = 0; off < slots_per_stream_; ++off) {
          const int32_t gs = s_base + off;
          if (gs != chosen) {
            try_reclaim_slot(s, gs, /*wait_gpu_write=*/true);
          }
        }
        p = pop_free_page();
      }
    }
    if (p < 0) {
      // Roll back already-acquired pages on failure
      for (int32_t j = 0; j < keep_pages + i; ++j) {
        release_page_ref(host_page_table_[hdr.page_table_offset + j]);
        host_page_table_[hdr.page_table_offset + j] = -1;
      }
      hdr.state.store(SLOT_FREE, std::memory_order_seq_cst);
      throw std::runtime_error("TSC PagedTensorPool exhausted");
    }
    host_page_table_[hdr.page_table_offset + keep_pages + i] = p;
  }
  for (int32_t i = keep_pages + frame_pages; i < max_pages_per_slot_; ++i) {
    host_page_table_[hdr.page_table_offset + i] = -1;
  }

  slot_write_start_page_[chosen] = keep_pages;
  hdr.num_pages = keep_pages + frame_pages;
  hdr.seq_len = keep_pages * page_size_ + frame_seq_len;
  return chosen;
}

void TemporalStateCache::prepare_write_batch(
    const std::vector<int32_t>& stream_ids,
    int32_t frame_seq_len,
    torch::Tensor& out_slot_page_table_gpu,
    torch::Tensor& out_start_page_idx_gpu,
    std::vector<int32_t>& out_global_slots,
    cudaStream_t copy_stream) {
  const int32_t B = static_cast<int32_t>(stream_ids.size());
  if (B < 0 || B > pinned_capacity_) {
    throw std::invalid_argument("prepare_write_batch B exceeds pinned_capacity");
  }
  c10::cuda::CUDAGuard guard(device_id_);
  while (true) {
    cudaError_t st = cudaEventQuery(h2d_write_staging_done_ev_);
    if (st == cudaSuccess) break;
    if (st != cudaErrorNotReady) TS_CUDA_CHECK(st);
    std::this_thread::sleep_for(std::chrono::microseconds(15));
  }
  out_global_slots.resize(B);

  for (int32_t b = 0; b < B; ++b) {
    int32_t gs = begin_write(stream_ids[b], frame_seq_len, copy_stream);
    out_global_slots[b] = gs;
    const SlotHeader& hdr = headers_[gs];
    h_pinned_write_start_[b] = slot_write_start_page_[gs];
    std::memcpy(
        &h_pinned_write_pt_[b * max_pages_per_slot_],
        &host_page_table_[hdr.page_table_offset],
        sizeof(int32_t) * max_pages_per_slot_);
  }

  TS_CUDA_CHECK(cudaMemcpyAsync(
      out_slot_page_table_gpu.data_ptr<int32_t>(),
      h_pinned_write_pt_,
      sizeof(int32_t) * B * max_pages_per_slot_,
      cudaMemcpyHostToDevice,
      copy_stream));
  TS_CUDA_CHECK(cudaMemcpyAsync(
      out_start_page_idx_gpu.data_ptr<int32_t>(),
      h_pinned_write_start_,
      sizeof(int32_t) * B,
      cudaMemcpyHostToDevice,
      copy_stream));
  TS_CUDA_CHECK(cudaEventRecord(h2d_write_staging_done_ev_, copy_stream));
}

void TemporalStateCache::write_layer_kv(
    int32_t global_slot,
    int32_t layer_idx,
    const torch::Tensor& k_in,
    const torch::Tensor& v_in,
    int64_t t_obs_ns,
    bool bake_spatial_rope,
    cudaStream_t write_stream) {
  if (!kv_mode_) {
    throw std::logic_error("write_layer_kv called on hidden-mode TemporalStateCache");
  }
  c10::cuda::CUDAStreamGuard stream_guard(c10::cuda::getStreamFromExternal(write_stream, device_id_));
  const SlotHeader& hdr = headers_[global_slot];
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_id_);
  auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, device_id_);
  auto pt_gpu = torch::empty({1, max_pages_per_slot_}, opts_i32);
  auto start_gpu = torch::full({1}, slot_write_start_page_[global_slot], opts_i32);
  auto t_gpu = torch::full({1}, t_obs_ns, opts_i64);
  TS_CUDA_CHECK(cudaMemcpyAsync(
      pt_gpu.data_ptr<int32_t>(),
      &host_page_table_[hdr.page_table_offset],
      sizeof(int32_t) * max_pages_per_slot_,
      cudaMemcpyHostToDevice,
      write_stream));
  int32_t grid_w = static_cast<int32_t>(std::sqrt(static_cast<double>(k_in.size(1))));
  launch_write_paged_kv(
      k_in,
      v_in,
      k_pool_,
      v_pool_,
      page_t_obs_ns_,
      pt_gpu,
      start_gpu,
      t_gpu,
      layer_idx,
      bake_spatial_rope,
      grid_w,
      write_stream);
}

void TemporalStateCache::write_hidden_state(
    int32_t global_slot,
    const torch::Tensor& hidden_in,
    int64_t t_obs_ns,
    cudaStream_t write_stream) {
  if (kv_mode_) {
    throw std::logic_error("write_hidden_state called on kv-mode TemporalStateCache");
  }
  c10::cuda::CUDAStreamGuard stream_guard(c10::cuda::getStreamFromExternal(write_stream, device_id_));
  const SlotHeader& hdr = headers_[global_slot];
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_id_);
  auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, device_id_);
  auto pt_gpu = torch::empty({1, max_pages_per_slot_}, opts_i32);
  auto start_gpu = torch::full({1}, slot_write_start_page_[global_slot], opts_i32);
  auto t_gpu = torch::full({1}, t_obs_ns, opts_i64);
  TS_CUDA_CHECK(cudaMemcpyAsync(
      pt_gpu.data_ptr<int32_t>(),
      &host_page_table_[hdr.page_table_offset],
      sizeof(int32_t) * max_pages_per_slot_,
      cudaMemcpyHostToDevice,
      write_stream));
  launch_write_paged_hidden(hidden_in, v_pool_, page_t_obs_ns_, pt_gpu, start_gpu, t_gpu, write_stream);
}

void TemporalStateCache::publish_write(
    int32_t stream_id,
    int32_t global_slot,
    int64_t t_obs_ns,
    int32_t frame_seq_len,
    cudaStream_t write_stream) {
  c10::cuda::CUDAGuard guard(device_id_);
  TS_CUDA_CHECK(cudaEventRecord(write_ready_events_[global_slot], write_stream));

  SlotHeader& hdr = headers_[global_slot];
  hdr.t_obs_ns = t_obs_ns;
  hdr.seq_len = slot_write_start_page_[global_slot] * page_size_ + frame_seq_len;
  hdr.version = stream_version_gen_[stream_id].fetch_add(1, std::memory_order_relaxed) + 1;
  publishing_slot_[stream_id].store(global_slot, std::memory_order_seq_cst);
  hdr.state.store(SLOT_READY, std::memory_order_seq_cst);

  const int32_t prev_slot = latest_slot_[stream_id].exchange(global_slot, std::memory_order_seq_cst);
  publishing_slot_[stream_id].store(-1, std::memory_order_seq_cst);
  if (prev_slot >= 0 && prev_slot != global_slot) {
    try_reclaim_slot(stream_id, prev_slot, /*wait_gpu_write=*/false);
  }
}

void TemporalStateCache::allocate_write_slots(
    const std::vector<int32_t>& stream_ids,
    int32_t frame_seq_len,
    std::vector<int32_t>& out_dst_slots,
    cudaStream_t init_stream) {
  const int32_t B = static_cast<int32_t>(stream_ids.size());
  out_dst_slots.assign(B, -1);
  if (B <= 0) return;
  c10::cuda::CUDAGuard guard(device_id_);
  try {
    for (int32_t b = 0; b < B; ++b) {
      out_dst_slots[b] = begin_write(stream_ids[b], frame_seq_len, init_stream);
    }
  } catch (...) {
    abort_write_batch(out_dst_slots, init_stream);
    out_dst_slots.clear();
    throw;
  }
}

void TemporalStateCache::enqueue_peer_copy_batch(
    const TemporalStateCache& src_tsc,
    const std::vector<int32_t>& src_global_slots,
    const std::vector<int32_t>& dst_global_slots,
    int32_t dst_layer_offset,
    int32_t layer_count,
    int32_t frame_seq_len,
    bool copy_timestamps,
    cudaStream_t push_stream,
    cudaEvent_t completion_event) {
  const int32_t B = static_cast<int32_t>(src_global_slots.size());
  if (B <= 0) return;
  if (static_cast<int32_t>(dst_global_slots.size()) != B) {
    throw std::invalid_argument("enqueue_peer_copy_batch vector size mismatch");
  }
  if (kv_mode_ != src_tsc.kv_mode_ || page_size_ != src_tsc.page_size_ ||
      num_kv_heads_ != src_tsc.num_kv_heads_ || head_dim_ != src_tsc.head_dim_) {
    throw std::invalid_argument("enqueue_peer_copy_batch source and destination TSC shape mismatch");
  }
  if (kv_mode_) {
    if (dst_layer_offset < 0 || layer_count <= 0 || dst_layer_offset + layer_count > num_layers_ ||
        layer_count > src_tsc.num_layers_) {
      throw std::invalid_argument("enqueue_peer_copy_batch invalid layer slice bounds");
    }
  } else if (dst_layer_offset != 0 || layer_count != 1) {
    throw std::invalid_argument("enqueue_peer_copy_batch hidden mode requires offset=0, count=1");
  }

  c10::cuda::CUDAGuard guard(device_id_);
  const int32_t frame_pages = (frame_seq_len + page_size_ - 1) / page_size_;
  const size_t dst_v_page_bytes = static_cast<size_t>(v_pool_.stride(0)) * v_pool_.element_size();
  const size_t src_v_page_bytes = static_cast<size_t>(src_tsc.v_pool_.stride(0)) * src_tsc.v_pool_.element_size();
  const size_t dst_k_page_bytes = kv_mode_ ? (static_cast<size_t>(k_pool_.stride(0)) * k_pool_.element_size()) : 0;
  const size_t src_k_page_bytes =
      kv_mode_ ? (static_cast<size_t>(src_tsc.k_pool_.stride(0)) * src_tsc.k_pool_.element_size()) : 0;
  const size_t per_layer_bytes =
      kv_mode_ ? (static_cast<size_t>(v_pool_.stride(1)) * v_pool_.element_size()) : dst_v_page_bytes;

  char* dst_v_base = static_cast<char*>(v_pool_.data_ptr());
  const char* src_v_base = static_cast<const char*>(src_tsc.v_pool_.data_ptr());
  char* dst_k_base = kv_mode_ ? static_cast<char*>(k_pool_.data_ptr()) : nullptr;
  const char* src_k_base = kv_mode_ ? static_cast<const char*>(src_tsc.k_pool_.data_ptr()) : nullptr;
  int64_t* dst_ts_base = page_t_obs_ns_.data_ptr<int64_t>();
  const int64_t* src_ts_base = src_tsc.page_t_obs_ns_.data_ptr<int64_t>();

  for (int32_t b = 0; b < B; ++b) {
    TS_CUDA_CHECK(cudaStreamWaitEvent(push_stream, src_tsc.write_ready_events_[src_global_slots[b]], 0));
  }

  const bool full_page_copy =
      (!kv_mode_) || (dst_layer_offset == 0 && layer_count == num_layers_ && src_tsc.num_layers_ == num_layers_);

  for (int32_t b = 0; b < B; ++b) {
    const int32_t dst_gs = dst_global_slots[b];
    const int32_t src_gs = src_global_slots[b];
    const int32_t dst_pt_off = headers_[dst_gs].page_table_offset + slot_write_start_page_[dst_gs];
    const int32_t src_pt_off = src_tsc.headers_[src_gs].page_table_offset + src_tsc.slot_write_start_page_[src_gs];

    if (full_page_copy) {
      int32_t p = 0;
      while (p < frame_pages) {
        const int32_t dst_p0 = host_page_table_[dst_pt_off + p];
        const int32_t src_p0 = src_tsc.host_page_table_[src_pt_off + p];
        int32_t run = 1;
        while (p + run < frame_pages &&
               host_page_table_[dst_pt_off + p + run] == dst_p0 + run &&
               src_tsc.host_page_table_[src_pt_off + p + run] == src_p0 + run) {
          ++run;
        }
        if (kv_mode_) {
          TS_CUDA_CHECK(cudaMemcpyPeerAsync(
              dst_k_base + static_cast<size_t>(dst_p0) * dst_k_page_bytes,
              device_id_,
              src_k_base + static_cast<size_t>(src_p0) * src_k_page_bytes,
              src_tsc.device_id_,
              static_cast<size_t>(run) * dst_k_page_bytes,
              push_stream));
        }
        TS_CUDA_CHECK(cudaMemcpyPeerAsync(
            dst_v_base + static_cast<size_t>(dst_p0) * dst_v_page_bytes,
            device_id_,
            src_v_base + static_cast<size_t>(src_p0) * src_v_page_bytes,
            src_tsc.device_id_,
            static_cast<size_t>(run) * dst_v_page_bytes,
            push_stream));
        if (copy_timestamps) {
          TS_CUDA_CHECK(cudaMemcpyPeerAsync(
              dst_ts_base + dst_p0,
              device_id_,
              src_ts_base + src_p0,
              src_tsc.device_id_,
              sizeof(int64_t) * static_cast<size_t>(run),
              push_stream));
        }
        p += run;
      }
    } else {
      const size_t dst_layer_byte_off = static_cast<size_t>(dst_layer_offset) * per_layer_bytes;
      const size_t slice_bytes = static_cast<size_t>(layer_count) * per_layer_bytes;
      for (int32_t p = 0; p < frame_pages; ++p) {
        const int32_t dst_p = host_page_table_[dst_pt_off + p];
        const int32_t src_p = src_tsc.host_page_table_[src_pt_off + p];
        TS_CUDA_CHECK(cudaMemcpyPeerAsync(
            dst_k_base + static_cast<size_t>(dst_p) * dst_k_page_bytes + dst_layer_byte_off,
            device_id_,
            src_k_base + static_cast<size_t>(src_p) * src_k_page_bytes,
            src_tsc.device_id_,
            slice_bytes,
            push_stream));
        TS_CUDA_CHECK(cudaMemcpyPeerAsync(
            dst_v_base + static_cast<size_t>(dst_p) * dst_v_page_bytes + dst_layer_byte_off,
            device_id_,
            src_v_base + static_cast<size_t>(src_p) * src_v_page_bytes,
            src_tsc.device_id_,
            slice_bytes,
            push_stream));
      }
      if (copy_timestamps) {
        int32_t p = 0;
        while (p < frame_pages) {
          const int32_t dst_p0 = host_page_table_[dst_pt_off + p];
          const int32_t src_p0 = src_tsc.host_page_table_[src_pt_off + p];
          int32_t run = 1;
          while (p + run < frame_pages &&
                 host_page_table_[dst_pt_off + p + run] == dst_p0 + run &&
                 src_tsc.host_page_table_[src_pt_off + p + run] == src_p0 + run) {
            ++run;
          }
          TS_CUDA_CHECK(cudaMemcpyPeerAsync(
              dst_ts_base + dst_p0,
              device_id_,
              src_ts_base + src_p0,
              src_tsc.device_id_,
              sizeof(int64_t) * static_cast<size_t>(run),
              push_stream));
          p += run;
        }
      }
    }
  }

  if (completion_event != nullptr) {
    TS_CUDA_CHECK(cudaEventRecord(completion_event, push_stream));
  }
}

size_t TemporalStateCache::stage_slice_bounce_bytes(int32_t batch_size, int32_t frame_seq_len) const {
  if (batch_size <= 0 || frame_seq_len <= 0) return 0;
  const size_t frame_pages = static_cast<size_t>((frame_seq_len + page_size_ - 1) / page_size_);
  const size_t v_page_bytes = static_cast<size_t>(v_pool_.stride(0)) * v_pool_.element_size();
  const size_t k_page_bytes = kv_mode_ ? (static_cast<size_t>(k_pool_.stride(0)) * k_pool_.element_size()) : 0;
  const size_t per_stream_bytes = frame_pages * (k_page_bytes + v_page_bytes + sizeof(int64_t));
  return static_cast<size_t>(batch_size) * per_stream_bytes;
}

void TemporalStateCache::stage_slice_to_host_pinned(
    const std::vector<int32_t>& src_global_slots,
    int32_t frame_seq_len,
    bool copy_timestamps,
    void* h_pinned_dst,
    cudaStream_t d2h_stream) const {
  const int32_t B = static_cast<int32_t>(src_global_slots.size());
  if (B <= 0) return;
  c10::cuda::CUDAGuard guard(device_id_);
  const int32_t frame_pages = (frame_seq_len + page_size_ - 1) / page_size_;
  const size_t src_v_page_bytes = static_cast<size_t>(v_pool_.stride(0)) * v_pool_.element_size();
  const size_t src_k_page_bytes = kv_mode_ ? (static_cast<size_t>(k_pool_.stride(0)) * k_pool_.element_size()) : 0;
  const char* src_v_base = static_cast<const char*>(v_pool_.data_ptr());
  const char* src_k_base = kv_mode_ ? static_cast<const char*>(k_pool_.data_ptr()) : nullptr;
  const int64_t* src_ts_base = page_t_obs_ns_.data_ptr<int64_t>();

  char* dst_cursor = static_cast<char*>(h_pinned_dst);
  for (int32_t b = 0; b < B; ++b) {
    TS_CUDA_CHECK(cudaStreamWaitEvent(d2h_stream, write_ready_events_[src_global_slots[b]], 0));
  }

  for (int32_t b = 0; b < B; ++b) {
    const int32_t src_gs = src_global_slots[b];
    const int32_t src_pt_off = headers_[src_gs].page_table_offset + slot_write_start_page_[src_gs];

    if (kv_mode_) {
      int32_t p = 0;
      while (p < frame_pages) {
        const int32_t p0 = host_page_table_[src_pt_off + p];
        int32_t run = 1;
        while (p + run < frame_pages && host_page_table_[src_pt_off + p + run] == p0 + run) ++run;
        const size_t chunk_bytes = static_cast<size_t>(run) * src_k_page_bytes;
        TS_CUDA_CHECK(cudaMemcpyAsync(
            dst_cursor, src_k_base + static_cast<size_t>(p0) * src_k_page_bytes, chunk_bytes, cudaMemcpyDeviceToHost, d2h_stream));
        dst_cursor += chunk_bytes;
        p += run;
      }
    }

    {
      int32_t p = 0;
      while (p < frame_pages) {
        const int32_t p0 = host_page_table_[src_pt_off + p];
        int32_t run = 1;
        while (p + run < frame_pages && host_page_table_[src_pt_off + p + run] == p0 + run) ++run;
        const size_t chunk_bytes = static_cast<size_t>(run) * src_v_page_bytes;
        TS_CUDA_CHECK(cudaMemcpyAsync(
            dst_cursor, src_v_base + static_cast<size_t>(p0) * src_v_page_bytes, chunk_bytes, cudaMemcpyDeviceToHost, d2h_stream));
        dst_cursor += chunk_bytes;
        p += run;
      }
    }

    if (copy_timestamps) {
      int32_t p = 0;
      while (p < frame_pages) {
        const int32_t p0 = host_page_table_[src_pt_off + p];
        int32_t run = 1;
        while (p + run < frame_pages && host_page_table_[src_pt_off + p + run] == p0 + run) ++run;
        const size_t ts_bytes = sizeof(int64_t) * static_cast<size_t>(run);
        TS_CUDA_CHECK(cudaMemcpyAsync(dst_cursor, src_ts_base + p0, ts_bytes, cudaMemcpyDeviceToHost, d2h_stream));
        dst_cursor += ts_bytes;
        p += run;
      }
    }
  }
}

void TemporalStateCache::enqueue_host_pinned_copy_batch(
    const void* h_pinned_src,
    const std::vector<int32_t>& dst_global_slots,
    int32_t dst_layer_offset,
    int32_t layer_count,
    int32_t frame_seq_len,
    bool copy_timestamps,
    cudaStream_t push_stream,
    cudaEvent_t completion_event) {
  const int32_t B = static_cast<int32_t>(dst_global_slots.size());
  if (B <= 0) return;
  if (kv_mode_) {
    if (dst_layer_offset < 0 || layer_count <= 0 || dst_layer_offset + layer_count > num_layers_) {
      throw std::invalid_argument("enqueue_host_pinned_copy_batch invalid layer slice bounds");
    }
  } else if (dst_layer_offset != 0 || layer_count != 1) {
    throw std::invalid_argument("enqueue_host_pinned_copy_batch hidden mode requires offset=0, count=1");
  }

  c10::cuda::CUDAGuard guard(device_id_);
  const int32_t frame_pages = (frame_seq_len + page_size_ - 1) / page_size_;
  const size_t dst_v_page_bytes = static_cast<size_t>(v_pool_.stride(0)) * v_pool_.element_size();
  const size_t dst_k_page_bytes = kv_mode_ ? (static_cast<size_t>(k_pool_.stride(0)) * k_pool_.element_size()) : 0;
  const size_t per_layer_bytes =
      kv_mode_ ? (static_cast<size_t>(v_pool_.stride(1)) * v_pool_.element_size()) : dst_v_page_bytes;
  const size_t slice_bytes = static_cast<size_t>(layer_count) * per_layer_bytes;
  const size_t dst_layer_byte_off = static_cast<size_t>(dst_layer_offset) * per_layer_bytes;

  char* dst_v_base = static_cast<char*>(v_pool_.data_ptr());
  char* dst_k_base = kv_mode_ ? static_cast<char*>(k_pool_.data_ptr()) : nullptr;
  int64_t* dst_ts_base = page_t_obs_ns_.data_ptr<int64_t>();
  const char* src_cursor = static_cast<const char*>(h_pinned_src);

  const bool full_page_copy = (!kv_mode_) || (dst_layer_offset == 0 && layer_count == num_layers_);

  for (int32_t b = 0; b < B; ++b) {
    const int32_t dst_gs = dst_global_slots[b];
    const int32_t dst_pt_off = headers_[dst_gs].page_table_offset + slot_write_start_page_[dst_gs];

    if (full_page_copy) {
      if (kv_mode_) {
        int32_t p = 0;
        while (p < frame_pages) {
          const int32_t dst_p0 = host_page_table_[dst_pt_off + p];
          int32_t run = 1;
          while (p + run < frame_pages && host_page_table_[dst_pt_off + p + run] == dst_p0 + run) ++run;
          const size_t chunk_bytes = static_cast<size_t>(run) * dst_k_page_bytes;
          TS_CUDA_CHECK(cudaMemcpyAsync(
              dst_k_base + static_cast<size_t>(dst_p0) * dst_k_page_bytes,
              src_cursor,
              chunk_bytes,
              cudaMemcpyHostToDevice,
              push_stream));
          src_cursor += chunk_bytes;
          p += run;
        }
      }
      {
        int32_t p = 0;
        while (p < frame_pages) {
          const int32_t dst_p0 = host_page_table_[dst_pt_off + p];
          int32_t run = 1;
          while (p + run < frame_pages && host_page_table_[dst_pt_off + p + run] == dst_p0 + run) ++run;
          const size_t chunk_bytes = static_cast<size_t>(run) * dst_v_page_bytes;
          TS_CUDA_CHECK(cudaMemcpyAsync(
              dst_v_base + static_cast<size_t>(dst_p0) * dst_v_page_bytes,
              src_cursor,
              chunk_bytes,
              cudaMemcpyHostToDevice,
              push_stream));
          src_cursor += chunk_bytes;
          p += run;
        }
      }
    } else {
      for (int32_t p = 0; p < frame_pages; ++p) {
        const int32_t dst_p = host_page_table_[dst_pt_off + p];
        TS_CUDA_CHECK(cudaMemcpyAsync(
            dst_k_base + static_cast<size_t>(dst_p) * dst_k_page_bytes + dst_layer_byte_off,
            src_cursor,
            slice_bytes,
            cudaMemcpyHostToDevice,
            push_stream));
        src_cursor += slice_bytes;
      }
      for (int32_t p = 0; p < frame_pages; ++p) {
        const int32_t dst_p = host_page_table_[dst_pt_off + p];
        TS_CUDA_CHECK(cudaMemcpyAsync(
            dst_v_base + static_cast<size_t>(dst_p) * dst_v_page_bytes + dst_layer_byte_off,
            src_cursor,
            slice_bytes,
            cudaMemcpyHostToDevice,
            push_stream));
        src_cursor += slice_bytes;
      }
    }

    if (copy_timestamps) {
      int32_t p = 0;
      while (p < frame_pages) {
        const int32_t dst_p0 = host_page_table_[dst_pt_off + p];
        int32_t run = 1;
        while (p + run < frame_pages && host_page_table_[dst_pt_off + p + run] == dst_p0 + run) ++run;
        const size_t ts_bytes = sizeof(int64_t) * static_cast<size_t>(run);
        TS_CUDA_CHECK(cudaMemcpyAsync(dst_ts_base + dst_p0, src_cursor, ts_bytes, cudaMemcpyHostToDevice, push_stream));
        src_cursor += ts_bytes;
        p += run;
      }
    }
  }

  if (completion_event != nullptr) {
    TS_CUDA_CHECK(cudaEventRecord(completion_event, push_stream));
  }
}

void TemporalStateCache::finalize_peer_publish_batch(
    const std::vector<int32_t>& stream_ids,
    const std::vector<int32_t>& dst_global_slots,
    const std::vector<int64_t>& t_obs_ns,
    const std::vector<int32_t>& frame_seq_lens,
    cudaStream_t push_stream,
    cudaEvent_t push_done_event) {
  const int32_t B = static_cast<int32_t>(stream_ids.size());
  if (B <= 0) return;
  if (static_cast<int32_t>(dst_global_slots.size()) != B || static_cast<int32_t>(t_obs_ns.size()) != B ||
      static_cast<int32_t>(frame_seq_lens.size()) != B) {
    throw std::invalid_argument("finalize_peer_publish_batch vector size mismatch");
  }
  c10::cuda::CUDAGuard guard(device_id_);
  while (true) {
    cudaError_t st = cudaEventQuery(push_done_event);
    if (st == cudaSuccess) break;
    if (st != cudaErrorNotReady) TS_CUDA_CHECK(st);
    std::this_thread::sleep_for(std::chrono::microseconds(15));
  }
  for (int32_t b = 0; b < B; ++b) {
    publish_write(stream_ids[b], dst_global_slots[b], t_obs_ns[b], frame_seq_lens[b], push_stream);
  }
}

void TemporalStateCache::abort_write_batch(const std::vector<int32_t>& global_slots, cudaStream_t write_stream) {
  c10::cuda::CUDAGuard guard(device_id_);
  if (write_stream != nullptr) {
    TS_CUDA_CHECK(cudaStreamSynchronize(write_stream));
  }
  for (int32_t gs : global_slots) {
    if (gs < 0 || gs >= total_slots_) continue;
    uint32_t expected = SLOT_WRITING;
    if (!headers_[gs].state.compare_exchange_strong(
            expected, SLOT_RECLAIMING, std::memory_order_seq_cst, std::memory_order_seq_cst)) {
      continue;
    }
    SlotHeader& hdr = headers_[gs];
    for (int32_t i = 0; i < hdr.num_pages; ++i) {
      int32_t& p = host_page_table_[hdr.page_table_offset + i];
      if (p >= 0) {
        release_page_ref(p);
        p = -1;
      }
    }
    hdr.num_pages = 0;
    hdr.seq_len = 0;
    slot_write_start_page_[gs] = 0;
    hdr.state.store(SLOT_FREE, std::memory_order_seq_cst);
  }
}

void TemporalStateCache::reset_stream(int32_t stream_id) {
  if (stream_id < 0 || stream_id >= num_streams_) {
    throw std::out_of_range("stream_id out of range");
  }
  c10::cuda::CUDAGuard guard(device_id_);
  latest_slot_[stream_id].store(-1, std::memory_order_seq_cst);
  publishing_slot_[stream_id].store(-1, std::memory_order_seq_cst);
  reclaim_retired_reads();
  const int32_t base = stream_id * slots_per_stream_;
  for (int32_t offset = 0; offset < slots_per_stream_; ++offset) {
    try_reclaim_slot(stream_id, base + offset, /*wait_gpu_write=*/true);
  }
}

void TemporalStateCache::acquire_read_batch(
    int32_t reader_id,
    const std::vector<int32_t>& stream_ids,
    const std::vector<int64_t>& t_now_ns,
    int32_t* d_batch_page_table,
    int32_t* d_batch_seq_lens,
    int64_t* d_batch_t_now_ns,
    std::vector<int64_t>& out_slow_t_obs_ns,
    std::vector<int32_t>& out_pinned_global_slots,
    cudaStream_t fast_stream) {
  if (reader_id < 0 || reader_id >= num_readers_) {
    throw std::out_of_range("reader_id out of range");
  }
  if (t_now_ns.size() != stream_ids.size()) {
    throw std::invalid_argument("acquire_read_batch requires t_now_ns.size() == stream_ids.size()");
  }
  const int32_t B = static_cast<int32_t>(stream_ids.size());
  if (B < 0 || B > pinned_capacity_) {
    throw std::invalid_argument("acquire_read_batch batch size exceeds pinned_capacity");
  }
  for (int32_t b = 0; b < B; ++b) {
    if (stream_ids[b] < 0 || stream_ids[b] >= num_streams_) {
      throw std::out_of_range("acquire_read_batch stream_id out of range");
    }
  }

  c10::cuda::CUDAGuard guard(device_id_);
  // Retire any prior batch of this reader_id so reader_id holds at most 1 bit per stream
  retire_reader(reader_id, /*sync_if_busy=*/true);

  const uint32_t r_bit = (1u << static_cast<uint32_t>(reader_id));
  out_slow_t_obs_ns.resize(B, 0);
  out_pinned_global_slots.assign(B, -1);
  std::vector<bool> stream_already_pinned(num_streams_, false);
  std::vector<int32_t> stream_pinned_slot(num_streams_, -1);

  int32_t* r_pinned_pt =
      h_pinned_page_table_ + static_cast<int64_t>(reader_id) * pinned_capacity_ * max_pages_per_slot_;
  int32_t* r_pinned_sl = h_pinned_seq_lens_ + static_cast<int64_t>(reader_id) * pinned_capacity_;
  int64_t* r_pinned_tn = h_pinned_t_now_ + static_cast<int64_t>(reader_id) * pinned_capacity_;

  try {
    for (int32_t b = 0; b < B; ++b) {
      const int32_t s = stream_ids[b];
      int32_t pinned_gs = -1;
      bool cold_start = false;

      if (stream_already_pinned[s]) {
        pinned_gs = stream_pinned_slot[s];
        cold_start = (pinned_gs < 0);
      } else {
        for (int attempt = 0; attempt < 16; ++attempt) {
          const int32_t gs = latest_slot_[s].load(std::memory_order_seq_cst);
          if (gs < 0) {
            cold_start = true;
            pinned_gs = -1;
            break;
          }
          headers_[gs].reader_mask.fetch_or(r_bit, std::memory_order_seq_cst);
          if (latest_slot_[s].load(std::memory_order_seq_cst) == gs &&
              headers_[gs].state.load(std::memory_order_seq_cst) == SLOT_READY) {
            pinned_gs = gs;
            break;
          }
          const uint32_t prev = headers_[gs].reader_mask.fetch_and(~r_bit, std::memory_order_seq_cst);
          if ((prev & ~r_bit) == 0u) {
            try_reclaim_slot(s, gs, /*wait_gpu_write=*/false);
          }
        }
        if (!cold_start && pinned_gs < 0) {
          throw std::runtime_error("TSC acquire_read_batch failed to pin published slot");
        }
        stream_already_pinned[s] = true;
        stream_pinned_slot[s] = pinned_gs;
        out_pinned_global_slots[b] = pinned_gs;
      }

      r_pinned_tn[b] = t_now_ns[b];
      if (pinned_gs >= 0) {
        const SlotHeader& hdr = headers_[pinned_gs];
        out_slow_t_obs_ns[b] = hdr.t_obs_ns;
        r_pinned_sl[b] = hdr.seq_len;
        std::memcpy(
            &r_pinned_pt[b * max_pages_per_slot_],
            &host_page_table_[hdr.page_table_offset],
            sizeof(int32_t) * max_pages_per_slot_);
        TS_CUDA_CHECK(cudaStreamWaitEvent(fast_stream, write_ready_events_[pinned_gs], 0));
      } else {
        out_slow_t_obs_ns[b] = 0;
        r_pinned_sl[b] = 0;
        std::fill_n(&r_pinned_pt[b * max_pages_per_slot_], max_pages_per_slot_, -1);
      }
    }

    TS_CUDA_CHECK(cudaMemcpyAsync(
        d_batch_page_table,
        r_pinned_pt,
        sizeof(int32_t) * B * max_pages_per_slot_,
        cudaMemcpyHostToDevice,
        fast_stream));
    TS_CUDA_CHECK(cudaMemcpyAsync(
        d_batch_seq_lens,
        r_pinned_sl,
        sizeof(int32_t) * B,
        cudaMemcpyHostToDevice,
        fast_stream));
    TS_CUDA_CHECK(cudaMemcpyAsync(
        d_batch_t_now_ns,
        r_pinned_tn,
        sizeof(int64_t) * B,
        cudaMemcpyHostToDevice,
        fast_stream));
  } catch (...) {
    abort_read_batch(reader_id, out_pinned_global_slots);
    throw;
  }
}

void TemporalStateCache::abort_read_batch(
    int32_t reader_id,
    const std::vector<int32_t>& pinned_global_slots) {
  if (reader_id < 0 || reader_id >= num_readers_) return;
  const uint32_t r_bit = (1u << static_cast<uint32_t>(reader_id));
  for (int32_t gs : pinned_global_slots) {
    if (gs >= 0 && gs < total_slots_) {
      const uint32_t prev = headers_[gs].reader_mask.fetch_and(~r_bit, std::memory_order_seq_cst);
      if ((prev & ~r_bit) == 0u) {
        const int32_t s = gs / slots_per_stream_;
        try_reclaim_slot(s, gs, /*wait_gpu_write=*/false);
      }
    }
  }
}

void TemporalStateCache::release_read_batch(
    int32_t reader_id,
    const std::vector<int32_t>& pinned_global_slots,
    cudaStream_t fast_stream) {
  if (reader_id < 0 || reader_id >= num_readers_) {
    throw std::out_of_range("reader_id out of range");
  }
  c10::cuda::CUDAGuard guard(device_id_);
  ReaderState& rs = *reader_states_[reader_id];
  rs.pinned_slots = pinned_global_slots;
  TS_CUDA_CHECK(cudaEventRecord(rs.done_event, fast_stream));
  rs.state.store(READER_ACTIVE, std::memory_order_release);
}

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
    cudaStream_t stream) {
  if (!k_in.is_contiguous() || !v_in.is_contiguous()) {
    throw std::invalid_argument("launch_write_paged_kv requires contiguous k_in and v_in");
  }
  c10::cuda::CUDAGuard guard(k_in.device());
  const int32_t B = static_cast<int32_t>(k_in.size(0));
  const int32_t S = static_cast<int32_t>(k_in.size(1));
  const int32_t H_kv = static_cast<int32_t>(k_in.size(2));
  const int32_t D = static_cast<int32_t>(k_in.size(3));
  const int32_t L = static_cast<int32_t>(k_pool.size(1));
  if (layer_idx < 0 || layer_idx >= L) {
    throw std::out_of_range("launch_write_paged_kv layer_idx out of range");
  }
  const int32_t P = static_cast<int32_t>(k_pool.size(2));
  const int32_t max_pages_per_slot = static_cast<int32_t>(page_table.size(1));

  const int32_t total_pairs = B * S * H_kv * (D / 2);
  const int32_t threads = 256;
  const int32_t blocks = (total_pairs + threads - 1) / threads;

  write_paged_kv_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_in.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(v_in.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(k_pool.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(v_pool.data_ptr<at::BFloat16>()),
      page_t_obs_ns.data_ptr<int64_t>(),
      page_table.data_ptr<int32_t>(),
      start_page_idx.data_ptr<int32_t>(),
      t_obs_ns.data_ptr<int64_t>(),
      B,
      S,
      L,
      P,
      H_kv,
      D,
      max_pages_per_slot,
      layer_idx,
      bake_spatial_rope,
      spatial_grid_w);
  TS_CUDA_CHECK(cudaGetLastError());
}

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
    cudaStream_t stream) {
  const int32_t B = static_cast<int32_t>(k_out.size(0));
  const int32_t hist_S = static_cast<int32_t>(k_out.size(1));
  if (hist_S <= 0) return;
  c10::cuda::CUDAGuard guard(k_pool.device());
  const int32_t H_kv = static_cast<int32_t>(k_out.size(2));
  const int32_t D = static_cast<int32_t>(k_out.size(3));
  const int32_t L = static_cast<int32_t>(k_pool.size(1));
  if (layer_idx < 0 || layer_idx >= L) {
    throw std::out_of_range("launch_gather_paged_kv_history layer_idx out of range");
  }
  const int32_t P = static_cast<int32_t>(k_pool.size(2));
  const int32_t max_pages_per_slot = static_cast<int32_t>(page_table.size(1));

  const int64_t total_pairs = static_cast<int64_t>(B) * hist_S * H_kv * (D / 2);
  const int32_t threads = 256;
  const int32_t blocks = static_cast<int32_t>((total_pairs + threads - 1) / threads);

  gather_paged_kv_history_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_pool.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(v_pool.data_ptr<at::BFloat16>()),
      page_t_obs_ns.data_ptr<int64_t>(),
      page_table.data_ptr<int32_t>(),
      start_page_idx.data_ptr<int32_t>(),
      t_obs_ns.data_ptr<int64_t>(),
      reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(v_out.data_ptr<at::BFloat16>()),
      B,
      hist_S,
      L,
      P,
      H_kv,
      D,
      max_pages_per_slot,
      layer_idx,
      apply_rel_time_rope);
  TS_CUDA_CHECK(cudaGetLastError());
}

void launch_write_paged_hidden(
    const torch::Tensor& hidden_in,
    torch::Tensor& v_pool,
    torch::Tensor& page_t_obs_ns,
    const torch::Tensor& page_table,
    const torch::Tensor& start_page_idx,
    const torch::Tensor& t_obs_ns,
    cudaStream_t stream) {
  if (!hidden_in.is_contiguous()) {
    throw std::invalid_argument("launch_write_paged_hidden requires contiguous hidden_in");
  }
  c10::cuda::CUDAGuard guard(hidden_in.device());
  const int32_t B = static_cast<int32_t>(hidden_in.size(0));
  const int32_t S = static_cast<int32_t>(hidden_in.size(1));
  const int32_t feat_dim = static_cast<int32_t>(hidden_in.numel() / (B * S));
  const int32_t P = static_cast<int32_t>(v_pool.size(2));
  const int32_t max_pages_per_slot = static_cast<int32_t>(page_table.size(1));

  const int64_t total = static_cast<int64_t>(B) * S * feat_dim;
  const int32_t threads = 256;
  const int32_t blocks = static_cast<int32_t>((total + threads - 1) / threads);

  write_paged_hidden_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden_in.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(v_pool.data_ptr<at::BFloat16>()),
      page_t_obs_ns.data_ptr<int64_t>(),
      page_table.data_ptr<int32_t>(),
      start_page_idx.data_ptr<int32_t>(),
      t_obs_ns.data_ptr<int64_t>(),
      B,
      S,
      P,
      feat_dim,
      max_pages_per_slot);
  TS_CUDA_CHECK(cudaGetLastError());
}

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
    cudaStream_t stream) {
  if (!out.is_contiguous() || out.scalar_type() != torch::kBFloat16) {
    throw std::invalid_argument("launch_gather_paged_hidden requires contiguous bfloat16 out tensor");
  }
  if (!out_delta_sec.is_contiguous() || out_delta_sec.scalar_type() != torch::kFloat32) {
    throw std::invalid_argument("launch_gather_paged_hidden requires contiguous float32 out_delta_sec tensor");
  }
  c10::cuda::CUDAGuard guard(v_pool.device());
  const int32_t P = static_cast<int32_t>(v_pool.size(2));
  const int32_t feat_dim = static_cast<int32_t>(v_pool.size(3) * v_pool.size(4));
  const int64_t total = static_cast<int64_t>(batch_size) * max_seq_len * feat_dim;
  const int32_t threads = 256;
  const int32_t blocks = static_cast<int32_t>((total + threads - 1) / threads);

  gather_paged_hidden_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(v_pool.data_ptr<at::BFloat16>()),
      page_t_obs_ns.data_ptr<int64_t>(),
      d_page_table,
      d_seq_lens,
      d_t_now_ns,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      out_delta_sec.data_ptr<float>(),
      batch_size,
      max_seq_len,
      P,
      feat_dim,
      max_pages_per_slot);
  TS_CUDA_CHECK(cudaGetLastError());
}

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
    float* d_out_lse) {
  const int32_t Q_len = static_cast<int32_t>(q.size(1));
  const int32_t H_q = static_cast<int32_t>(q.size(2));
  const int32_t D = static_cast<int32_t>(q.size(3));
  const int32_t L = static_cast<int32_t>(k_pool.size(1));
  const int32_t P = static_cast<int32_t>(k_pool.size(2));
  const int32_t H_kv = static_cast<int32_t>(k_pool.size(3));
  const int32_t pool_D = static_cast<int32_t>(k_pool.size(4));

  if (!q.is_contiguous() || !out.is_contiguous()) {
    throw std::invalid_argument("paged_rel_rope_cross_attn requires contiguous q and out");
  }
  if (q.device() != k_pool.device() || out.device() != k_pool.device()) {
    throw std::invalid_argument("paged_rel_rope_cross_attn requires q, out, and k_pool on the same CUDA device");
  }
  if (P != 16 || D != pool_D || (D != 64 && D != 128 && D != 256)) {
    throw std::invalid_argument(
        "paged_rel_rope_cross_attn requires page_size == 16 and matching head_dim in {64, 128, 256}");
  }
  if (H_kv <= 0 || H_q <= 0 || (H_q % H_kv) != 0) {
    throw std::invalid_argument("paged_rel_rope_cross_attn requires H_q divisible by H_kv");
  }
  if (layer_idx < 0 || layer_idx >= L) {
    throw std::out_of_range("paged_rel_rope_cross_attn layer_idx out of range");
  }
  if (partial_out_workspace.dim() == 4 &&
      (Q_len > partial_out_workspace.size(1) || H_q > partial_out_workspace.size(2))) {
    throw std::invalid_argument(
        "paged_rel_rope_cross_attn query exceeds Split-KV workspace (increase Engine max_q_len / max_q_heads)");
  }
  c10::cuda::CUDAGuard guard(q.device());

  const int32_t gqa_group = H_q / H_kv;
  const int32_t total_q_per_kv = Q_len * gqa_group;
  const float sm_scale = 1.0f / std::sqrt(static_cast<float>(D));
  const uint32_t q_tiles = static_cast<uint32_t>((total_q_per_kv + 31) / 32);

  const int32_t v_splits = (D == 256) ? 2 : 1;
  const int32_t base_blocks = batch_size * H_kv * static_cast<int32_t>(q_tiles) * v_splits;
  const int32_t max_bsk_avail = (partial_out_workspace.dim() == 4)
      ? static_cast<int32_t>(partial_out_workspace.size(0))
      : 0;
  int32_t sk = 1;
  for (int32_t cand : {16, 8, 4, 2}) {
    if (base_blocks * cand <= 768 &&
        max_pages_per_slot >= cand * 2 &&
        batch_size * cand <= max_bsk_avail) {
      sk = cand;
      break;
    }
  }
  if (sk > 1) {
    const int64_t needed_out = static_cast<int64_t>(batch_size) * sk * Q_len * H_q * D;
    if (needed_out > partial_out_workspace.numel()) {
      throw std::invalid_argument("Split-KV workspace capacity exceeded; increase max_q_len / max_q_heads");
    }
  }

  float* d_pout = (sk > 1) ? partial_out_workspace.data_ptr<float>() : nullptr;
  float* d_plse = (sk > 1) ? partial_lse_workspace.data_ptr<float>() : nullptr;

  auto dispatch_sk = [&](auto d_tag) {
    constexpr int kD = decltype(d_tag)::value;
    if (sk == 16) {
      launch_wmma_splitk_impl<16, kD>(
          q, k_pool, v_pool, page_t_obs, d_pout, d_plse, d_page_table, d_seq_lens, d_t_now_ns, out, d_out_lse,
          batch_size, Q_len, H_q, H_kv, L, max_pages_per_slot, layer_idx, sm_scale, apply_rel_time_rope, q_tiles, stream);
    } else if (sk == 8) {
      launch_wmma_splitk_impl<8, kD>(
          q, k_pool, v_pool, page_t_obs, d_pout, d_plse, d_page_table, d_seq_lens, d_t_now_ns, out, d_out_lse,
          batch_size, Q_len, H_q, H_kv, L, max_pages_per_slot, layer_idx, sm_scale, apply_rel_time_rope, q_tiles, stream);
    } else if (sk == 4) {
      launch_wmma_splitk_impl<4, kD>(
          q, k_pool, v_pool, page_t_obs, d_pout, d_plse, d_page_table, d_seq_lens, d_t_now_ns, out, d_out_lse,
          batch_size, Q_len, H_q, H_kv, L, max_pages_per_slot, layer_idx, sm_scale, apply_rel_time_rope, q_tiles, stream);
    } else if (sk == 2) {
      launch_wmma_splitk_impl<2, kD>(
          q, k_pool, v_pool, page_t_obs, d_pout, d_plse, d_page_table, d_seq_lens, d_t_now_ns, out, d_out_lse,
          batch_size, Q_len, H_q, H_kv, L, max_pages_per_slot, layer_idx, sm_scale, apply_rel_time_rope, q_tiles, stream);
    } else {
      launch_wmma_splitk_impl<1, kD>(
          q, k_pool, v_pool, page_t_obs, d_pout, d_plse, d_page_table, d_seq_lens, d_t_now_ns, out, d_out_lse,
          batch_size, Q_len, H_q, H_kv, L, max_pages_per_slot, layer_idx, sm_scale, apply_rel_time_rope, q_tiles, stream);
    }
  };

  if (D == 64) {
    dispatch_sk(std::integral_constant<int, 64>{});
  } else if (D == 128) {
    dispatch_sk(std::integral_constant<int, 128>{});
  } else {
    dispatch_sk(std::integral_constant<int, 256>{});
  }
  TS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace tempo
