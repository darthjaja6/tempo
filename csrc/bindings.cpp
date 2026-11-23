#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "rtc_ring.h"
#include "scheduler.h"
#include "tsc_pool.h"

namespace py = pybind11;

namespace tempo {

PYBIND11_MODULE(_tempo_C, m) {
  m.doc() = "Tempo C++/CUDA runtime extension";

  m.def("steady_now_ns", &steady_now_ns, "Monotonic steady clock in nanoseconds");

  m.def(
      "write_paged_kv",
      [](const torch::Tensor& k_in,
         const torch::Tensor& v_in,
         torch::Tensor& k_pool,
         torch::Tensor& v_pool,
         torch::Tensor& page_t_obs_ns,
         const torch::Tensor& page_table,
         const torch::Tensor& start_page_idx,
         const torch::Tensor& t_obs_ns,
         int32_t layer_idx,
         bool bake_spatial_rope,
         int32_t spatial_grid_w) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(k_in.get_device()).stream();
        launch_write_paged_kv(
            k_in,
            v_in,
            k_pool,
            v_pool,
            page_t_obs_ns,
            page_table,
            start_page_idx,
            t_obs_ns,
            layer_idx,
            bake_spatial_rope,
            spatial_grid_w,
            stream);
      },
      "Capturable paged KV write with spatial (h, w) RoPE baking and sliding-window append offset");

  m.def(
      "gather_paged_kv_history",
      [](const torch::Tensor& k_pool,
         const torch::Tensor& v_pool,
         const torch::Tensor& page_t_obs_ns,
         const torch::Tensor& page_table,
         const torch::Tensor& start_page_idx,
         const torch::Tensor& t_obs_ns,
         torch::Tensor& k_out,
         torch::Tensor& v_out,
         int32_t layer_idx,
         bool apply_rel_time_rope) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(k_pool.get_device()).stream();
        launch_gather_paged_kv_history(
            k_pool,
            v_pool,
            page_t_obs_ns,
            page_table,
            start_page_idx,
            t_obs_ns,
            k_out,
            v_out,
            layer_idx,
            apply_rel_time_rope,
            stream);
      },
      "Capturable sliding-window history KV gather with relative-time RoPE from co-pooled TSC pages");

  m.def(
      "write_paged_hidden",
      [](const torch::Tensor& hidden_in,
         torch::Tensor& v_pool,
         torch::Tensor& page_t_obs_ns,
         const torch::Tensor& page_table,
         const torch::Tensor& start_page_idx,
         const torch::Tensor& t_obs_ns) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(hidden_in.get_device()).stream();
        launch_write_paged_hidden(hidden_in, v_pool, page_t_obs_ns, page_table, start_page_idx, t_obs_ns, stream);
      },
      "Capturable paged dense hidden-state write into TSC pool");

  m.def(
      "gather_paged_hidden",
      [](const torch::Tensor& v_pool,
         const torch::Tensor& page_t_obs_ns,
         const torch::Tensor& page_table,
         const torch::Tensor& seq_lens,
         const torch::Tensor& t_now_ns,
         torch::Tensor& out,
         torch::Tensor& out_delta_sec) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(v_pool.get_device()).stream();
        launch_gather_paged_hidden(
            v_pool,
            page_t_obs_ns,
            page_table.data_ptr<int32_t>(),
            seq_lens.data_ptr<int32_t>(),
            t_now_ns.data_ptr<int64_t>(),
            out,
            out_delta_sec,
            static_cast<int32_t>(out.size(0)),
            static_cast<int32_t>(out.size(1)),
            static_cast<int32_t>(page_table.size(1)),
            stream);
      },
      "Capturable zero-copy paged dense hidden-state gather with per-token staleness delta_t_sec");

  m.def(
      "paged_rel_rope_cross_attn",
      [](const torch::Tensor& q,
         const torch::Tensor& k_pool,
         const torch::Tensor& v_pool,
         const torch::Tensor& page_t_obs_ns,
         torch::Tensor& partial_out_workspace,
         torch::Tensor& partial_lse_workspace,
         const torch::Tensor& page_table,
         const torch::Tensor& seq_lens,
         const torch::Tensor& t_now_ns,
         torch::Tensor& out,
         int32_t layer_idx,
         bool apply_rel_time_rope,
         c10::optional<torch::Tensor> out_lse) {
        float* d_out_lse = nullptr;
        if (out_lse.has_value() && out_lse->defined()) {
          if (!out_lse->is_cuda() || !out_lse->is_contiguous() || out_lse->scalar_type() != torch::kFloat32 ||
              out_lse->device() != q.device() || out_lse->dim() != 3 || out_lse->size(0) != q.size(0) ||
              out_lse->size(1) != q.size(1) || out_lse->size(2) != q.size(2)) {
            throw std::invalid_argument(
                "paged_rel_rope_cross_attn out_lse must be contiguous CUDA float32 [B, Q_len, H_q] on q's device");
          }
          d_out_lse = out_lse->data_ptr<float>();
        }
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
        launch_paged_rel_rope_cross_attn(
            q,
            k_pool,
            v_pool,
            page_t_obs_ns,
            partial_out_workspace,
            partial_lse_workspace,
            page_table.data_ptr<int32_t>(),
            seq_lens.data_ptr<int32_t>(),
            t_now_ns.data_ptr<int64_t>(),
            out,
            static_cast<int32_t>(q.size(0)),
            static_cast<int32_t>(page_table.size(1)),
            layer_idx,
            apply_rel_time_rope,
            stream,
            d_out_lse);
      },
      py::arg("q"),
      py::arg("k_pool"),
      py::arg("v_pool"),
      py::arg("page_t_obs_ns"),
      py::arg("partial_out_workspace"),
      py::arg("partial_lse_workspace"),
      py::arg("page_table"),
      py::arg("seq_lens"),
      py::arg("t_now_ns"),
      py::arg("out"),
      py::arg("layer_idx"),
      py::arg("apply_rel_time_rope"),
      py::arg("out_lse") = py::none(),
      "Capturable zero-copy paged cross-attention with on-chip SRAM relative-time RoPE and optional LSE output");

  py::class_<TemporalStateCache, std::shared_ptr<TemporalStateCache>>(m, "TemporalStateCache")
      .def(py::init<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, bool, int32_t, int32_t, int32_t>(),
           py::arg("num_streams"),
           py::arg("num_readers"),
           py::arg("max_pages"),
           py::arg("num_layers"),
           py::arg("page_size"),
           py::arg("num_kv_heads"),
           py::arg("head_dim"),
           py::arg("max_seq_len"),
           py::arg("kv_mode") = true,
           py::arg("max_q_len") = 64,
           py::arg("max_q_heads") = 16,
           py::arg("device_id") = 0)
      .def("reclaim_retired_reads", &TemporalStateCache::reclaim_retired_reads)
      .def("reset_stream", &TemporalStateCache::reset_stream)
      .def(
           "write_single_stream_kv",
           [](TemporalStateCache& self,
              int32_t stream_id,
              const torch::Tensor& k_all_layers,
              const torch::Tensor& v_all_layers,
              int64_t t_obs_ns,
              bool bake_spatial_rope) {
             if (k_all_layers.dim() != 5 || v_all_layers.sizes() != k_all_layers.sizes() ||
                 k_all_layers.size(0) != self.num_layers()) {
               throw std::invalid_argument(
                   "write_single_stream_kv requires matching 5D [num_layers, 1, S, H_kv, D] tensors");
             }
             c10::cuda::CUDAGuard guard(self.device_id());
             cudaStream_t st = at::cuda::getCurrentCUDAStream(self.device_id()).stream();
             const int32_t S = static_cast<int32_t>(k_all_layers.size(2));
             const int32_t gs = self.begin_write(stream_id, S, st);
             try {
               for (int32_t l = 0; l < self.num_layers(); ++l) {
                 self.write_layer_kv(gs, l, k_all_layers[l], v_all_layers[l], t_obs_ns, bake_spatial_rope, st);
               }
               self.publish_write(stream_id, gs, t_obs_ns, S, st);
             } catch (...) {
               self.abort_write_batch({gs}, st);
               throw;
             }
           })
      .def(
           "write_single_stream_hidden",
           [](TemporalStateCache& self,
              int32_t stream_id,
              const torch::Tensor& hidden_in,
              int64_t t_obs_ns) {
             if (hidden_in.dim() != 3) {
               throw std::invalid_argument("write_single_stream_hidden requires 3D [1, S, D] tensor");
             }
             c10::cuda::CUDAGuard guard(self.device_id());
             cudaStream_t st = at::cuda::getCurrentCUDAStream(self.device_id()).stream();
             const int32_t S = static_cast<int32_t>(hidden_in.size(1));
             const int32_t gs = self.begin_write(stream_id, S, st);
             try {
               self.write_hidden_state(gs, hidden_in, t_obs_ns, st);
               self.publish_write(stream_id, gs, t_obs_ns, S, st);
             } catch (...) {
               self.abort_write_batch({gs}, st);
               throw;
             }
           })
      .def(
           "read_batch_cross_attn",
           [](TemporalStateCache& self,
              int32_t reader_id,
              const std::vector<int32_t>& stream_ids,
              const std::vector<int64_t>& t_now_ns,
              const torch::Tensor& q,
              int32_t layer_idx,
              bool apply_rel_time_rope) {
             if (!q.is_cuda() || q.get_device() != self.device_id() || q.dim() != 4) {
               throw std::invalid_argument("read_batch_cross_attn requires 4D q on TemporalStateCache device");
             }
             if (t_now_ns.size() != stream_ids.size() || static_cast<int64_t>(stream_ids.size()) != q.size(0)) {
               throw std::invalid_argument(
                   "read_batch_cross_attn requires stream_ids.size() == t_now_ns.size() == q.size(0)");
             }
             c10::cuda::CUDAGuard guard(self.device_id());
             cudaStream_t st = at::cuda::getCurrentCUDAStream(self.device_id()).stream();
             const int32_t B = static_cast<int32_t>(stream_ids.size());
             auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(q.device());
             auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(q.device());
             auto d_pt = torch::empty({B, self.max_pages_per_slot()}, i32);
             auto d_sl = torch::empty({B}, i32);
             auto d_tn = torch::empty({B}, i64);
             auto out = torch::empty_like(q);
             auto pout = self.splitk_partial_out(reader_id);
             auto plse = self.splitk_partial_lse(reader_id);

             std::vector<int64_t> slow_tobs;
             std::vector<int32_t> pinned;
             self.acquire_read_batch(
                 reader_id,
                 stream_ids,
                 t_now_ns,
                 d_pt.data_ptr<int32_t>(),
                 d_sl.data_ptr<int32_t>(),
                 d_tn.data_ptr<int64_t>(),
                 slow_tobs,
                 pinned,
                 st);
             try {
               launch_paged_rel_rope_cross_attn(
                   q,
                   self.k_pool(),
                   self.v_pool(),
                   self.page_timestamps(),
                   pout,
                   plse,
                   d_pt.data_ptr<int32_t>(),
                   d_sl.data_ptr<int32_t>(),
                   d_tn.data_ptr<int64_t>(),
                   out,
                   B,
                   self.max_pages_per_slot(),
                   layer_idx,
                   apply_rel_time_rope,
                   st);
               self.release_read_batch(reader_id, pinned, st);
             } catch (...) {
               self.abort_read_batch(reader_id, pinned);
               throw;
             }
             return std::make_pair(out, slow_tobs);
           })
      .def(
           "read_batch_hidden",
           [](TemporalStateCache& self,
              int32_t reader_id,
              const std::vector<int32_t>& stream_ids,
              const std::vector<int64_t>& t_now_ns,
              int32_t seq_len) {
             if (t_now_ns.size() != stream_ids.size() || seq_len <= 0) {
               throw std::invalid_argument(
                   "read_batch_hidden requires stream_ids.size() == t_now_ns.size() and seq_len > 0");
             }
             c10::cuda::CUDAGuard guard(self.device_id());
             cudaStream_t st = at::cuda::getCurrentCUDAStream(self.device_id()).stream();
             const int32_t B = static_cast<int32_t>(stream_ids.size());
             auto dev = self.v_pool().device();
             auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
             auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(dev);
             auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(dev);
             auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
             auto d_pt = torch::empty({B, self.max_pages_per_slot()}, i32);
             auto d_sl = torch::empty({B}, i32);
             auto d_tn = torch::empty({B}, i64);
             const int32_t feat_dim = self.num_kv_heads() * self.head_dim();
             auto out = torch::empty({B, seq_len, feat_dim}, bf16);
             auto out_dt = torch::empty({B, seq_len}, f32);

             std::vector<int64_t> slow_tobs;
             std::vector<int32_t> pinned;
             self.acquire_read_batch(
                 reader_id,
                 stream_ids,
                 t_now_ns,
                 d_pt.data_ptr<int32_t>(),
                 d_sl.data_ptr<int32_t>(),
                 d_tn.data_ptr<int64_t>(),
                 slow_tobs,
                 pinned,
                 st);
             try {
               launch_gather_paged_hidden(
                   self.v_pool(),
                   self.page_timestamps(),
                   d_pt.data_ptr<int32_t>(),
                   d_sl.data_ptr<int32_t>(),
                   d_tn.data_ptr<int64_t>(),
                   out,
                   out_dt,
                   B,
                   seq_len,
                   self.max_pages_per_slot(),
                   st);
               self.release_read_batch(reader_id, pinned, st);
             } catch (...) {
               self.abort_read_batch(reader_id, pinned);
               throw;
             }
             return std::make_tuple(out, out_dt, slow_tobs);
           })
      .def_property_readonly("k_pool", &TemporalStateCache::k_pool)
      .def_property_readonly("v_pool", &TemporalStateCache::v_pool)
      .def_property_readonly("page_timestamps", &TemporalStateCache::page_timestamps)
      .def("splitk_partial_out", &TemporalStateCache::splitk_partial_out, py::arg("reader_id") = 0)
      .def("splitk_partial_lse", &TemporalStateCache::splitk_partial_lse, py::arg("reader_id") = 0)
      .def_property_readonly("free_page_count", &TemporalStateCache::free_page_count)
      .def_property_readonly("max_pages_per_slot", &TemporalStateCache::max_pages_per_slot)
      .def_property_readonly("kv_mode", &TemporalStateCache::kv_mode);

  py::class_<RTCStats>(m, "RTCStats")
      .def_readonly("total_commits", &RTCStats::total_commits)
      .def_readonly("soft_overruns", &RTCStats::soft_overruns)
      .def_readonly("hard_underruns", &RTCStats::hard_underruns);

  py::class_<RTCHorizonRing, std::shared_ptr<RTCHorizonRing>>(m, "RTCHorizonRing")
      .def(py::init<int32_t, int32_t, int32_t, int32_t, int64_t, int32_t>(),
           py::arg("num_streams"),
           py::arg("horizon_steps"),
           py::arg("prefix_steps"),
           py::arg("action_dim"),
           py::arg("step_dt_ns"),
           py::arg("device_id") = 0)
      .def("reset_all", &RTCHorizonRing::reset_all, py::arg("init_pose") = py::none())
      .def("reset_stats", &RTCHorizonRing::reset_stats)
      .def("sample_at", &RTCHorizonRing::sample_at, py::call_guard<py::gil_scoped_release>())
      .def("stats", &RTCHorizonRing::stats);

  py::class_<SchedulerTelemetry>(m, "SchedulerTelemetry")
      .def_readonly("fast_ttac_ms", &SchedulerTelemetry::fast_ttac_ms)
      .def_readonly("slow_latency_ms", &SchedulerTelemetry::slow_latency_ms)
      .def_readonly("fast_batches_launched", &SchedulerTelemetry::fast_batches_launched)
      .def_readonly("fast_requests_served", &SchedulerTelemetry::fast_requests_served)
      .def_readonly("slow_batches_completed", &SchedulerTelemetry::slow_batches_completed)
      .def_readonly("slow_slices_launched", &SchedulerTelemetry::slow_slices_launched)
      .def_readonly("fast_preemptions", &SchedulerTelemetry::fast_preemptions)
      .def_readonly("fast_coalesced_drops", &SchedulerTelemetry::fast_coalesced_drops)
      .def_readonly("slow_coalesced_drops", &SchedulerTelemetry::slow_coalesced_drops)
      .def_readonly("p2p_prepushes", &SchedulerTelemetry::p2p_prepushes);

  py::class_<TimeAwareScheduler, std::shared_ptr<TimeAwareScheduler>>(m, "TimeAwareScheduler")
      .def(py::init<
               int32_t,
               const std::vector<std::vector<int32_t>>&,
               const std::vector<int32_t>&,
               int32_t,
               int32_t,
               int32_t,
               double,
               double,
               double>(),
           py::arg("num_streams"),
           py::arg("slow_devices"),
           py::arg("fast_devices"),
           py::arg("slow_batch_size"),
           py::arg("max_fast_batch_size"),
           py::arg("slow_frame_seq_len"),
           py::arg("fast_deadline_ms"),
           py::arg("max_slice_ms"),
           py::arg("max_slow_aoi_ms"))
      .def("num_fast_shards", &TimeAwareScheduler::num_fast_shards)
      .def("num_slow_replicas", &TimeAwareScheduler::num_slow_replicas)
      .def("num_slow_stages", &TimeAwareScheduler::num_slow_stages)
      .def("fast_shard_num_streams", &TimeAwareScheduler::fast_shard_num_streams)
      .def("slow_replica_num_streams", &TimeAwareScheduler::slow_replica_num_streams)
      .def("is_disaggregated", &TimeAwareScheduler::is_disaggregated)
      .def("p2p_direct_enabled", &TimeAwareScheduler::p2p_direct_enabled)
      .def("init_fast_shard", &TimeAwareScheduler::init_fast_shard)
      .def("init_slow_stage", &TimeAwareScheduler::init_slow_stage)
      .def("fast_exec_cuda_stream_raw", &TimeAwareScheduler::fast_exec_cuda_stream_raw)
      .def("slow_exec_cuda_stream_raw", &TimeAwareScheduler::slow_exec_cuda_stream_raw)
      .def("begin_slow_slice_capture", &TimeAwareScheduler::begin_slow_slice_capture)
      .def("end_slow_slice_capture", &TimeAwareScheduler::end_slow_slice_capture)
      .def("profile_slow_slices", &TimeAwareScheduler::profile_slow_slices)
      .def("begin_fast_bucket_capture", &TimeAwareScheduler::begin_fast_bucket_capture)
      .def("end_fast_bucket_capture", &TimeAwareScheduler::end_fast_bucket_capture)
      .def("abort_active_captures", &TimeAwareScheduler::abort_active_captures)
      .def("start", &TimeAwareScheduler::start, py::call_guard<py::gil_scoped_release>())
      .def("stop", &TimeAwareScheduler::stop, py::call_guard<py::gil_scoped_release>())
      .def("reset_all_streams", &TimeAwareScheduler::reset_all_streams, py::arg("init_pose") = py::none())
      .def("reset_telemetry", &TimeAwareScheduler::reset_telemetry)
      .def("reset_rtc_stats", &TimeAwareScheduler::reset_rtc_stats)
      .def("submit_slow_obs", &TimeAwareScheduler::submit_slow_obs, py::call_guard<py::gil_scoped_release>())
      .def("submit_fast_obs", &TimeAwareScheduler::submit_fast_obs, py::call_guard<py::gil_scoped_release>())
      .def("submit_remote_hidden", &TimeAwareScheduler::submit_remote_hidden, py::call_guard<py::gil_scoped_release>())
      .def(
           "sample_actions_at",
           &TimeAwareScheduler::sample_actions_at,
           py::arg("t_query_ns"),
           py::arg("return_cpu") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("rtc_stats", &TimeAwareScheduler::rtc_stats)
      .def("telemetry", &TimeAwareScheduler::telemetry)
      .def("now_ns", &TimeAwareScheduler::now_ns);
}

}  // namespace tempo
