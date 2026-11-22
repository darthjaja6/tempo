"""
TCP transport for edge-cloud System 1 + System 2 serving (remote Slow VLM server + on-device Fast engine).
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .channel import Engine, _to_bf16_obs

_MAX_NDIM = 6
_HEADER_STRUCT = struct.Struct(f"!4siiqi{_MAX_NDIM}i")
_MAGIC = b"TSRM"
_VERSION = 1


def _pack_frame(stream_id: int, t_obs_ns: int, tensor_bf16_cpu: torch.Tensor) -> Tuple[bytes, memoryview]:
    if tensor_bf16_cpu.dtype != torch.bfloat16 or not tensor_bf16_cpu.is_contiguous() or tensor_bf16_cpu.is_cuda:
        raise ValueError("_pack_frame requires contiguous CPU bfloat16 tensor")
    ndim = tensor_bf16_cpu.ndim
    if ndim < 1 or ndim > _MAX_NDIM or tensor_bf16_cpu.numel() <= 0:
        raise ValueError(f"Expected non-empty 1D..{_MAX_NDIM}D tensor, got shape {tuple(tensor_bf16_cpu.shape)}")
    dims = list(tensor_bf16_cpu.shape) + [0] * (_MAX_NDIM - ndim)
    hdr = _HEADER_STRUCT.pack(_MAGIC, _VERSION, int(stream_id), int(t_obs_ns), ndim, *dims)
    mv = memoryview(tensor_bf16_cpu.view(torch.uint8).numpy())
    return hdr, mv


def _recv_exact_into(sock: socket.socket, buf: bytearray) -> bool:
    mv = memoryview(buf)
    offset = 0
    total = len(buf)
    while offset < total:
        n = sock.recv_into(mv[offset:])
        if n == 0:
            return False
        offset += n
    return True


def _recv_frame(sock: socket.socket) -> Optional[Tuple[int, int, torch.Tensor]]:
    hdr_buf = bytearray(_HEADER_STRUCT.size)
    if not _recv_exact_into(sock, hdr_buf):
        return None
    unpacked = _HEADER_STRUCT.unpack(hdr_buf)
    magic, ver, stream_id, t_obs_ns, ndim = unpacked[:5]
    raw_dims = unpacked[5:]
    if magic != _MAGIC or ver != _VERSION:
        raise RuntimeError(f"Invalid TSRM frame header (magic={magic!r}, version={ver})")
    if ndim < 1 or ndim > _MAX_NDIM:
        raise RuntimeError(f"Invalid TSRM tensor ndim={ndim}")
    shape = tuple(raw_dims[:ndim])
    if any(d <= 0 for d in shape):
        raise RuntimeError(f"Invalid TSRM tensor shape={shape}")
    numel = 1
    for d in shape:
        numel *= d
    payload = bytearray(numel * 2)
    if not _recv_exact_into(sock, payload):
        return None
    tensor = torch.frombuffer(payload, dtype=torch.bfloat16).reshape(shape)
    return stream_id, t_obs_ns, tensor


class _ClientConn:
    def __init__(self, conn_id: int, sock: socket.socket) -> None:
        self.conn_id = conn_id
        self.sock = sock
        self.send_lock = threading.Lock()
        self.closed = False

    def send_frame(self, stream_id: int, t_obs_ns: int, hidden_cpu: torch.Tensor) -> None:
        hdr, mv = _pack_frame(stream_id, t_obs_ns, hidden_cpu)
        with self.send_lock:
            if self.closed:
                return
            self.sock.sendall(hdr)
            self.sock.sendall(mv)

    def close(self) -> None:
        with self.send_lock:
            if self.closed:
                return
            self.closed = True
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.sock.close()
            except OSError:
                pass


class RemoteSlowServer:
    """AoI-batched Slow VLM server echoing client observation timestamps alongside BF16 latent states."""

    def __init__(
        self,
        slow_fn: Callable[[torch.Tensor], torch.Tensor],
        host: str = "0.0.0.0",
        port: int = 0,
        max_batch_size: int = 4,
        device: str | torch.device = "cuda:0",
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.slow_fn = slow_fn
        self.host = host
        self._bind_port = port
        self.max_batch_size = max_batch_size
        self.device = torch.device(device)

        self._srv_sock: Optional[socket.socket] = None
        self.port: int = port
        self._running = False

        self._mu = threading.Lock()
        self._cv = threading.Condition(self._mu)
        # Keyed by (conn_id, stream_id) -> (arrival_seq, t_obs_ns, obs_cpu, _ClientConn)
        self._pending: Dict[Tuple[int, int], Tuple[int, int, torch.Tensor, _ClientConn]] = {}
        self._last_t_obs: Dict[Tuple[int, int], int] = {}
        self._seq_gen = 0
        self._conns: Dict[int, _ClientConn] = {}
        self._next_conn_id = 1

        self._accept_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._client_threads: List[threading.Thread] = []

    def start(self) -> "RemoteSlowServer":
        if self._running:
            return self
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self._bind_port))
        srv.listen(64)
        self.port = int(srv.getsockname()[1])
        self._srv_sock = srv
        self._running = True

        self._worker_thread = threading.Thread(target=self._batch_worker_loop, daemon=True)
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._worker_thread.start()
        self._accept_thread.start()
        return self

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._srv_sock is not None:
            try:
                self._srv_sock.close()
            except OSError:
                pass
            self._srv_sock = None
        with self._cv:
            conns = list(self._conns.values())
            self._conns.clear()
            self._pending.clear()
            self._cv.notify_all()
        for c in conns:
            c.close()
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=3.0)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=3.0)
        for t in self._client_threads:
            t.join(timeout=2.0)
        self._client_threads.clear()

    def __enter__(self) -> "RemoteSlowServer":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def _accept_loop(self) -> None:
        assert self._srv_sock is not None
        while self._running:
            try:
                client_sock, _ = self._srv_sock.accept()
            except OSError:
                break
            client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._cv:
                cid = self._next_conn_id
                self._next_conn_id += 1
                conn = _ClientConn(cid, client_sock)
                self._conns[cid] = conn
                t = threading.Thread(target=self._client_reader_loop, args=(conn,), daemon=True)
                self._client_threads.append(t)
            t.start()

    def _client_reader_loop(self, conn: _ClientConn) -> None:
        try:
            while self._running:
                frame = _recv_frame(conn.sock)
                if frame is None:
                    break
                stream_id, t_obs_ns, obs_cpu = frame
                key = (conn.conn_id, stream_id)
                with self._cv:
                    prev_t = self._last_t_obs.get(key)
                    if prev_t is not None and t_obs_ns <= prev_t:
                        continue
                    self._last_t_obs[key] = t_obs_ns
                    if key in self._pending:
                        seq = self._pending[key][0]
                    else:
                        self._seq_gen += 1
                        seq = self._seq_gen
                    self._pending[key] = (seq, t_obs_ns, obs_cpu, conn)
                    self._cv.notify()
        except OSError:
            pass
        finally:
            conn.close()
            with self._cv:
                self._conns.pop(conn.conn_id, None)
                stale_keys = [k for k in self._pending if k[0] == conn.conn_id]
                for k in stale_keys:
                    self._pending.pop(k, None)
                for k in [k for k in self._last_t_obs if k[0] == conn.conn_id]:
                    self._last_t_obs.pop(k, None)

    def _batch_worker_loop(self) -> None:
        while self._running:
            with self._cv:
                while self._running and not self._pending:
                    self._cv.wait(timeout=0.1)
                if not self._running:
                    break
                # Oldest arrival_seq first (preserves fairness across heterogeneous client clocks)
                ordered = sorted(self._pending.items(), key=lambda kv: kv[1][0])[: self.max_batch_size]
                batch_entries = []
                for k, val in ordered:
                    self._pending.pop(k, None)
                    batch_entries.append((k[1], val[1], val[2], val[3]))

            batch_obs = torch.stack([e[2] for e in batch_entries], dim=0).to(self.device, dtype=torch.bfloat16)
            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        hidden_out = self.slow_fn(batch_obs)
                else:
                    hidden_out = self.slow_fn(batch_obs)
                if hidden_out.dtype != torch.bfloat16:
                    hidden_out = hidden_out.to(torch.bfloat16)
                hidden_cpu = hidden_out.detach().contiguous().cpu()

            for i, (stream_id, echoed_t_obs_ns, _obs, conn) in enumerate(batch_entries):
                try:
                    conn.send_frame(stream_id, echoed_t_obs_ns, hidden_cpu[i])
                except OSError:
                    conn.close()


class RemoteSlowClient:
    """Non-blocking Echo-Timestamp client bridging an on-device Fast Engine to a RemoteSlowServer."""

    def __init__(
        self,
        engine: Engine,
        host: str,
        port: int,
        reconnect_sec: float = 0.1,
    ) -> None:
        self.engine = engine
        self.host = host
        self.port = port
        self.reconnect_sec = reconnect_sec

        self._mu = threading.Lock()
        self._cv = threading.Condition(self._mu)
        # stream_id -> (t_obs_ns, obs_cpu_bf16) latest-only outbound slot
        self._outbound: Dict[int, Tuple[int, torch.Tensor]] = {}
        self._last_pushed_ns: Dict[int, int] = {}
        self._active_sock: Optional[socket.socket] = None
        self._worker_error: Optional[BaseException] = None
        self._running = False
        self._io_thread: Optional[threading.Thread] = None

    def start(self) -> "RemoteSlowClient":
        if self._running:
            return self
        self._worker_error = None
        self._running = True
        self._io_thread = threading.Thread(target=self._io_loop, daemon=True)
        self._io_thread.start()
        return self

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        with self._cv:
            sock = self._active_sock
            self._cv.notify_all()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        if self._io_thread is not None:
            self._io_thread.join(timeout=3.0)
            self._io_thread = None

    def __enter__(self) -> "RemoteSlowClient":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def reset(self) -> None:
        """Clears queued outbound observations and timestamp watermarks between episodes."""
        with self._cv:
            self._outbound.clear()
            self._last_pushed_ns.clear()

    def push_slow(self, stream_id: int, obs: torch.Tensor, t_obs_ns: Optional[int] = None) -> int:
        if stream_id < 0 or stream_id >= self.engine.num_streams:
            raise IndexError(f"stream_id {stream_id} out of range [0, {self.engine.num_streams})")
        if obs.ndim < 1 or obs.ndim > _MAX_NDIM or obs.numel() <= 0:
            raise ValueError(f"Expected non-empty 1D..{_MAX_NDIM}D tensor, got shape {tuple(obs.shape)}")
        obs_cpu = _to_bf16_obs(obs.detach(), "RemoteSlowClient.push_slow", device="cpu")
        stamp_ns = self.engine.now_ns() if t_obs_ns is None else int(t_obs_ns)
        with self._cv:
            if self._worker_error is not None:
                raise RuntimeError(f"RemoteSlowClient worker failed: {self._worker_error}") from self._worker_error
            prev_ns = self._last_pushed_ns.get(stream_id)
            if prev_ns is None or stamp_ns > prev_ns:
                self._last_pushed_ns[stream_id] = stamp_ns
                self._outbound[stream_id] = (stamp_ns, obs_cpu)
                self._cv.notify()
        return stamp_ns

    def _recv_loop(self, sock: socket.socket) -> None:
        try:
            while self._running:
                frame = _recv_frame(sock)
                if frame is None:
                    break
                stream_id, echoed_t_obs_ns, hidden_cpu = frame
                self.engine.publish_remote_hidden(stream_id, hidden_cpu, echoed_t_obs_ns)
        except OSError:
            pass
        except BaseException as exc:
            with self._cv:
                self._worker_error = exc
            self._running = False
        finally:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            with self._cv:
                self._cv.notify_all()

    def _io_loop(self) -> None:
        while self._running:
            sock: Optional[socket.socket] = None
            recv_t: Optional[threading.Thread] = None
            try:
                sock = socket.create_connection((self.host, self.port), timeout=2.0)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(None)
                with self._cv:
                    if not self._running:
                        sock.close()
                        break
                    self._active_sock = sock

                recv_t = threading.Thread(target=self._recv_loop, args=(sock,), daemon=True)
                recv_t.start()

                while self._running and recv_t.is_alive():
                    with self._cv:
                        while self._running and not self._outbound and recv_t.is_alive():
                            self._cv.wait(timeout=0.05)
                        if not self._running or not recv_t.is_alive():
                            break
                        stream_id, (t_obs_ns, obs_cpu) = min(
                            self._outbound.items(), key=lambda kv: kv[1][0]
                        )
                        self._outbound.pop(stream_id, None)

                    hdr, mv = _pack_frame(stream_id, t_obs_ns, obs_cpu)
                    sock.sendall(hdr)
                    sock.sendall(mv)
            except OSError:
                if self._running:
                    time.sleep(self.reconnect_sec)
            except BaseException as exc:
                with self._cv:
                    self._worker_error = exc
                self._running = False
                break
            finally:
                with self._cv:
                    if self._active_sock is sock:
                        self._active_sock = None
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
                if recv_t is not None:
                    recv_t.join(timeout=1.0)
