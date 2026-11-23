from __future__ import annotations

from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union
import urllib.error
import urllib.request

from safetensors import safe_open
import torch


def _cache_root() -> Path:
    env_dir = os.environ.get("TEMPO_CACHE_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "tempo"
    return Path.home() / ".cache" / "tempo" / "hub"


def _hf_headers() -> Dict[str, str]:
    headers = {"User-Agent": "tempo/0.1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def download_hf_file(repo_id: str, filename: str, *, revision: str = "main") -> Path:
    """Downloads a single file from Hugging Face Hub with atomic local caching."""
    if importlib.util.find_spec("huggingface_hub") is not None:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        return Path(local)

    safe_repo = repo_id.replace("/", "--")
    dest = _cache_root() / safe_repo / revision / filename
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    tmp = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
    req = urllib.request.Request(url, headers=_hf_headers())
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out_f:
            shutil.copyfileobj(resp, out_f, length=16 * 1024 * 1024)
        tmp.replace(dest)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return dest


def _hf_file_exists(repo_id: str, filename: str, *, revision: str = "main") -> bool:
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    req = urllib.request.Request(url, method="HEAD", headers=_hf_headers())
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 400
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return False
        raise


def resolve_checkpoint(model_id_or_path: Union[str, Path]) -> Tuple[Dict[str, Any], Path]:
    """Resolves a local directory or Hugging Face repo ID to (config_dict, checkpoint_dir)."""
    cand = Path(model_id_or_path).expanduser()
    if cand.is_dir():
        cfg_path = cand / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing config.json in checkpoint directory: {cand}")
        return json.loads(cfg_path.read_text(encoding="utf-8")), cand

    repo_id = str(model_id_or_path)
    if importlib.util.find_spec("huggingface_hub") is not None:
        from huggingface_hub import snapshot_download

        snap_dir = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.txt",
                    "*.model",
                ],
            )
        )
        cfg_path = snap_dir / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing config.json in snapshot directory: {snap_dir}")
        return json.loads(cfg_path.read_text(encoding="utf-8")), snap_dir

    safe_repo = repo_id.replace("/", "--")
    cached_dir = _cache_root() / safe_repo / "main"
    cached_cfg = cached_dir / "config.json"
    if cached_cfg.is_file():
        cached_idx = cached_dir / "model.safetensors.index.json"
        if cached_idx.is_file():
            idx_data = json.loads(cached_idx.read_text(encoding="utf-8"))
            shards = sorted(set(idx_data.get("weight_map", {}).values()))
            if shards and all((cached_dir / s).is_file() for s in shards):
                return json.loads(cached_cfg.read_text(encoding="utf-8")), cached_dir
        elif (cached_dir / "model.safetensors").is_file():
            return json.loads(cached_cfg.read_text(encoding="utf-8")), cached_dir

    cfg_file = download_hf_file(repo_id, "config.json")
    config = json.loads(cfg_file.read_text(encoding="utf-8"))

    if _hf_file_exists(repo_id, "model.safetensors.index.json"):
        idx_file = download_hf_file(repo_id, "model.safetensors.index.json")
        idx_data = json.loads(idx_file.read_text(encoding="utf-8"))
        shards = sorted(set(idx_data.get("weight_map", {}).values()))
        if not shards:
            raise RuntimeError(f"Empty weight_map in {idx_file}")
        ckpt_dir = idx_file.parent
        for shard in shards:
            download_hf_file(repo_id, shard)
        return config, ckpt_dir

    st_file = download_hf_file(repo_id, "model.safetensors")
    return config, st_file.parent


def load_hf_tokenizer(repo_or_path: Union[str, Path], *, checkpoint_dir: Optional[Path] = None):
    """Loads ``tokenizer.json`` from a local checkpoint directory or Hugging Face repo."""
    from tokenizers import Tokenizer

    if checkpoint_dir is not None:
        local_tok = Path(checkpoint_dir).expanduser() / "tokenizer.json"
        if local_tok.is_file():
            return Tokenizer.from_file(str(local_tok))

    cand = Path(repo_or_path).expanduser()
    if cand.is_file():
        return Tokenizer.from_file(str(cand))
    if cand.is_dir() and (cand / "tokenizer.json").is_file():
        return Tokenizer.from_file(str(cand / "tokenizer.json"))

    tok_path = download_hf_file(str(repo_or_path), "tokenizer.json")
    return Tokenizer.from_file(str(tok_path))


class SafetensorLoader:
    """Safetensors reader supporting single-file and sharded checkpoints."""

    def __init__(self, ckpt_dir: Union[str, Path]) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self._stack = ExitStack()
        self._handles: Dict[str, Any] = {}
        self._key_to_file: Dict[str, str] = {}
        self.consumed: Set[str] = set()

        idx_path = self.ckpt_dir / "model.safetensors.index.json"
        if idx_path.is_file():
            idx_data = json.loads(idx_path.read_text(encoding="utf-8"))
            for k, shard_name in idx_data["weight_map"].items():
                self._key_to_file[k] = shard_name
        else:
            single = self.ckpt_dir / "model.safetensors"
            if not single.is_file():
                raise FileNotFoundError(f"No model.safetensors or index found in {self.ckpt_dir}")
            handle = self._stack.enter_context(safe_open(str(single), framework="pt", device="cpu"))
            self._handles["model.safetensors"] = handle
            for k in handle.keys():
                self._key_to_file[k] = "model.safetensors"

    def close(self) -> None:
        self._stack.close()
        self._handles.clear()

    def __enter__(self) -> "SafetensorLoader":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def keys(self) -> Set[str]:
        return set(self._key_to_file.keys())

    def has(self, key: str) -> bool:
        return key in self._key_to_file

    def _handle_for(self, key: str):
        shard = self._key_to_file.get(key)
        if shard is None:
            raise KeyError(f"Missing weight tensor '{key}' in {self.ckpt_dir}")
        handle = self._handles.get(shard)
        if handle is None:
            shard_path = self.ckpt_dir / shard
            handle = self._stack.enter_context(safe_open(str(shard_path), framework="pt", device="cpu"))
            self._handles[shard] = handle
        return handle

    def tensor(
        self,
        key: str,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> torch.Tensor:
        t = self._handle_for(key).get_tensor(key)
        self.consumed.add(key)
        if device is not None or (dtype is not None and t.dtype != dtype):
            t = t.to(device=device if device is not None else t.device, dtype=dtype if dtype is not None else t.dtype)
        return t.contiguous()

    def load_linear(
        self,
        mod: torch.nn.Linear,
        prefix: str,
        *,
        device: Union[str, torch.device],
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        with torch.no_grad():
            w = self.tensor(f"{prefix}.weight", device=device, dtype=dtype)
            if w.shape != mod.weight.shape:
                raise ValueError(
                    f"Shape mismatch for '{prefix}.weight': expected {tuple(mod.weight.shape)}, got {tuple(w.shape)}"
                )
            mod.weight.copy_(w)
            if mod.bias is not None:
                b = self.tensor(f"{prefix}.bias", device=device, dtype=dtype)
                if b.shape != mod.bias.shape:
                    raise ValueError(
                        f"Shape mismatch for '{prefix}.bias': expected {tuple(mod.bias.shape)}, got {tuple(b.shape)}"
                    )
                mod.bias.copy_(b)

    def verify_all_consumed(self, *, ignore_keys: Iterable[str] = (), ignore_prefixes: Iterable[str] = ()) -> None:
        ignored = set(ignore_keys)
        prefixes = tuple(ignore_prefixes)
        unconsumed = sorted(
            k
            for k in self._key_to_file
            if k not in self.consumed and k not in ignored and not k.startswith(prefixes)
        )
        if unconsumed:
            sample = ", ".join(unconsumed[:8])
            raise RuntimeError(
                f"{len(unconsumed)} checkpoint tensor(s) in {self.ckpt_dir} were not loaded: {sample}"
            )
