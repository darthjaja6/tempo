import concurrent.futures
import os
import pathlib
import re
import shutil
import subprocess
import sys
import sysconfig
from typing import List

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


def _resolve_nvcc(ce) -> str:
    found = shutil.which("nvcc")
    if found:
        return found
    cuda_home = os.environ.get("CUDA_HOME") or getattr(ce, "CUDA_HOME", None)
    if cuda_home:
        candidate = pathlib.Path(cuda_home) / "bin" / "nvcc"
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("nvcc not found in PATH or CUDA_HOME. Install CUDA Toolkit >= 12.0 or set CUDA_HOME.")


def _nvcc_max_arch(nvcc_bin: str) -> int:
    out = subprocess.check_output([nvcc_bin, "--version"], text=True)
    m = re.search(r"release\s+(\d+)\.(\d+)", out)
    if not m:
        return 90
    ver = (int(m.group(1)), int(m.group(2)))
    return 120 if ver >= (12, 8) else 90


def _resolve_arch_flags(nvcc_bin: str, torch) -> List[str]:
    max_arch = _nvcc_max_arch(nvcc_bin)
    arch_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if arch_env:
        flags: List[str] = []
        for tok in re.split(r"[;,\s]+", arch_env):
            tok = tok.strip()
            if not tok:
                continue
            want_ptx = tok.endswith("+PTX")
            clean = tok.replace("+PTX", "")
            parts = clean.split(".")
            num = min(int(parts[0]) * 10 + (int(parts[1]) if len(parts) > 1 else 0), max_arch)
            flags.extend(["-gencode", f"arch=compute_{num},code=sm_{num}"])
            if want_ptx:
                flags.extend(["-gencode", f"arch=compute_{num},code=compute_{num}"])
        if flags:
            return flags

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. Run on a GPU host or set TORCH_CUDA_ARCH_LIST (e.g. '8.9' or '9.0') for headless builds."
        )
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    arch_num = min(cc_major * 10 + cc_minor, max_arch)
    return [f"-arch=sm_{arch_num}"]


def _build_cuda_extension(out_so: pathlib.Path) -> None:
    import torch
    import torch.utils.cpp_extension as ce

    root = pathlib.Path(__file__).resolve().parent
    build_dir = root / "build"
    build_dir.mkdir(exist_ok=True)
    out_so.parent.mkdir(parents=True, exist_ok=True)

    nvcc_bin = _resolve_nvcc(ce)
    arch_flags = _resolve_arch_flags(nvcc_bin, torch)
    cxx_bin = os.environ.get("CXX", "g++")

    cxx11_abi = int(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", 1))
    py_inc = sysconfig.get_paths()["include"]

    inc_dirs = [str(root / "csrc"), py_inc] + ce.include_paths("cuda")
    inc_flags = [f"-I{p}" for p in inc_dirs]
    defs = [
        f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DTORCH_EXTENSION_NAME=_tempo_C",
    ]

    header_mtime = max(
        [pathlib.Path(__file__).stat().st_mtime] + [h.stat().st_mtime for h in (root / "csrc").glob("*.h")]
    )

    cu_sources = ["csrc/tsc_pool.cu", "csrc/rtc_ring.cu"]
    cpp_sources = ["csrc/scheduler.cpp", "csrc/bindings.cpp"]
    obj_files: List[str] = []
    compile_jobs = []

    for src in cu_sources:
        src_path = root / src
        obj = build_dir / (src_path.stem + ".cu.o")
        obj_files.append(str(obj))
        cmd = [
            nvcc_bin,
            "-O3",
            "-std=c++17",
            *arch_flags,
            "--expt-relaxed-constexpr",
            "-Xcompiler",
            "-fPIC",
            *defs,
            *inc_flags,
            "-c",
            str(src_path),
            "-o",
            str(obj),
        ]
        compile_jobs.append((src_path, obj, cmd))

    for src in cpp_sources:
        src_path = root / src
        obj = build_dir / (src_path.stem + ".cpp.o")
        obj_files.append(str(obj))
        cmd = [
            cxx_bin,
            "-O3",
            "-std=c++17",
            "-fPIC",
            "-pthread",
            *defs,
            *inc_flags,
            "-c",
            str(src_path),
            "-o",
            str(obj),
        ]
        compile_jobs.append((src_path, obj, cmd))

    pending = []
    for src_path, obj, cmd in compile_jobs:
        sig_file = build_dir / f"{obj.name}.cmd"
        cmd_sig = " ".join(cmd)
        up_to_date = (
            obj.exists()
            and sig_file.exists()
            and sig_file.read_text() == cmd_sig
            and obj.stat().st_mtime >= max(src_path.stat().st_mtime, header_mtime)
        )
        if not up_to_date:
            pending.append((obj, sig_file, cmd, cmd_sig))

    if pending:
        print(
            f"[tempo] Compiling {len(pending)}/{len(compile_jobs)} unit(s) "
            f"({' '.join(arch_flags)}, CXX11_ABI={cxx11_abi})..."
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(subprocess.run, cmd, check=True, capture_output=True, text=True) for _, _, cmd, _ in pending]
            for f, (_, sig_file, cmd, cmd_sig) in zip(futs, pending):
                try:
                    f.result()
                    sig_file.write_text(cmd_sig)
                except subprocess.CalledProcessError as e:
                    print(f"FAILED: {' '.join(cmd)}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}", file=sys.stderr)
                    raise

    newest_obj_mtime = max(pathlib.Path(o).stat().st_mtime for o in obj_files)
    if pending or not out_so.exists() or out_so.stat().st_mtime < newest_obj_mtime:
        lib_dirs = ce.library_paths("cuda")
        lib_flags: List[str] = []
        for ld in lib_dirs:
            lib_flags.extend([f"-L{ld}", f"-Wl,-rpath,{ld}"])

        link_cmd = [
            cxx_bin,
            "-shared",
            "-pthread",
            *obj_files,
            *lib_flags,
            "-ltorch",
            "-ltorch_cpu",
            "-ltorch_cuda",
            "-ltorch_python",
            "-lc10",
            "-lc10_cuda",
            "-lcudart",
            "-o",
            str(out_so),
        ]
        res = subprocess.run(link_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Link failed for {out_so}:\n{res.stderr}")
        print(f"[tempo] Built {out_so}")
    else:
        print(f"[tempo] Up to date: {out_so}")


class CUDABuildExt(_build_ext):
    def build_extension(self, ext: Extension) -> None:
        dst = pathlib.Path(self.get_ext_fullpath(ext.name)).resolve()
        _build_cuda_extension(dst)


setup(
    ext_modules=[Extension("tempo._tempo_C", sources=[])],
    cmdclass={"build_ext": CUDABuildExt},
)
