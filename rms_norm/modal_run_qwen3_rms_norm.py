"""Compile and run Qwen3RMSNorm reference cases on a Modal GPU."""
from pathlib import Path
import subprocess
import sys
import time

import modal


def find_project_root():
    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / "requirements.txt").is_file():
            return candidate

    modal_project_root = Path("/root/project")
    if (modal_project_root / "requirements.txt").is_file():
        return modal_project_root

    raise RuntimeError(
        "Could not locate the project root containing requirements.txt"
    )


ROOT = find_project_root()
WEEK_03 = ROOT
MODULE_DIR = WEEK_03 / "rms_norm"
REMOTE_ROOT = "/root/project"
REMOTE_WEEK_03 = REMOTE_ROOT
REMOTE_MODULE_DIR = f"{REMOTE_WEEK_03}/rms_norm"

app = modal.App("qwen3-rms-norm")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.3.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .entrypoint([])
    .pip_install_from_requirements(str(WEEK_03 / "requirements.txt"))
    .add_local_dir(
        str(ROOT),
        remote_path=REMOTE_ROOT,
        ignore=[
            ".git", "**/.git/**", ".venv", "**/.venv/**",
            ".venv312", "**/.venv312/**", "__pycache__", "**/__pycache__/**",
        ],
    )
)


@app.function(image=image, gpu="A100")
def run_qwen3_rms_norm():
    library_path = "/tmp/libqwen3_rms_norm.so"
    subprocess.run(
        [
            "nvcc", "--enable-tile", "-std=c++20", "-arch=sm_80",
            "--shared", "-Xcompiler", "-fPIC",
            f"{REMOTE_MODULE_DIR}/qwen3_rms_norm.cu", "-o", library_path,
        ],
        cwd=REMOTE_WEEK_03,
        check=True,
    )
    started = time.perf_counter()
    subprocess.run(
        [
            sys.executable, "-u", f"{REMOTE_MODULE_DIR}/run_qwen3_rms_norm.py",
            "--database", f"{REMOTE_MODULE_DIR}/qwen3_rms_norm_cases.db",
            "--library", library_path,
        ],
        cwd=REMOTE_WEEK_03,
        check=True,
    )
    print(f"Qwen3RMSNorm cases passed in {time.perf_counter() - started:.1f}s", flush=True)


@app.local_entrypoint()
def main():
    run_qwen3_rms_norm.remote()
