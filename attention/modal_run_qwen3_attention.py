"""Compile and run the Qwen3Attention cases on a Modal GPU."""
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
print(f"Loaded attention launcher from {Path(__file__).resolve()}; project root: {ROOT}", flush=True)
WEEK_03 = ROOT
ATTENTION_DIR = WEEK_03 / "attention"
REMOTE_ROOT = "/root/project"
REMOTE_WEEK_03 = REMOTE_ROOT
REMOTE_ATTENTION_DIR = f"{REMOTE_WEEK_03}/attention"

app = modal.App("qwen3-attention")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.3.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .entrypoint([])
    .pip_install_from_requirements(str(WEEK_03 / "requirements.txt"))
    # Mount the shared schema/models as well as the attention source and case database.
    .add_local_dir(
        str(ROOT),
        remote_path=REMOTE_ROOT,
        ignore=[
            ".git",
            "**/.git/**",
            ".venv",
            "**/.venv/**",
            ".venv312",
            "**/.venv312/**",
            "__pycache__",
            "**/__pycache__/**",
        ],
    )
)


@app.function(image=image, gpu="A100")
def run_qwen3_attention():
    library_path = "/tmp/libqwen3_attention.so"
    database_path = f"{REMOTE_ATTENTION_DIR}/qwen3_attention_cases.db"
    source_path = f"{REMOTE_ATTENTION_DIR}/qwen3_attention.cu"

    print("Compiling Qwen3Attention CUDA Tile implementation...", flush=True)
    started = time.perf_counter()
    subprocess.run(
        [
            "nvcc",
            "--enable-tile",
            "-std=c++20",
            "-arch=sm_80",
            "--shared",
            "-Xcompiler",
            "-fPIC",
            source_path,
            "-o",
            library_path,
        ],
        cwd=REMOTE_WEEK_03,
        check=True,
    )
    print(f"Compilation completed in {time.perf_counter() - started:.1f}s", flush=True)

    subprocess.run(
        [
            sys.executable,
            "-u",
            f"{REMOTE_ATTENTION_DIR}/run_attention.py",
            "--database",
            database_path,
            "--library",
            library_path,
        ],
        cwd=REMOTE_WEEK_03,
        check=True,
    )


@app.local_entrypoint()
def main():
    run_qwen3_attention.remote()
