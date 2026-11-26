"""Compile and run the Qwen3MLP cases on a Modal GPU."""
from pathlib import Path
import subprocess
import sys
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
WEEK_03 = ROOT
MLP_DIR = WEEK_03 / "mlp"
REMOTE_ROOT = "/root/project"
REMOTE_WEEK_03 = REMOTE_ROOT
REMOTE_MLP_DIR = f"{REMOTE_WEEK_03}/mlp"

app = modal.App("qwen3-mlp")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.3.0-devel-ubuntu24.04",
        add_python="3.12",
    )
    .entrypoint([])
    .pip_install_from_requirements(str(WEEK_03 / "requirements.txt"))
    # Mount the project root so the runner can reuse schema.sql and sql_models.py.
    .add_local_dir(
        str(ROOT),
        remote_path=REMOTE_ROOT,
        ignore=[
            ".git",
            "**/.git/**",
            ".venv",
            "**/.venv/**",
            "__pycache__",
            "**/__pycache__/**",
        ],
    )
)


@app.function(image=image, gpu="A100")
def run_qwen3_mlp():
    library_path = "/tmp/libqwen3_mlp.so"
    database_path = f"{REMOTE_MLP_DIR}/qwen3_mlp_cases.db"
    source_path = f"{REMOTE_MLP_DIR}/qwen3_mlp.cu"

    print("Compiling Qwen3MLP CUDA Tile implementation...", flush=True)
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

    print("Running SQLite reference cases...", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-u",
            f"{REMOTE_MLP_DIR}/run_mlp.py",
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
    run_qwen3_mlp.remote()
