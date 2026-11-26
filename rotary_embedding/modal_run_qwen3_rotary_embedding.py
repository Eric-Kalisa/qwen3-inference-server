"""Compile and run Qwen3RotaryEmbedding reference cases on a Modal GPU."""
from pathlib import Path
import subprocess
import sys

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
MODULE_DIR = WEEK_03 / "rotary_embedding"
REMOTE_ROOT = "/root/project"
REMOTE_WEEK_03 = REMOTE_ROOT
REMOTE_MODULE_DIR = f"{REMOTE_WEEK_03}/rotary_embedding"

app = modal.App("qwen3-rotary-embedding")
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
def run_qwen3_rotary_embedding():
    library_path = "/tmp/libqwen3_rotary_embedding.so"
    subprocess.run(
        [
            "nvcc", "--enable-tile", "-std=c++20", "-arch=sm_80",
            "--shared", "-Xcompiler", "-fPIC",
            f"{REMOTE_MODULE_DIR}/qwen3_rotary_embedding.cu", "-o", library_path,
        ],
        cwd=REMOTE_WEEK_03,
        check=True,
    )
    subprocess.run(
        [
            sys.executable, "-u", f"{REMOTE_MODULE_DIR}/run_qwen3_rotary_embedding.py",
            "--database", f"{REMOTE_MODULE_DIR}/qwen3_rotary_embedding_cases.db",
            "--library", library_path,
        ],
        cwd=REMOTE_WEEK_03,
        check=True,
    )


@app.local_entrypoint()
def main():
    run_qwen3_rotary_embedding.remote()
