"""Compile the reused Qwen3 components and run decoder-layer cases on Modal."""
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
REMOTE_ROOT = "/root/project"
REMOTE_WEEK_03 = REMOTE_ROOT
REMOTE_DECODER_DIR = f"{REMOTE_WEEK_03}/decoder_layer"

app = modal.App("qwen3-decoder-layer")
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
def run_qwen3_decoder_layer():
    # Keep each component library separate; the runner composes their host C ABIs.
    sources = [
        ("rms_norm/qwen3_rms_norm.cu", "libqwen3_rms_norm.so"),
        ("rotary_embedding/qwen3_rotary_embedding.cu", "libqwen3_rotary_embedding.so"),
        ("attention/qwen3_attention.cu", "libqwen3_attention.so"),
        ("mlp/qwen3_mlp.cu", "libqwen3_mlp.so"),
        ("decoder_layer/qwen3_decoder_layer.cu", "libqwen3_decoder_layer.so"),
    ]
    for source, library_name in sources:
        subprocess.run(
            [
                "nvcc", "--enable-tile", "-std=c++20", "-arch=sm_80",
                "--shared", "-Xcompiler", "-fPIC",
                f"{REMOTE_WEEK_03}/{source}", "-o", f"/tmp/{library_name}",
            ],
            cwd=REMOTE_WEEK_03,
            check=True,
        )

    subprocess.run(
        [
            sys.executable, "-u", f"{REMOTE_DECODER_DIR}/run_qwen3_decoder_layer.py",
            "--database", f"{REMOTE_DECODER_DIR}/qwen3_decoder_layer_cases.db",
            "--library-dir", "/tmp",
        ],
        cwd=REMOTE_WEEK_03,
        check=True,
    )


@app.local_entrypoint()
def main():
    run_qwen3_decoder_layer.remote()
