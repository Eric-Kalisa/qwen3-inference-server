"""Build and serve the minimal Qwen3 CUDA Tile inference path on Modal."""

from pathlib import Path
import os
import subprocess
import sys

import modal


def find_project_root():
    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / "inference" / "requirements.txt").is_file():
            return candidate

    modal_project_root = Path("/root/project")
    if (modal_project_root / "inference" / "requirements.txt").is_file():
        return modal_project_root

    raise RuntimeError("Could not locate requirements.txt")


ROOT = find_project_root()
WEEK_03 = ROOT
INFERENCE_DIR = WEEK_03 / "inference"
REMOTE_ROOT = "/root/project"
REMOTE_WEEK_03 = REMOTE_ROOT
REMOTE_INFERENCE_DIR = f"{REMOTE_WEEK_03}/inference"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
QWEN3_4B_MODEL = "Qwen/Qwen3-4B"

app = modal.App("qwen3-inference-server")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.3.0-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("cmake", "g++", "make")
    .pip_install_from_requirements(str(INFERENCE_DIR / "requirements.txt"))
    .add_local_dir(
        str(ROOT),
        remote_path=REMOTE_ROOT,
        ignore=[
            ".git", "**/.git/**", ".venv", "**/.venv/**",
            ".venv312", "**/.venv312/**", "__pycache__", "**/__pycache__/**",
            "_repo", "_repo/**", ".cuda-tile-verify", ".cuda-tile-verify/**",
        ],
    )
)


def build_libraries():
    subprocess.run(
        [
            "cmake", "-S", str(REMOTE_INFERENCE_DIR),
            "-B", "/tmp/qwen3-tokenizer-build", "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake", "--build", "/tmp/qwen3-tokenizer-build",
            "--target", "qwen3_tokenizer", "--config", "Release", "--parallel",
        ],
        check=True,
    )

    sources = [
        ("rms_norm/qwen3_rms_norm.cu", "libqwen3_rms_norm.so"),
        ("rotary_embedding/qwen3_rotary_embedding.cu", "libqwen3_rotary_embedding.so"),
        ("attention/qwen3_attention.cu", "libqwen3_attention.so"),
        ("mlp/qwen3_mlp.cu", "libqwen3_mlp.so"),
        ("decoder_layer/qwen3_decoder_layer.cu", "libqwen3_decoder_layer.so"),
        ("inference/qwen3_inference.cu", "libqwen3_inference.so"),
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

    tokenizer_library = Path("/tmp/qwen3-tokenizer-build/libqwen3_tokenizer.so")
    if not tokenizer_library.exists():
        tokenizer_library = Path(
            "/tmp/qwen3-tokenizer-build/Release/libqwen3_tokenizer.so"
        )
    Path("/tmp/libqwen3_tokenizer.so").write_bytes(tokenizer_library.read_bytes())


@app.function(image=image, timeout=3600)
def test_qwen3_build_load():
    build_libraries()
    environment = os.environ.copy()
    environment["QWEN3_LIBRARY_DIR"] = "/tmp"
    subprocess.run(
        [sys.executable, "-m", "unittest", "test_build_load"],
        cwd=REMOTE_INFERENCE_DIR,
        env=environment,
        check=True,
    )


@app.function(image=image, gpu="A100", memory=24576, timeout=3600)
def run_qwen3_tests(model_id: str = DEFAULT_MODEL):
    build_libraries()
    environment = os.environ.copy()
    environment["QWEN3_LIBRARY_DIR"] = "/tmp"
    environment["QWEN3_MODEL_ID"] = model_id
    subprocess.run(
        [sys.executable, "-m", "unittest", "test_inference_server"],
        cwd=REMOTE_INFERENCE_DIR,
        env=environment,
        check=True,
    )


@app.function(image=image, gpu="A100", memory=24576, timeout=3600)
def compare_qwen3(model_id: str, prompt: str, max_new_tokens: int):
    build_libraries()
    subprocess.run(
        [
            sys.executable, "-u", f"{REMOTE_INFERENCE_DIR}/inference_server.py",
            "--library-directory", "/tmp",
            "--model", model_id,
            "--compare-prompt", prompt,
            "--max-new-tokens", str(max_new_tokens),
        ],
        cwd=REMOTE_WEEK_03,
        check=True,
    )


@app.function(image=image, gpu="A100", timeout=3600)
def check_qwen3_hidden_state_positions(model_id: str, prompt: str):
    build_libraries()
    subprocess.run(
        [
            sys.executable, "-u", f"{REMOTE_INFERENCE_DIR}/inference_server.py",
            "--library-directory", "/tmp",
            "--model", model_id,
            "--check-hidden-state-positions", prompt,
        ],
        cwd=REMOTE_WEEK_03,
        check=True,
    )


@app.function(image=image, gpu="A100", timeout=3600)
def diagnose_qwen3_layer_hidden_states(model_id: str, prompt: str):
    build_libraries()
    subprocess.run(
        [
            sys.executable, "-u", f"{REMOTE_INFERENCE_DIR}/inference_server.py",
            "--library-directory", "/tmp",
            "--model", model_id,
            "--diagnose-layer-hidden-states", prompt,
        ],
        cwd=REMOTE_WEEK_03,
        check=True,
    )


def start_qwen3_server(model_id: str):
    build_libraries()
    subprocess.Popen(
        [
            sys.executable, "-u", f"{REMOTE_INFERENCE_DIR}/inference_server.py",
            "--host", "0.0.0.0", "--port", "8000",
            "--library-directory", "/tmp",
            "--model", model_id,
        ],
        cwd=REMOTE_WEEK_03,
    )


@app.function(image=image, gpu="A100", timeout=3600)
@modal.web_server(port=8000, startup_timeout=900)
def serve_qwen3():
    start_qwen3_server(DEFAULT_MODEL)


@app.function(image=image, gpu="A100", timeout=3600)
@modal.web_server(port=8000, startup_timeout=900)
def serve_qwen3_4b():
    start_qwen3_server(QWEN3_4B_MODEL)


@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_MODEL,
    prompt: str = "The capital of France is",
    max_new_tokens: int = 3,
    build_only: bool = False,
    check_hidden_state_positions: bool = False,
    diagnose_layer_hidden_states: bool = False,
    run_tests: bool = False,
):
    if run_tests:
        run_qwen3_tests.remote(model_id)
        return
    if build_only:
        test_qwen3_build_load.remote()
        return
    if check_hidden_state_positions:
        check_qwen3_hidden_state_positions.remote(model_id, prompt)
        return
    if diagnose_layer_hidden_states:
        diagnose_qwen3_layer_hidden_states.remote(model_id, prompt)
        return
    compare_qwen3.remote(model_id, prompt, max_new_tokens)
