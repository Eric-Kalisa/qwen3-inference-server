"""H100 build/test/serve entry points. Model download is separate from execution."""
from pathlib import Path
import subprocess
import modal

ROOT = Path(__file__).resolve().parent.parent
app = modal.App("qwen3-stage-2")
volume = modal.Volume.from_name("qwen3-week-7-8-models", create_if_missing=True)
image = (modal.Image.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", add_python="3.12")
         .entrypoint([]).apt_install("cmake", "g++", "make")
         .pip_install_from_requirements(str(ROOT/"inference/requirements.txt"))
         .add_local_dir(str(ROOT), "/project", ignore=["**/__pycache__/**", "**/.git/**", "**/build/**"]))


def build():
    subprocess.run(["cmake", "-S", "/project/inference", "-B", "/tmp/build",
                    "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=90"], check=True)
    subprocess.run(["cmake", "--build", "/tmp/build", "--target", "qwen3_runtime",
                    "qwen3_tokenizer", "--parallel", "4"], check=True)


@app.function(image=image, gpu="H100", timeout=1800)
def check(sanitize: bool = False):
    build()
    import os
    env = dict(os.environ, QWEN_CUDA_LIBRARY="/tmp/build/libqwen3_runtime.so")
    subprocess.run(["python", "-m", "unittest", "discover", "-s", "tests", "-v"],
                   cwd="/project/inference", env=env, check=True)
    if sanitize:
        for tool in ("memcheck", "racecheck", "synccheck", "initcheck"):
            subprocess.run(["compute-sanitizer", "--tool", tool, "--error-exitcode", "99",
                            "python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_cuda.py"],
                           cwd="/project/inference", env=env, check=True)


@app.function(image=image, volumes={"/models": volume}, timeout=1800)
def download(model: str = "Qwen/Qwen3-0.6B", revision: str = "main"):
    from huggingface_hub import snapshot_download
    path = snapshot_download(model, revision=revision, local_dir=f"/models/{model.split('/')[-1]}",
                             allow_patterns=["*.json", "*.safetensors"])
    volume.commit()
    return path


def start_server(model, mode="paged"):
    build()
    subprocess.Popen(["python", "/project/inference/server.py", "--model-dir",
                      f"/models/{model.split('/')[-1]}", "--model", model,
                      "--library-dir", "/tmp/build", "--host", "0.0.0.0", "--mode", mode])


@app.function(image=image, gpu="H100", volumes={"/models": volume}, timeout=3600)
@modal.web_server(8000, startup_timeout=900)
def serve():
    start_server("Qwen/Qwen3-0.6B")


@app.function(image=image, gpu="H100", volumes={"/models": volume}, timeout=3600)
@modal.web_server(8000, startup_timeout=900)
def serve_4b():
    start_server("Qwen/Qwen3-4B")
