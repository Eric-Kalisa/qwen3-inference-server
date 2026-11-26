"""Explicit H100 jobs for full-model validation and serial AIPerf comparisons."""
from pathlib import Path
import json
import subprocess
import uuid
import modal
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
app=modal.App("qwen3-phase2")
models=modal.Volume.from_name("qwen3-week-7-8-models",create_if_missing=True)
RESULTS_VOLUME_NAME="qwen3-week-7-8-results"
results=modal.Volume.from_name(RESULTS_VOLUME_NAME,create_if_missing=True)
base=(modal.Image.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04",add_python="3.12")
      .entrypoint([]).apt_install("cmake","g++","make","python3-venv")
      .pip_install("numpy==2.2.6","huggingface-hub==0.36.0"))
# Separate environments keep the custom CUDA server free of HF/PyTorch runtime
# dependencies while allowing vLLM and AIPerf to use their own dependency trees.
bench_base=base.run_commands(
    "python -m venv /opt/vllm",
    "/opt/vllm/bin/python -m pip install vllm==0.11.0 transformers==4.57.3 tokenizers==0.22.1 huggingface-hub==0.36.0",
    "/opt/vllm/bin/python -m pip check",
    "/opt/vllm/bin/python -c \"from transformers import PreTrainedTokenizerBase; assert hasattr(PreTrainedTokenizerBase, 'all_special_tokens_extended'), 'vLLM tokenizer API is incompatible'\"",
    "python -m venv /opt/aiperf",
    "/opt/aiperf/bin/python -m pip install aiperf==0.13.0")
ignore=["**/__pycache__/**","**/.git/**","**/build/**","**/results/**","**/.vscode/**",
        "**/*.safetensors","**/*.db","**/*.log"]
def sources(image):
    return (image.add_local_dir(str(ROOT/"inference"),"/project/inference",ignore=ignore)
            .add_local_dir(str(ROOT/"benchmarks"),"/project/benchmarks",ignore=ignore))
validation_image=sources(base)
benchmark_image=sources(bench_base)


def checkpoint(model,revision):
    from huggingface_hub import HfApi,snapshot_download
    if model not in ("Qwen/Qwen3-0.6B","Qwen/Qwen3-4B"):
        raise ValueError("Only Qwen3-0.6B/4B are configured")
    sha=HfApi().model_info(model,revision=revision).sha
    path=snapshot_download(model,revision=sha,local_dir=f"/models/phase2/{model.split('/')[-1]}/{sha}",
                           allow_patterns=["*.json","*.safetensors"])
    models.commit()
    return path,sha


def build():
    subprocess.run(["cmake","-S","/project/inference","-B","/tmp/build",
                    "-DCMAKE_BUILD_TYPE=Release","-DCMAKE_CUDA_ARCHITECTURES=90"],check=True)
    subprocess.run(["cmake","--build","/tmp/build","--target","qwen3_runtime","qwen3_tokenizer",
                    "--parallel","4"],check=True)


@app.function(image=validation_image,gpu="H100",volumes={"/models":models,"/results":results},timeout=3600)
def validate(model: str="Qwen/Qwen3-0.6B",revision: str="main"):
    build(); path,sha=checkpoint(model,revision)
    out=Path(f"/results/validation-{model.split('/')[-1]}-{uuid.uuid4().hex[:8]}.json")
    try:
        subprocess.run(["python","/project/inference/validate_model.py","--model-dir",path,
                        "--library-dir","/tmp/build","--output",str(out)],check=True)
        data=json.loads(out.read_text()); data.update(model=model,revision=sha)
        out.write_text(json.dumps(data,indent=2))
        return dict(path=str(out),result=data)
    except Exception as exc:
        out.write_text(json.dumps(dict(passed=False,model=model,revision=sha,error=str(exc)),indent=2))
        raise
    finally: results.commit()


def run_checkpoint_tests(revision, test_file):
    import os

    path, sha = checkpoint("Qwen/Qwen3-0.6B", revision)
    print(f"Qwen/Qwen3-0.6B revision: {sha}", flush=True)
    build()
    env = dict(os.environ, QWEN_KV_MODEL_DIR=path,
               QWEN_CUDA_LIBRARY="/tmp/build/libqwen3_runtime.so")
    subprocess.run(
        ["python", "-m", "unittest", "discover", "-s", "tests", "-p", test_file, "-v"],
        cwd="/project/inference", env=env, check=True,
    )


@app.function(image=validation_image, gpu="H100", volumes={"/models": models}, timeout=1800)
def check_checkpoint_kv(revision: str = "main"):
    run_checkpoint_tests(revision, "test_checkpoint_kv.py")


@app.function(image=validation_image, gpu="H100", volumes={"/models": models}, timeout=1800)
def check_paged_attention(revision: str = "main"):
    run_checkpoint_tests(revision, "test_paged_attention.py")


@app.function(image=validation_image, gpu="H100", volumes={"/models": models}, timeout=1800)
def check_mixed_batch(revision: str = "main"):
    run_checkpoint_tests(revision, "test_mixed_batch.py")


@app.function(image=validation_image, gpu="H100", volumes={"/models": models}, timeout=1800)
def check_kv_cancellation(revision: str = "main"):
    run_checkpoint_tests(revision, "test_kv_cancellation.py")


@app.function(image=validation_image, gpu="H100", volumes={"/models": models}, timeout=1800)
def check_kv_pressure(revision: str = "main"):
    run_checkpoint_tests(revision, "test_kv_pressure.py")


@app.function(image=benchmark_image,gpu="H100",volumes={"/models":models,"/results":results},timeout=86400)
def benchmark(model: str="Qwen/Qwen3-4B",revision: str="main",preset: str="smoke",
              variants: str="uncached,contiguous,paged,paged_batch,vllm",
              requests: int=0,warmup: int=-1,repeats: int=0,case_timeout: int=0):
    # Modal places this entry-point module under /root, while the shared source
    # directory is mounted at /project. Resolve the plan only after mounting.
    sys.path.insert(0,"/project")
    from benchmarks.plan import settings
    effective=settings(preset,requests,warmup,repeats,case_timeout)
    build(); path,sha=checkpoint(model,revision)
    out=Path(f"/results/{model.split('/')[-1]}-{preset}-{uuid.uuid4().hex[:8]}")
    # A failed full-checkpoint gate prevents any benchmark numbers being reported.
    gate=Path("/tmp/model-validation.json")
    subprocess.run(["python","/project/inference/validate_model.py","--model-dir",path,
                    "--library-dir","/tmp/build","--output",str(gate)],check=True)
    cmd=["python","/project/benchmarks/run_suite.py","--model-dir",path,"--model",model,
         "--revision",sha,"--library-dir","/tmp/build","--output",str(out),"--preset",preset,
         "--variants",*variants.split(",")]
    for key,value in effective.items(): cmd += ["--"+key.replace("_","-"),str(value)]
    try:
        subprocess.run(cmd,check=True)
        summary=json.loads((out/"summary.json").read_text())
        print(f"Benchmark completed. Saved results: {out}",flush=True)
        return dict(path=str(out),summary=summary)
    finally:
        if out.exists(): (out/"checkpoint-validation.json").write_bytes(gate.read_bytes())
        results.commit()
