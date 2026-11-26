"""Compare the custom CUDA server and vLLM on the shared-prefix workload.

Run from stage_2: python -m modal run benchmarks\\modal_shared_benchmark.py --size compact
Use --size full or --size both for the larger run or both sizes on one H100.
"""
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
import uuid

import modal

from benchmarks.modal_phase2 import (
    RESULTS_VOLUME_NAME, benchmark_image, build, checkpoint, models, results,
)
from benchmarks.plan import DEFAULTS
from benchmarks.shared_smoke import CONCURRENCIES, MODEL, REVISION

app = modal.App("qwen3-shared-prefix-benchmark")
image = benchmark_image.add_local_python_source("benchmarks")
VARIANTS = ("paged_shared", "vllm_shared")
SIZES = ("compact", "full")
METRICS = {
    "output_token_throughput.avg": "Output throughput",
    "time_to_first_token.avg": "Mean TTFT",
    "inter_token_latency.avg": "Mean ITL",
}


def compare(summary, requests):
    rows = {
        (row["variant"], row["concurrency"]): row
        for row in summary["runs"]
        if row["status"] == "exported"
        and row["workload"] == "shared_prefix"
        and row["request_count.avg"] == requests
    }
    pairs = []
    for concurrency in CONCURRENCIES:
        custom, vllm = (rows.get((variant, concurrency)) for variant in VARIANTS)
        metrics = {}
        if custom and vllm:
            for key, name in METRICS.items():
                left, right = custom.get(key), vllm.get(key)
                unit = custom.get(key.removesuffix(".avg") + ".unit")
                if (unit and unit == vllm.get(key.removesuffix(".avg") + ".unit")
                        and all(isinstance(value, (int, float)) and math.isfinite(value)
                                and value > 0 for value in (left, right))):
                    metrics[name] = {
                        "custom": left, "vllm": right, "unit": unit,
                        "custom_divided_by_vllm": left / right,
                    }
        pairs.append({"concurrency": concurrency, "metrics": metrics})
    return pairs


@app.function(image=image, gpu="H100!", volumes={"/models": models, "/results": results},
              timeout=14400)
def run_benchmark(size: str, revision: str):
    sizes = SIZES if size == "both" else (size,)
    model_dir, sha = checkpoint(MODEL, revision)
    build()
    gate = Path("/tmp/shared-benchmark-validation.json")
    subprocess.run(
        ["python", "/project/inference/validate_model.py", "--model-dir", model_dir,
         "--library-dir", "/tmp/build", "--output", str(gate)],
        check=True,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path("/results/shared-benchmark") / MODEL.split("/")[-1] / sha
    root = root / f"{stamp}-{uuid.uuid4().hex[:8]}"
    reports = []
    for name in sizes:
        out = root / name
        requests = DEFAULTS[name]["requests"]
        # Keep the smoke workload matrix, warmup, repeat count, and timeout.
        completed = subprocess.run(
            ["python", "/project/benchmarks/run_suite.py", "--model-dir", model_dir,
             "--model", MODEL, "--revision", sha, "--library-dir", "/tmp/build",
             "--output", str(out), "--preset", "smoke", "--requests", str(requests),
             "--variants", *VARIANTS],
        )
        pairs = []
        if out.exists():
            (out / "checkpoint-validation.json").write_bytes(gate.read_bytes())
            summary_path = out / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                pairs = compare(summary, requests)
            report = {"size": name, "requests_per_case": requests, "cases": pairs}
            (out / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            results.commit()
        else:
            completed.check_returncode()
            raise RuntimeError(f"Benchmark produced no result directory: {out}")
        complete = (completed.returncode == 0 and len(pairs) == len(CONCURRENCIES)
                    and all(len(pair["metrics"]) == len(METRICS) for pair in pairs))
        reports.append({"path": str(out), "complete": complete, **report})
    return reports


@app.local_entrypoint()
def main(size: str = "compact", revision: str = REVISION):
    if size not in (*SIZES, "both"):
        raise ValueError("Size must be compact, full, or both")
    reports = run_benchmark.remote(size, revision)
    for report in reports:
        relative = PurePosixPath(report["path"]).relative_to("/results")
        local = Path(__file__).resolve().parent.joinpath("results", *relative.parts)
        local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "modal", "volume", "get", RESULTS_VOLUME_NAME,
             "/" + relative.as_posix(), str(local)],
            check=True,
        )
        print(f"{report['size']} ({report['requests_per_case']} measured requests per case)")
        print(f"Modal results: {report['path']}")
        print(f"Local results: {local}")
        for pair in report["cases"]:
            c = pair["concurrency"]
            for name, values in pair["metrics"].items():
                print(f"C{c} | {name} | custom {values['custom']:.2f} {values['unit']} | "
                      f"vLLM {values['vllm']:.2f} {values['unit']} | "
                      f"custom / vLLM {values['custom_divided_by_vllm']:.2f}x")
    if any(not report["complete"] for report in reports):
        raise RuntimeError("One or more comparisons were incomplete. See the downloaded run logs.")
