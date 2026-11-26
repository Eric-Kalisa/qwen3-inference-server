"""Run shared-prefix smoke tests independently and compare saved results."""
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

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
THROUGHPUT = "output_token_throughput.avg"
CONCURRENCIES = (1, 4)
LOCAL_RESULTS = Path(__file__).resolve().parent.parent / "results"

app = modal.App("qwen3-shared-prefix-smoke")
image = benchmark_image.add_local_python_source("benchmarks")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_settings(run_dir):
    manifest = read_json(run_dir / "manifest.json")
    gpu, _, driver, memory = (value.strip() for value in manifest["gpu"].split(","))
    args = manifest["settings"]
    return {
        "model": manifest["model"],
        "revision": manifest["revision"],
        "gpu": [gpu, memory, driver],
        "settings": {key: args[key] for key in
                     ("preset", "blocks", "max_context", "requests", "warmup", "repeats")},
        "workload_sources": {key: manifest["source_sha256"][key] for key in
                             ("benchmarks/plan.py", "benchmarks/profile.py", "benchmarks/workloads.json")},
        "client_packages": (run_dir / "client-packages.txt").read_text().splitlines(),
    }


def throughput_by_concurrency(summary, variant, requests):
    rates = {}
    for row in summary["runs"]:
        rate = row.get(THROUGHPUT)
        if (row["variant"] == variant and row["workload"] == "shared_prefix"
                and row["status"] == "exported" and row.get("request_count.avg") == requests
                and isinstance(rate, (int, float)) and math.isfinite(rate) and rate > 0):
            rates[row["concurrency"]] = rate
    return rates


def compare(run_dir, variant, summary):
    settings = summary["comparison_settings"]
    requests = settings["settings"]["requests"]
    own_rates = throughput_by_concurrency(summary, variant, requests)
    peer = "vllm_shared" if variant == "paged_shared" else "paged_shared"
    peer_root = run_dir.parent.parent / peer
    for path in sorted(peer_root.glob("*/summary.json"), reverse=True):
        saved = read_json(path)
        if saved.get("comparison_settings") != settings:
            continue
        peer_rates = throughput_by_concurrency(saved, peer, requests)
        if not all(c in peer_rates for c in CONCURRENCIES):
            continue
        custom, vllm = (own_rates, peer_rates) if variant == "paged_shared" else (peer_rates, own_rates)
        custom_summary, vllm_summary = (summary, saved) if variant == "paged_shared" else (saved, summary)
        ttft = ttft_comparison(custom_summary, vllm_summary, requests)
        return {
            "paired_run": str(path.parent),
            "cases": [
                {"concurrency": c, "paged_shared_tokens_per_second": custom[c],
                 "vllm_shared_tokens_per_second": vllm[c],
                 "vllm_to_paged_shared_x": vllm[c] / custom[c],
                 "percent_of_vllm": 100 * custom[c] / vllm[c],
                 "ttft": ttft.get(c, {})}
                for c in CONCURRENCIES if c in own_rates
            ],
        }
    return {"paired_run": None, "cases": []}


def ttft_comparison(custom, vllm, requests):
    def selected(summary, variant):
        return {row["concurrency"]: row for row in summary["runs"]
                if row["variant"] == variant and row["workload"] == "shared_prefix"
                and row["status"] == "exported" and row.get("request_count.avg") == requests}

    peers = selected(vllm, "vllm_shared")
    comparisons = {}
    for c, row in selected(custom, "paged_shared").items():
        baseline = peers.get(c, {})
        unit = row.get("time_to_first_token.unit")
        if unit and unit == baseline.get("time_to_first_token.unit"):
            metrics = {}
            for percentile in ("p50", "p95", "p99"):
                key = f"time_to_first_token.{percentile}"
                left, right = row.get(key), baseline.get(key)
                if all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                       for value in (left, right)):
                    metrics[percentile] = {"custom": left, "vllm": right, "unit": unit,
                                           "custom_divided_by_vllm": left / right}
            comparisons[c] = metrics
    return comparisons


@app.function(image=image, gpu="H100!", volumes={"/models": models, "/results": results}, timeout=7200)
def run_smoke(variant: str, revision: str):
    if variant not in ("paged_shared", "vllm_shared"):
        raise ValueError(f"Unknown shared-prefix variant: {variant}")

    model_dir, sha = checkpoint(MODEL, revision)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = Path("/results/shared-smoke") / MODEL.split("/")[-1] / sha / variant / f"{stamp}-{uuid.uuid4().hex[:8]}"
    gate = Path("/tmp/shared-smoke-validation.json")
    print(f"Saving {variant} artifacts to {out}", flush=True)
    try:
        if variant == "paged_shared":
            build()
            subprocess.run(
                ["python", "/project/inference/validate_model.py", "--model-dir", model_dir,
                 "--library-dir", "/tmp/build", "--output", str(gate)],
                check=True,
            )

        subprocess.run(
            ["python", "/project/benchmarks/run_suite.py", "--model-dir", model_dir,
             "--model", MODEL, "--revision", sha, "--library-dir", "/tmp/build",
             "--output", str(out), "--preset", "smoke", "--variants", variant],
            check=True,
        )
        path = out / "summary.json"
        summary = read_json(path)
        summary["comparison_settings"] = comparison_settings(out)
        summary["vllm_throughput_comparison"] = compare(out, variant, summary)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        temporary.replace(path)
        return {"variant": variant, "path": str(out), "summary": summary}
    finally:
        if out.exists():
            if variant == "paged_shared":
                (out / "checkpoint-validation.json").write_bytes(gate.read_bytes())
            results.commit()


def show_summary(result):
    print(f"Modal volume results: {result['path']}")
    print(f"Local results: {result['local_path']}")
    summary = result["summary"]
    for row in summary["runs"]:
        print(f"Concurrency {row['concurrency']} | {row['status']} | "
              f"{row.get(THROUGHPUT)} output tok/s | "
              f"TTFT {row.get('time_to_first_token.avg')} {row.get('time_to_first_token.unit')} | "
              f"ITL {row.get('inter_token_latency.avg')} {row.get('inter_token_latency.unit')}")
        if row.get("artifact") and result["variant"] == "paged_shared":
            breakdown = Path(result["local_path"]) / row["artifact"] / "latency-breakdown.json"
            if breakdown.is_file():
                print(f"  TTFT breakdown: {breakdown}")

    comparison = summary["vllm_throughput_comparison"]
    if not comparison["paired_run"]:
        print("vLLM comparison: unavailable until both implementations have a matching smoke run.")
        return
    print(f"Paired run: {comparison['paired_run']}")
    for row in comparison["cases"]:
        print(f"Concurrency {row['concurrency']} | vLLM / custom = {row['vllm_to_paged_shared_x']:.2f}x "
              f"| custom delivers {row['percent_of_vllm']:.1f}% of vLLM throughput")
        for percentile, metric in row.get("ttft", {}).items():
            print(f"  {percentile} TTFT custom / vLLM: {metric['custom_divided_by_vllm']:.2f}x")


def download_run(remote, parent):
    remote_path = PurePosixPath(remote)
    try:
        relative_path = remote_path.relative_to("/results")
    except ValueError as exc:
        raise RuntimeError(f"Unexpected Modal results path: {remote_path}") from exc
    if not relative_path.parts:
        raise RuntimeError(f"Expected a run directory below /results, got: {remote_path}")

    parent.mkdir(parents=True, exist_ok=True)
    local_path = parent / relative_path.name
    if local_path.exists():
        raise FileExistsError(f"Result destination already exists: {local_path}")
    # Modal adds the remote directory basename to an existing local directory.
    subprocess.run(
        [sys.executable, "-m", "modal", "volume", "get", RESULTS_VOLUME_NAME,
         f"/{relative_path.as_posix()}", str(parent)],
        check=True,
    )
    if not local_path.is_dir():
        raise RuntimeError(f"Modal did not create the expected download directory: {local_path}")
    return local_path


def run_smoke_and_download(variant, revision):
    result = run_smoke.remote(variant, revision)
    result["local_path"] = str(download_run(result["path"], LOCAL_RESULTS / "shared-smoke" / variant))
    show_summary(result)
