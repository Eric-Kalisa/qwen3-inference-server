"""Generate/replay AIPerf commands. Dry-run by default; one endpoint per run.

Use the same workload, seed, offered rate and token policy for CUDA and vLLM.
Do not run both servers on the benchmark GPU at the same time.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys

WORKLOADS = json.loads((Path(__file__).parent/"workloads.json").read_text())


def command(args, workload, repeat):
    spec = WORKLOADS[workload]
    cmd = ["aiperf", "profile", "--model", args.model, "--tokenizer", args.tokenizer or args.model,
           "--url", args.url, "--endpoint-type", "completions", "--endpoint", "/v1/completions",
           "--streaming", "--random-seed", "42", "--request-count", str(args.requests),
           "--warmup-request-count", str(args.warmup),
           "--output-artifact-dir", str(args.output/args.label/workload/f"repeat-{repeat}"),
           "--extra-inputs", "temperature:0.0", "--extra-inputs", "ignore_eos:true",
           "--extra-inputs", "skip_special_tokens:false"]
    if "distribution" in spec:
        cmd += ["--sequence-distribution", spec["distribution"]]
    else:
        cmd += ["--synthetic-input-tokens-mean", str(spec["input"]),
                "--synthetic-input-tokens-stddev", "0", "--output-tokens-mean", str(spec["output"]),
                "--output-tokens-stddev", "0"]
    if "prefix" in spec:
        cmd += ["--prefix-prompt-length", str(spec["prefix"]),
                "--prefix-prompt-pool-size", str(spec["pool"])]
    if args.input_file:
        # Real/custom prompt replay replaces synthetic input generation.
        for option in ("--synthetic-input-tokens-mean", "--synthetic-input-tokens-stddev",
                       "--sequence-distribution", "--prefix-prompt-length", "--prefix-prompt-pool-size"):
            if option in cmd:
                i=cmd.index(option); del cmd[i:i+2]
        cmd += ["--input-file", str(args.input_file), "--custom-dataset-type", "single_turn"]
    if args.rate is not None:
        cmd += ["--request-rate", str(args.rate), "--arrival-pattern", args.arrival]
        if args.arrival == "gamma":
            cmd += ["--arrival-smoothness", "0.3"]
    elif args.concurrency is not None:
        cmd += ["--concurrency", str(args.concurrency)]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--label", required=True, help="Unique name, e.g. paged-batched-c8")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--tokenizer")
    p.add_argument("--workload", choices=[*WORKLOADS,"all"], default="short")
    load=p.add_mutually_exclusive_group()
    load.add_argument("--concurrency", type=int)
    load.add_argument("--rate", type=float)
    p.add_argument("--arrival", choices=["constant","poisson","gamma"], default="poisson")
    p.add_argument("--requests", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--input-file", type=Path)
    p.add_argument("--run", action="store_true")
    args=p.parse_args()
    if args.requests<1 or args.repeats<1 or args.warmup<0:
        p.error("Invalid run counts")
    if args.concurrency is not None and args.concurrency<1 or args.rate is not None and args.rate<=0:
        p.error("Load must be positive")
    names=list(WORKLOADS) if args.workload=="all" else [args.workload]
    commands=[command(args,name,r) for name in names for r in range(args.repeats)]
    if not args.run:
        for cmd in commands: print(shlex.join(cmd))
        return
    args.output.mkdir(parents=True,exist_ok=True)
    versions={}
    for package in ("aiperf","vllm","numpy"):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    metadata=dict(platform=platform.platform(),python=sys.version,versions=versions,commands=commands,
                  model=args.model, note="Record model SHA, server flags, H100 variant and GPU telemetry separately")
    manifest=args.output/f"{args.label}-manifest.json"
    if manifest.exists(): raise FileExistsError("Choose a new label to preserve previous results")
    manifest.write_text(json.dumps(metadata,indent=2))
    for cmd in commands: subprocess.run(cmd,check=True)


if __name__=="__main__": main()
