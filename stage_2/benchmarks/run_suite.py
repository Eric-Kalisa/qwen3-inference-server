"""Run each server sequentially on ONE Linux H100, saving failures and raw data.

Subprocess groups are terminated and reaped before a new server starts. The
client, vLLM and custom server can use separate Python environments.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.request import Request, urlopen
from urllib.error import URLError
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from benchmarks.profile import command
from benchmarks.artifacts import profile_export
from benchmarks.latency import write_breakdown
from benchmarks.plan import cases, settings, DEFAULTS
VARIANTS={
    "uncached": ("none", False, False),
    "contiguous": ("contiguous", False, False),
    "paged": ("paged", False, False),
    "contiguous_batch": ("contiguous", True, False),
    "paged_batch": ("paged", True, False),
    "paged_shared": ("paged", True, True),
    "vllm": (None, True, False),
    "vllm_shared": (None, True, True),
}


def server_command(args, variant, logdir):
    mode,batch,sharing=VARIANTS[variant]
    cfg=json.loads((args.model_dir/"config.json").read_text())
    cache_bytes=(2*cfg["num_hidden_layers"]*args.blocks*16*cfg["num_key_value_heads"]*
                 cfg.get("head_dim", cfg["hidden_size"]//cfg["num_attention_heads"])*2)
    if mode is None:
        return [args.vllm_python,"-m","vllm.entrypoints.openai.api_server",
                "--model",str(args.model_dir),"--served-model-name",args.model,
                "--dtype","bfloat16","--kv-cache-dtype","auto",
                "--max-model-len",str(args.max_context),"--max-num-seqs","64",
                "--max-num-batched-tokens","256","--block-size","16",
                "--kv-cache-memory-bytes",str(cache_bytes),
                "--enable-prefix-caching" if sharing else "--no-enable-prefix-caching",
                "--generation-config","vllm","--seed","42","--host","127.0.0.1","--port","8000"]
    cmd=[args.server_python,str(ROOT/"inference/server.py"),"--model-dir",str(args.model_dir),
         "--model",args.model,"--library-dir",str(args.library_dir),"--mode",mode,
         "--blocks",str(args.blocks),"--max-context",str(args.max_context),
         "--max-sequences","64","--max-requests","128","--max-rows","256",
         "--prefix-capacity","64" if sharing else "0",
         "--telemetry",str(logdir/"server-telemetry.jsonl")]
    if not batch: cmd.append("--no-batching")
    return cmd


def request_json(url, payload=None, timeout=10):
    data=None if payload is None else json.dumps(payload).encode()
    req=Request(url,data=data,headers={"Content-Type":"application/json"})
    with urlopen(req,timeout=timeout) as response:
        return json.load(response)


@contextmanager
def managed_process(cmd, log):
    with log.open("w",encoding="utf-8") as stream:
        process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            yield process
        finally:
            # Reap the whole server tree (including vLLM engine workers), even if
            # the parent has already exited or a benchmark timed out.
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait(timeout=10)


def wait_ready(process, timeout=900):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if process.poll() is not None: raise RuntimeError("Server exited; inspect server.log")
        try:
            with urlopen("http://127.0.0.1:8000/health",timeout=2) as response:
                if response.status == 200: return
        except (URLError,TimeoutError,ValueError): pass
        time.sleep(.5)
    raise TimeoutError("Server readiness timed out")


def api_sanity(model, custom):
    body=dict(model=model,prompt="The capital of France is",max_tokens=8,
              temperature=0,ignore_eos=True,skip_special_tokens=False)
    result=request_json("http://127.0.0.1:8000/v1/completions",body,timeout=300)
    if result.get("usage",{}).get("completion_tokens") != 8:
        raise AssertionError("Server did not honor the fixed output-token count")
    if not result.get("choices") or not result["choices"][0].get("text"):
        raise AssertionError("Server produced no text")
    if custom:
        for extra in (dict(n=3,temperature=.8,seed=42),dict(beam_width=3)):
            output=request_json("http://127.0.0.1:8000/v1/completions",dict(body,**extra),timeout=300)
            if len(output.get("choices",[])) != 3: raise AssertionError("Branch validation failed")
    return result


def execute(args):
    if os.name != "posix": raise RuntimeError("The H100 suite requires a Linux host")
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    # Never send measurements to an old listener left on the benchmark port.
    try:
        with urlopen("http://127.0.0.1:8000/health",timeout=2): pass
    except URLError:
        pass
    else:
        raise RuntimeError("Port 8000 already has a server; stop it before benchmarking")
    wait_gpu_idle()
    plan=[]
    for variant in args.variants:
        for workload,concurrency,rate,arrival in cases(args.preset):
            if variant.endswith("shared"): workload="shared_prefix"
            # Deduplicate the shared workload after replacing base names.
            for repeat in range(args.repeats):
                item=dict(variant=variant,workload=workload,concurrency=concurrency,
                          rate=rate,arrival=arrival,repeat=repeat)
                if item not in plan: plan.append(item)
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,driver_version,memory.total",
                                 "--format=csv,noheader"],text=True)
    if "H100" not in gpu or len(gpu.strip().splitlines()) != 1:
        raise RuntimeError(
            f"Expected one visible H100. nvidia-smi reported: {gpu.strip() or 'no GPUs'}"
        )
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob("*")) if p.is_file() and
            p.suffix in (".py",".cu",".cpp",".json") and
            not any(x in p.parts for x in ("build","results","__pycache__","third_party")) and
            not p.is_relative_to(args.output)}
    manifest=dict(model=args.model,model_dir=str(args.model_dir),revision=args.revision,gpu=gpu,
                  platform=platform.platform(),settings=vars(args).copy(),source_sha256=hashes,plan=plan)
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2,default=str))
    print(f"Plan: {len(plan)} AIPerf runs, {len(plan)*args.requests} measured requests, "
          f"{args.case_timeout}s timeout per run. Results: {args.output}",flush=True)
    for label,python in (("server",args.server_python),("vllm",args.vllm_python),("client",args.client_python)):
        with (args.output/f"{label}-packages.txt").open("w") as f:
            subprocess.run([python,"-m","pip","freeze"],stdout=f,check=True)
    failures=[]
    baseline_text=None
    for variant in args.variants:
        logdir=args.output/variant; logdir.mkdir()
        cmd=server_command(args,variant,logdir)
        (logdir/"command.json").write_text(json.dumps(cmd,indent=2))
        # GPU telemetry is captured for every variant, outside the server process.
        telemetry=["nvidia-smi","--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw",
                   "--format=csv","-lms","500"]
        with managed_process(telemetry,logdir/"gpu.csv"):
            try:
                with managed_process(cmd,logdir/"server.log") as process:
                    wait_ready(process)
                    sanity=api_sanity(args.model,VARIANTS[variant][0] is not None)
                    text=sanity["choices"][0]["text"]
                    if baseline_text is None: baseline_text=text
                    sanity["matches_first_variant_text"]=text==baseline_text
                    (logdir/"sanity.json").write_text(json.dumps(sanity,indent=2))
                    for item in (p for p in plan if p["variant"]==variant):
                        name=f'{item["workload"]}-c{item["concurrency"]}-r{item["rate"]}-{item["arrival"]}'
                        client=SimpleNamespace(model=args.model,tokenizer=str(args.model_dir),url="http://127.0.0.1:8000",
                                               requests=args.requests,warmup=args.warmup,output=logdir,
                                               label=name,input_file=None,rate=item["rate"],
                                               concurrency=item["concurrency"],arrival=item["arrival"])
                        invocation=command(client,item["workload"],item["repeat"])
                        invocation[:1]=[args.client_python,"-m","aiperf"]
                        run_dir=logdir/name/item["workload"]/f'repeat-{item["repeat"]}'
                        run_dir.mkdir(parents=True)
                        (run_dir/"run.json").write_text(json.dumps(dict(item,command=invocation),indent=2))
                        print(f"[{variant}] starting {name}, repeat {item['repeat']+1}/{args.repeats}",flush=True)
                        try:
                            with managed_process(invocation,run_dir/"client.log") as client_process:
                                status=client_process.wait(timeout=args.case_timeout)
                                if status: raise subprocess.CalledProcessError(status,invocation)
                            export=profile_export(run_dir)
                            if export is None:
                                raise RuntimeError(f"No measured AIPerf export in {run_dir}; inspect client.log")
                            (run_dir/"selected-export.json").write_text(json.dumps(
                                {"path":str(export.relative_to(run_dir))},indent=2))
                            if variant == "paged_shared":
                                try:
                                    breakdown = write_breakdown(run_dir, export, logdir/"server-telemetry.jsonl")
                                    print(f"[{variant}] TTFT breakdown: {breakdown['status']}, "
                                          f"{breakdown['matched']} matched requests", flush=True)
                                except (ValueError, KeyError, TypeError, OSError) as exc:
                                    print(f"[{variant}] Could not build TTFT breakdown in {run_dir}: {exc}",
                                          flush=True)
                            print(f"[{variant}] completed {name}, repeat {item['repeat']+1}",flush=True)
                        except (subprocess.SubprocessError,RuntimeError) as exc:
                            failures.append(dict(item,error=str(exc)))
                            print(f"[{variant}] {name}: {exc}",flush=True)
                            print((run_dir/"client.log").read_text(errors="replace")[-6000:],flush=True)
                            # A killed client may leave requests running. Restart
                            # on the next variant; never contaminate another case.
                            raise RuntimeError(f"Case failed: {name}: {exc}") from exc
            except Exception as exc:
                failures.append(dict(variant=variant,error=str(exc)))
                print(f"[{variant}] failed: {exc}",flush=True)
                print((logdir/"server.log").read_text(errors="replace")[-6000:],flush=True)
            finally:
                (args.output/"failures.json").write_text(json.dumps(failures,indent=2))
        # A reaped API parent alone does not establish that its GPU worker has
        # released the device. Refuse to start the next variant until it has.
        wait_gpu_idle()
    return failures


def wait_gpu_idle(timeout=30):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        pids=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True)
        if not pids.strip(): return
        time.sleep(.5)
    raise RuntimeError("GPU processes remain after server teardown; refusing overlapping measurements")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--model",default="Qwen/Qwen3-4B")
    p.add_argument("--revision",required=True,help="Resolved immutable checkpoint commit SHA")
    p.add_argument("--library-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--server-python",default=sys.executable)
    p.add_argument("--vllm-python",default="/opt/vllm/bin/python")
    p.add_argument("--client-python",default="/opt/aiperf/bin/python")
    p.add_argument("--preset",choices=list(DEFAULTS),default="smoke")
    p.add_argument("--variants",nargs="+",choices=VARIANTS,default=["uncached","contiguous","paged","paged_batch","vllm"])
    p.add_argument("--requests",type=int,default=0,help="0 selects the preset default")
    p.add_argument("--warmup",type=int,default=-1,help="-1 selects the preset default; 0 disables warmup")
    p.add_argument("--repeats",type=int,default=0,help="0 selects the preset default")
    p.add_argument("--blocks",type=int,default=4096)
    p.add_argument("--max-context",type=int,default=8192)
    p.add_argument("--case-timeout",type=int,default=0,help="Seconds; 0 selects the preset default")
    args=p.parse_args()
    try:
        effective=settings(args.preset,args.requests,args.warmup,args.repeats,args.case_timeout)
    except ValueError as exc:
        p.error(str(exc))
    for key,value in effective.items(): setattr(args,key,value)
    if min(args.requests,args.repeats,args.blocks,args.max_context,args.case_timeout)<1 or args.warmup<0:
        p.error("Run sizes must be positive; warmup must be nonnegative")
    if len(args.revision)!=40 or any(c not in "0123456789abcdef" for c in args.revision.lower()):
        p.error("revision must be a 40-character checkpoint commit SHA")
    if len(args.variants)!=len(set(args.variants)): p.error("Duplicate variants")
    failures=execute(args)
    from benchmarks.summarize import summarize
    summarize(args.output)
    if failures: raise SystemExit("Some variants failed; see failures.json and server/client logs")


if __name__=="__main__": main()
