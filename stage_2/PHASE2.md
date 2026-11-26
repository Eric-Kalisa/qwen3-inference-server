# Phase 2: full-checkpoint validation and H100 benchmarks

Phase 1 already implements cache modes, paged attention, prefix sharing/COW and
batching. Phase 2 adds a reproducible full-model validation and performance
comparison pipeline. It does not add an HF/PyTorch reference implementation.
vLLM and AIPerf have isolated dependency environments; the custom CUDA process
continues to use NumPy, ctypes, cuBLAS and the C++ tokenizer.

## 1. Validate the real 0.6B checkpoint

From PowerShell in `stage_2`, using the existing Modal environment:

```powershell
& '..\.venv\Scripts\python.exe' -m modal run benchmarks/modal_phase2.py::validate --model Qwen/Qwen3-0.6B
```

This resolves the requested model revision to an immutable SHA, downloads only
JSON and safetensors, builds on H100, and compares full-checkpoint logits across
uncached, contiguous and paged modes. Every step uses the same teacher-forced
history; differences cannot be hidden by diverging generated prefixes. It also
checks prefix reuse after the originating sequence releases its references.
Default tolerances are `atol=0.1, rtol=0.02`, and all logits must be finite.
This establishes cache equivalence, not equivalence to an independent full-model
implementation; the independent NumPy oracle remains in the tiny-model suite.

Repeat with `--model Qwen/Qwen3-4B`. For exact reruns, pass `--revision SHA` using
the SHA returned by the first job. A moving revision such as `main` is resolved
and recorded before it is used by either server.

## 2. Run a small end-to-end comparison

```powershell
& '..\.venv\Scripts\python.exe' -m modal run benchmarks/modal_phase2.py::benchmark --model Qwen/Qwen3-0.6B
```

The default smoke run uses 20 requests, 3 warmup requests, one repetition,
32/32 input/output tokens and concurrency 1 and 4. It runs five server variants
sequentially on the same H100: uncached, contiguous KV, paged KV, paged KV with
continuous batching, and vLLM. It is a functional test of the pipeline, not a
statistically stable performance conclusion. No other public service is started.

Then run the same command with `--model Qwen/Qwen3-4B`. A full-checkpoint
validation gate must pass before AIPerf starts. Per-server API checks also verify
fixed output counts and nonempty real generation; the custom server additionally
checks three samples and three beam candidates. Greedy text equality with the
first variant is recorded, but any difference from vLLM requires examination and
is not silently presented as numerical equivalence.

## 3. Run the measurement matrix

```powershell
& '..\.venv\Scripts\python.exe' -m modal run benchmarks/modal_phase2.py::benchmark --model Qwen/Qwen3-4B --revision MODEL_COMMIT_SHA --preset full
```

Full mode uses 1,000 requests, 20 warmup requests and three repetitions per case.
It covers prompt-heavy, decode-heavy, long-context and mixed workloads;
concurrency 1–64; and fixed request rates 0.25–16 requests/s with constant,
Poisson and bursty gamma arrivals. Rates are the same across servers, not scaled
independently to make each server appear saturated. Open-loop cases do not use
a client concurrency cap. Cases time out after 30 minutes; a timed-out client
and its server group are torn down, and the rest of that variant is skipped.
Failures and incomplete coverage remain visible. A full matrix can exceed the
24-hour Modal job limit for slow baselines; run selected variants separately
when needed and retain incomplete results as such. It may consume substantial
GPU time. The smoke preset is the intended starting point.

To compare contiguous/paged allocation with batching held constant, include
`contiguous_batch,paged_batch`. For a separate shared-prefix comparison:

```powershell
& '..\.venv\Scripts\python.exe' -m modal run benchmarks/modal_phase2.py::benchmark --model Qwen/Qwen3-4B --variants paged_shared,vllm_shared
```

Shared variants use the same four-prefix pool and warmup policy. The primary
five variants disable cross-request prefix reuse; paging and batching gains are
therefore not conflated with reuse of identical prompts. vLLM always retains its
normal KV cache; this is not a claimed vLLM-without-KV-cache comparison.

## Fairness and lifecycle

* CUDA and vLLM use identical checkpoint files, BF16, block size 16, context
  limits and an equal byte budget for the KV pool (computed from model config).
* CUDA uses at most 256 rows and 64 sequences per GPU batch. vLLM is explicitly
  given the same token and sequence caps, a fixed seed and no speculative model.
  vLLM keeps its production kernel/graph optimizations, which are recorded in
  its package versions and launch flags; this is not a kernel-only comparison.
* Request generation uses a fixed AIPerf seed, greedy decoding, fixed requested
  lengths, `ignore_eos=true`, `skip_special_tokens=false`, and the local tokenizer.
* CUDA, vLLM and AIPerf run in separate Python environments. Versions are pinned
  to vLLM 0.11.0 and AIPerf 0.13.0, with complete installed package manifests
  recorded per experiment. These are reproducible baseline choices, not claims
  to benchmark the newest vLLM release.
* The custom server uses bounded admission (128 requests); overload rejection
  is part of its measured behavior. Record errors alongside throughput and
  latency. Successful-request latency alone can conceal rejected traffic.
* The runner refuses an occupied benchmark port or another GPU compute process.
  It kills/reaps entire server/client process groups on failure and waits for
  GPU compute contexts to disappear before starting the next variant.
* One H100 must be visible. GPU model, UUID, memory, driver, source hashes,
  model SHA, commands and dependency versions are recorded. nvidia-smi samples
  utilization, used memory and power every 500 ms; CUDA's JSONL records cache
  occupancy, batch sizes, COW, prefix hits, preemption and worker timings.
* Tokenization differences or actual sequence-length mismatches must be examined
  before interpreting comparisons. Qwen's C++ tokenizer is still checked only by
  real-model/API smoke checks here; equality to vLLM tokenization is not assumed
  from matching model filenames.

## Shared-prefix latency diagnosis

Run the custom smoke from `stage_2/` with `python -m modal run modal_aipref.py`.
For each concurrency, open the `latency-breakdown.json` printed beside the
smoke summary. The report joins measured AIPerf requests to server telemetry
using the HTTP request ID and ranks the same requests by client TTFT. It
exposes admission queue wait, tokenization, prefix lookup and hit tokens,
prefill KV preparation, prefill forward, scheduling gaps, first token
selection, first text flush, and decode wall/forward time. Mixed batch forward
measurements are already included in the corresponding forward total.

Use the client TTFT percentile ratios from matching custom and vLLM runs to
check the 10x target. Use the request rows to diagnose those results. A smoke
run's p95/p99 rank selects a small number of requests and may not equal the
interpolated percentile in AIPerf's aggregate export.

To run the same shared-prefix workload at larger sample counts, use
`python -m modal run -m benchmarks.modal_shared_benchmark::main --size compact`
for 30 requests per case or substitute `--size full` for 1,000. Both retain the
smoke workload, concurrencies, and warmup. Full allows up to three hours per case.
Downloaded runs are stored under `results/shared-benchmark/<run-id>/`.

## Outputs

Results live in the persistent Modal volume `qwen3-week-7-8-results`. The shared
smoke entrypoints automatically download their run directory into the local
`results/shared-smoke/...` folder after the Modal job completes. Other jobs
print their `/results/...` directory; download one with:

```powershell
& '..\.venv\Scripts\python.exe' -m modal volume get qwen3-week-7-8-results /RUN_DIRECTORY ./results
```

Each experiment includes `manifest.json`, package versions, validation results,
per-server commands, sanity results, `server.log`, `gpu.csv`, CUDA telemetry,
per-case `run.json`, `client.log`, and raw AIPerf JSON/CSV exports. Errors go to
`failures.json`; failed or missing cases must remain in the final interpretation.

`summary.csv` and `summary.json` preserve per-run TTFT/ITL/end-to-end statistics,
throughputs, error counts, observed lengths and units. Throughput speedup is
computed only against a matching clean uncached run. Repetition aggregates show
mean and standard deviation; p99 values are never averaged and called a pooled
p99. Missing metrics stay missing. Tails with fewer than 1,000 observations are
flagged. AIPerf's raw exports remain authoritative.

To regenerate the summary after downloading results:

```powershell
& '..\.venv\Scripts\python.exe' benchmarks/summarize.py ./results/RUN_DIRECTORY
```

## Local tests and current evidence

From `inference`, the existing command discovers both phases:

```powershell
& '..\..\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

`tests/test_phase2.py` adds ten offline tests for readiness, process-group
teardown/escalation, GPU-release gating, equal cache budgets, traffic controls,
failure-preserving summaries, the workload matrix, and teacher-forced validation.
These use mocks and explicitly synthetic metric fixtures, not benchmark results.
The phase 1 CUDA suite passed in the H100 output supplied by the user. Phase 2
real-checkpoint and benchmark jobs have not yet been executed by the agent.

Running either Modal job uploads the inference/benchmark source folders to the
configured Modal account, downloads public model files and uses billable H100
time. It excludes local model weights, result directories and editor settings.
An earlier automatic approval review required explicit permission for the agent
to export source code; no subsequent cloud operation is assumed approved merely
because local implementation is complete.

## Smoke-run corrections (November 5)

Run Qwen3-4B-smoke-c69a98e9 exposed two integration issues. AIPerf writes top-level, warmup and profiling exports; the runner and summarizer now select phases/profiling/profile_export_aiperf.json explicitly, with a root-only fallback for older layouts. Warmup-only results never count as measured results. vLLM 0.11.0 installed Transformers 5.22.1, which removed its required tokenizer attribute. The vLLM environment now pins Transformers 4.57.3, tokenizers 0.22.1 and huggingface-hub 0.36.0, with pip check and a tokenizer-API check during image build. These packages remain isolated from the custom CUDA server. Failures now print client/server log tails to the Modal console. The saved 20-request CUDA measured export parses successfully with the corrected summarizer. Rerun the smoke comparison to verify the rebuilt vLLM environment and both concurrency levels. Saved evidence is in diagnostics/Qwen3-4B-smoke-c69a98e9.

