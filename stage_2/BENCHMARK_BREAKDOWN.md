# Benchmark breakdown and smaller jobs

## What benchmark() does

1. Select effective request counts, warmup, repetitions and case timeout.
2. Build the CUDA runtime and C++ tokenizer on the allocated H100.
3. Resolve/download a Qwen3 checkpoint, record its immutable SHA and reuse cached
   files when available.
4. Run the full-checkpoint correctness gate: eight teacher-forced generation
   steps in uncached, contiguous and paged modes, plus shared-prefix reuse.
   This gate is separate from the unittest suite; benchmark() does not rerun
   all CPU/GPU unit tests.
5. For each requested server variant, sequentially: start it, wait for health,
   check fixed-length real generation, then run the selected AIPerf cases. The
   custom server also checks multiple samples and beam search before timing.
6. For each case, send warmup requests, then measured requests. Save the exact
   command, client log, measured export, server log and GPU/cache telemetry.
7. Stop/reap the server and verify its GPU contexts disappeared before starting
   the next variant. Server startup/loading and correctness gates are outside
   AIPerf's measurements but still consume GPU job time.
8. Summarize measured profiling exports (exclude warmup), preserve failures and
   unattempted cases, and commit results to the persistent Modal volume.

The same server stays loaded across its cases, which avoids a model reload for
every AIPerf invocation. A new process/client is launched for every case.

## Server variants

* uncached: persistent GPU weights, full-prefix recomputation, one request group
  at a time. This is the no-persistent-KV baseline.
* contiguous: retain KV in a reserved contiguous block run; serial request groups.
* paged: allocate pages on demand; serial request groups. Cross-request prefix
  reuse is disabled in this primary comparison.
* paged_batch: paged caching plus iteration-level continuous batching.
* vllm: the external optimized serving baseline, with its KV cache enabled.
* Optional contiguous_batch controls for batching when comparing allocation.
  Optional paged_shared/vllm_shared enable prefix reuse in separate experiments.

All primary variants use the same checkpoint, BF16, token policy and KV-pool byte
budget. vLLM retains its production kernels/graphs, so the comparison includes
implementation differences beyond the three individual optimizations.

## Why full was expensive

Per variant, the full preset includes eight workloads at seven concurrency levels:
short (32 input/32 output), prefill_512 (512/1), prefill_4096 (4096/1), decode_128
(128/128), decode_512 (128/512), context_2048 (2048/128), context_7680 (7680/128),
and mixed (60% 128/64, 30% 1024/128, 10% 4096/256).

Concurrency levels are 1, 2, 4, 8, 16, 32, 64: 56 cases. Another 21 cases cross
seven request rates (0.25–16 requests/s) with constant, Poisson and gamma arrivals.
That gives 77 cases per variant. Five variants, three repetitions and 1,000
measured requests produce 1,155 AIPerf invocations and 1,155,000 measured requests,
plus 23,100 warmup requests. Long sequences make individual requests expensive;
low arrival rates add intentional idle time. At 0.25 requests/s, launching 1,000
requests alone takes about 67 minutes, exceeding the old 30-minute case timeout.
The full plan is preserved explicitly for compatibility, not recommended for an
unattended complete run with those defaults.

## Recommended compact preset

Four cases per variant:

1. prefill_512, concurrency 1: 512 input tokens, one output token. TTFT diagnoses
   prompt processing. ITL is not meaningful with only one output token. KV cache
   is primarily a decode optimization; do not expect a large TTFT gain here.
2. decode_128, concurrency 1: 128 input and 128 output tokens. Isolates generation
   latency and the cost of repeated-prefix recomputation.
3. decode_128, concurrency 4: identical lengths with concurrent clients. Separates
   batching throughput and queuing from the single-user result.
4. mixed_small, concurrency 4: 60% 64/32, 30% 256/64, 10% 1024/128. A small,
   synthetic test of heterogeneous lengths and interference, not a production
   trace. With 30 requests, the longest category is sparsely sampled.

Defaults: 30 measured requests, three warmups, one repetition, five-minute timeout
per case. Five variants produce 20 AIPerf invocations, 600 measured requests and
60 warmups: 57.75x fewer invocations and 1,925x fewer measured requests than full.
These reductions are not runtime speedup predictions. Startup/gates still take
time and longer requests have different costs.

This is a diagnostic comparison, not a p99/SLA study. Report mean/median and
throughput; treat tail percentiles as provisional. It does not establish maximum
concurrency, long-context limits, fragmentation benefits, prefix-cache hit gains,
or burst recovery. Add those selectively after finding a specific bottleneck.
To isolate paging with batching held fixed, compare contiguous_batch/paged_batch.

## Commands

From stage_2 in PowerShell:

```powershell
& '..\.venv\Scripts\python.exe' -m modal run --detach benchmarks/modal_phase2.py::benchmark --model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --preset compact
```

Even smaller: only the uncached baseline, final optimized server and vLLM,
with ten measured requests per case:

```powershell
& '..\.venv\Scripts\python.exe' -m modal run --detach benchmarks/modal_phase2.py::benchmark --model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --preset compact --variants uncached,paged_batch,vllm --requests 10
```

That produces 12 invocations, 120 measured requests and 36 warmups. It answers
the required before/after/vLLM comparison, while dropping attribution between
contiguous KV and paging. Use it first if budget matters most.

Options are now exposed on the Modal function: --requests, --warmup, --repeats,
--case-timeout. Zero requests/repeats/timeout means use the preset default;
warmup=-1 means default and warmup=0 disables warmup. Prefer a small nonzero
warmup when comparing steady-state performance. Each invocation prints its
plan size, results path and case start/completion progress.

The code update does not change or stop any already running Modal app. Stop the
old full job yourself before launching its replacement, if you no longer want
it consuming GPU time. Detached jobs can continue after the terminal disconnects.
