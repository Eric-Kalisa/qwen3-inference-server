                                                           # Stage 2 Breakdown

Stage 2 is the optimized CUDA inference server in [the Stage 2 folder](stage_2/README.md).
It adds a GPU-resident KV cache, paged attention, prefix sharing, and continuous
batching. The module-level CUDA Tile and PyTorch comparisons are documented in
[Stage 1 Breakdown](Stage_1_Breakdown.md).

Run the commands below from `stage_2` unless the section specifies another
directory. See [Workflow](stage_2/WORKFLOW.md) for runtime ownership and
[Validation](stage_2/VALIDATION.md) for recorded checks.

## Implemented path

* BF16 Qwen3 weights are read directly from local safetensors; no Torch or
  Transformers is imported by inference. `huggingface-hub` is only a downloader.
* Model weights, activations, workspaces and KV tensors remain on the GPU.
  cuBLAS executes projection GEMMs. CUDA Tile kernels handle normalization,
  RoPE, residual/MLP operations, KV-cache writes, and paged attention.
* Prefill is causal and chunked. Decode processes one new token per sequence.
  Attention uses an FP32 online softmax and reads physical KV pages directly.
* Exact token prefixes share complete read-only blocks across requests. An LRU
  prefix registry holds its own references; eviction does not invalidate live
  sequences. The last prompt token is recomputed to obtain fresh logits.
* Parallel samples and beam candidates fork block tables. A shared partial tail
  is copied before its first write; full preceding blocks remain shared. Parent
  references are retained until all surviving children have been created.
* Beam search keeps a fixed-width pool of finished/live candidates, ranked by
  cumulative log probability without length normalization. `beam_width` is an
  extension parameter and returns the surviving beams; it is not a drop-in
  implementation of every vLLM beam-search option.
* A single owner thread controls the GPU, tokenizer and cache. HTTP threads
  communicate through a queue, futures and cancellation events. One CUDA stream
  completes before metadata publication, copies, eviction or reuse.
* Continuous batching packs ragged prompt/decode rows, reserves decode work
  ahead of bounded prompt chunks and admits requests between iterations. Under
  pressure it evicts unused prefixes and preempts younger unselected request
  groups, which later recompute their cached history. A group that cannot fit
  alone fails explicitly. CPU swapping and distributed execution are excluded.

## Build on an H100 Linux host

Use Python 3.12, CUDA Toolkit 13.3+, a CUDA 13.3-compatible NVIDIA driver,
cuBLAS, and CMake 3.24+. The runtime builds its custom kernels with CUDA Tile
C++ and uses cuBLAS for the BF16 linear layers. From this folder:

```sh
python -m pip install -r inference/requirements.txt
cmake -S inference -B inference/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build inference/build --target qwen3_runtime qwen3_tokenizer --parallel 4
```

Download a pinned Qwen model revision into a local directory containing its
config, tokenizer and BF16 safetensors. Only dense Qwen3 with default unscaled
RoPE and bias-free attention is accepted. Start with 0.6B, then use 4B:

```sh
python inference/server.py --model-dir /models/Qwen3-0.6B --model Qwen/Qwen3-0.6B
```

For Modal, run from this folder with an authenticated account:

```sh
python -m modal run inference/modal_run.py::check --sanitize
python -m modal run inference/modal_run.py::download --model Qwen/Qwen3-0.6B --revision MODEL_COMMIT_SHA
python -m modal serve inference/modal_run.py
```

The check builds the runtime, runs all tests on H100 and invokes all four CUDA
sanitizers. The serve command exposes separate 0.6B and 4B routes; download each
model first. Model downloading and serving are separate operations.

## API

`POST /v1/completions` accepts a string `prompt`, `model`, `max_tokens`,
`temperature`, `seed`, `n`, `beam_width`, `stream`, `stream_options` and
`ignore_eos`, and `skip_special_tokens`. Unknown parameters are rejected rather than silently ignored.

```json
{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":32,"stream":true,"stream_options":{"include_usage":true}}
```

Set `n:4, temperature:0.8, seed:42` for four reproducible stochastic samples.
Set `beam_width:4, stream:false` for beam search; a winning beam can change its
earlier tokens, so beam streaming is explicitly rejected. Normal completions
stream SSE deltas, completion reasons, optional usage, then `[DONE]`. Incomplete
UTF-8 tails are withheld until they decode cleanly. Slow consumers are cancelled
when their bounded event queue fills. Queue overflow returns HTTP 429.

`GET /health` reports worker availability. `GET /v1/models` reports the model.
This minimal HTTP server has no authentication and is intended for a controlled
benchmark host, not deployment as a public production service.

## Controlled optimization modes

* `--mode none`: discard K/V after every output token; recompute the full prefix.
  Temporary K/V are still needed within each full forward pass.
* `--mode contiguous`: reserve a contiguous run of pages for each sequence's
  maximum context; forks copy only computed blocks. Prefix sharing is disabled.
* `--mode paged`: allocate on demand, share exact prefixes and use reference
  counts/COW. `--prefix-capacity 0` disables cross-request prefix reuse while
  retaining fork sharing, useful for controlled comparisons.
* `--no-batching`: execute one request group at a time. Samples/beams within a
  group remain batchable; use `n=1, beam_width=1` for a single-stream baseline.

All modes share the same attention math/addressing kernel. Contiguous mode
isolates allocation/sharing effects, not the overhead of an independently
optimized contiguous-attention kernel. The historical wrappers' transfer-heavy
behavior is not reproduced by `none`: GPU residency is held constant.

The defaults allocate 1,024 blocks of 16 tokens. KV bytes are
`2 * layers * blocks * block_size * kv_heads * head_dim * 2`.
Increase `--blocks` for memory stress/concurrency tests; reserve room for weights
and workspaces. `--max-context` counts prompt plus requested output tokens.
Defaults use 8,192 context tokens, 256 scheduled rows, 32 sequences and 32 queued
or running requests. These are limits, not a promise all requests fit together.

## Tests and race protection

From `inference`, run:

```sh
python -m unittest discover -s tests -v
QWEN_CUDA_LIBRARY="$PWD/build/libqwen3_runtime.so" python -m unittest discover -s tests -v
```

CPU tests cover 2,000 randomized fork/append/free/eviction operations with a
complete reference-count audit; COW sibling isolation; page boundaries; prefix
lifetime; failed allocation/copy rollback; owner-thread guards; parallel clients;
cancel-during-execution; shutdown/submit ordering; request limits; injected device
failure; preemption/recomputation; fixed-seed sampling; exhaustive small beam
selection; and HTTP/SSE equivalence, validation and usage.

The GPU suite uses a tiny generated BF16 model and an independent full-attention
NumPy oracle. It covers chunked/full prefill, fragmented pages, >128-token
contexts, mixed branch batches, COW, block reuse and native alias rejection.
No model download or HF/PyTorch installation is needed for these tests.

Run `compute-sanitizer --tool TOOL --error-exitcode 99 python -m unittest discover
-s tests -p test_cuda.py` with QWEN_CUDA_LIBRARY set, for each of `memcheck`,
`racecheck`, `synccheck`, `initcheck`. Racecheck checks shared-memory hazards;
it does not prove absence of all global-memory or application scheduling races.
Ownership assertions, rejection of overlapping native writes/read-write aliases,
deterministic barriers, and output-equivalence tests cover those additional risks.
Tests provide evidence for tested schedules, not a mathematical race-free proof.

## AIPerf benchmark preparation

`benchmarks/profile.py` generates commands by default; `--run` executes them.
Install `benchmarks/requirements.txt` in a **separate Python 3.12 client
environment**. AIPerf's own dependencies may include Transformers; they are not
part of this CUDA server. vLLM is also a separate external baseline environment.

```sh
python benchmarks/profile.py --label paged-c8 --workload mixed --concurrency 8
python benchmarks/profile.py --label paged-qps2 --workload mixed --rate 2 --arrival poisson --run
```

Workloads include short baseline, prefill, long decoding, context-length,
heterogeneous lengths and shared-prefix pools. Replay real prompts with
`--input-file prompts.jsonl` (AIPerf single_turn records: `{"texts":["..."]}`).
Use the same tokenizer/revision, seed and commands for each server. Run
concurrency 1,2,4,8,16,32,64 and constant/Poisson/bursty rate sweeps around measured
capacity. Increase server request limits for the 64-concurrency sweep. Request
rate runs omit a client concurrency cap so they do not hide offered load behind
client-side queuing. Record successful, rejected and timed-out requests.

Primary comparison: `none`, `contiguous`, `paged` with `--prefix-capacity 0`,
then paged+batching; compare vLLM with prefix reuse and speculative decoding
disabled. Separate warm shared-prefix runs enable reuse on both servers. vLLM's
KV cache remains enabled; disabling prefix caching is not disabling KV caching.
Run servers sequentially on the same dedicated H100. Match BF16, context length,
model revision and greedy/fixed-output policy (`ignore_eos`, `skip_special_tokens=false`). Pin the vLLM image
by digest and record it in the run manifest; do not use a floating latest tag for
reported results. Use a low-noise local network path for the client.

Each run saves AIPerf artifacts and a command/version manifest. Collect GPU memory,
utilization and Nsight traces separately; profiling runs must not be mixed with
headline measurements. Report TTFT, ITL, end-to-end latency, aggregate output
tokens/s, requests/s, errors and p50/p95/p99 with sample counts. The default is
20 warmup requests and 1,000 measured requests, repeated three times. Inspect
actual token lengths, confidence intervals, client saturation and tail sample
counts before making performance claims. No benchmark measurements are fabricated
or bundled as if they were H100 results.

Optional `--telemetry measurements.jsonl` on the CUDA server records iteration
batch sizes, active requests, used blocks, prefix hits, COW copies and preemptions,
plus request admission/worker-first-token/end timings. Correlate completion IDs
with HTTP results. Forward durations are host wall time including transfers and
synchronization, not CUDA kernel timings. Worker-first-token time precedes
transport delivery and is distinct from AIPerf's client-observed TTFT.



## Phase 2

See [PHASE2.md](stage_2/PHASE2.md) for full-checkpoint validation, isolated vLLM/AIPerf environments, H100 comparison jobs, reproducibility records, and result summaries.


## Smaller benchmark jobs

See [BENCHMARK_BREAKDOWN.md](stage_2/BENCHMARK_BREAKDOWN.md). The compact preset runs four focused cases across five variants: 20 invocations, 600 measured requests and 60 warmups. Requests, warmup, repetitions and per-case timeout are now configurable from the Modal command. Existing running jobs are unaffected.

