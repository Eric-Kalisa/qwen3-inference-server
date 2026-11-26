# Validation status

Date: November 5, 2026.

The required source copy into `stage_2` succeeded before any new
implementation was started. Copied CUDA source files were hash-checked. New
implementation files are synchronized with hash verification as well.

## Executed locally

The CPU/API suite uses NumPy and simulated GPU storage with prefix fingerprints;
it does not silently substitute CPU inference for the actual CUDA executor.
Tests exercise ownership, numerical scheduling equivalence and concurrent HTTP.

The final test output is recorded in `test-results.txt` after synchronization.
Six tests in `test_cuda.py` deliberately skip unless QWEN_CUDA_LIBRARY points to
the built native library. All other tests must pass.

## Pending

This local Windows environment has no accessible CUDA compiler or NVIDIA GPU.
The CUDA library has not yet been compiled here. Tiny-model CUDA numerical tests,
real Qwen3-0.6B/4B generation, CUDA sanitizers, and AIPerf/vLLM H100 measurements
therefore remain unverified. AIPerf command generation has been exercised, but
the AIPerf benchmark itself has not been run.

The H100 test runner was prepared and attempted using the configured Modal
account. Automatic approval review rejected cloud execution because it uploads
the copied project to an external service and explicit permission to export the
code had not been granted. No workaround was used after that rejection.

The reviewable next operation is:

`python -m modal run inference/modal_run.py::check --sanitize`

It uploads this project (source code and tokenizer dependencies; no model
weights or user prompts) to Modal, builds on an H100, runs the suite with a tiny
generated model, then runs memcheck, racecheck, synccheck and initcheck. This uses
the user's configured Modal account and incurs its normal compute charges.
Full-model download/serving and performance benchmarking are separate operations.

Passing CPU tests does not establish CUDA correctness or prove absence of every
race. Do not report this implementation as GPU-validated until those checks pass.

## Phase 2 update

The user supplied an H100 run showing all 34 phase 1 tests passed with no skips. This supersedes the earlier pending status for the native build and six tiny-model CUDA tests. No sanitizer summaries were supplied. Phase 2 adds ten offline orchestration/reporting tests. Full-checkpoint validation and real AIPerf comparisons remain pending; no further cloud upload was attempted without export approval. See PHASE2.md and phase2-test-results.txt.


## Compact preset validation

The suite ran 51 tests locally: 45 passed and six GPU tests skipped. New checks enforce the compact request/time budgets, the matched decode workload at concurrency 1/4, no low-rate waiting, configurable repetitions and zero warmup, and rejection of invalid settings. The Modal definitions imported successfully; no new GPU job was started. The compact preset has not yet been measured on H100.

