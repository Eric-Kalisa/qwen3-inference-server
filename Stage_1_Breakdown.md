# CUDA Tile implementations for Hugging Face modules

## Stage 1  goal

This project builds and validates CUDA Tile C++ forward implementations of selected
modules from Hugging Face Transformers. PyTorch generates trusted reference cases
from initialized modules. Each case records the exact weights, configuration,
invocation inputs, and expected outputs in SQLite. A small Python runner loads those
rows and compares a CUDA implementation with the saved PyTorch result.

The initial modules are `Qwen3MLP`, `Qwen3Attention`, `Qwen3RMSNorm`,
`Qwen3RotaryEmbedding`, and `Qwen3DecoderLayer`. We will add more modules/modify the existing ones as separate implementations while reusing the shared schema, SQLAlchemy models, and case-generation and validation conventions wherever they fit.

## Project layout

- `schema.sql` defines the shared SQLite schema.
- `sql_models.py` maps that schema for Python generators and runners.
- `<module>/generate_<module>_cases.py` constructs PyTorch modules and writes
  reference cases.
- `<module>/<module>.cu` implements the forward computation using CUDA Tile.
- `<module>/run_<module>.py` loads module state and invocation data from SQLite,
  calls the compiled library, and compares its outputs with the references.
- `<module>/modal_run_<module>.py` compiles and runs the implementation on Modal.
- `<module>/<module>_cases.db` is the generated database used by the runner.

Each module folder follows the same generator, CUDA source, runner, launcher, and
generated database pattern. Keep shared database definitions at the project root
and each module's artifacts in its own folder. The decoder-layer runner
composes the RMSNorm, rotary embedding, attention, and MLP libraries and uses its
own CUDA Tile residual-add kernel; this reuses the component implementations
instead of duplicating their operations. It calls each component's host C ABI in
sequence, so this simple composition adds host/device copies and synchronizations;
it is a correctness-focused starting point, not a fused performance kernel.

The initial component cases use BF16, dimensions no larger than 128, default
Qwen3 RoPE, eager full attention, and no cache. Expand these limits deliberately
by adding reference cases and updating the corresponding CUDA implementation.

## Shared case format

The database separates data by its role:

- `module_instance` stores one concrete initialized module and its Safetensors
  state dictionary.
- `module_config` stores the scalar construction settings needed to rebuild or
  interpret that module.
- `test_case` identifies one invocation of a particular module instance.
- `test_case_input` stores named forward arguments. Tensor values are Safetensors
  bundles; ordinary scalar arguments can use SQLite scalar values.
- `test_case_expected` stores the reference result bundle for that invocation.

Several invocations can share one module instance and its exact weights. Keep
configuration on the module instance and invocation-specific values on the test
case. Continue using the shared schema and models unless a concrete module feature
cannot be represented by them; explain any schema change and keep it useful across
modules.

## Workflow for each module

1. **Inspect the reference implementation.** Identify the module's constructor,
   forward arguments, output structure, weight names and shapes, dtype behavior,
   optional paths, and numerical details. Decide and document the initial supported
   scope, such as eager attention, no cache, or a limited shape range.
2. **Generate reference cases in PyTorch.** Build the actual Hugging Face module,
   initialize it deterministically, set evaluation behavior where appropriate, and
   save its state dictionary. Record scalar construction settings as relational
   configuration rows. Generate multiple representative invocations per initialized
   module and save every tensor input and expected output. Keep tensor dtypes
   consistent with the CUDA interface; for example, this project currently uses
   BF16 tensors. Include cases that exercise distinct branches and shapes rather
   than relying on a large number of near-identical random cases.
3. **Implement the CUDA Tile forward path.** Follow the saved weight layout and
   tensor shapes, and match the reference's operation order, rounding points,
   masking, and output structure. Reuse proven project patterns when appropriate,
   especially the memory-backed FP32 accumulation used by the MLP projection after
   loop-carried Tile accumulation proved unreliable. Add comments for pivotal
   choices, supported limits, and any deliberate differences from the reference.
4. **Connect the minimal runner.** Use `sql_models.py` to load module instances,
   construction settings, named inputs, and expected outputs. Pass tensors through
   a clearly defined C ABI, call the CUDA implementation, and compare each returned
   output using the runner's fixed tolerances. Keep execution and comparison policy
   in the runner, not as per-case database fields.
5. **Check the generated artifacts and Python code.** Regenerate the module database
   after changing its generator or reference behavior. Verify the database can be
   opened, has the expected module and case rows, and stores tensors with the
   intended dtype and shape. Check Python syntax and review diffs before handing off.
6. **Run and diagnose on the GPU.** The project owner runs the Modal/CUDA command and
   owns GPU execution. Treat successful kernel completion and numerical agreement
   as separate results. Use per-stage logs to locate the first failing computation;
   fix the underlying math, indexing, layout, or dtype issue instead of loosening
   tolerances to conceal a mismatch. Repeat until every intended case passes.
7. **Review and deliver.** Keep work on a feature branch and do not push directly to
   the default branch. Include the generator, CUDA source, runner, launcher, and the
   generated database when appropriate. Summarize supported behavior, checks run,
   and any remaining limits in the pull request.

## Running a module workflow

From the project root, generate reference cases with that module's generator,
then run its Modal launcher yourself when ready. For example:

```powershell
python attention\generate_qwen3_attention_cases.py
modal run attention\modal_run_qwen3_attention.py
```

Use the matching module path under `mlp`, `rms_norm`, `rotary_embedding`, or `decoder_layer`.

Modal may import a launcher remotely from a temporary path such as
`/root/modal_run_<module>.py`, rather than from its repository location. Avoid
deriving the project root by indexing a fixed number of parents from `__file__`.
Each launcher should search its local ancestor paths for the `requirements.txt`
sentinel, then fall back to `/root/project`, which is the project-root mount used by
these Modal images. Raise a clear error if neither location contains the sentinel.
This keeps local Modal imports and remote function startup pointed at the same
project files. CUDA runs are intentionally initiated and controlled by the project
owner.

## Limitations as of 11/25/26

Stress tests were added to compare intermediate tensor values at each layer of
`Qwen/Qwen3-0.6B` against the Hugging Face reference under several tolerance
settings. The first divergence appears at layer 2, while by layer 9 the
intermediate values exceed the selected tolerance. This indicates that the
discrepancy accumulates as the model runs, rather than appearing only in the
final logits.

The MLP down-projection weights and their associated operations are a likely
source of the mismatch, but this has not yet been confirmed. Next steps are to
establish concrete, justified tolerance levels and investigate the MLP
implementation more closely to isolate the source of the growing numerical
difference.

