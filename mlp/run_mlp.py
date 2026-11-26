import argparse
import ctypes
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

# The shared SQLAlchemy models stay alongside schema.sql in Week_03.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sql_models import (
    ModuleConfig,
    ModuleInstance,
    TestCase,
    TestCaseExpected,
    TestCaseInput,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR / "qwen3_mlp_cases.db"
LIBRARY_PATH = SCRIPT_DIR / "libqwen3_mlp.so"
RTOL = 1e-2
ATOL = 1e-2


def tensor_pointer(tensor):
    # BF16 tensors store 16-bit values; the CUDA C ABI reads the same bits as __nv_bfloat16.
    return ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_uint16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DB_PATH)
    parser.add_argument("--library", type=Path, default=LIBRARY_PATH)
    args = parser.parse_args()

    print(f"Opening test database: {args.database}", flush=True)
    engine = create_engine(f"sqlite:///{args.database}")
    library = ctypes.CDLL(str(args.library))
    run_qwen3_mlp = library.qwen3_mlp_cuda
    bf16_pointer = ctypes.POINTER(ctypes.c_uint16)
    run_qwen3_mlp.argtypes = [
        bf16_pointer,
        bf16_pointer,
        bf16_pointer,
        bf16_pointer,
        bf16_pointer,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    run_qwen3_mlp.restype = ctypes.c_int

    with Session(engine) as session:
        module_instances = session.scalars(select(ModuleInstance)).all()
        print(f"Loaded {len(module_instances)} module instance(s)", flush=True)

        for module_instance in module_instances:
            # A module row owns construction config and weights; each case supplies only its
            # invocation input and expected output, so cases reuse the same initialized module.
            config_rows = session.scalars(
                select(ModuleConfig).where(
                    ModuleConfig.module_id == module_instance.id
                )
            ).all()
            config = {row.name: row.value for row in config_rows}
            hidden_size = int(config["hidden_size"])
            intermediate_size = int(config["intermediate_size"])
            # Load the saved module state once; case rows contain only invocation data.
            weights = load(module_instance.state)
            gate_weight = weights["gate_proj.weight"].contiguous()
            up_weight = weights["up_proj.weight"].contiguous()
            down_weight = weights["down_proj.weight"].contiguous()
            cases = session.scalars(
                select(TestCase).where(TestCase.module_id == module_instance.id)
            ).all()

            print(
                f"Module {module_instance.id}: hidden_size={hidden_size}, "
                f"intermediate_size={intermediate_size}, {len(cases)} case(s)",
                flush=True,
            )

            for case_index, case in enumerate(cases, start=1):
                case_input = session.scalars(
                    select(TestCaseInput).where(TestCaseInput.case_id == case.id)
                ).one()
                case_expected = session.get(TestCaseExpected, case.id)
                inputs = load(case_input.value)["x"].contiguous()
                expected = load(case_expected.value)["output"]
                actual = torch.empty_like(inputs)
                # Flatten leading batch dimensions into rows for the CUDA ABI.
                rows = inputs.numel() // hidden_size

                print(
                    f"Case {case_index}/{len(cases)} ({case.id}): "
                    f"rows={rows}; running CUDA",
                    flush=True,
                )
                started = time.perf_counter()
                status = run_qwen3_mlp(
                    tensor_pointer(inputs),
                    tensor_pointer(gate_weight),
                    tensor_pointer(up_weight),
                    tensor_pointer(down_weight),
                    tensor_pointer(actual),
                    rows,
                    hidden_size,
                    intermediate_size,
                )
                print(
                    f"CUDA returned status={status} in "
                    f"{time.perf_counter() - started:.2f}s",
                    flush=True,
                )
                if status != 0:
                    raise RuntimeError(
                        f"Qwen3MLP failed for case {case.id}: CUDA status {status}"
                    )

                # BF16 reduction order can differ slightly from the reference.
                torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
                print(f"Case {case.id}: passed", flush=True)

    engine.dispose()


if __name__ == "__main__":
    main()
