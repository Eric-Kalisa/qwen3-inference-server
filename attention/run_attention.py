import argparse
import ctypes
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sql_models import (
    ModuleConfig,
    ModuleInstance,
    TestCase,
    TestCaseExpected,
    TestCaseInput,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR / "qwen3_attention_cases.db"
LIBRARY_PATH = SCRIPT_DIR / "libqwen3_attention.so"
RTOL = 1e-2
ATOL = 1e-2


def tensor_pointer(tensor):
    # BF16 data crosses the C ABI unchanged as 16-bit CUDA bfloat values.
    return ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_uint16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DB_PATH)
    parser.add_argument("--library", type=Path, default=LIBRARY_PATH)
    args = parser.parse_args()

    print(f"Opening test database: {args.database}", flush=True)
    engine = create_engine(f"sqlite:///{args.database}")
    library = ctypes.CDLL(str(args.library))
    run_attention = library.qwen3_attention_cuda
    bf16_pointer = ctypes.POINTER(ctypes.c_uint16)
    run_attention.argtypes = [
        bf16_pointer,
        bf16_pointer,
        bf16_pointer,
        bf16_pointer,
        bf16_pointer,
        bf16_pointer,
        bf16_pointer,
        bf16_pointer,
        bf16_pointer,
        bf16_pointer,
        bf16_pointer,
        bf16_pointer,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    run_attention.restype = ctypes.c_int

    mismatches = []
    with Session(engine) as session:
        module_instances = session.scalars(select(ModuleInstance)).all()
        print(f"Loaded {len(module_instances)} attention module instance(s)", flush=True)

        for module_instance in module_instances:
            # Each module row owns weights/config; invocation rows hold x, RoPE, and mask.
            config_rows = session.scalars(
                select(ModuleConfig).where(
                    ModuleConfig.module_id == module_instance.id
                )
            ).all()
            config = {row.name: row.value for row in config_rows}
            hidden_size = int(config["hidden_size"])
            num_heads = int(config["num_attention_heads"])
            num_kv_heads = int(config["num_key_value_heads"])
            head_dim = int(config["head_dim"])
            rms_norm_eps = float(config["rms_norm_eps"])

            state = load(module_instance.state)
            q_weight = state["q_proj.weight"].contiguous()
            k_weight = state["k_proj.weight"].contiguous()
            v_weight = state["v_proj.weight"].contiguous()
            o_weight = state["o_proj.weight"].contiguous()
            q_norm = state["q_norm.weight"].contiguous()
            k_norm = state["k_norm.weight"].contiguous()
            cases = session.scalars(
                select(TestCase).where(TestCase.module_id == module_instance.id)
            ).all()

            print(
                f"Module {module_instance.id}: hidden={hidden_size}, heads={num_heads}, "
                f"kv_heads={num_kv_heads}, head_dim={head_dim}, {len(cases)} case(s)",
                flush=True,
            )

            for case_index, case in enumerate(cases, start=1):
                case_inputs = session.scalars(
                    select(TestCaseInput).where(TestCaseInput.case_id == case.id)
                ).all()
                input_values = {row.name: load(row.value) for row in case_inputs}
                hidden_states = input_values["hidden_states"]["value"].contiguous()
                cos = input_values["position_embeddings"]["cos"].contiguous()
                sin = input_values["position_embeddings"]["sin"].contiguous()
                attention_mask = input_values["attention_mask"]["value"].contiguous()
                expected_bundle = load(session.get(TestCaseExpected, case.id).value)
                expected_output = expected_bundle["attention_output"]
                expected_weights = expected_bundle["attention_weights"]

                batch_size = hidden_states.shape[0]
                sequence_length = hidden_states.shape[1]
                actual_output = torch.empty_like(hidden_states)
                actual_weights = torch.empty_like(expected_weights)
                print(
                    f"Case {case_index}/{len(cases)} ({case.id}): batch={batch_size}, "
                    f"sequence={sequence_length}; running CUDA",
                    flush=True,
                )
                started = time.perf_counter()
                status = run_attention(
                    tensor_pointer(hidden_states),
                    tensor_pointer(cos),
                    tensor_pointer(sin),
                    tensor_pointer(attention_mask),
                    tensor_pointer(q_weight),
                    tensor_pointer(k_weight),
                    tensor_pointer(v_weight),
                    tensor_pointer(o_weight),
                    tensor_pointer(q_norm),
                    tensor_pointer(k_norm),
                    tensor_pointer(actual_output),
                    tensor_pointer(actual_weights),
                    batch_size,
                    sequence_length,
                    hidden_size,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    rms_norm_eps,
                )
                print(
                    f"CUDA returned status={status} in "
                    f"{time.perf_counter() - started:.2f}s",
                    flush=True,
                )
                if status != 0:
                    raise RuntimeError(
                        f"Qwen3Attention failed for case {case.id}: CUDA status {status}"
                    )

                case_mismatches = []
                comparisons = (
                    ("attention output", actual_output, expected_output),
                    ("attention weights", actual_weights, expected_weights),
                )
                for name, actual, expected in comparisons:
                    try:
                        torch.testing.assert_close(
                            actual, expected, rtol=RTOL, atol=ATOL
                        )
                    except AssertionError as error:
                        detail = f"Case {case.id} {name} mismatch:\n{error}"
                        case_mismatches.append(detail)
                        mismatches.append(detail)
                        print(detail, flush=True)

                if case_mismatches:
                    print(
                        f"Case {case.id}: completed with "
                        f"{len(case_mismatches)} mismatch(es)",
                        flush=True,
                    )
                else:
                    print(f"Case {case.id}: passed", flush=True)

    engine.dispose()
    if mismatches:
        print(
            f"Completed all attention cases; found {len(mismatches)} "
            "numerical mismatch(es).",
            flush=True,
        )
        raise AssertionError(
            f"{len(mismatches)} numerical mismatch(es) across attention cases"
        )


if __name__ == "__main__":
    main()
