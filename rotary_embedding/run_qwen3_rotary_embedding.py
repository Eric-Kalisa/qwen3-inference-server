import argparse
import ctypes
from pathlib import Path
import sys

import torch
from safetensors.torch import load
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sql_models import ModuleConfig, ModuleInstance, TestCase, TestCaseExpected, TestCaseInput


SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR / "qwen3_rotary_embedding_cases.db"
LIBRARY_PATH = SCRIPT_DIR / "libqwen3_rotary_embedding.so"
RTOL = 1e-2
ATOL = 1e-2


def tensor_pointer(tensor):
    return ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_uint16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DB_PATH)
    parser.add_argument("--library", type=Path, default=LIBRARY_PATH)
    args = parser.parse_args()

    engine = create_engine(f"sqlite:///{args.database}")
    library = ctypes.CDLL(str(args.library))
    run_rotary = library.qwen3_rotary_embedding_cuda
    position_pointer = ctypes.POINTER(ctypes.c_int64)
    bf16_pointer = ctypes.POINTER(ctypes.c_uint16)
    run_rotary.argtypes = [
        position_pointer, bf16_pointer, bf16_pointer,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
    ]
    run_rotary.restype = ctypes.c_int

    with Session(engine) as session:
        module_instances = session.scalars(select(ModuleInstance)).all()
        for module_instance in module_instances:
            config_rows = session.scalars(
                select(ModuleConfig).where(ModuleConfig.module_id == module_instance.id)
            ).all()
            config = {row.name: row.value for row in config_rows}
            head_dim = int(config["head_dim"])
            rope_theta = float(config["rope_theta"])
            attention_scaling = float(config["attention_scaling"])
            cases = session.scalars(
                select(TestCase).where(TestCase.module_id == module_instance.id)
            ).all()
            print(f"Module {module_instance.id}: head_dim={head_dim}", flush=True)

            for case in cases:
                inputs = session.scalars(
                    select(TestCaseInput).where(TestCaseInput.case_id == case.id)
                ).all()
                input_values = {row.name: load(row.value)["value"] for row in inputs}
                hidden_states = input_values["hidden_states"]
                position_ids = input_values["position_ids"].to(torch.int64).contiguous()
                batch_size, sequence_length = hidden_states.shape[:2]
                rows = batch_size * sequence_length
                cos = torch.empty((batch_size, sequence_length, head_dim), dtype=torch.bfloat16)
                sin = torch.empty_like(cos)
                status = run_rotary(
                    ctypes.cast(position_ids.data_ptr(), position_pointer),
                    tensor_pointer(cos), tensor_pointer(sin), rows, head_dim,
                    rope_theta, attention_scaling,
                )
                if status != 0:
                    raise RuntimeError(f"Qwen3RotaryEmbedding failed for {case.id}: {status}")

                expected = load(session.get(TestCaseExpected, case.id).value)
                torch.testing.assert_close(cos, expected["cos"], rtol=RTOL, atol=ATOL)
                torch.testing.assert_close(sin, expected["sin"], rtol=RTOL, atol=ATOL)
                print(f"Case {case.id}: passed", flush=True)

    engine.dispose()


if __name__ == "__main__":
    main()
