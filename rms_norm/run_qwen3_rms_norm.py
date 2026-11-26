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
DB_PATH = SCRIPT_DIR / "qwen3_rms_norm_cases.db"
LIBRARY_PATH = SCRIPT_DIR / "libqwen3_rms_norm.so"
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
    run_rms_norm = library.qwen3_rms_norm_cuda
    bf16_pointer = ctypes.POINTER(ctypes.c_uint16)
    run_rms_norm.argtypes = [
        bf16_pointer, bf16_pointer, bf16_pointer,
        ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ]
    run_rms_norm.restype = ctypes.c_int

    with Session(engine) as session:
        module_instances = session.scalars(select(ModuleInstance)).all()
        for module_instance in module_instances:
            config_rows = session.scalars(
                select(ModuleConfig).where(ModuleConfig.module_id == module_instance.id)
            ).all()
            config = {row.name: row.value for row in config_rows}
            hidden_size = int(config["hidden_size"])
            epsilon = float(config["eps"])
            state = load(module_instance.state)
            weight = state["weight"].contiguous()
            cases = session.scalars(
                select(TestCase).where(TestCase.module_id == module_instance.id)
            ).all()
            print(f"Module {module_instance.id}: hidden_size={hidden_size}", flush=True)

            for case in cases:
                row = session.scalars(
                    select(TestCaseInput).where(TestCaseInput.case_id == case.id)
                ).one()
                hidden_states = load(row.value)["value"].contiguous()
                expected = load(session.get(TestCaseExpected, case.id).value)["output"]
                rows = hidden_states.numel() // hidden_size
                actual = torch.empty_like(hidden_states)
                status = run_rms_norm(
                    tensor_pointer(hidden_states), tensor_pointer(weight),
                    tensor_pointer(actual), rows, hidden_size, epsilon,
                )
                if status != 0:
                    raise RuntimeError(f"Qwen3RMSNorm failed for {case.id}: {status}")
                torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
                print(f"Case {case.id}: passed", flush=True)

    engine.dispose()


if __name__ == "__main__":
    main()
