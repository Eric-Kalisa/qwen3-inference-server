import sqlite3
import sys
import uuid
from pathlib import Path

import torch
from safetensors.torch import save
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

# Reuse Week_03's shared schema models rather than creating MLP-specific copies.
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
# Each size gets its own initialized module; its invocation cases share those saved weights.
MODULE_SIZES = [(8, 16), (64, 128), (128, 256)]
# BF16 matches the intended Qwen execution precision at the linear/activation boundaries.
DTYPE = torch.bfloat16
ACTIVATION = "silu"


if __name__ == "__main__":
    # Fixed seed makes generated random weights and inputs repeatable.
    torch.manual_seed(0)
    # Tiny reference matrices are faster without launching many CPU worker threads.
    torch.set_num_threads(1)

    # Rebuild the generated database so rerunning this script replaces old cases.
    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as connection:
        # Keep the common relational schema in Week_03; only MLP artifacts live here.
        schema_path = SCRIPT_DIR.parent / "schema.sql"
        connection.executescript(schema_path.read_text(encoding="utf-8"))

    engine = create_engine(f"sqlite:///{DB_PATH}")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    with Session(engine) as session:
        for hidden_size, intermediate_size in MODULE_SIZES:
            config = Qwen3Config(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=ACTIVATION,
            )
            module = Qwen3MLP(config).to(dtype=DTYPE)
            module.eval()

            # Save one concrete state per size. Invocation cases below reference
            # this module so every expected output uses exactly the same weights.
            state = {}
            for name, tensor in module.state_dict().items():
                state[name] = tensor.detach().cpu().contiguous()

            module_id = uuid.uuid4().hex
            module_instance = ModuleInstance(
                id=module_id,
                target="transformers.models.qwen3.modeling_qwen3.Qwen3MLP",
                state=save(state),
            )
            session.add(module_instance)
            session.add_all(
                [
                    ModuleConfig(module_id=module_id, name="hidden_size", value=hidden_size),
                    ModuleConfig(
                        module_id=module_id,
                        name="intermediate_size",
                        value=intermediate_size,
                    ),
                    ModuleConfig(module_id=module_id, name="hidden_act", value=ACTIVATION),
                    ModuleConfig(module_id=module_id, name="torch_dtype", value="bfloat16"),
                ]
            )

            case_shapes = [
                (1, hidden_size),
                (2, hidden_size),
                (2, 3, hidden_size),
            ]
            for shape in case_shapes:
                # The smallest [1, 8] input is zero; all other cases are random.
                if hidden_size == 8 and shape == (1, 8):
                    inputs = torch.zeros(shape, dtype=DTYPE)
                else:
                    inputs = torch.randn(shape, dtype=DTYPE)

                with torch.inference_mode():
                    # Store the actual module result as the reference for this
                    # input and the module's saved weights.
                    expected = module(inputs)

                case_id = uuid.uuid4().hex
                case = TestCase(id=case_id, module_id=module_id)
                case_input = TestCaseInput(
                    case_id=case_id,
                    name="x",
                    value=save({"x": inputs.contiguous()}),
                )
                case_expected = TestCaseExpected(
                    case_id=case_id,
                    value=save({"output": expected.contiguous()}),
                )
                session.add(case)
                session.flush()
                session.add_all([case_input, case_expected])

        session.commit()

    engine.dispose()
