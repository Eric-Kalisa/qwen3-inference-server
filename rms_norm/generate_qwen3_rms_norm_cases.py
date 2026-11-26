import sqlite3
import sys
import uuid
from pathlib import Path

import torch
from safetensors.torch import save
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sql_models import ModuleConfig, ModuleInstance, TestCase, TestCaseExpected, TestCaseInput


SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR / "qwen3_rms_norm_cases.db"
HIDDEN_SIZES = [8, 64, 128]
DTYPE = torch.bfloat16
EPSILON = 1e-6


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_num_threads(1)

    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as connection:
        schema_path = SCRIPT_DIR.parent / "schema.sql"
        connection.executescript(schema_path.read_text(encoding="utf-8"))

    engine = create_engine(f"sqlite:///{DB_PATH}")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    with Session(engine) as session:
        for hidden_size in HIDDEN_SIZES:
            module = Qwen3RMSNorm(hidden_size, eps=EPSILON).to(dtype=DTYPE)
            # Non-uniform scale values exercise the learned affine parameter.
            with torch.no_grad():
                module.weight.copy_(1.0 + 0.1 * torch.randn(hidden_size, dtype=DTYPE))
            module.eval()

            module_id = uuid.uuid4().hex
            session.add(
                ModuleInstance(
                    id=module_id,
                    target="transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm",
                    state=save(
                        {name: value.detach().cpu().contiguous()
                         for name, value in module.state_dict().items()}
                    ),
                )
            )
            config_values = [
                ("hidden_size", hidden_size),
                ("eps", EPSILON),
                ("torch_dtype", "bfloat16"),
            ]
            for name, value in config_values:
                session.add(ModuleConfig(module_id=module_id, name=name, value=value))

            for batch_size, sequence_length in [(1, 1), (1, 3), (2, 3)]:
                shape = (batch_size, sequence_length, hidden_size)
                if hidden_size == HIDDEN_SIZES[0] and sequence_length == 1:
                    hidden_states = torch.zeros(shape, dtype=DTYPE)
                else:
                    hidden_states = torch.randn(shape, dtype=DTYPE)

                with torch.inference_mode():
                    expected = module(hidden_states)

                case_id = uuid.uuid4().hex
                session.add(TestCase(id=case_id, module_id=module_id))
                session.flush()
                session.add(
                    TestCaseInput(
                        case_id=case_id,
                        name="hidden_states",
                        value=save({"value": hidden_states.contiguous()}),
                    )
                )
                session.add(
                    TestCaseExpected(
                        case_id=case_id,
                        value=save({"output": expected.contiguous()}),
                    )
                )

        session.commit()

    engine.dispose()
