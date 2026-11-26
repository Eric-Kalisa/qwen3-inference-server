import sqlite3
import sys
import uuid
from pathlib import Path

import torch
from safetensors.torch import save
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sql_models import ModuleConfig, ModuleInstance, TestCase, TestCaseExpected, TestCaseInput


SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR / "qwen3_rotary_embedding_cases.db"
HEAD_DIMS = [8, 16, 32]
DTYPE = torch.bfloat16


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
        for head_dim in HEAD_DIMS:
            config = Qwen3Config(
                hidden_size=head_dim,
                num_attention_heads=1,
                num_key_value_heads=1,
                head_dim=head_dim,
            )
            module = Qwen3RotaryEmbedding(config)
            # RoPE has no trainable state; an empty Safetensors state keeps the
            # shared module_instance contract while all behavior is in config+input.
            module_id = uuid.uuid4().hex
            session.add(
                ModuleInstance(
                    id=module_id,
                    target="transformers.models.qwen3.modeling_qwen3.Qwen3RotaryEmbedding",
                    state=save({}),
                )
            )
            config_values = [
                ("head_dim", head_dim),
                ("rope_theta", float(config.rope_parameters["rope_theta"])),
                ("rope_type", str(config.rope_parameters["rope_type"])),
                ("attention_scaling", float(module.attention_scaling)),
                ("torch_dtype", "bfloat16"),
            ]
            for name, value in config_values:
                session.add(ModuleConfig(module_id=module_id, name=name, value=value))

            for batch_size, sequence_length, position_offset in [
                (1, 1, 0), (1, 4, 0), (2, 3, 7),
            ]:
                hidden_states = torch.randn(
                    batch_size, sequence_length, head_dim, dtype=DTYPE
                )
                position_ids = torch.arange(sequence_length).expand(batch_size, -1)
                position_ids = position_ids + position_offset
                with torch.inference_mode():
                    cos, sin = module(hidden_states, position_ids)

                case_id = uuid.uuid4().hex
                session.add(TestCase(id=case_id, module_id=module_id))
                session.flush()
                session.add_all(
                    [
                        TestCaseInput(
                            case_id=case_id,
                            name="hidden_states",
                            value=save({"value": hidden_states.contiguous()}),
                        ),
                        TestCaseInput(
                            case_id=case_id,
                            name="position_ids",
                            value=save({"value": position_ids.to(torch.int64).contiguous()}),
                        ),
                        TestCaseExpected(
                            case_id=case_id,
                            value=save({"cos": cos.contiguous(), "sin": sin.contiguous()}),
                        ),
                    ]
                )

        session.commit()

    engine.dispose()
