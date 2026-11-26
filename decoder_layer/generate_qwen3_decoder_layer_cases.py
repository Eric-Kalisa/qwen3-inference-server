import sqlite3
import sys
import uuid
from pathlib import Path

import torch
from safetensors.torch import save
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sql_models import ModuleConfig, ModuleInstance, TestCase, TestCaseExpected, TestCaseInput


SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR / "qwen3_decoder_layer_cases.db"
MODULE_SIZES = [(8, 2, 1, 4), (64, 4, 2, 16), (128, 8, 4, 16)]
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
        for hidden_size, num_heads, num_kv_heads, head_dim in MODULE_SIZES:
            config = Qwen3Config(
                hidden_size=hidden_size,
                intermediate_size=hidden_size * 2,
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                head_dim=head_dim,
                num_hidden_layers=1,
                attention_bias=False,
                attention_dropout=0.0,
                _attn_implementation="eager",
            )
            config._attn_implementation = "eager"
            module = Qwen3DecoderLayer(config, layer_idx=0).to(dtype=DTYPE)
            module.eval()
            rotary = Qwen3RotaryEmbedding(config)

            module_id = uuid.uuid4().hex
            state = {
                name: value.detach().cpu().contiguous()
                for name, value in module.state_dict().items()
            }
            session.add(
                ModuleInstance(
                    id=module_id,
                    target="transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer",
                    state=save(state),
                )
            )
            config_values = [
                ("hidden_size", hidden_size),
                ("intermediate_size", hidden_size * 2),
                ("num_attention_heads", num_heads),
                ("num_key_value_heads", num_kv_heads),
                ("head_dim", head_dim),
                ("rms_norm_eps", config.rms_norm_eps),
                ("attention_bias", False),
                ("attention_dropout", 0.0),
                ("attn_implementation", "eager"),
                ("rope_theta", float(config.rope_parameters["rope_theta"])),
                ("rope_type", str(config.rope_parameters["rope_type"])),
                ("attention_scaling", float(rotary.attention_scaling)),
                ("torch_dtype", "bfloat16"),
            ]
            for name, value in config_values:
                session.add(ModuleConfig(module_id=module_id, name=name, value=value))

            for batch_size, sequence_length in [(1, 1), (1, 3), (2, 3)]:
                shape = (batch_size, sequence_length, hidden_size)
                if hidden_size == 8 and sequence_length == 1:
                    hidden_states = torch.zeros(shape, dtype=DTYPE)
                else:
                    hidden_states = torch.randn(shape, dtype=DTYPE)
                position_ids = torch.arange(sequence_length).expand(batch_size, -1)
                position_embeddings = rotary(hidden_states, position_ids)
                attention_mask = torch.triu(
                    torch.full(
                        (sequence_length, sequence_length),
                        float("-inf"),
                        dtype=DTYPE,
                    ),
                    diagonal=1,
                )
                attention_mask = attention_mask.expand(
                    batch_size, 1, sequence_length, sequence_length
                ).contiguous()

                with torch.inference_mode():
                    expected = module(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                        position_embeddings=position_embeddings,
                    )

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
                        TestCaseInput(
                            case_id=case_id,
                            name="attention_mask",
                            value=save({"value": attention_mask}),
                        ),
                        TestCaseExpected(
                            case_id=case_id,
                            value=save({"output": expected.contiguous()}),
                        ),
                    ]
                )

        session.commit()

    engine.dispose()
