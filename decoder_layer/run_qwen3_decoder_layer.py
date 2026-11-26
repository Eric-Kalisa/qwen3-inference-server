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

from sql_models import ModuleConfig, ModuleInstance, TestCase, TestCaseExpected, TestCaseInput


SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR / "qwen3_decoder_layer_cases.db"
RTOL = 1e-2
ATOL = 1e-2


def bf16_pointer(tensor):
    return ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctypes.c_uint16))


def load_library(path):
    return ctypes.CDLL(str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DB_PATH)
    parser.add_argument("--library-dir", type=Path, default=SCRIPT_DIR)
    args = parser.parse_args()

    rms_library = load_library(args.library_dir / "libqwen3_rms_norm.so")
    rotary_library = load_library(args.library_dir / "libqwen3_rotary_embedding.so")
    attention_library = load_library(args.library_dir / "libqwen3_attention.so")
    mlp_library = load_library(args.library_dir / "libqwen3_mlp.so")
    decoder_library = load_library(args.library_dir / "libqwen3_decoder_layer.so")

    bf16 = ctypes.POINTER(ctypes.c_uint16)
    int64 = ctypes.POINTER(ctypes.c_int64)
    run_norm = rms_library.qwen3_rms_norm_cuda
    run_norm.argtypes = [bf16, bf16, bf16, ctypes.c_int, ctypes.c_int, ctypes.c_float]
    run_norm.restype = ctypes.c_int
    run_rotary = rotary_library.qwen3_rotary_embedding_cuda
    run_rotary.argtypes = [
        int64, bf16, bf16, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
    ]
    run_rotary.restype = ctypes.c_int
    run_attention = attention_library.qwen3_attention_cuda
    run_attention.argtypes = [
        bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_float,
    ]
    run_attention.restype = ctypes.c_int
    run_mlp = mlp_library.qwen3_mlp_cuda
    run_mlp.argtypes = [bf16, bf16, bf16, bf16, bf16, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    run_mlp.restype = ctypes.c_int
    run_residual = decoder_library.qwen3_residual_add_cuda
    run_residual.argtypes = [bf16, bf16, bf16, ctypes.c_int]
    run_residual.restype = ctypes.c_int

    engine = create_engine(f"sqlite:///{args.database}")
    with Session(engine) as session:
        module_instances = session.scalars(select(ModuleInstance)).all()
        for module_instance in module_instances:
            config_rows = session.scalars(
                select(ModuleConfig).where(ModuleConfig.module_id == module_instance.id)
            ).all()
            config = {row.name: row.value for row in config_rows}
            hidden_size = int(config["hidden_size"])
            intermediate_size = int(config["intermediate_size"])
            num_heads = int(config["num_attention_heads"])
            num_kv_heads = int(config["num_key_value_heads"])
            head_dim = int(config["head_dim"])
            epsilon = float(config["rms_norm_eps"])
            rope_theta = float(config["rope_theta"])
            attention_scaling = float(config["attention_scaling"])

            state = load(module_instance.state)
            input_norm_weight = state["input_layernorm.weight"].contiguous()
            post_norm_weight = state["post_attention_layernorm.weight"].contiguous()
            q_weight = state["self_attn.q_proj.weight"].contiguous()
            k_weight = state["self_attn.k_proj.weight"].contiguous()
            v_weight = state["self_attn.v_proj.weight"].contiguous()
            o_weight = state["self_attn.o_proj.weight"].contiguous()
            q_norm_weight = state["self_attn.q_norm.weight"].contiguous()
            k_norm_weight = state["self_attn.k_norm.weight"].contiguous()
            gate_weight = state["mlp.gate_proj.weight"].contiguous()
            up_weight = state["mlp.up_proj.weight"].contiguous()
            down_weight = state["mlp.down_proj.weight"].contiguous()
            cases = session.scalars(
                select(TestCase).where(TestCase.module_id == module_instance.id)
            ).all()
            print(
                f"Module {module_instance.id}: hidden={hidden_size}, "
                f"intermediate={intermediate_size}, {len(cases)} case(s)",
                flush=True,
            )

            for case_index, case in enumerate(cases, start=1):
                rows = session.scalars(
                    select(TestCaseInput).where(TestCaseInput.case_id == case.id)
                ).all()
                inputs = {row.name: load(row.value)["value"] for row in rows}
                hidden_states = inputs["hidden_states"].contiguous()
                position_ids = inputs["position_ids"].to(torch.int64).contiguous()
                attention_mask = inputs["attention_mask"].contiguous()
                batch_size, sequence_length, _ = hidden_states.shape
                row_count = batch_size * sequence_length

                normalized = torch.empty_like(hidden_states)
                cos = torch.empty(
                    (batch_size, sequence_length, head_dim), dtype=torch.bfloat16
                )
                sin = torch.empty_like(cos)
                attention_output = torch.empty_like(hidden_states)
                attention_weights = torch.empty(
                    (batch_size, num_heads, sequence_length, sequence_length),
                    dtype=torch.bfloat16,
                )
                after_attention = torch.empty_like(hidden_states)
                mlp_input = torch.empty_like(hidden_states)
                mlp_output = torch.empty_like(hidden_states)
                actual = torch.empty_like(hidden_states)

                print(
                    f"Case {case_index}/{len(cases)} ({case.id}): "
                    f"batch={batch_size}, sequence={sequence_length}",
                    flush=True,
                )
                started = time.perf_counter()
                status = run_norm(
                    bf16_pointer(hidden_states), bf16_pointer(input_norm_weight),
                    bf16_pointer(normalized), row_count, hidden_size, epsilon,
                )
                if status != 0:
                    raise RuntimeError(f"Input RMSNorm failed for {case.id}: {status}")
                status = run_rotary(
                    ctypes.cast(position_ids.data_ptr(), int64),
                    bf16_pointer(cos), bf16_pointer(sin), row_count, head_dim,
                    rope_theta, attention_scaling,
                )
                if status != 0:
                    raise RuntimeError(f"Rotary embedding failed for {case.id}: {status}")
                status = run_attention(
                    bf16_pointer(normalized), bf16_pointer(cos), bf16_pointer(sin),
                    bf16_pointer(attention_mask), bf16_pointer(q_weight),
                    bf16_pointer(k_weight), bf16_pointer(v_weight), bf16_pointer(o_weight),
                    bf16_pointer(q_norm_weight), bf16_pointer(k_norm_weight),
                    bf16_pointer(attention_output), bf16_pointer(attention_weights),
                    batch_size, sequence_length, hidden_size, num_heads, num_kv_heads,
                    head_dim, epsilon,
                )
                if status != 0:
                    raise RuntimeError(f"Self attention failed for {case.id}: {status}")
                status = run_residual(
                    bf16_pointer(hidden_states), bf16_pointer(attention_output),
                    bf16_pointer(after_attention), after_attention.numel(),
                )
                if status != 0:
                    raise RuntimeError(f"Attention residual failed for {case.id}: {status}")
                status = run_norm(
                    bf16_pointer(after_attention), bf16_pointer(post_norm_weight),
                    bf16_pointer(mlp_input), row_count, hidden_size, epsilon,
                )
                if status != 0:
                    raise RuntimeError(f"Post-attention RMSNorm failed for {case.id}: {status}")
                status = run_mlp(
                    bf16_pointer(mlp_input), bf16_pointer(gate_weight),
                    bf16_pointer(up_weight), bf16_pointer(down_weight),
                    bf16_pointer(mlp_output), row_count, hidden_size, intermediate_size,
                )
                if status != 0:
                    raise RuntimeError(f"MLP failed for {case.id}: {status}")
                status = run_residual(
                    bf16_pointer(after_attention), bf16_pointer(mlp_output),
                    bf16_pointer(actual), actual.numel(),
                )
                if status != 0:
                    raise RuntimeError(f"MLP residual failed for {case.id}: {status}")

                expected = load(session.get(TestCaseExpected, case.id).value)["output"]
                torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
                print(
                    f"Case {case.id}: passed in {time.perf_counter() - started:.2f}s",
                    flush=True,
                )

    engine.dispose()


if __name__ == "__main__":
    main()
