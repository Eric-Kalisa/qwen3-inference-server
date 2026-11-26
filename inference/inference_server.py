import argparse
import copy
from http.server import BaseHTTPRequestHandler, HTTPServer
import ctypes
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM


MAX_SEQUENCE_LENGTH = 128
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
QWEN3_4B_MODEL = "Qwen/Qwen3-4B"
LOGIT_ATOL = 0.1
QWEN3_4B_LOGIT_ATOL = 5e-1
LOGIT_RTOL = 0.02
LOGIT_MISMATCH_LIMIT = 10
BF16_POINTER = ctypes.POINTER(ctypes.c_uint16)
INT64_POINTER = ctypes.POINTER(ctypes.c_int64)
def bf16_pointer(tensor):
    return ctypes.cast(tensor.data_ptr(), BF16_POINTER)


class CppTokenizer:
    def __init__(self, library_path, tokenizer_directory):
        self.library = ctypes.CDLL(str(library_path))
        self.library.qwen3_tokenizer_load.argtypes = [ctypes.c_char_p]
        self.library.qwen3_tokenizer_load.restype = ctypes.c_void_p
        self.library.qwen3_tokenizer_free.argtypes = [ctypes.c_void_p]
        self.library.qwen3_tokenizer_encode.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ]
        self.library.qwen3_tokenizer_encode.restype = ctypes.c_int
        self.library.qwen3_tokenizer_decode.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int,
        ]
        self.library.qwen3_tokenizer_decode.restype = ctypes.c_int
        self.handle = self.library.qwen3_tokenizer_load(
            str(tokenizer_directory).encode("utf-8")
        )
        if not self.handle:
            raise RuntimeError(f"Could not load tokenizer from {tokenizer_directory}")

    def encode(self, text):
        encoded_text = text.encode("utf-8")
        count = self.library.qwen3_tokenizer_encode(
            self.handle, encoded_text, None, 0
        )
        token_ids = (ctypes.c_int * count)()
        self.library.qwen3_tokenizer_encode(
            self.handle, encoded_text, token_ids, count
        )
        return list(token_ids)

    def decode(self, token_ids):
        if not token_ids:
            return ""
        ids = (ctypes.c_int * len(token_ids))(*token_ids)
        text_size = self.library.qwen3_tokenizer_decode(
            self.handle, ids, len(token_ids), None, 0
        )
        text = ctypes.create_string_buffer(text_size + 1)
        self.library.qwen3_tokenizer_decode(
            self.handle, ids, len(token_ids), text, len(text)
        )
        return text.value.decode("utf-8")

    def close(self):
        if self.handle:
            self.library.qwen3_tokenizer_free(self.handle)
            self.handle = None


class CudaModules:
    def __init__(self, library_directory):
        self.rms_library = ctypes.CDLL(str(library_directory / "libqwen3_rms_norm.so"))
        self.rotary_library = ctypes.CDLL(
            str(library_directory / "libqwen3_rotary_embedding.so")
        )
        self.attention_library = ctypes.CDLL(
            str(library_directory / "libqwen3_attention.so")
        )
        self.mlp_library = ctypes.CDLL(str(library_directory / "libqwen3_mlp.so"))
        self.decoder_library = ctypes.CDLL(
            str(library_directory / "libqwen3_decoder_layer.so")
        )
        self.inference_library = ctypes.CDLL(
            str(library_directory / "libqwen3_inference.so")
        )

        self.run_norm = self.rms_library.qwen3_rms_norm_cuda
        self.run_norm.argtypes = [
            BF16_POINTER, BF16_POINTER, BF16_POINTER,
            ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ]
        self.run_norm.restype = ctypes.c_int

        self.run_rotary = self.rotary_library.qwen3_rotary_embedding_cuda
        self.run_rotary.argtypes = [
            INT64_POINTER, BF16_POINTER, BF16_POINTER,
            ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
        ]
        self.run_rotary.restype = ctypes.c_int

        self.run_attention = self.attention_library.qwen3_attention_cuda
        self.run_attention.argtypes = [
            BF16_POINTER, BF16_POINTER, BF16_POINTER, BF16_POINTER,
            BF16_POINTER, BF16_POINTER, BF16_POINTER, BF16_POINTER,
            BF16_POINTER, BF16_POINTER, BF16_POINTER, BF16_POINTER,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ]
        self.run_attention.restype = ctypes.c_int
        self.run_mlp = self.mlp_library.qwen3_mlp_cuda
        self.run_mlp.argtypes = [
            BF16_POINTER, BF16_POINTER, BF16_POINTER, BF16_POINTER,
            BF16_POINTER, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.run_mlp.restype = ctypes.c_int
        self.run_residual = self.decoder_library.qwen3_residual_add_cuda
        self.run_residual.argtypes = [BF16_POINTER, BF16_POINTER, BF16_POINTER, ctypes.c_int]
        self.run_residual.restype = ctypes.c_int

        self.run_embedding = self.inference_library.qwen3_embedding_cuda
        self.run_embedding.argtypes = [
            INT64_POINTER, BF16_POINTER, BF16_POINTER,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.run_embedding.restype = ctypes.c_int

        self.run_vocab_projection = self.inference_library.qwen3_vocab_projection_cuda
        self.run_vocab_projection.argtypes = [
            BF16_POINTER, BF16_POINTER, BF16_POINTER, ctypes.c_int, ctypes.c_int,
        ]
        self.run_vocab_projection.restype = ctypes.c_int


class Qwen3Inference:
    def __init__(self, model_id, library_directory, tokenizer_directory=None):
        self.logit_atol = QWEN3_4B_LOGIT_ATOL if model_id == QWEN3_4B_MODEL else LOGIT_ATOL
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            # The existing attention library implements eager full attention.
            attn_implementation="eager",
        )
        self.model.eval()
        self.reference_model = copy.deepcopy(self.model).to("cuda")
        self.reference_model.eval()
        self.config = self.model.config

        if tokenizer_directory is None:
            model_path = Path(model_id)
            if model_path.is_dir():
                tokenizer_directory = model_path
            else:
                hf_hub_download(model_id, "tokenizer.json")
                tokenizer_config = hf_hub_download(model_id, "tokenizer_config.json")
                tokenizer_directory = Path(tokenizer_config).parent

        self.tokenizer = CppTokenizer(
            library_directory / "libqwen3_tokenizer.so", tokenizer_directory
        )
        self.cuda = CudaModules(library_directory)
        generation_eos = self.model.generation_config.eos_token_id
        if isinstance(generation_eos, int):
            generation_eos = [generation_eos]
        self.eos_token_ids = set(generation_eos or [])

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def check_hidden_state_positions(self, prompt):
        token_ids = self.encode(prompt)
        sequence_length = len(token_ids)
        hidden_size = self.config.hidden_size
        input_ids_cpu = torch.tensor(token_ids, dtype=torch.int64).reshape(1, -1)
        cuda_hidden = torch.empty(
            (1, sequence_length, hidden_size), dtype=torch.bfloat16
        )

        status = self.cuda.run_embedding(
            ctypes.cast(input_ids_cpu.data_ptr(), INT64_POINTER),
            bf16_pointer(self.model.model.embed_tokens.weight),
            bf16_pointer(cuda_hidden),
            sequence_length,
            self.config.vocab_size,
            hidden_size,
        )
        if status != 0:
            raise RuntimeError(f"Token embedding CUDA call failed: {status}")

        input_ids_gpu = input_ids_cpu.to("cuda")
        with torch.inference_mode():
            reference_hidden = self.reference_model.model.embed_tokens(input_ids_gpu)
        reference_hidden = reference_hidden.cpu()

        differences = (cuda_hidden.float() - reference_hidden.float()).abs()
        position_differences = differences.amax(dim=2)[0]
        mismatch_count = int(torch.count_nonzero(position_differences).item())
        max_abs_diff = float(differences.max().item())

        print(f"Input token count: {sequence_length}")
        print(f"CUDA hidden-state shape: {tuple(cuda_hidden.shape)}")
        print(f"HF hidden-state shape: {tuple(reference_hidden.shape)}")
        for position, token_id in enumerate(token_ids):
            print(
                f"position={position}, token_id={token_id}, "
                f"max_abs_diff={float(position_differences[position]):.8g}"
            )
        print(
            f"Positions differing: {mismatch_count}/{sequence_length}; "
            f"overall max_abs_diff={max_abs_diff:.8g}"
        )
        if mismatch_count != 0:
            raise AssertionError("CUDA embedding rows do not match HF positions")

        return {
            "token_count": sequence_length,
            "shape": tuple(cuda_hidden.shape),
            "mismatch_count": mismatch_count,
            "max_abs_diff": max_abs_diff,
        }

    def diagnose_layer_hidden_states(self, prompt):
        token_ids = self.encode(prompt)
        if len(token_ids) > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"The current attention kernel supports at most "
                f"{MAX_SEQUENCE_LENGTH} tokens per sequence"
            )
        input_ids = torch.tensor(
            [token_ids], dtype=torch.int64, device="cuda"
        )
        with torch.inference_mode():
            reference = self.reference_model.model(
                input_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

        config = self.config
        print(
            f"Input: batch=1, sequence={len(token_ids)}, "
            f"positions=0..{len(token_ids) - 1}, dtype={self.model.dtype}"
        )
        print(
            f"Model sizes: hidden={config.hidden_size}, "
            f"intermediate={config.intermediate_size}, "
            f"attention_heads={config.num_attention_heads}, "
            f"kv_heads={config.num_key_value_heads}, head_dim={config.head_dim}"
        )
        self.forward_logits(token_ids, reference.hidden_states)
        print("All layer hidden-state boundaries are within tolerance")

    def compare_generation(self, prompt, max_new_tokens):
        initial_token_ids = self.encode(prompt)
        if len(initial_token_ids) + max_new_tokens > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"Prompt plus generation limit must not exceed "
                f"{MAX_SEQUENCE_LENGTH} tokens"
            )

        cuda_token_ids = initial_token_ids.copy()
        pytorch_token_ids = initial_token_ids.copy()
        cuda_generated = []
        pytorch_generated = []
        step_results = []
        cuda_active = True
        pytorch_active = True

        for step in range(max_new_tokens):
            same_input = (
                cuda_active
                and pytorch_active
                and cuda_token_ids == pytorch_token_ids
            )
            cuda_logits = None
            pytorch_logits = None
            cuda_next_token = None
            pytorch_next_token = None

            if cuda_active:
                cuda_logits = self.forward_logits(cuda_token_ids)
                cuda_next_token = int(cuda_logits[0].argmax().item())
                cuda_token_ids.append(cuda_next_token)
                cuda_generated.append(cuda_next_token)
                if cuda_next_token in self.eos_token_ids:
                    cuda_active = False

            if pytorch_active:
                input_ids = torch.tensor(
                    [pytorch_token_ids], dtype=torch.int64, device="cuda"
                )
                with torch.inference_mode():
                    pytorch_logits = self.reference_model(
                        input_ids, use_cache=False
                    ).logits[:, -1, :]
                pytorch_next_token = int(pytorch_logits[0].argmax().item())
                pytorch_token_ids.append(pytorch_next_token)
                pytorch_generated.append(pytorch_next_token)
                if pytorch_next_token in self.eos_token_ids:
                    pytorch_active = False

            step_result = {
                "step": step + 1,
                "cuda_token_id": cuda_next_token,
                "pytorch_token_id": pytorch_next_token,
                "logits": None,
            }
            if same_input:
                step_result["logits"] = self.compare_logits(
                    cuda_logits, pytorch_logits
                )
            step_results.append(step_result)
            if not cuda_active and not pytorch_active:
                break

        return {
            "prompt_tokens": len(initial_token_ids),
            "cuda": {
                "token_ids": cuda_generated,
                "text": self.decode(cuda_generated),
            },
            "pytorch": {
                "token_ids": pytorch_generated,
                "text": self.decode(pytorch_generated),
            },
            "steps": step_results,
        }

    def compare_logits(self, cuda_logits, pytorch_logits):
        cuda_values = cuda_logits.float().cpu()
        pytorch_values = pytorch_logits.float().cpu()
        differences = (cuda_values - pytorch_values).abs()
        tolerance = self.logit_atol + LOGIT_RTOL * pytorch_values.abs()
        mismatch_mask = differences > tolerance
        mismatch_indices = mismatch_mask[0].nonzero().flatten()
        mismatch_differences = differences[0, mismatch_indices]
        count = min(LOGIT_MISMATCH_LIMIT, mismatch_indices.numel())
        if count:
            worst = torch.topk(mismatch_differences, count).indices
            worst_indices = mismatch_indices[worst]
            worst_logits = [
                {
                    "index": int(index),
                    "cuda": float(cuda_values[0, index]),
                    "pytorch": float(pytorch_values[0, index]),
                    "abs_diff": float(differences[0, index]),
                }
                for index in worst_indices
            ]
        else:
            worst_logits = []
        return {
            "compared": cuda_values.numel(),
            "outside_tolerance": int(mismatch_mask.sum()),
            "max_abs_diff": float(differences.max()),
            "atol": self.logit_atol,
            "rtol": LOGIT_RTOL,
            "largest_mismatches": worst_logits,
        }

    @staticmethod
    def compare_hidden_states(actual, expected, stage):
        if actual.shape != expected.shape:
            print(
                f"FIRST_DIFFERENCE {stage}: shape mismatch; "
                f"CUDA={tuple(actual.shape)}, HF={tuple(expected.shape)}",
                flush=True,
            )
            raise AssertionError(f"Hidden-state shape mismatch at {stage}")

        actual_values = actual.float().cpu()
        expected_values = expected.float().cpu()
        differences = (actual_values - expected_values).abs()
        sort_differences = torch.nan_to_num(
            differences, nan=float("inf"), posinf=float("inf"), neginf=float("inf")
        )
        tolerance = LOGIT_ATOL + LOGIT_RTOL * expected_values.abs()
        outside = (differences > tolerance) | ~torch.isfinite(actual_values)
        outside |= ~torch.isfinite(expected_values)
        outside_count = int(torch.count_nonzero(outside).item())
        max_index = tuple(
            int(value)
            for value in torch.unravel_index(sort_differences.argmax(), differences.shape)
        )
        max_difference = float(differences[max_index].item())
        print(
            f"DIAGNOSTIC {stage}: shape={tuple(actual.shape)}, "
            f"outside_tolerance={outside_count}/{differences.numel()}, "
            f"max_abs_diff={max_difference:.8g}, index={max_index}, "
            f"CUDA={float(actual_values[max_index]):.8g}, "
            f"HF={float(expected_values[max_index]):.8g}, "
            f"atol={LOGIT_ATOL}, rtol={LOGIT_RTOL}",
            flush=True,
        )
        if outside_count:
            first_index = tuple(int(value) for value in torch.nonzero(outside)[0])
            print(
                f"FIRST_DIFFERENCE {stage}: index={first_index}, "
                f"CUDA={float(actual_values[first_index]):.8g}, "
                f"HF={float(expected_values[first_index]):.8g}, "
                f"abs_diff={float(differences[first_index]):.8g}, "
                f"allowed={float(tolerance[first_index]):.8g}",
                flush=True,
            )
        return outside_count

    def forward_logits(self, token_ids, expected_hidden_states=None):
        batch_size = 1
        sequence_length = len(token_ids)
        if sequence_length > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"The current attention kernel supports at most "
                f"{MAX_SEQUENCE_LENGTH} tokens per sequence"
            )

        model = self.model
        config = self.config
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        rows = batch_size * sequence_length
        input_ids = torch.tensor(token_ids, dtype=torch.int64).reshape(batch_size, -1)
        hidden_states = torch.empty(
            (batch_size, sequence_length, hidden_size), dtype=torch.bfloat16
        )
        status = self.cuda.run_embedding(
            ctypes.cast(input_ids.data_ptr(), INT64_POINTER),
            bf16_pointer(model.model.embed_tokens.weight),
            bf16_pointer(hidden_states),
            rows,
            config.vocab_size,
            hidden_size,
        )
        if status != 0:
            raise RuntimeError(f"Token embedding CUDA call failed: {status}")
        position_ids = torch.arange(sequence_length, dtype=torch.int64).reshape(1, -1)
        cos = torch.empty((batch_size, sequence_length, head_dim), dtype=torch.bfloat16)
        sin = torch.empty_like(cos)
        status = self.cuda.run_rotary(
            ctypes.cast(position_ids.data_ptr(), INT64_POINTER),
            bf16_pointer(cos), bf16_pointer(sin), rows, head_dim,
            float(config.rope_parameters["rope_theta"]),
            float(model.model.rotary_emb.attention_scaling),
        )
        if status != 0:
            raise RuntimeError(f"Rotary embedding CUDA call failed: {status}")
        attention_mask = torch.triu(
            torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                dtype=torch.bfloat16,
            ),
            diagonal=1,
        )
        attention_mask = attention_mask.reshape(1, 1, sequence_length, sequence_length)

        if expected_hidden_states is not None:
            outside_count = self.compare_hidden_states(
                hidden_states, expected_hidden_states[0], "embedding.output"
            )
            if outside_count:
                raise AssertionError("Embedding output exceeded tolerance")

        for layer_index, layer in enumerate(model.model.layers):
            if expected_hidden_states is not None:
                q_size = config.num_attention_heads * config.head_dim
                kv_size = config.num_key_value_heads * config.head_dim
                print(
                    f"BEFORE layer={layer_index}: hidden_state="
                    f"{tuple(hidden_states.shape)}, hidden={hidden_size}, "
                    f"intermediate={intermediate_size}, "
                    f"q_projection={hidden_size}->{q_size}, "
                    f"k_projection={hidden_size}->{kv_size}, "
                    f"v_projection={hidden_size}->{kv_size}, "
                    f"o_projection={q_size}->{hidden_size}, "
                    f"gate/up={hidden_size}->{intermediate_size}, "
                    f"down={intermediate_size}->{hidden_size}",
                    flush=True,
                )
                outside_count = self.compare_hidden_states(
                    hidden_states,
                    expected_hidden_states[layer_index],
                    f"layer.{layer_index}.input",
                )
                if outside_count:
                    raise AssertionError(
                        f"Layer {layer_index} input exceeded tolerance"
                    )
            residual = hidden_states
            normalized = torch.empty_like(hidden_states)
            status = self.cuda.run_norm(
                bf16_pointer(hidden_states),
                bf16_pointer(layer.input_layernorm.weight),
                bf16_pointer(normalized),
                rows,
                hidden_size,
                float(config.rms_norm_eps),
            )
            if status != 0:
                raise RuntimeError(f"Input RMSNorm CUDA call failed: {status}")
            attention_output = torch.empty_like(hidden_states)
            attention_weights = torch.empty(
                (batch_size, num_heads, sequence_length, sequence_length),
                dtype=torch.bfloat16,
            )
            attention = layer.self_attn
            status = self.cuda.run_attention(
                bf16_pointer(normalized), bf16_pointer(cos), bf16_pointer(sin),
                bf16_pointer(attention_mask),
                bf16_pointer(attention.q_proj.weight),
                bf16_pointer(attention.k_proj.weight),
                bf16_pointer(attention.v_proj.weight),
                bf16_pointer(attention.o_proj.weight),
                bf16_pointer(attention.q_norm.weight),
                bf16_pointer(attention.k_norm.weight),
                bf16_pointer(attention_output), bf16_pointer(attention_weights),
                batch_size, sequence_length, hidden_size,
                num_heads, num_kv_heads, head_dim, float(config.rms_norm_eps),
            )
            if status != 0:
                raise RuntimeError(f"Attention CUDA call failed: {status}")
            after_attention = torch.empty_like(hidden_states)
            status = self.cuda.run_residual(
                bf16_pointer(residual), bf16_pointer(attention_output),
                bf16_pointer(after_attention), after_attention.numel(),
            )
            if status != 0:
                raise RuntimeError(f"Attention residual CUDA call failed: {status}")
            mlp_input = torch.empty_like(hidden_states)
            status = self.cuda.run_norm(
                bf16_pointer(after_attention),
                bf16_pointer(layer.post_attention_layernorm.weight),
                bf16_pointer(mlp_input), rows, hidden_size,
                float(config.rms_norm_eps),
            )
            if status != 0:
                raise RuntimeError(f"Post-attention RMSNorm CUDA call failed: {status}")
            mlp_output = torch.empty_like(hidden_states)
            mlp = layer.mlp
            status = self.cuda.run_mlp(
                bf16_pointer(mlp_input),
                bf16_pointer(mlp.gate_proj.weight),
                bf16_pointer(mlp.up_proj.weight),
                bf16_pointer(mlp.down_proj.weight),
                bf16_pointer(mlp_output), rows, hidden_size, intermediate_size,
            )
            if status != 0:
                raise RuntimeError(f"MLP CUDA call failed: {status}")
            hidden_states = torch.empty_like(hidden_states)
            status = self.cuda.run_residual(
                bf16_pointer(after_attention), bf16_pointer(mlp_output),
                bf16_pointer(hidden_states), hidden_states.numel(),
            )
            if status != 0:
                raise RuntimeError(f"MLP residual CUDA call failed: {status}")
            if expected_hidden_states is not None:
                outside_count = self.compare_hidden_states(
                    hidden_states,
                    expected_hidden_states[layer_index + 1],
                    f"layer.{layer_index}.output",
                )
                if outside_count:
                    print(
                        f"Stopping before layer {layer_index + 1} calculations",
                        flush=True,
                    )
                    raise AssertionError(
                        f"Layer {layer_index} output exceeded tolerance"
                    )

        if expected_hidden_states is not None:
            return hidden_states

        normalized = torch.empty_like(hidden_states)
        status = self.cuda.run_norm(
            bf16_pointer(hidden_states), bf16_pointer(model.model.norm.weight),
            bf16_pointer(normalized), rows, hidden_size, float(config.rms_norm_eps),
        )
        if status != 0:
            raise RuntimeError(f"Final RMSNorm CUDA call failed: {status}")
        last_hidden_state = normalized[:, -1, :].contiguous()
        logits = torch.empty((1, config.vocab_size), dtype=torch.bfloat16)
        status = self.cuda.run_vocab_projection(
            bf16_pointer(last_hidden_state),
            bf16_pointer(model.lm_head.weight),
            bf16_pointer(logits), hidden_size, config.vocab_size,
        )
        if status != 0:
            raise RuntimeError(f"Vocabulary projection CUDA call failed: {status}")
        return logits

    def generate_token_ids(self, prompt, max_new_tokens):
        token_ids = self.encode(prompt)
        if len(token_ids) + max_new_tokens > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"Prompt plus generation limit must not exceed "
                f"{MAX_SEQUENCE_LENGTH} tokens"
            )
        prompt_length = len(token_ids)
        generated_ids = []
        for _ in range(max_new_tokens):
            logits = self.forward_logits(token_ids)
            next_token = int(logits[0].argmax().item())
            token_ids.append(next_token)
            generated_ids.append(next_token)
            if next_token in self.eos_token_ids:
                break
        return generated_ids, self.decode(generated_ids), prompt_length


def make_handler(engine, max_new_tokens):
    class InferenceHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/generate":
                self.send_error(404)
                return
            request_size = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(request_size))
            prompt = request["prompt"]
            token_limit = int(request.get("max_new_tokens", max_new_tokens))
            result = engine.compare_generation(prompt, token_limit)
            response = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    return InferenceHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer-directory", type=Path)
    parser.add_argument("--library-directory", type=Path, default=Path(__file__).parent)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--compare-prompt")
    parser.add_argument("--check-hidden-state-positions")
    parser.add_argument("--diagnose-layer-hidden-states")
    args = parser.parse_args()

    engine = Qwen3Inference(
        args.model, args.library_directory, args.tokenizer_directory
    )
    if args.check_hidden_state_positions is not None:
        engine.check_hidden_state_positions(args.check_hidden_state_positions)
        engine.tokenizer.close()
        return
    if args.diagnose_layer_hidden_states is not None:
        engine.diagnose_layer_hidden_states(args.diagnose_layer_hidden_states)
        engine.tokenizer.close()
        return
    if args.compare_prompt is not None:
        result = engine.compare_generation(
            args.compare_prompt, args.max_new_tokens
        )
        print(f"Prompt: {args.compare_prompt}")
        print(f"Prompt tokens: {result['prompt_tokens']}")
        print(f"CUDA: {result['cuda']['text']!r}")
        print(f"PyTorch: {result['pytorch']['text']!r}")
        for step in result["steps"]:
            print(
                f"step {step['step']}: CUDA token={step['cuda_token_id']}, "
                f"PyTorch token={step['pytorch_token_id']}"
            )
            logits = step["logits"]
            if logits is None:
                print("  logit comparison skipped: token prefixes have diverged")
                continue
            print(
                "  logits: "
                f"{logits['outside_tolerance']}/{logits['compared']} outside "
                f"atol={logits['atol']} rtol={logits['rtol']}; "
                f"max_abs_diff={logits['max_abs_diff']:.8g}"
            )
            for mismatch in logits["largest_mismatches"]:
                print(
                    f"  logit[{mismatch['index']}]: "
                    f"CUDA={mismatch['cuda']:.8g}, "
                    f"PyTorch={mismatch['pytorch']:.8g}, "
                    f"abs_diff={mismatch['abs_diff']:.8g}"
                )
        engine.tokenizer.close()
        return

    server = HTTPServer(
        (args.host, args.port), make_handler(engine, args.max_new_tokens)
    )
    print(f"Qwen3 inference server listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
