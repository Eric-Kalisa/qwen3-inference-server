"""NumPy/ctypes bridge; safetensors are read directly, without Torch/Transformers."""
import ctypes as C
import json
import mmap
from pathlib import Path
import struct
import threading
import numpy as np


class CudaBackend:
    def __init__(self, model_dir, library, blocks=1024, block_size=16,
                 max_context=8192, max_rows=256, max_sequences=32):
        self.owner = threading.get_ident()
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text())
        if cfg.get("model_type") != "qwen3" or cfg.get("attention_bias", False):
            raise ValueError("Only dense, bias-free Qwen3 is supported")
        rope = cfg.get("rope_parameters") or {}
        if cfg.get("rope_scaling") or rope.get("rope_type", "default") != "default":
            raise ValueError("Scaled RoPE is not implemented")
        if max_context > cfg["max_position_embeddings"]:
            raise ValueError("Context exceeds the model limit")
        self.vocab = cfg["vocab_size"]
        self.max_rows, self.max_sequences = max_rows, max_sequences
        self.block_size, self.max_context = block_size, max_context
        self.max_blocks = (max_context + block_size - 1) // block_size
        generation = model_dir / "generation_config.json"
        eos = (json.loads(generation.read_text()) if generation.exists() else cfg).get("eos_token_id", [])
        self.eos = {eos} if isinstance(eos, int) else set(eos or [])
        self.lib = C.CDLL(str(library))
        ptr = C.c_void_p
        self.lib.q_error.restype = C.c_char_p
        self.lib.q_create.argtypes = [C.c_int]*12 + [C.c_float]*2
        self.lib.q_create.restype = ptr
        self.lib.q_weight.argtypes = [ptr, C.c_char_p, ptr, C.c_size_t]
        self.lib.q_forward.argtypes = [ptr]*6 + [C.c_int]*2 + [ptr]
        self.lib.q_copy.argtypes = [ptr, C.c_int, C.c_int]
        self.lib.q_destroy.argtypes = [ptr]
        h, mid, layers = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
        qh, kh = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        dim = cfg.get("head_dim", h // qh)
        self.handle = self.lib.q_create(h, mid, layers, qh, kh, dim, self.vocab,
                                       blocks, block_size, max_context, max_rows,
                                       max_sequences, cfg["rms_norm_eps"],
                                       rope.get("rope_theta", cfg.get("rope_theta", 1000000.0)))
        if not self.handle:
            raise RuntimeError(self.lib.q_error().decode())
        shapes = {"model.embed_tokens.weight": (self.vocab, h),
                  "model.norm.weight": (h,), "lm_head.weight": (self.vocab, h)}
        for i in range(layers):
            p = f"model.layers.{i}."
            shapes.update({p+"input_layernorm.weight": (h,), p+"post_attention_layernorm.weight": (h,),
                           p+"self_attn.q_proj.weight": (qh*dim,h), p+"self_attn.k_proj.weight": (kh*dim,h),
                           p+"self_attn.v_proj.weight": (kh*dim,h), p+"self_attn.o_proj.weight": (h,qh*dim),
                           p+"self_attn.q_norm.weight": (dim,), p+"self_attn.k_norm.weight": (dim,),
                           p+"mlp.gate_proj.weight": (mid,h), p+"mlp.up_proj.weight": (mid,h),
                           p+"mlp.down_proj.weight": (h,mid)})
        loaded = set()
        tied_embedding = None
        try:
            for file in sorted(model_dir.glob("*.safetensors")):
                with file.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
                    header_len = struct.unpack("<Q", data[:8])[0]
                    header = json.loads(data[8:8+header_len])
                    for name, meta in header.items():
                        if name not in shapes:
                            continue
                        if name in loaded or meta["dtype"] != "BF16" or tuple(meta["shape"]) != shapes[name]:
                            raise ValueError(f"Invalid/duplicate BF16 tensor: {name}")
                        start, end = meta["data_offsets"]
                        if end-start != int(np.prod(shapes[name]))*2:
                            raise ValueError(f"Invalid tensor size: {name}")
                        tensor = np.frombuffer(data, dtype=np.uint16, count=(end-start)//2,
                                               offset=8+header_len+start).copy()
                        self._upload(name, tensor)
                        loaded.add(name)
                        if name == "model.embed_tokens.weight" and cfg.get("tie_word_embeddings"):
                            tied_embedding = tensor
            if tied_embedding is not None and "lm_head.weight" not in loaded:
                self._upload("lm_head.weight", tied_embedding)
                loaded.add("lm_head.weight")
            if loaded != set(shapes):
                raise ValueError(f"Missing weights: {sorted(set(shapes)-loaded)}")
        except BaseException:
            self.close()
            raise

    def _check(self, status):
        if status:
            raise RuntimeError(self.lib.q_error().decode())

    def _upload(self, name, tensor):
        self._check(self.lib.q_weight(self.handle, name.encode(), tensor.ctypes.data, tensor.size))

    def copy_block(self, src, dst):
        self._check(self.lib.q_copy(self.handle, src, dst))

    def forward(self, sequences, ends):
        ids, positions, owners, last = [], [], [], []
        tables = np.full((len(sequences), self.max_blocks), -1, np.int32)
        for i, (s, end) in enumerate(zip(sequences, ends)):
            ids.extend(s.tokens[s.computed:end])
            positions.extend(range(s.computed, end))
            owners.extend([i]*(end-s.computed))
            last.append(len(ids)-1)
            tables[i, :len(s.blocks)] = s.blocks
        arrays = [np.asarray(a, dtype=np.int32) for a in (ids, positions, owners)]
        arrays += [tables, np.asarray(last, dtype=np.int32)]
        result = np.empty((len(sequences), self.vocab), np.float32)
        self._check(self.lib.q_forward(self.handle, *(a.ctypes.data for a in arrays),
                                      len(ids), len(sequences), result.ctypes.data))
        return result

    def close(self):
        if self.handle:
            self._check(self.lib.q_destroy(self.handle))
            self.handle = None


class Tokenizer:
    """The scheduler also owns the C++ tokenizer; HTTP threads never call it."""
    def __init__(self, library, directory):
        self.lib = C.CDLL(str(library))
        self.lib.qwen3_tokenizer_load.argtypes = [C.c_char_p]
        self.lib.qwen3_tokenizer_load.restype = C.c_void_p
        self.lib.qwen3_tokenizer_free.argtypes = [C.c_void_p]
        self.lib.qwen3_tokenizer_encode.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_int]
        self.lib.qwen3_tokenizer_decode.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_int]
        self.lib.qwen3_tokenizer_decode_flags.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_int, C.c_int]
        self.handle = self.lib.qwen3_tokenizer_load(str(directory).encode())
        if not self.handle:
            raise ValueError("Tokenizer could not be loaded")

    def encode(self, text):
        n = self.lib.qwen3_tokenizer_encode(self.handle, text.encode(), None, 0)
        if n < 0:
            raise ValueError("Tokenizer encoding failed")
        buf = (C.c_int*n)()
        self.lib.qwen3_tokenizer_encode(self.handle, text.encode(), buf, n)
        return list(buf)

    def decode(self, ids, skip_special_tokens=True):
        if not ids:
            return ""
        buf = (C.c_int*len(ids))(*ids)
        n = self.lib.qwen3_tokenizer_decode_flags(self.handle, buf, len(ids), None, 0, int(skip_special_tokens))
        if n < 0:
            raise ValueError("Tokenizer decoding failed")
        out = C.create_string_buffer(n+1)
        self.lib.qwen3_tokenizer_decode_flags(self.handle, buf, len(ids), out, n+1, int(skip_special_tokens))
        return out.value.decode("utf-8", errors="replace")

    def close(self):
        if self.handle:
            self.lib.qwen3_tokenizer_free(self.handle)
            self.handle = None
