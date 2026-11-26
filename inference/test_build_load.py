"""Smoke-test that the compiled inference libraries load and export their C APIs."""

import ctypes
import os
from pathlib import Path
import unittest


LIBRARY_DIR = Path(
    os.environ.get("QWEN3_LIBRARY_DIR", Path(__file__).resolve().parent)
)
LIBRARY_SYMBOLS = {
    "libqwen3_tokenizer.so": (
        "qwen3_tokenizer_load",
        "qwen3_tokenizer_free",
        "qwen3_tokenizer_encode",
        "qwen3_tokenizer_decode",
    ),
    "libqwen3_rms_norm.so": ("qwen3_rms_norm_cuda",),
    "libqwen3_rotary_embedding.so": ("qwen3_rotary_embedding_cuda",),
    "libqwen3_attention.so": ("qwen3_attention_cuda",),
    "libqwen3_mlp.so": ("qwen3_mlp_cuda",),
    "libqwen3_decoder_layer.so": ("qwen3_residual_add_cuda",),
    "libqwen3_inference.so": (
        "qwen3_embedding_cuda",
        "qwen3_vocab_projection_cuda",
    ),
}


class SharedLibraryLoadTests(unittest.TestCase):
    def test_all_libraries_load_and_export_required_functions(self):
        for library_name, symbol_names in LIBRARY_SYMBOLS.items():
            library_path = LIBRARY_DIR / library_name
            library = ctypes.CDLL(str(library_path))
            for symbol_name in symbol_names:
                self.assertTrue(getattr(library, symbol_name))
            print(f"Loaded {library_name} and checked {len(symbol_names)} function(s)")


if __name__ == "__main__":
    unittest.main()
