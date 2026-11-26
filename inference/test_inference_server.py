"""GPU integration checks against Hugging Face using real Qwen3 weights."""

import os
from pathlib import Path
import unittest

import torch
from transformers import AutoTokenizer

from inference_server import DEFAULT_MODEL, LOGIT_RTOL, QWEN3_4B_LOGIT_ATOL, Qwen3Inference


INFERENCE_DIR = Path(__file__).resolve().parent
LIBRARY_DIR = Path(os.environ.get("QWEN3_LIBRARY_DIR", INFERENCE_DIR))
TOKENIZER_DIR = os.environ.get("QWEN3_TOKENIZER_DIR")
MODEL_ID = os.environ.get("QWEN3_MODEL_ID", DEFAULT_MODEL)
PROMPTS = ("The capital of France is", "A short poem about the moon:")


class Qwen3InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"Testing {MODEL_ID} with rtol={LOGIT_RTOL}, atol={QWEN3_4B_LOGIT_ATOL}")
        cls.engine = Qwen3Inference(
            MODEL_ID,
            LIBRARY_DIR,
            Path(TOKENIZER_DIR) if TOKENIZER_DIR else None,
        )
        cls.reference_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    @classmethod
    def tearDownClass(cls):
        cls.engine.tokenizer.close()

    def test_cpp_tokenizer_matches_hugging_face(self):
        for prompt in PROMPTS:
            expected_ids = self.reference_tokenizer(
                prompt, add_special_tokens=True
            )["input_ids"]
            self.assertEqual(self.engine.encode(prompt), expected_ids)

    def test_logits_and_greedy_generation_match_hugging_face(self):
        prompt = PROMPTS[0]
        token_ids = self.engine.encode(prompt)
        input_ids = torch.tensor([token_ids], dtype=torch.long)

        with torch.inference_mode():
            reference_logits = self.engine.model(input_ids).logits[:, -1, :]
            reference_generation = self.engine.model.generate(
                input_ids,
                max_new_tokens=3,
                do_sample=False,
                use_cache=False,
            )[:, input_ids.shape[1] :]

        actual_logits = self.engine.forward_logits(token_ids)
        torch.testing.assert_close(
            actual_logits.float(),
            reference_logits.float(),
            rtol=LOGIT_RTOL,
            atol=QWEN3_4B_LOGIT_ATOL,
        )
        actual_ids, _, _ = self.engine.generate_token_ids(prompt, 3)
        self.assertEqual(actual_ids, reference_generation[0].tolist())


if __name__ == "__main__":
    unittest.main()
