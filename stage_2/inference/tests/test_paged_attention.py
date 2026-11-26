"""Check paged CUDA attention with an explicitly scattered block table."""
import ctypes as C
import os
import unittest
from unittest.mock import patch

import numpy as np

from backend import CudaBackend
from cache import Sequence
from test_checkpoint_kv import ATOL, BLOCK_SIZE, PREFIX_LENGTHS, RTOL, TOKEN_IDS


ADJACENT_BLOCKS = (0, 1)
SCATTERED_BLOCKS = (4, 2)


def run_to(backend, sequence, end):
    logits = backend.forward([sequence], [end])[0]
    sequence.computed = end
    return logits


def trace_layout(backend, blocks):
    # Assign the layout directly so this test controls the table CUDA receives.
    sequence = Sequence(list(TOKEN_IDS), blocks=list(blocks))
    calls = []
    native_forward = backend.lib.q_forward

    def record_forward(handle, ids, positions, owners, table, last, rows, sequences, output):
        calls.append((
            tuple((C.c_int32 * rows).from_address(positions)),
            tuple((C.c_int32 * backend.max_blocks).from_address(table)),
        ))
        return native_forward(handle, ids, positions, owners, table, last, rows, sequences, output)

    with patch.object(backend.lib, "q_forward", side_effect=record_forward):
        logits = {end: run_to(backend, sequence, end) for end in PREFIX_LENGTHS}
    return logits, calls


@unittest.skipUnless(
    os.environ.get("QWEN_CUDA_LIBRARY") and os.environ.get("QWEN_KV_MODEL_DIR"),
    "Set QWEN_CUDA_LIBRARY and QWEN_KV_MODEL_DIR to run paged attention on CUDA",
)
class PagedAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        length = len(TOKEN_IDS)
        backend = CudaBackend(
            os.environ["QWEN_KV_MODEL_DIR"], os.environ["QWEN_CUDA_LIBRARY"],
            blocks=6, block_size=BLOCK_SIZE, max_context=length,
            max_rows=length, max_sequences=1,
        )
        try:
            cls.adjacent_logits, cls.adjacent_calls = trace_layout(backend, ADJACENT_BLOCKS)

            # Replace baseline data with a different history. A kernel that
            # ignores the scattered table must not find the target's old KV.
            resident_ids = list(reversed(TOKEN_IDS))
            resident = Sequence(resident_ids, blocks=list(ADJACENT_BLOCKS))
            run_to(backend, resident, length - 1)
            cls.resident_before = run_to(backend, resident, length)

            cls.scattered_logits, cls.scattered_calls = trace_layout(backend, SCATTERED_BLOCKS)

            # Re-evaluate the last token using the still-resident prefix.
            resident.computed = length - 1
            cls.resident_after = run_to(backend, resident, length)
        finally:
            backend.close()

    def assert_logits_match(self, actual, expected):
        self.assertTrue(np.isfinite(actual).all(), "Nonfinite logits")
        self.assertTrue(np.isfinite(expected).all(), "Nonfinite reference logits")
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL, equal_nan=False)

    def test_cuda_receives_adjacent_and_scattered_tables_at_same_positions(self):
        positions = [tuple(range(15)), (15,), (16,), (17,)]
        self.assertEqual(self.adjacent_calls, [(pos, ADJACENT_BLOCKS) for pos in positions])
        self.assertEqual(self.scattered_calls, [(pos, SCATTERED_BLOCKS) for pos in positions])

    def test_scattered_blocks_match_adjacent_logits_at_each_step(self):
        for end in PREFIX_LENGTHS:
            with self.subTest(prefix_length=end):
                self.assert_logits_match(self.scattered_logits[end], self.adjacent_logits[end])

    def test_scattered_request_preserves_resident_prefix(self):
        self.assert_logits_match(self.resident_after, self.resident_before)
