"""Run a real-Qwen3 continuous batch with one decode and one prefill."""
import ctypes as C
import os
import threading
import unittest

import numpy as np

from backend import CudaBackend
from scheduler import Worker
from test_checkpoint_kv import ATOL, BLOCK_SIZE, RTOL, TOKEN_IDS
from test_scheduler import FakeTokenizer


FIRST_PROMPT = TOKEN_IDS[:BLOCK_SIZE]
SECOND_PROMPT = (41, 43, 47, 53, 59)
CAPACITY = dict(blocks=4, block_size=BLOCK_SIZE, max_context=BLOCK_SIZE + 2,
                max_rows=BLOCK_SIZE, max_sequences=2)


def run_requests(mixed):
    calls = []
    entered = threading.Event()
    resume = threading.Event()

    def make_backend():
        backend = CudaBackend(os.environ["QWEN_KV_MODEL_DIR"],
                              os.environ["QWEN_CUDA_LIBRARY"], **CAPACITY)
        native_forward = backend.lib.q_forward
        max_blocks, vocab = backend.max_blocks, backend.vocab

        def record_forward(handle, ids, positions, owners, table, last, rows, sequences, output):
            if mixed and not calls:
                # Hold A's prefill call open until B is in the incoming queue.
                entered.set()
                if not resume.wait(60):
                    raise RuntimeError("Timed out waiting for the second request")
            status = native_forward(handle, ids, positions, owners, table, last, rows, sequences, output)
            if status == 0:
                calls.append(dict(
                    rows=rows, sequences=sequences,
                    ids=tuple((C.c_int32 * rows).from_address(ids)),
                    positions=tuple((C.c_int32 * rows).from_address(positions)),
                    owners=tuple((C.c_int32 * rows).from_address(owners)),
                    last=tuple((C.c_int32 * sequences).from_address(last)),
                    blocks=np.ctypeslib.as_array(
                        (C.c_int32 * (sequences * max_blocks)).from_address(table)
                    ).reshape(sequences, max_blocks).copy(),
                    logits=np.ctypeslib.as_array(
                        (C.c_float * (sequences * vocab)).from_address(output)
                    ).reshape(sequences, vocab).copy(),
                ))
            return status

        backend.lib.q_forward = record_forward
        return backend

    worker = Worker(make_backend, FakeTokenizer, batch=mixed, mode="paged",
                    prefix_capacity=0, audit=True, **CAPACITY)
    try:
        first = worker.submit(dict(prompt=" ".join(map(str, FIRST_PROMPT)),
                                   max_tokens=2, ignore_eos=True))
        if mixed:
            if not entered.wait(60):
                raise RuntimeError("Timed out waiting for the first prefill")
        else:
            first.future.result(timeout=60)
        second = worker.submit(dict(prompt=" ".join(map(str, SECOND_PROMPT)),
                                    max_tokens=1, ignore_eos=True))
        resume.set()
        outputs = [ticket.future.result(timeout=60) for ticket in (first, second)]
    finally:
        resume.set()
        worker.close()
    return calls, [result["choices"][0]["token_ids"] for result in outputs]


@unittest.skipUnless(
    os.environ.get("QWEN_CUDA_LIBRARY") and os.environ.get("QWEN_KV_MODEL_DIR"),
    "Set QWEN_CUDA_LIBRARY and QWEN_KV_MODEL_DIR to run the real-checkpoint mixed batch",
)
class MixedBatchTests(unittest.TestCase):
    def test_decode_and_new_prefill_share_cuda_call_and_match_serial(self):
        serial, serial_ids = run_requests(mixed=False)
        mixed, mixed_ids = run_requests(mixed=True)
        self.assertEqual([(call["rows"], call["sequences"]) for call in serial],
                         [(BLOCK_SIZE, 1), (1, 1), (len(SECOND_PROMPT), 1)])
        self.assertEqual([(call["rows"], call["sequences"]) for call in mixed],
                         [(BLOCK_SIZE, 1), (1 + len(SECOND_PROMPT), 2)])

        batch = mixed[1]
        self.assertEqual(batch["ids"], serial[1]["ids"] + serial[2]["ids"])
        self.assertEqual(batch["positions"], (BLOCK_SIZE, *range(len(SECOND_PROMPT))))
        self.assertEqual(batch["owners"], (0,) + (1,) * len(SECOND_PROMPT))
        self.assertEqual(batch["last"], (0, len(SECOND_PROMPT)))
        decode_blocks, prefill_blocks = (set(row) - {-1} for row in batch["blocks"])
        self.assertEqual(len(decode_blocks), 2)
        self.assertEqual(len(prefill_blocks), 1)
        self.assertTrue(decode_blocks.isdisjoint(prefill_blocks))

        for label, actual, expected in (
            ("A decode", batch["logits"][0], serial[1]["logits"][0]),
            ("B prefill", batch["logits"][1], serial[2]["logits"][0]),
        ):
            with self.subTest(request=label):
                self.assertTrue(np.isfinite(actual).all() and np.isfinite(expected).all())
                np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL, equal_nan=False)
                print(f"{label}: max abs logit difference {np.max(np.abs(actual - expected)):.6g}",
                      flush=True)
        self.assertEqual(mixed_ids, serial_ids)
