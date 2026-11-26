"""Check KV block lifetime when one request in a mixed CUDA batch is cancelled."""
import ctypes as C
import os
import threading
import unittest

import numpy as np

from backend import CudaBackend
from scheduler import Worker
from test_checkpoint_kv import ATOL, BLOCK_SIZE, RTOL
from test_mixed_batch import CAPACITY as MIXED_CAPACITY, FIRST_PROMPT, SECOND_PROMPT
from test_scheduler import FakeTokenizer


THIRD_PROMPT = (61, 67, 71, 73, 79)
CAPACITY = dict(MIXED_CAPACITY, blocks=3, max_context=BLOCK_SIZE + 3)
WAIT = 60


@unittest.skipUnless(
    os.environ.get("QWEN_CUDA_LIBRARY") and os.environ.get("QWEN_KV_MODEL_DIR"),
    "Set QWEN_CUDA_LIBRARY and QWEN_KV_MODEL_DIR to run the CUDA cancellation test",
)
class KVCancellationTests(unittest.TestCase):
    def run_requests(self, cancel_in_batch):
        calls, releases = [], []
        prefill_entered, prefill_resume = threading.Event(), threading.Event()
        mixed_entered, mixed_resume = threading.Event(), threading.Event()

        def make_backend():
            backend = CudaBackend(os.environ["QWEN_KV_MODEL_DIR"],
                                  os.environ["QWEN_CUDA_LIBRARY"], **CAPACITY)
            native_forward = backend.lib.q_forward

            def record_forward(handle, ids, positions, owners, table, last, rows, sequences, output):
                if cancel_in_batch and not calls:
                    prefill_entered.set()
                    if not prefill_resume.wait(WAIT):
                        raise RuntimeError("Timed out waiting to queue request B")
                if cancel_in_batch and len(calls) == 1:
                    # The worker has reserved the mixed batch before this gate.
                    mixed_entered.set()
                    if not mixed_resume.wait(WAIT):
                        raise RuntimeError("Timed out waiting to cancel request A")

                status = native_forward(handle, ids, positions, owners, table, last, rows, sequences, output)
                if status == 0:
                    block_ids = (C.c_int32 * (sequences * backend.max_blocks)).from_address(table)
                    block_rows = np.ctypeslib.as_array(block_ids).reshape(sequences, backend.max_blocks)
                    logits = (C.c_float * (sequences * backend.vocab)).from_address(output)
                    calls.append(dict(
                        rows=rows,
                        positions=tuple((C.c_int32 * rows).from_address(positions)),
                        blocks=tuple(frozenset(row) - {-1} for row in block_rows),
                        logits=np.ctypeslib.as_array(logits).reshape(sequences, backend.vocab).copy(),
                    ))
                return status

            backend.lib.q_forward = record_forward
            return backend

        worker = Worker(make_backend, FakeTokenizer, batch=cancel_in_batch, mode="paged",
                        prefix_capacity=0, audit=True, **CAPACITY)
        release = worker.cache.release

        def record_release(blocks):
            release(blocks)
            if blocks:
                releases.append((len(calls), frozenset(blocks)))

        worker.cache.release = record_release

        def submit(prompt, max_tokens):
            return worker.submit(dict(prompt=" ".join(map(str, prompt)),
                                      max_tokens=max_tokens, ignore_eos=True))

        try:
            if cancel_in_batch:
                first = submit(FIRST_PROMPT, 3)
                self.assertTrue(prefill_entered.wait(WAIT), "A never reached prefill")
            second = submit(SECOND_PROMPT, 2)
            if cancel_in_batch:
                prefill_resume.set()
                self.assertTrue(mixed_entered.wait(WAIT), "The mixed forward never started")
                first.cancelled.set()
            else:
                second.future.result(timeout=WAIT)

            third = submit(THIRD_PROMPT, 1)
            if cancel_in_batch:
                for ticket in (first, second, third):
                    self.assertFalse(ticket.future.done())
                mixed_resume.set()
                with self.assertRaisesRegex(RuntimeError, "Request cancelled"):
                    first.future.result(timeout=WAIT)

            outputs = [ticket.future.result(timeout=WAIT) for ticket in (second, third)]
        finally:
            prefill_resume.set()
            mixed_resume.set()
            worker.close()

        self.assertIsNone(getattr(worker, "failure", None))
        self.assertEqual(worker.cache.refs, [0] * CAPACITY["blocks"])
        self.assertEqual(worker.cache.free, set(range(CAPACITY["blocks"])))
        return calls, releases, [result["choices"][0]["token_ids"] for result in outputs]

    def test_cancel_mixed_forward_then_reuse_kv_blocks(self):
        serial, _, serial_ids = self.run_requests(cancel_in_batch=False)
        mixed, releases, mixed_ids = self.run_requests(cancel_in_batch=True)

        self.assertEqual([call["rows"] for call in serial], [len(SECOND_PROMPT), 1, len(THIRD_PROMPT)])
        self.assertEqual([call["rows"] for call in mixed],
                         [BLOCK_SIZE, 1 + len(SECOND_PROMPT), 1 + len(THIRD_PROMPT)])
        _, first_mixed, next_mixed = mixed
        self.assertEqual(first_mixed["positions"], (BLOCK_SIZE, *range(len(SECOND_PROMPT))))
        self.assertEqual(next_mixed["positions"], (len(SECOND_PROMPT), *range(len(THIRD_PROMPT))))

        first_blocks, second_blocks = first_mixed["blocks"]
        surviving_blocks, third_blocks = next_mixed["blocks"]
        self.assertEqual(len(first_blocks), 2)
        self.assertEqual(len(second_blocks), 1)
        self.assertTrue(first_blocks.isdisjoint(second_blocks))
        self.assertEqual(first_blocks | second_blocks, set(range(CAPACITY["blocks"])))
        self.assertEqual(surviving_blocks, second_blocks)
        self.assertEqual(len(third_blocks), 1)
        self.assertTrue(third_blocks <= first_blocks)
        # A's two pages must be released together after the second native call.
        self.assertEqual(releases[0], (2, first_blocks))

        for label, actual, expected in (
            ("B prefill", first_mixed["logits"][1], serial[0]["logits"][0]),
            ("B decode", next_mixed["logits"][0], serial[1]["logits"][0]),
            ("C prefill", next_mixed["logits"][1], serial[2]["logits"][0]),
        ):
            with self.subTest(step=label):
                self.assertTrue(np.isfinite(actual).all() and np.isfinite(expected).all())
                np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL, equal_nan=False)
        self.assertEqual(mixed_ids, serial_ids)
