"""Trace real CUDA KV preemption with a two-page pool and two requests."""

import json
import os
from pathlib import Path
import threading
import unittest

import numpy as np

from backend import CudaBackend, Tokenizer
from scheduler import Worker
from test_checkpoint_kv import ATOL, BLOCK_SIZE, PROMPT_IDS, RTOL
from test_mixed_batch import SECOND_PROMPT


PROMPTS = {"A": (PROMPT_IDS, 3), "B": (SECOND_PROMPT, 2)}
REQUEST_BY_FIRST_TOKEN = {prompt[0]: name for name, (prompt, _) in PROMPTS.items()}
WAIT = 60


class TokenIdTokenizer(Tokenizer):
    """Accept fixed input token IDs and decode the output with the model tokenizer."""

    def encode(self, text):
        return [int(token) for token in text.split()]


@unittest.skipUnless(
    os.environ.get("QWEN_CUDA_LIBRARY") and os.environ.get("QWEN_KV_MODEL_DIR"),
    "Set QWEN_CUDA_LIBRARY and QWEN_KV_MODEL_DIR to run the CUDA KV pressure test",
)
class KVPressureTests(unittest.TestCase):
    def run_case(self, case, blocks, batch):
        settings = dict(blocks=blocks, block_size=BLOCK_SIZE, max_context=32,
                        max_rows=16, max_sequences=2)
        calls, preemptions = [], []
        histories = {name: [] for name in PROMPTS}
        generated = {name: [] for name in PROMPTS}
        results, errors = {}, {}
        entered, resume = threading.Event(), threading.Event()

        def log(event, **fields):
            print("kv_pressure " + json.dumps(dict(case=case, event=event, **fields)),
                  flush=True)

        class TracingBackend(CudaBackend):
            def forward(self, sequences, ends):
                if batch and not calls:
                    entered.set()
                    if not resume.wait(WAIT):
                        raise RuntimeError("Timed out waiting to queue request B")

                entries = []
                for seq, end in zip(sequences, ends):
                    entries.append(dict(
                        request=REQUEST_BY_FIRST_TOKEN[seq.tokens[0]],
                        start=seq.computed, end=end, pages=list(seq.blocks),
                        input_token_ids=seq.tokens[seq.computed:end],
                    ))
                calls.append(entries)
                log("batch", step=len(calls),
                    rows=sum(row["end"] - row["start"] for row in entries),
                    sequences=entries)
                logits = super().forward(sequences, ends)
                for entry, seq, end, result in zip(entries, sequences, ends, logits):
                    histories[entry["request"]].append((tuple(seq.tokens[:end]), result.copy()))
                return logits

        class TracingWorker(Worker):
            def reset_job(self, job):
                seq = job.seqs[0]
                event = dict(
                    before_step=len(calls) + 1, request=REQUEST_BY_FIRST_TOKEN[seq.tokens[0]],
                    computed_before=seq.computed, released_pages=list(seq.blocks),
                    generated_token_ids=list(seq.generated),
                )
                super().reset_job(job)
                event["computed_after"] = seq.computed
                preemptions.append(event)
                log("preempt", **event)

            def expand(self, job):
                before = len(job.seqs[0].generated)
                super().expand(job)
                seq = job.seqs[0]
                name = REQUEST_BY_FIRST_TOKEN[seq.tokens[0]]
                for index, token in enumerate(seq.generated[before:], start=before + 1):
                    generated[name].append(token)
                    log("token", step=len(calls), request=name, index=index, token_id=token)

        library = Path(os.environ["QWEN_CUDA_LIBRARY"])
        model_dir = Path(os.environ["QWEN_KV_MODEL_DIR"])
        log("start", pages=blocks, tokens_per_page=BLOCK_SIZE, batching=batch)
        worker = TracingWorker(
            lambda: TracingBackend(model_dir, library, **settings),
            lambda: TokenIdTokenizer(library.with_name("libqwen3_tokenizer.so"), model_dir),
            **settings, mode="paged", batch=batch, prefix_capacity=0, audit=True,
        )

        def submit(name):
            ids, count = PROMPTS[name]
            try:
                return worker.submit(dict(prompt=" ".join(map(str, ids)), max_tokens=count,
                                          ignore_eos=True, skip_special_tokens=False))
            except Exception as exc:
                errors[name] = str(exc)
                log("result", request=name, status="error", token_ids=generated[name],
                    error=str(exc))
                return None

        def collect(name, ticket):
            if ticket is None:
                return
            try:
                result = ticket.future.result(timeout=WAIT)["choices"][0]
            except Exception as exc:
                errors[name] = str(exc)
                log("result", request=name, status="error", token_ids=generated[name],
                    error=str(exc) or type(exc).__name__)
                return
            results[name] = result["token_ids"]
            log("result", request=name, status="ok", token_ids=result["token_ids"],
                text=result["text"], finish_reason=result["finish_reason"])

        try:
            first = submit("A")
            if batch:
                self.assertTrue(entered.wait(WAIT), "A did not reach its prefill gate")
            else:
                collect("A", first)
            second = submit("B")
            resume.set()
            if batch:
                collect("A", first)
            collect("B", second)
        finally:
            resume.set()
            worker.close()

        log("summary", preemptions=worker.stats["preemptions"], forward_calls=len(calls),
            free_pages=sorted(worker.cache.free))
        self.assertFalse(errors, f"{case} request errors: {errors}")
        self.assertIsNone(getattr(worker, "failure", None))
        self.assertEqual(worker.stats["preemptions"], len(preemptions))
        self.assertEqual(worker.cache.refs, [0] * blocks)
        self.assertEqual(worker.cache.free, set(range(blocks)))
        self.assertEqual(results, generated)
        for name, (_, count) in PROMPTS.items():
            self.assertEqual(len(results[name]), count)
        return calls, preemptions, histories, results

    def test_paged_batch_preempts_and_replays_with_two_kv_pages(self):
        pressure, evicted, pressured_logits, pressured_tokens = self.run_case("two_pages", 2, True)
        control, control_evicted, control_logits, control_tokens = self.run_case("three_pages", 3, True)
        _, serial_evicted, serial_logits, serial_tokens = self.run_case("serial", 2, False)

        def schedule(calls):
            return [[(s["request"], s["start"], s["end"]) for s in call] for call in calls]

        first_passes = [[("A", 0, 15)], [("A", 15, 16), ("B", 0, 5)]]
        self.assertEqual(schedule(pressure), first_passes + [[("A", 16, 17)], [("B", 0, 6)]])
        self.assertEqual(schedule(control), first_passes + [[("A", 16, 17), ("B", 5, 6)]])
        self.assertEqual(evicted, [dict(
            before_step=3, request="B", computed_before=5, computed_after=0,
            released_pages=pressure[1][1]["pages"], generated_token_ids=serial_tokens["B"][:1],
        )])
        self.assertEqual(len(evicted[0]["released_pages"]), 1)
        self.assertEqual(pressure[2][0]["pages"],
                         pressure[1][0]["pages"] + evicted[0]["released_pages"])
        self.assertEqual(pressure[3][0]["input_token_ids"], [*SECOND_PROMPT, serial_tokens["B"][0]])
        self.assertEqual(control_evicted, [])
        self.assertEqual(serial_evicted, [])

        for case, histories, tokens in (
            ("two_pages", pressured_logits, pressured_tokens),
            ("three_pages", control_logits, control_tokens),
        ):
            self.assertEqual(tokens, serial_tokens)
            for name in PROMPTS:
                self.assertEqual(len(histories[name]), len(serial_logits[name]))
                for step, ((history, actual), (serial_history, expected)) in enumerate(
                    zip(histories[name], serial_logits[name]), start=1
                ):
                    with self.subTest(case=case, request=name, generated_step=step):
                        self.assertEqual(history, serial_history)
                        self.assertTrue(np.isfinite(actual).all() and np.isfinite(expected).all())
                        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL,
                                                   equal_nan=False)
