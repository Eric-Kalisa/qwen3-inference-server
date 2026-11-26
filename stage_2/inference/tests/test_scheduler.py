from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest
import numpy as np
from scheduler import Worker, Busy


class FakeTokenizer:
    def encode(self, text): return [int(x) for x in text.split()]
    def decode(self, tokens, skip_special_tokens=True): return "".join(chr(65+t) for t in tokens)
    def close(self): pass


class FakeBackend:
    """Writes prefix fingerprints into simulated slots to detect cross-talk.

    Every backend call asserts the owning thread and checks serial execution.
    """
    eos=set()
    def __init__(self, block_size=4, gate=None, entered=None, fail=False):
        self.owner=threading.get_ident(); self.bs=block_size; self.data={}
        self.gate,self.entered,self.fail=gate,entered,fail
        self.inflight=False; self.calls=0
    def check(self):
        assert self.owner==threading.get_ident()
        assert not self.inflight, "GPU call/copy overlapped"
    def copy_block(self, a,b):
        self.check(); self.data[b]=list(self.data[a])
    def forward(self, seqs, ends):
        self.check(); self.inflight=True; self.calls+=1
        try:
            if self.entered: self.entered.set()
            if self.gate: assert self.gate.wait(10), "Test gate timed out"
            if self.fail: raise RuntimeError("injected device failure")
            result=[]; writes=set()
            for s,end in zip(seqs,ends):
                for p in range(s.computed,end):
                    slot=(s.blocks[p//self.bs],p%self.bs)
                    assert slot not in writes, "Aliased write destination"
                    writes.add(slot)
                    block=self.data.setdefault(slot[0],[None]*self.bs)
                    block[slot[1]]=tuple(s.tokens[:p+1])
                for p in range(end):
                    assert self.data[s.blocks[p//self.bs]][p%self.bs]==tuple(s.tokens[:p+1]), "Corrupt/shared cache"
                target=sum((p+1)*t for p,t in enumerate(s.tokens[:end]))%13
                result.append(-np.abs(np.arange(13,dtype=float)-target))
            return np.asarray(result)
        finally: self.inflight=False
    def close(self): self.check()


def worker(**kw):
    return Worker(FakeBackend,FakeTokenizer,blocks=128,block_size=4,max_context=128,
                  max_rows=32,max_sequences=8,max_requests=64,audit=True,**kw)


class SchedulerTests(unittest.TestCase):
    def test_modes_match_with_long_prompt_and_partial_prefill(self):
        outputs=[]
        for mode in ("none","contiguous","paged"):
            w=worker(mode=mode)
            try:
                outputs.append(w.submit(dict(prompt=" ".join(["1"]*40),max_tokens=8)).future.result(10))
            finally: w.close()
        self.assertEqual(outputs[0],outputs[1]); self.assertEqual(outputs[1],outputs[2])

    def test_concurrent_clients_match_serial_and_no_interleaved_backend_calls(self):
        requests=[dict(prompt=f"1 2 {i%10}",max_tokens=8) for i in range(48)]
        serial=worker(batch=False)
        try: expected=[serial.submit(r).future.result(10) for r in requests]
        finally: serial.close()
        w=worker()
        try:
            with ThreadPoolExecutor(16) as ex:
                actual=list(ex.map(lambda r:w.submit(r).future.result(20),requests))
            self.assertEqual(actual,expected)
        finally: w.close()
        self.assertGreater(w.stats["max_batch_sequences"],1)

    def test_seeded_samples_and_prefix_hits(self):
        w=worker()
        req=dict(prompt="1 2 3 4 5",max_tokens=6,n=4,temperature=0.8,seed=42)
        try:
            a=w.submit(req).future.result(10); b=w.submit(req).future.result(10)
            self.assertEqual(a,b); self.assertEqual(len(a["choices"]),4)
            self.assertGreater(len({tuple(c["token_ids"]) for c in a["choices"]}),1)
        finally: w.close()
        self.assertGreater(w.stats["prefix_hit_tokens"],0)
        self.assertGreater(w.stats["cow_copies"],0)

    def test_beam_pruning_matches_exhaustive_small_search(self):
        w=worker()
        try: actual=w.submit(dict(prompt="1 2 3",max_tokens=4,beam_width=3)).future.result(10)
        finally: w.close()
        beams=[([1,2,3],[],0.)]
        for _ in range(4):
            candidates=[]
            for tokens,generated,score in beams:
                target=sum((p+1)*t for p,t in enumerate(tokens))%13
                logits=-np.abs(np.arange(13)-target).astype(float)
                z=logits-logits.max(); lp=z-np.log(np.exp(z).sum())
                for t in range(13): candidates.append((tokens+[t],generated+[t],score+lp[t]))
            beams=sorted(candidates,key=lambda b:-b[2])[:3]
        self.assertEqual([c["token_ids"] for c in actual["choices"]],[b[1] for b in beams])
        np.testing.assert_allclose([c["score"] for c in actual["choices"]],[b[2] for b in beams])

    def test_cancel_during_gpu_call_waits_before_reuse(self):
        gate,entered=threading.Event(),threading.Event()
        w=Worker(lambda:FakeBackend(gate=gate,entered=entered),FakeTokenizer,
                 blocks=16,block_size=4,max_context=64,max_rows=16,max_sequences=4,audit=True)
        try:
            first=w.submit(dict(prompt="1 2 3",max_tokens=10))
            self.assertTrue(entered.wait(5))
            first.cancelled.set()
            second=w.submit(dict(prompt="4 5 6",max_tokens=3))
            self.assertFalse(first.future.done()); self.assertFalse(second.future.done())
            gate.set()
            with self.assertRaisesRegex(RuntimeError,"cancelled"): first.future.result(10)
            self.assertEqual(len(second.future.result(10)["choices"][0]["token_ids"]),3)
        finally: gate.set(); w.close()

    def test_queue_limit_and_shutdown_submit_race(self):
        gate,entered=threading.Event(),threading.Event()
        w=Worker(lambda:FakeBackend(gate=gate,entered=entered),FakeTokenizer,
                 blocks=16,block_size=4,max_context=64,max_rows=16,max_sequences=4,max_requests=1,audit=True)
        t=w.submit(dict(prompt="1",max_tokens=3)); self.assertTrue(entered.wait(5))
        with self.assertRaises(Busy): w.submit(dict(prompt="2"))
        closer=threading.Thread(target=w.close); closer.start()
        time.sleep(.02)
        with self.assertRaises(Busy): w.submit(dict(prompt="3"))
        gate.set(); closer.join(10); self.assertFalse(closer.is_alive())
        self.assertTrue(t.future.done())
        with self.assertRaises(RuntimeError): t.future.result()

    def test_device_failure_fails_all_tickets_and_poisoned_worker_stops(self):
        gate,entered=threading.Event(),threading.Event()
        w=Worker(lambda:FakeBackend(gate=gate,entered=entered,fail=True),FakeTokenizer,
                 blocks=16,block_size=4,max_context=64,max_rows=16,max_sequences=4,audit=True)
        try:
            a=w.submit(dict(prompt="1")); self.assertTrue(entered.wait(5))
            b=w.submit(dict(prompt="2")); gate.set()
            for t in (a,b):
                with self.assertRaises(RuntimeError): t.future.result(10)
        finally: gate.set(); w.close()
        self.assertIsInstance(w.failure,RuntimeError)

    def test_preemption_recomputes_without_corrupting_outputs(self):
        w=Worker(FakeBackend,FakeTokenizer,blocks=5,block_size=4,max_context=64,
                 max_rows=8,max_sequences=4,max_requests=16,audit=True)
        try:
            ts=[w.submit(dict(prompt=f"1 2 {i}",max_tokens=10)) for i in range(8)]
            actual=[t.future.result(20) for t in ts]
        finally: w.close()
        serial=worker(batch=False)
        try:
            expected=[serial.submit(dict(prompt=f"1 2 {i}",max_tokens=10)).future.result(10) for i in range(8)]
        finally: serial.close()
        self.assertEqual(actual,expected); self.assertGreater(w.stats["preemptions"],0)

    def test_invalid_request_does_not_kill_worker(self):
        w=worker()
        try:
            for req in (dict(prompt=""),dict(prompt="1",n=0),dict(prompt="1",temperature=float("nan")),
                        dict(prompt="1",beam_width=2,n=2),dict(prompt="1",max_tokens=True)):
                with self.assertRaises(ValueError): w.submit(req).future.result(10)
            self.assertEqual(len(w.submit(dict(prompt="1",max_tokens=2)).future.result(10)["choices"]),1)
        finally: w.close()

    def test_slow_stream_consumer_is_cancelled_without_leaking_cache(self):
        w=Worker(FakeBackend,FakeTokenizer,blocks=80,block_size=4,max_context=256,
                 max_rows=32,max_sequences=4,max_requests=4,audit=True)
        try:
            ticket=w.submit(dict(prompt="1",max_tokens=160,stream=True))
            # Deliberately never drain the bounded stream queue.
            import queue
            with self.assertRaises(queue.Full): ticket.future.result(10)
            result=w.submit(dict(prompt="2",max_tokens=3)).future.result(10)
            self.assertEqual(result["completion_tokens"],3)
        finally: w.close()

    def test_eos_stops_and_ignore_eos_continues(self):
        class EosBackend(FakeBackend): eos={0}
        w=Worker(EosBackend,FakeTokenizer,blocks=32,block_size=4,max_context=64,
                 max_rows=16,max_sequences=4,audit=True)
        try:
            a=w.submit(dict(prompt="0",max_tokens=5,n=3)).future.result(10)
            self.assertEqual([c["token_ids"] for c in a["choices"]],[[0]]*3)
            self.assertTrue(all(c["finish_reason"]=="stop" for c in a["choices"]))
            b=w.submit(dict(prompt="0",max_tokens=5,ignore_eos=True)).future.result(10)
            self.assertEqual(b["choices"][0]["token_ids"],[0]*5)
            beam=w.submit(dict(prompt="0",max_tokens=4,beam_width=3)).future.result(10)
            self.assertEqual(beam["choices"][0]["token_ids"],[0])
            self.assertTrue(all(c["token_ids"].count(0)<=1 for c in beam["choices"]))
        finally: w.close()

    def test_concurrent_submit_cancel_stress_all_futures_settle(self):
        w=worker()
        try:
            def client(i):
                ticket=w.submit(dict(prompt=f"1 2 {i%10}",max_tokens=12,n=2,temperature=.8,seed=i))
                if i%3==0: ticket.cancelled.set()
                try: return ticket.future.result(20)
                except RuntimeError as e:
                    self.assertIn("cancelled",str(e)); return None
            with ThreadPoolExecutor(16) as ex:
                results=list(ex.map(client,range(64)))
            self.assertTrue(any(x is not None for x in results))
        finally: w.close()

    def test_telemetry_correlates_requests_and_iterations(self):
        import tempfile
        import json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"events.jsonl"; w=worker(telemetry_path=path)
            try:
                ticket=w.submit(dict(prompt="1 2 3",max_tokens=3,stream=True),
                                client_request_id="aiperf-id")
                ticket.future.result(10)
            finally: w.close()
            events=[json.loads(s) for s in path.read_text().splitlines()]
            end=next(e for e in events if e["event"]=="request_end")
            self.assertEqual(end["id"],ticket.ident)
            self.assertEqual(end["generated_tokens"],3)
            self.assertEqual(end["client_request_id"],"aiperf-id")
            self.assertLessEqual(end["first_logits_seconds"],end["ttft_worker_seconds"])
            self.assertLessEqual(end["ttft_worker_seconds"],end["first_delta_seconds"])
            self.assertGreater(end["prefill_forward_seconds"],0)
            self.assertGreater(end["decode_forward_seconds"],0)
            self.assertGreaterEqual(end["ttft_worker_seconds"],end["queue_and_tokenize_seconds"])
            self.assertTrue(any(e["event"]=="iteration" and e["used_blocks"]>0 for e in events))


if __name__=="__main__": unittest.main()
