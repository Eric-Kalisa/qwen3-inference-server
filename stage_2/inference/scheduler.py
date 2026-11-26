"""
Scheduler contract
------------------
Receives: a request dict through Worker.submit().  Required: "prompt" (non-empty string).
Optional: "max_tokens" (default 32), "n" or "beam_width", "temperature", "seed",
"ignore_eos", "stream", and "skip_special_tokens"; invalid values raise ValueError.

Delivers: a concurrent.futures.Future on the Ticket returned by submit().  On success, the
Future result contains {choices, prompt_tokens, completion_tokens}; each choice contains
{text, token_ids, finish_reason, index, and score}.  Streaming also puts
{index, text, finish_reason} on Ticket.events as generated text becomes available.

Tickets: submit() raises Busy when shutdown has begun or no request slot is available.
Otherwise it returns Ticket(request, ident="cmpl-...", future=..., events=..., cancelled=...,
submitted=...).  Future exceptions use ValueError for bad requests, RuntimeError for cancel/worker
failure, CacheFull for exhausted capacity, and Busy for submit-time rejection.

Block responsibilities
----------------------
- Ticket/Job: hold request identity and mutable generation state.
- Worker.__init__: start the CUDA-owning worker and wait until it is ready.
- submit/close: admit or shut down producers and their request slots.
- admit: validate a request, tokenize it, allocate a sequence, and build a Job.
- finish: release KV cache references, report completion, set the Future, and free a slot.
- reset_job: preempt a request and invalidate all active logits and cache references.
- expand: turn logits into sampled or beam branches and fork KV-cache parents safely.
- decode: convert generated token IDs to text with the configured tokenizer options.
- record: append one JSON telemetry event, with telemetry failures made non-fatal.
- run: own the batch loop, forward selected sequences, expand jobs, and drain/shutdown safely.
- _admit: isolate admission errors and make them visible through Ticket.future.
"""
from concurrent.futures import Future
from dataclasses import dataclass, field
import itertools
import json
import queue
import threading
import time
import uuid
import numpy as np
from cache import BlockManager, CacheFull


class Busy(Exception):
    """Raised when submissions are shut down or all request slots are occupied."""

    pass


@dataclass
class Ticket:
    request: dict
    ident: str = field(default_factory=lambda: "cmpl-"+uuid.uuid4().hex)
    future: Future = field(default_factory=Future)
    events: queue.Queue = field(default_factory=lambda: queue.Queue(128))
    cancelled: threading.Event = field(default_factory=threading.Event)
    submitted: float = field(default_factory=time.perf_counter)


@dataclass(eq=False)
class Job:
    ticket: Ticket
    seqs: list
    prompt_length: int
    max_tokens: int
    width: int
    beam: bool
    temperature: float
    rng: object
    ignore_eos: bool
    logits: dict = field(default_factory=dict)
    texts: dict = field(default_factory=dict)
    first_token: float = None


class Worker:
    def __init__(self, backend_factory, tokenizer_factory, *, blocks=1024,
                 block_size=16, mode="paged", max_context=8192, max_rows=256,
                 max_sequences=32, max_requests=32, batch=True, prefix_capacity=64,
                 audit=False, telemetry_path=None):
        self.factories = backend_factory, tokenizer_factory
        self.options = dict(count=blocks, block_size=block_size, mode=mode,
                            max_context=max_context, prefix_capacity=prefix_capacity)
        self.max_context, self.max_rows = max_context, max_rows
        self.max_sequences = max_sequences
        self.batch, self.audit_enabled = batch, audit
        self.telemetry_path, self.telemetry = telemetry_path, None
        if min(blocks, block_size, max_context, max_rows, max_sequences, max_requests) < 1:
            raise ValueError("Capacities must be positive")
        if max_sequences > max_rows:
            raise ValueError("Sequence capacity cannot exceed token capacity")
        self.incoming = queue.Queue()
        self.slots = threading.BoundedSemaphore(max_requests)
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.ready = Future()
        self.closed = False
        self.stats = {"iterations": 0, "preemptions": 0, "max_batch_sequences": 0,
                      "prefill_tokens": 0, "decode_tokens": 0, "forward_wall_seconds": 0.0,
                      "peak_used_blocks": 0}
        self.thread = threading.Thread(target=self.run, name="qwen-cuda-owner", daemon=True)
        self.thread.start()
        self.ready.result(timeout=900)

    def submit(self, request):
        with self.lock:
            if self.closed or self.stopping.is_set():
                raise Busy("Server is shutting down")
            if not self.slots.acquire(blocking=False):
                raise Busy("Request queue is full")
            ticket = Ticket(dict(request))
            self.incoming.put(ticket)
            return ticket

    def finish(self, job, error=None):
        self.record(dict(event="request_end", id=job.ticket.ident, error=str(error) if error else None,
                         queue_and_tokenize_seconds=job.ticket.started-job.ticket.submitted,
                         ttft_worker_seconds=(job.first_token-job.ticket.submitted) if job.first_token else None,
                         total_worker_seconds=time.perf_counter()-job.ticket.submitted,
                         generated_tokens=sum(len(s.generated) for s in job.seqs)))
        for seq in job.seqs:
            self.cache.reset(seq)
        if not job.ticket.future.done():
            if error:
                job.ticket.future.set_exception(error)
            else:
                choices = []
                seqs = sorted(job.seqs, key=lambda s: s.score, reverse=True) if job.beam else job.seqs
                for i, seq in enumerate(seqs):
                    choices.append(dict(index=i, text=self.decode(job, seq),
                                        token_ids=list(seq.generated), finish_reason=seq.reason,
                                        score=seq.score))
                job.ticket.future.set_result(dict(choices=choices, prompt_tokens=job.prompt_length,
                                                  completion_tokens=sum(len(s.generated) for s in seqs)))
        self.slots.release()

    def admit(self, ticket):
        req = ticket.request
        def integer(name, default, low, high):
            value = req.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
            return value
        if not isinstance(req.get("prompt"), str):
            raise ValueError("prompt must be a string")
        maximum = integer("max_tokens", 32, 1, self.max_context)
        samples = integer("n", 1, 1, self.max_sequences)
        beams = integer("beam_width", 1, 1, self.max_sequences)
        if beams > 1 and samples != 1:
            raise ValueError("Use n or beam_width, not both")
        temperature = float(req.get("temperature", 0))
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if beams > 1 and temperature != 0:
            raise ValueError("Beam search uses untempered log probabilities; temperature must be 0")
        seed = integer("seed", 0, 0, 2**32-1)
        tokens = self.tokenizer.encode(req["prompt"])
        if not tokens or len(tokens) + maximum > self.max_context:
            raise ValueError("Empty prompt or prompt + output exceeds context limit")
        width = max(samples, beams)
        seq = self.cache.new(tokens)
        ticket.started = time.perf_counter()
        return Job(ticket, [seq], len(tokens), maximum, width, beams > 1,
                   temperature, np.random.default_rng(seed), bool(req.get("ignore_eos", False)))

    def reset_job(self, job):
        for seq in job.seqs:
            self.cache.reset(seq)
        job.logits.clear()
        self.stats["preemptions"] += 1

    def expand(self, job):
        """Fork all surviving branches before releasing their parent references.

        Beam score is cumulative log probability, without a length penalty.
        Completed beams compete with live candidates in a fixed-width beam.
        """
        if job.first_token is None:
            job.first_token = time.perf_counter()
        candidates = []
        serial = itertools.count()
        for i, seq in enumerate(job.seqs):
            if seq.finished:
                candidates.append((seq.score, next(serial), i, None))
                continue
            logits = np.asarray(job.logits[id(seq)], np.float64)
            if not np.isfinite(logits).all():
                raise RuntimeError("Non-finite logits")
            if job.beam:
                z = logits - logits.max()
                logp = z - np.log(np.exp(z).sum())
                top = np.argsort(-logp, kind="stable")[:job.width]
                candidates.extend((seq.score+float(logp[t]), next(serial), i, int(t)) for t in top)
            else:
                count = job.width if len(job.seqs) == 1 and not seq.generated else 1
                if job.temperature == 0:
                    tokens = [int(logits.argmax())]*count
                else:
                    z = (logits-logits.max())/job.temperature
                    p = np.exp(z); p /= p.sum()
                    tokens = job.rng.choice(len(p), size=count, p=p)
                candidates.extend((seq.score, next(serial), i, int(t)) for t in tokens)
        chosen = sorted(candidates, key=lambda c: (-c[0], c[1]))[:job.width] if job.beam else candidates
        children, fresh, reused = [], [], set()
        try:
            for score, _, parent, token in chosen:
                if parent not in reused:
                    child = job.seqs[parent]
                    reused.add(parent)
                else:
                    child = self.cache.fork(job.seqs[parent])
                    fresh.append(child)
                children.append(child)
        except BaseException:
            for child in fresh:
                self.cache.reset(child)
            raise
        for child, (score, _, _, token) in zip(children, chosen):
            child.score = score
            if token is not None:
                child.tokens.append(token); child.generated.append(token)
                stop = token in self.backend.eos and not job.ignore_eos
                child.finished = stop or len(child.generated) >= job.max_tokens
                child.reason = "stop" if stop else "length"
        for index, seq in enumerate(job.seqs):
            if index not in reused:
                self.cache.reset(seq)
        job.seqs = children
        job.logits.clear()
        if self.options["mode"] == "none":
            for child in children:
                self.cache.reset(child)
        # A winning beam can change its earlier text; emit beams only at completion.
        if not job.beam and job.ticket.request.get("stream", False):
            for i, seq in enumerate(children):
                text = self.decode(job, seq)
                # Hold incomplete UTF-8 tails instead of sending a replacement
                # character that would later need to be retracted.
                text = text.rstrip("\ufffd")
                old = job.texts.get(i, "")
                if not text.startswith(old):
                    raise RuntimeError("Tokenizer emitted a non-monotonic text prefix")
                delta = text[len(old):]
                if delta:
                    job.ticket.events.put_nowait(dict(index=i, text=delta, finish_reason=None))
                    job.texts[i] = text

    def decode(self, job, seq):
        return self.tokenizer.decode(seq.generated,
                                     skip_special_tokens=job.ticket.request.get("skip_special_tokens", True))

    def record(self, event):
        if self.telemetry:
            event["time"] = time.time()
            try:
                self.telemetry.write(json.dumps(event)+"\n")
                self.telemetry.flush()
            except OSError:
                # Diagnostics must never strand live cache references.
                try:
                    self.telemetry.close()
                except OSError:
                    pass
                self.telemetry = None
                self.stats["telemetry_errors"] = self.stats.get("telemetry_errors", 0)+1

    def run(self):
        active = []
        self.backend = self.tokenizer = self.cache = None
        try:
            self.backend = self.factories[0]()
            self.tokenizer = self.factories[1]()
            self.cache = BlockManager(copy_block=self.backend.copy_block, **self.options)
            if self.telemetry_path:
                self.telemetry = open(self.telemetry_path, "a", encoding="utf-8")
            self.ready.set_result(True)
            while not self.stopping.is_set():
                if not active:
                    try:
                        pending = self.incoming.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    self._admit(pending, active)
                while len(active) < 32:
                    try:
                        self._admit(self.incoming.get_nowait(), active)
                    except queue.Empty:
                        break
                for job in list(active):
                    if job.ticket.cancelled.is_set():
                        self.finish(job, RuntimeError("Request cancelled")); active.remove(job)
                selected, ends, jobs = [], [], []
                budget = self.max_rows
                # Reserve one decode row per runnable branch before prompt work.
                groups = active if self.batch else active[:1]
                ordered = sorted(((j,s) for j in groups for s in j.seqs
                                  if not s.finished and id(s) not in j.logits),
                                 key=lambda js: (len(js[1].tokens)-js[1].computed != 1,
                                                 active.index(js[0])))
                for job, seq in ordered:
                    if budget == 0 or len(selected) == self.max_sequences:
                        break
                    if job not in active:
                        continue
                    remaining_decodes = sum(len(s.tokens)-s.computed == 1 for j,s in ordered
                                            if not any(s is chosen for chosen in selected) and j in active)
                    take = 1 if len(seq.tokens)-seq.computed == 1 else min(32, max(1, budget-remaining_decodes))
                    end = min(len(seq.tokens), seq.computed + min(take, budget))
                    try:
                        self.cache.prepare(seq, end)
                    except CacheFull:
                        # Preempt whole younger request groups, never arbitrary
                        # pages. Skip already-selected groups until this GPU call
                        # finishes, so staged mappings cannot be invalidated.
                        victims = [j for j in reversed(active[active.index(job)+1:])
                                   if j not in jobs and any(s.blocks for s in j.seqs)]
                        for victim in victims:
                            self.reset_job(victim)
                        try:
                            self.cache.prepare(seq, end)
                        except CacheFull as exc:
                            if active[0] is job and not selected:
                                self.finish(job, exc); active.remove(job)
                            continue
                    selected.append(seq); ends.append(end); jobs.append(job)
                    budget -= end-seq.computed
                if selected:
                    start = time.perf_counter()
                    logits = self.backend.forward(selected, ends)
                    elapsed = time.perf_counter()-start
                    self.stats["forward_wall_seconds"] += elapsed
                    self.stats["max_batch_sequences"] = max(self.stats["max_batch_sequences"], len(selected))
                    self.stats["iterations"] += 1
                    used = len(self.cache.refs)-len(self.cache.free)
                    self.stats["peak_used_blocks"] = max(self.stats["peak_used_blocks"], used)
                    self.record(dict(event="iteration", sequences=len(selected), rows=self.max_rows-budget,
                                     active_requests=len(active), used_blocks=used,
                                     forward_wall_seconds=elapsed, cow_copies=self.cache.copies,
                                     prefix_hit_tokens=self.cache.hits, preemptions=self.stats["preemptions"]))
                    for seq, end, job, logit in zip(selected, ends, jobs, logits):
                        count = end-seq.computed
                        self.stats["decode_tokens" if count == 1 and seq.generated else "prefill_tokens"] += count
                        seq.computed = end
                        self.cache.publish(seq, job.prompt_length)
                        if end == len(seq.tokens):
                            job.logits[id(seq)] = logit
                for job in list(active):
                    if job.ticket.cancelled.is_set():
                        self.finish(job, RuntimeError("Request cancelled")); active.remove(job)
                        continue
                    if all(s.finished or id(s) in job.logits for s in job.seqs):
                        try:
                            self.expand(job)
                            if all(s.finished for s in job.seqs):
                                self.finish(job); active.remove(job)
                        except (CacheFull, ValueError, RuntimeError, queue.Full) as exc:
                            self.finish(job, exc); active.remove(job)
                if self.audit_enabled:
                    self.cache.audit([s for j in active for s in j.seqs])
        except BaseException as exc:
            if not self.ready.done():
                self.ready.set_exception(exc)
            for job in active:
                self.finish(job, exc)
            active.clear()
            self.failure = exc
        finally:
            # Close submissions before draining; no producer can enqueue after it.
            with self.lock:
                self.closed = True
            for job in active:
                self.finish(job, RuntimeError("Server stopped"))
            while True:
                try:
                    ticket = self.incoming.get_nowait()
                except queue.Empty:
                    break
                ticket.future.set_exception(RuntimeError("Server stopped")); self.slots.release()
            if self.cache:
                self.stats.update(prefix_hit_tokens=self.cache.hits, cow_copies=self.cache.copies,
                                  prefix_evictions=self.cache.evictions)
                self.cache.close()
            if self.tokenizer:
                self.tokenizer.close()
            if self.backend:
                self.backend.close()
            if self.telemetry:
                self.telemetry.close()

    def _admit(self, ticket, active):
        try:
            if ticket.cancelled.is_set():
                raise RuntimeError("Request cancelled")
            active.append(self.admit(ticket))
        except Exception as exc:
            ticket.future.set_exception(exc); self.slots.release()

    def close(self):
        with self.lock:
            self.closed = True
            self.stopping.set()
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            raise RuntimeError("GPU worker did not stop within 60 seconds")
