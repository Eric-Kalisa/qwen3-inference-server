"""Single-owner KV metadata. A reference is held by each sequence and prefix entry.

GPU operations complete before any metadata is published or any block is reused.
The scheduler owns this object; HTTP threads must only submit messages to it.
"""
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
import threading


class CacheFull(Exception):
    """Raised when the cache cannot allocate another KV block.

    This is used by BlockManager.allocate() when the buffer is exhausted and no
    prefix can be evicted to free space. It signals a full-cache condition rather
    than a logic error in the caller.
    """
    pass


@dataclass(eq=False)
class Sequence:
    """State for one prompt/generation request and its KV-block references.

    In this file, BlockManager.new() creates the initial Sequence and
    BlockManager.fork() creates a child. The scheduler/inference worker is
    expected to use these objects; the exact external caller cannot be
    identified from this file alone.
    """

    # Prompt/input token IDs. new() copies them and checks them against cached
    # prefixes; fork() copies them; publish() reads them to build a prefix key.
    tokens: list

    # IDs of KV blocks held by this sequence. new()/fork()/prepare() assign
    # them; retain/release manage references; publish() reads them; reset()
    # releases and clears them.
    blocks: list = field(default_factory=list)

    # Count of leading tokens whose KV state is computed. new() initializes it
    # on a prefix hit; prepare(), fork(), and publish() read it; reset() sets it
    # to zero
    computed: int = 0

    # Generated token IDs. fork() copies this list;
    generated: list = field(default_factory=list)

    # Accumulated sequence score. fork() copies it;
    score: float = 0.0

    # Whether generation has ended. fork() copies it;
    finished: bool = False

   # End condition, defaulting to the length limit. fork() copies it;
    reason: str = "length"


class BlockManager:
    def __init__(self, count, block_size, copy_block, mode="paged", max_context=8192,
                 prefix_capacity=64):
        # Validate the requested cache layout before creating any runtime state.
        # A cache must have at least one block, each block must be at least one
        # token wide, and the allocator mode must be one of the supported modes.
        if count < 1 or block_size < 1 or mode not in ("none", "contiguous", "paged"):
            raise ValueError("Invalid cache configuration")

        # Record the thread that owns this cache instance. The rest of the cache
        # API enforces single-thread access through check_owner(), preventing the
        # scheduler and worker threads from mutating the same metadata concurrently.
        self.owner = threading.get_ident()

        # Store the basic memory geometry for the block pool.
        # block_size is the width of each KV block in tokens; mode chooses how
        # allocation and eviction are handled.
        self.block_size, self.mode = block_size, mode

        # max_blocks is the maximum number of blocks we can index within the
        # context window: ceil(max_context / block_size). This is used when
        # validating append ranges and when contiguous allocation needs a full
        # window-sized reservation.
        self.max_blocks = (max_context + block_size - 1) // block_size

        # refs tracks the number of live references for each physical block.
        # A value of 0 means the block is free and may be allocated; values > 0
        # mean it is in use by one or more sequences/prefixes.
        self.refs = [0] * count

        # free is the pool of block IDs available for allocation. Since the set is
        # kept in sorted order only when iterated explicitly, allocation logic can
        # choose the lowest-numbered blocks first to keep reuse deterministic.
        self.free = set(range(count))

        # prefixes stores cached prompt-prefix keys and the blocks that back each
        # prefix. OrderedDict preserves insertion order so the oldest prefix can be
        # evicted first when the prefix cache is full.
        self.prefixes = OrderedDict()

        # Prefix caching is only enabled in paged mode. In none/contiguous modes,
        # the metadata is used differently and the prefix cache is intentionally
        # disabled.
        self.prefix_capacity = prefix_capacity if mode == "paged" else 0

        # copy_block is the callback used to duplicate a physical KV block. The
        # cache manager relies on this so that copy-on-write and fork/prepare
        # behaviors can keep the block manager generic over the actual storage.
        self.copy_block = copy_block

        # Statistics track how often blocks are copied, prefix hits occur, and
        # prefix eviction happened. These counters are used for debugging and
        # performance checks.
        self.copies = self.hits = self.evictions = 0

    def check_owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("Cache accessed outside its owning worker")

    def retain(self, blocks):
        self.check_owner()
        for b in blocks:
            assert self.refs[b] > 0
            self.refs[b] += 1

    def release(self, blocks):
        self.check_owner()
        for b in blocks:
            assert self.refs[b] > 0, "Double free"
            self.refs[b] -= 1
            if self.refs[b] == 0:
                self.free.add(b)

    def evict_prefix(self):
        self.check_owner()
        if not self.prefixes:
            return False
        _, blocks = self.prefixes.popitem(last=False)
        self.release(blocks)
        self.evictions += 1
        return True

    def allocate(self, count=1, contiguous=False):
        self.check_owner()
        while True:
            if contiguous:
                for start in sorted(self.free):
                    ids = list(range(start, start + count))
                    if all(b in self.free for b in ids):
                        break
                else:
                    ids = None
            else:
                ids = sorted(self.free)[:count] if len(self.free) >= count else None
            if ids is not None:
                for b in ids:
                    self.free.remove(b)
                    assert self.refs[b] == 0
                    self.refs[b] = 1
                return ids
            if not self.evict_prefix():
                raise CacheFull("KV block pool exhausted")

    def new(self, tokens):
        self.check_owner()
        seq = Sequence(list(tokens))
        # Leave at least one prompt token to recompute its next-token logits.
        for key in sorted(self.prefixes, key=len, reverse=True):
            if len(key) <= len(tokens) - 1 and tuple(tokens[:len(key)]) == key:
                seq.blocks = list(self.prefixes[key])
                self.retain(seq.blocks)
                seq.computed = len(key)
                self.prefixes.move_to_end(key)
                self.hits += len(key)
                break
        return seq

    def prepare(self, seq, end):
        """Reserve append slots; privatize a shared tail before the first write.

        Failed allocation/copy leaves the sequence and its ownership unchanged.
        Only the valid tail is semantically readable; copies include unused bytes.
        """
        self.check_owner()
        if end <= seq.computed or end > self.max_blocks * self.block_size:
            raise ValueError("Invalid append range")
        needed = (end + self.block_size - 1) // self.block_size
        if self.mode == "contiguous":
            if not seq.blocks:
                seq.blocks = self.allocate(self.max_blocks, contiguous=True)
            return
        tail = seq.computed // self.block_size
        cow = (seq.computed % self.block_size != 0 and tail < len(seq.blocks)
               and self.refs[seq.blocks[tail]] > 1)
        extra = max(0, needed - len(seq.blocks))
        fresh = self.allocate(extra + int(cow)) if extra or cow else []
        try:
            if cow:
                self.copy_block(seq.blocks[tail], fresh[0])
        except BaseException:
            self.release(fresh)
            raise
        if cow:
            self.release([seq.blocks[tail]])
            seq.blocks[tail] = fresh.pop(0)
            self.copies += 1
        seq.blocks.extend(fresh)

    def fork(self, seq):
        self.check_owner()
        child = Sequence(list(seq.tokens), [], seq.computed, list(seq.generated),
                         seq.score, seq.finished, seq.reason)
        if self.mode == "contiguous":
            child.blocks = self.allocate(self.max_blocks, contiguous=True)
            try:
                used = (seq.computed + self.block_size - 1) // self.block_size
                for src, dst in zip(seq.blocks[:used], child.blocks[:used]):
                    self.copy_block(src, dst)
            except BaseException:
                self.release(child.blocks)
                raise
        else:
            child.blocks = list(seq.blocks)
            self.retain(child.blocks)
        return child

    def publish(self, seq, prompt_length):
        self.check_owner()
        if not self.prefix_capacity:
            return
        n = min(seq.computed, prompt_length) // self.block_size * self.block_size
        if not n:
            return
        key = tuple(seq.tokens[:n])
        if key in self.prefixes:
            self.prefixes.move_to_end(key)
            return
        blocks = tuple(seq.blocks[:n // self.block_size])
        self.retain(blocks)
        self.prefixes[key] = blocks
        while len(self.prefixes) > self.prefix_capacity:
            self.evict_prefix()

    def reset(self, seq):
        self.check_owner()
        self.release(seq.blocks)
        seq.blocks.clear()
        seq.computed = 0

    def audit(self, sequences):
        self.check_owner()
        expected = Counter(b for s in sequences for b in s.blocks)
        expected.update(b for bs in self.prefixes.values() for b in bs)
        assert all(self.refs[b] == expected[b] for b in range(len(self.refs)))
        assert self.free == {b for b, n in enumerate(self.refs) if n == 0}

    def close(self):
        self.check_owner()
        while self.evict_prefix():
            pass
        assert not any(self.refs), "Live sequence leaked at shutdown"
