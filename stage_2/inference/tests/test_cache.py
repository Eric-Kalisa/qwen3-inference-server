import random
import threading
import unittest
from cache import BlockManager, CacheFull


class CacheTests(unittest.TestCase):
    def make(self, count=16, size=4, **kw):
        self.data = {}
        def copy(a, b):
            self.data[b] = list(self.data[a])
        return BlockManager(count, size, copy, max_context=16, **kw)

    def write(self, cache, seq, end):
        cache.prepare(seq, end)
        for p in range(seq.computed, end):
            self.data.setdefault(seq.blocks[p//cache.block_size], [None]*cache.block_size)[p%cache.block_size] = seq.tokens[p]
        seq.computed = end

    def test_shared_partial_tail_cow_preserves_parent_and_sibling(self):
        c = self.make(); a = c.new([1,2,3]); self.write(c,a,3)
        b, d = c.fork(a), c.fork(a)
        b.tokens.append(7); d.tokens.append(8)
        self.write(c,b,4); self.write(c,d,4)
        self.assertEqual(len({a.blocks[0],b.blocks[0],d.blocks[0]}),3)
        self.assertEqual(self.data[a.blocks[0]], [1,2,3,None])
        self.assertEqual(self.data[b.blocks[0]], [1,2,3,7])
        self.assertEqual(self.data[d.blocks[0]], [1,2,3,8])
        c.audit([a,b,d])
        for s in (a,b,d): c.reset(s)
        c.close()

    def test_full_boundary_appends_without_copy(self):
        c=self.make(); a=c.new([1,2,3,4]); self.write(c,a,4)
        b=c.fork(a); b.tokens.append(5); self.write(c,b,5)
        self.assertEqual(a.blocks[0],b.blocks[0]); self.assertEqual(c.copies,0)
        c.reset(a); c.reset(b); c.close()

    def test_longest_exact_prefix_and_lifetime(self):
        c=self.make(); a=c.new(list(range(12))); self.write(c,a,12); c.publish(a,12)
        short=c.new(list(range(4))); self.write(c,short,4); c.publish(short,4)
        b=c.new(list(range(12))+[55])
        self.assertEqual(b.computed,12)
        wrong=c.new([99]+list(range(1,12))); self.assertEqual(wrong.computed,0)
        c.reset(a); c.reset(short)
        while c.evict_prefix(): pass
        c.audit([b,wrong]); self.assertEqual(len(c.free),13)
        c.reset(b); c.reset(wrong); c.close()

    def test_exact_prompt_recomputes_last_block_for_logits(self):
        c=self.make(); a=c.new(list(range(8))); self.write(c,a,8); c.publish(a,8)
        b=c.new(list(range(8))); self.assertLess(b.computed,8)
        c.reset(a); c.reset(b); c.close()

    def test_oom_is_transactional(self):
        c=self.make(count=1); a=c.new([1,2,3]); self.write(c,a,3); b=c.fork(a)
        before=(list(b.blocks),list(c.refs)); b.tokens.append(9)
        with self.assertRaises(CacheFull): c.prepare(b,4)
        self.assertEqual((b.blocks,c.refs),before); c.audit([a,b])
        c.reset(a); c.reset(b); c.close()

    def test_copy_failure_rolls_back(self):
        c=self.make(); a=c.new([1]); self.write(c,a,1); b=c.fork(a)
        c.copy_block=lambda a,b: (_ for _ in ()).throw(RuntimeError("injected copy failure"))
        b.tokens.append(2)
        with self.assertRaises(RuntimeError): c.prepare(b,2)
        c.audit([a,b]); self.assertEqual(len(c.free),15)
        c.reset(a); c.reset(b); c.close()

    def test_non_owner_rejected(self):
        c=self.make(); failures=[]
        def mutate():
            try: c.new([1])
            except RuntimeError as e: failures.append(e)
        t=threading.Thread(target=mutate); t.start(); t.join()
        self.assertEqual(len(failures),1); c.close()

    def test_contiguous_reserves_run_and_copies_fork(self):
        c=self.make(mode="contiguous"); a=c.new([1,2]); self.write(c,a,2)
        # Model unused slots are initialized by the CUDA allocator.
        for b in a.blocks: self.data.setdefault(b,[None]*4)
        b=c.fork(a)
        self.assertEqual(a.blocks,list(range(4))); self.assertEqual(b.blocks,list(range(4,8)))
        self.assertEqual(self.data[b.blocks[0]],[1,2,None,None])
        c.audit([a,b]); c.reset(a); c.reset(b); c.close()

    def test_random_fork_append_free_eviction_invariants(self):
        c=self.make(count=40); rng=random.Random(9821); live=[c.new([3])]
        for _ in range(2000):
            s=rng.choice(live); op=rng.randrange(4)
            try:
                if op==0 and len(live)<12: live.append(c.fork(s))
                elif op==1 and len(s.tokens)<16:
                    if s.computed<len(s.tokens): self.write(c,s,len(s.tokens))
                    else: s.tokens.append(rng.randrange(50)); self.write(c,s,len(s.tokens))
                elif op==2: c.publish(s,len(s.tokens))
                elif len(live)>1: c.reset(s); live.remove(s)
            except CacheFull: pass
            c.audit(live)
            for seq in live:
                cached=[self.data[seq.blocks[p//4]][p%4] for p in range(seq.computed)]
                self.assertEqual(cached,seq.tokens[:seq.computed])
        for s in live: c.reset(s)
        c.close()


if __name__ == "__main__": unittest.main()
