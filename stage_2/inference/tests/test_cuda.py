"""Real CUDA tests, also run under all four Compute Sanitizer tools.

The independent tiny-model NumPy oracle needs no HF/PyTorch installation.
Set QWEN_CUDA_LIBRARY to the compiled runtime; absent CUDA is an explicit skip.
"""
import json
import os
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
from backend import CudaBackend
from cache import BlockManager


def bf(x):
    bits=np.asarray(x,dtype=np.float32).copy().view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)


def tiny_model(directory):
    cfg=dict(model_type="qwen3",hidden_size=16,intermediate_size=32,num_hidden_layers=2,
             num_attention_heads=4,num_key_value_heads=2,head_dim=8,vocab_size=32,
             rms_norm_eps=1e-6,rope_theta=1000000.,max_position_embeddings=256,
             eos_token_id=[],tie_word_embeddings=False)
    rng=np.random.default_rng(2081)
    def rand(shape): return bf(rng.normal(0,.12,shape))
    weights={"model.embed_tokens.weight":rand((32,16)),"lm_head.weight":rand((32,16)),
             "model.norm.weight":bf(np.ones(16))}
    for l in range(2):
        p=f"model.layers.{l}."
        for name in ("input_layernorm","post_attention_layernorm"):
            weights[p+name+".weight"]=bf(np.ones(16))
        for name,shape in {"q_proj":(32,16),"k_proj":(16,16),"v_proj":(16,16),"o_proj":(16,32)}.items():
            weights[p+"self_attn."+name+".weight"]=rand(shape)
        for name in ("q_norm","k_norm"): weights[p+"self_attn."+name+".weight"]=bf(np.ones(8))
        for name,shape in {"gate_proj":(32,16),"up_proj":(32,16),"down_proj":(16,32)}.items():
            weights[p+"mlp."+name+".weight"]=rand(shape)
    data=bytearray(); header={}
    for name,w in weights.items():
        raw=(w.view(np.uint32)>>16).astype("<u2").tobytes()
        header[name]=dict(dtype="BF16",shape=list(w.shape),data_offsets=[len(data),len(data)+len(raw)])
        data.extend(raw)
    encoded=json.dumps(header).encode(); encoded+=b" "*((-len(encoded))%8)
    (directory/"model.safetensors").write_bytes(struct.pack("<Q",len(encoded))+encoded+data)
    (directory/"config.json").write_text(json.dumps(cfg))
    return weights


def oracle(ids,w):
    """Full-prefix reference computes attention directly, with no cache or tables."""
    def norm(x,weight): return bf(x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)*weight)
    def linear(x,name): return bf(x@w[name].T)
    x=w["model.embed_tokens.weight"][ids]
    length=len(ids)
    angle=np.arange(length,dtype=np.float32)[:,None]*np.power(np.float32(1e6),-2*np.arange(4,dtype=np.float32)/8)
    co=bf(np.cos(angle))[:,None,:]; si=bf(np.sin(angle))[:,None,:]
    def rotate(t):
        a,b=t[:,:,:4],t[:,:,4:]
        return bf(np.concatenate([a*co-b*si,b*co+a*si],axis=-1))
    for l in range(2):
        p=f"model.layers.{l}."
        n=norm(x,w[p+"input_layernorm.weight"])
        q=rotate(norm(linear(n,p+"self_attn.q_proj.weight").reshape(length,4,8),w[p+"self_attn.q_norm.weight"]))
        k=rotate(norm(linear(n,p+"self_attn.k_proj.weight").reshape(length,2,8),w[p+"self_attn.k_norm.weight"]))
        v=linear(n,p+"self_attn.v_proj.weight").reshape(length,2,8)
        att=np.empty((length,4,8),np.float32)
        for pos in range(length):
            for head in range(4):
                scores=k[:pos+1,head//2]@q[pos,head]/np.sqrt(np.float32(8))
                prob=np.exp(scores-scores.max()); prob/=prob.sum()
                att[pos,head]=prob@v[:pos+1,head//2]
        x=bf(x+linear(bf(att.reshape(length,32)),p+"self_attn.o_proj.weight"))
        n=norm(x,w[p+"post_attention_layernorm.weight"])
        gate=linear(n,p+"mlp.gate_proj.weight"); up=linear(n,p+"mlp.up_proj.weight")
        x=bf(x+linear(bf(gate/(1+np.exp(-gate))*up),p+"mlp.down_proj.weight"))
    return linear(norm(x[-1:],w["model.norm.weight"]),"lm_head.weight")[0]


@unittest.skipUnless(os.environ.get("QWEN_CUDA_LIBRARY"),"CUDA runtime not configured")
class CudaTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.path=Path(self.tmp.name)
        self.weights=tiny_model(self.path)
        self.backend=CudaBackend(self.path,os.environ["QWEN_CUDA_LIBRARY"],blocks=128,
                                 block_size=4,max_context=160,max_rows=160,max_sequences=8)
        self.cache=BlockManager(128,4,self.backend.copy_block,max_context=160)
        self.live=[]
    def tearDown(self):
        for s in self.live: self.cache.reset(s)
        self.cache.close(); self.backend.close(); self.tmp.cleanup()
    def new(self,ids):
        s=self.cache.new(ids); self.live.append(s); return s
    def run_to(self,s,end):
        self.cache.prepare(s,end); out=self.backend.forward([s],[end])[0]; s.computed=end
        return out

    def test_full_and_chunked_prefill_match_independent_oracle(self):
        ids=[1,3,5,7,9,2,4,6,8]
        expected=oracle(ids,self.weights)
        full=self.run_to(self.new(ids),len(ids))
        s=self.new(ids)
        for end in (3,4,7,9): chunked=self.run_to(s,end)
        np.testing.assert_allclose(full,expected,atol=.04,rtol=.04)
        np.testing.assert_allclose(chunked,full,atol=.01,rtol=.01)

    def test_shared_tail_cow_and_mixed_batch_match_oracle(self):
        root=self.new([1,2,3]); self.run_to(root,3)
        children=[self.cache.fork(root) for _ in range(3)]; self.live.extend(children)
        for i,s in enumerate(children):
            s.tokens.extend([i+5,i+9]); self.cache.prepare(s,5)
        logits=self.backend.forward(children,[5]*3)
        for s,actual in zip(children,logits):
            s.computed=5
            np.testing.assert_allclose(actual,oracle(s.tokens,self.weights),atol=.04,rtol=.04)
        np.testing.assert_allclose(self.run_to(self.new(root.tokens),3),oracle(root.tokens,self.weights),atol=.04,rtol=.04)
        self.cache.audit(self.live)

    def test_fragmented_blocks_and_context_beyond_128(self):
        blockers=[self.new([i%31]) for i in range(80)]
        for s in blockers: self.run_to(s,1)
        for s in blockers[::2]: self.cache.reset(s)
        ids=[i%31 for i in range(130)]; s=self.new(ids)
        actual=self.run_to(s,130)
        self.assertTrue(any(b-a!=1 for a,b in zip(s.blocks,s.blocks[1:])))
        np.testing.assert_allclose(actual,oracle(ids,self.weights),atol=.04,rtol=.04)

    def test_native_rejects_aliases_before_gpu_launch(self):
        a=self.new([1,2]); self.run_to(a,2)
        b=self.cache.fork(a); self.live.append(b)
        for s in (a,b): s.tokens.append(3)
        with self.assertRaisesRegex(RuntimeError,"Duplicate cache write slot"):
            self.backend.forward([a,b],[3,3])
        # A write/read alias at distinct offsets must also fail before launch.
        b.computed=1
        with self.assertRaisesRegex(RuntimeError,"Shared block is writable"):
            self.backend.forward([a,b],[3,2])

    def test_native_owner_thread_guard(self):
        failures=[]
        def wrong_thread():
            try: self.backend.copy_block(0,1)
            except RuntimeError as e: failures.append(str(e))
        t=threading.Thread(target=wrong_thread); t.start(); t.join()
        self.assertEqual(len(failures),1); self.assertIn("non-owner",failures[0])

    def test_free_reuse_and_prefix_lifetime(self):
        for i in range(12):
            ids=[i%31,2,3,4,5]; root=self.new(ids); self.run_to(root,5)
            self.cache.publish(root,5); self.cache.reset(root)
            s=self.new(ids+[7]); self.assertGreaterEqual(s.computed,4)
            actual=self.run_to(s,6)
            np.testing.assert_allclose(actual,oracle(s.tokens,self.weights),atol=.04,rtol=.04)
            self.cache.reset(s)
        self.cache.audit(self.live)


@unittest.skipUnless(os.environ.get("QWEN_CUDA_LIBRARY"), "CUDA runtime not configured")
class ContiguousKVTests(unittest.TestCase):
    # Cross the 16-token attention tile boundary within a single cache page.
    capacity = 19

    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.weights = tiny_model(Path(directory))
        self.backend = CudaBackend(
            Path(directory), os.environ["QWEN_CUDA_LIBRARY"], blocks=1,
            block_size=self.capacity, max_context=self.capacity,
            max_rows=self.capacity, max_sequences=1,
        )
        self.addCleanup(self.backend.close)
        self.cache = BlockManager(
            1, self.capacity, self.backend.copy_block,
            mode="contiguous", max_context=self.capacity,
        )
        self.addCleanup(self.cache.close)

    def new_sequence(self, tokens):
        sequence = self.cache.new(tokens)
        self.addCleanup(self.cache.reset, sequence)
        return sequence

    def run_to(self, sequence, end):
        self.cache.prepare(sequence, end)
        logits = self.backend.forward([sequence], [end])[0]
        sequence.computed = end
        return logits

    def test_prefill_and_two_decode_steps_match_full_replay(self):
        tokens = [i % 31 for i in range(self.capacity)]
        prefixes = (self.capacity - 2, self.capacity - 1, self.capacity)
        sequence = self.new_sequence(tokens)
        replay = []

        with patch.object(self.backend.lib, "q_forward", wraps=self.backend.lib.q_forward) as forward:
            for end in prefixes:
                replay.append(self.run_to(sequence, end))
                self.cache.reset(sequence)
            incremental = [self.run_to(sequence, end) for end in prefixes]

        submitted_rows = [call.args[6] for call in forward.call_args_list]
        self.assertEqual(submitted_rows, [*prefixes, prefixes[0], 1, 1])
        for end, expected, actual in zip(prefixes, replay, incremental):
            with self.subTest(prefix_length=end):
                np.testing.assert_allclose(expected, oracle(tokens[:end], self.weights),
                                           atol=.04, rtol=.04, equal_nan=False)
                np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01, equal_nan=False)
                self.assertEqual(int(actual.argmax()), int(expected.argmax()))

    def test_request_reset_reuses_slot_without_stale_history(self):
        short = self.new_sequence([8, 4])
        expected = self.run_to(short, 2)
        self.cache.reset(short)

        long = self.new_sequence([i % 31 for i in range(self.capacity)])
        self.run_to(long, self.capacity)
        old_blocks = long.blocks.copy()
        self.cache.reset(long)
        self.assertEqual(long.computed, 0)

        next_request = self.new_sequence([8, 4])
        actual = self.run_to(next_request, 2)
        self.assertEqual(next_request.blocks, old_blocks)
        np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01, equal_nan=False)

    def test_exact_capacity_and_reject_one_more_before_forward(self):
        sequence = self.new_sequence([i % 31 for i in range(self.capacity)])
        self.run_to(sequence, self.capacity - 1)
        logits = self.run_to(sequence, self.capacity)
        self.assertTrue(np.isfinite(logits).all())

        sequence.tokens.append(4)
        with patch.object(self.backend.lib, "q_forward", wraps=self.backend.lib.q_forward) as forward:
            with self.assertRaisesRegex(ValueError, "Invalid append range"):
                self.run_to(sequence, self.capacity + 1)
        forward.assert_not_called()
        self.assertEqual(sequence.computed, self.capacity)


if __name__=="__main__": unittest.main()
