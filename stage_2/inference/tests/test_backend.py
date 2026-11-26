import ctypes as C
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from backend import CudaBackend
from cache import Sequence
from test_cuda import tiny_model


class Function:
    def __init__(self, fn): self.fn=fn; self.argtypes=None; self.restype=None
    def __call__(self,*args):
        if self.argtypes is not None and len(args)!=len(self.argtypes):
            raise TypeError("Declared C ABI argument count does not match invocation")
        return self.fn(*args)


class FakeLibrary:
    def __init__(self):
        self.weights={}; self.destroyed=False
        self.q_error=Function(lambda:b"injected failure")
        self.q_create=Function(self.create)
        self.q_weight=Function(self.upload)
        self.q_forward=Function(self.forward)
        self.q_copy=Function(lambda h,a,b:0)
        self.q_destroy=Function(self.destroy)
    def create(self,h,mid,layers,qh,kh,dim,vocab,blocks,bs,context,rows,sequences,eps,theta):
        self.vocab=vocab; self.mb=(context+bs-1)//bs
        return 1
    def upload(self,h,name,ptr,count):
        name=name.decode()
        if name in self.weights: return -1
        self.weights[name]=np.ctypeslib.as_array((C.c_uint16*count).from_address(ptr)).copy()
        return 0
    def forward(self,h,ids,pos,owners,tables,last,rows,ns,out):
        def ints(ptr,n): return list((C.c_int*n).from_address(ptr))
        self.metadata=[ints(ids,rows),ints(pos,rows),ints(owners,rows),ints(tables,ns*self.mb),ints(last,ns)]
        np.ctypeslib.as_array((C.c_float*(ns*self.vocab)).from_address(out))[:]=np.arange(ns*self.vocab)
        return 0
    def destroy(self,h): self.destroyed=True; return 0


class BackendTests(unittest.TestCase):
    def test_loader_and_packed_metadata_without_torch(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td); weights=tiny_model(path); lib=FakeLibrary()
            with patch("backend.C.CDLL",return_value=lib):
                b=CudaBackend(path,"unused",blocks=8,block_size=4,max_context=16,max_rows=8,max_sequences=2)
            self.assertEqual(set(lib.weights),set(weights))
            for name,w in weights.items():
                np.testing.assert_array_equal(lib.weights[name],(w.view(np.uint32)>>16).astype(np.uint16).ravel())
            a=Sequence([1,2,3],[2],1); c=Sequence([5],[7])
            logits=b.forward([a,c],[3,1])
            self.assertEqual(lib.metadata[:3],[[2,3,5],[1,2,0],[0,0,1]])
            self.assertEqual(lib.metadata[3],[2,-1,-1,-1,7,-1,-1,-1])
            self.assertEqual(lib.metadata[4],[1,2]); self.assertEqual(logits.shape,(2,32))
            b.close(); self.assertTrue(lib.destroyed)

    def test_missing_weight_cleans_up_native_handle(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td); tiny_model(path); (path/"model.safetensors").unlink()
            lib=FakeLibrary()
            with patch("backend.C.CDLL",return_value=lib):
                with self.assertRaisesRegex(ValueError,"Missing weights"):
                    CudaBackend(path,"unused",max_context=16)
            self.assertTrue(lib.destroyed)

    def test_tied_checkpoint_with_explicit_lm_head_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td); weights=tiny_model(path)
            cfg=json.loads((path/"config.json").read_text())
            cfg["tie_word_embeddings"]=True
            (path/"config.json").write_text(json.dumps(cfg))
            lib=FakeLibrary()
            with patch("backend.C.CDLL",return_value=lib):
                CudaBackend(path,"unused",max_context=16)
            self.assertEqual(set(lib.weights),set(weights))
            for name,w in weights.items():
                np.testing.assert_array_equal(
                    lib.weights[name],(w.view(np.uint32)>>16).astype(np.uint16).ravel())

    def test_tied_checkpoint_without_lm_head_uses_embedding(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td); weights=tiny_model(path)
            cfg=json.loads((path/"config.json").read_text())
            cfg["tie_word_embeddings"]=True
            (path/"config.json").write_text(json.dumps(cfg))
            checkpoint=path/"model.safetensors"
            data=checkpoint.read_bytes()
            header_len=int.from_bytes(data[:8],"little")
            header=json.loads(data[8:8+header_len])
            del header["lm_head.weight"]
            payload=data[8+header_len:]
            encoded=json.dumps(header).encode()
            encoded+=b" "*((-len(encoded))%8)
            checkpoint.write_bytes(len(encoded).to_bytes(8,"little")+encoded+payload)
            weights.pop("lm_head.weight")
            lib=FakeLibrary()
            with patch("backend.C.CDLL",return_value=lib):
                CudaBackend(path,"unused",max_context=16)
            expected=(weights["model.embed_tokens.weight"].view(np.uint32)>>16).astype(np.uint16).ravel()
            np.testing.assert_array_equal(lib.weights["lm_head.weight"],expected)

    def test_scaled_rope_rejected_before_native_context_creation(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td); tiny_model(path)
            cfg=json.loads((path/"config.json").read_text()); cfg["rope_scaling"]={"factor":2}
            (path/"config.json").write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,"Scaled RoPE"):
                CudaBackend(path,"unused",max_context=16)


if __name__=="__main__": unittest.main()
