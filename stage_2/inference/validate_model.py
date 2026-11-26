"""Full-checkpoint cache equivalence. No Torch/HF model execution is involved.

Every mode is fed the SAME token history, even if its greedy argmax differs.
This avoids comparing logits produced from diverging contexts.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from backend import CudaBackend, Tokenizer
from cache import BlockManager


def validate(model_dir, library_dir, steps=8, atol=.1, rtol=.02):
    tokenizer = Tokenizer(library_dir/"libqwen3_tokenizer.so", model_dir)
    prompt = ("Explain in simple terms how a GPU executes many operations at once. "
              "Use an example involving a large collection of independent numbers. ")*2
    ids = tokenizer.encode(prompt)
    context = len(ids)+steps
    reference, history, results = [], [], {}
    try:
        for mode in ("none", "contiguous", "paged"):
            backend = CudaBackend(model_dir, library_dir/"libqwen3_runtime.so", blocks=128,
                                  max_context=context, max_rows=64, max_sequences=4)
            cache = BlockManager(128, 16, backend.copy_block, mode=mode, max_context=context)
            seq = cache.new(ids)
            rows = []
            try:
                for step in range(steps):
                    while seq.computed < len(seq.tokens):
                        end = min(len(seq.tokens), seq.computed+32)
                        cache.prepare(seq, end)
                        logits = backend.forward([seq], [end])[0]
                        seq.computed = end
                    if not np.isfinite(logits).all():
                        raise AssertionError(f"{mode}: nonfinite logits at step {step}")
                    if mode == "none":
                        reference.append(logits.copy())
                        history.append(int(logits.argmax()))
                    diff = np.abs(logits-reference[step])
                    bad = ~np.isclose(logits, reference[step], atol=atol, rtol=rtol)
                    rows.append(dict(step=step, max_abs_diff=float(diff.max()),
                                     outside_tolerance=int(bad.sum()), greedy_token=int(logits.argmax())))
                    seq.tokens.append(history[step])
                    if mode == "none":
                        cache.reset(seq)
                results[mode] = rows
                if mode == "paged":
                    # Exercise actual checkpoint pages after parent release.
                    cache.publish(seq, len(ids))
                    cache.reset(seq)
                    seq = cache.new(ids)
                    hit_tokens = seq.computed
                    if not hit_tokens:
                        raise AssertionError("Validation prompt produced no shared-prefix hit")
                    while seq.computed < len(seq.tokens):
                        end = min(len(seq.tokens), seq.computed+32)
                        cache.prepare(seq, end)
                        actual = backend.forward([seq], [end])[0]
                        seq.computed = end
                    np.testing.assert_allclose(actual, reference[0], atol=atol, rtol=rtol)
                    results["prefix_hit_tokens"] = hit_tokens
                if any(row["outside_tolerance"] for row in rows):
                    raise AssertionError(f"{mode} exceeds cache-equivalence tolerances: {rows}")
            finally:
                cache.reset(seq); cache.close(); backend.close()
        return dict(passed=True, prompt_tokens=len(ids), forced_tokens=history,
                    continuation=tokenizer.decode(history), atol=atol, rtol=rtol, results=results)
    finally:
        tokenizer.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--library-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    args=p.parse_args()
    if not 1 <= args.steps <= 32: p.error("steps must be in [1,32]")
    result=validate(args.model_dir, args.library_dir, args.steps)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
