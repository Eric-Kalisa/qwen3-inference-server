"""Shared workload plans and run sizes for local and Modal entry points."""
DEFAULTS = {
    "smoke": dict(requests=20, warmup=3, repeats=1, case_timeout=1800),
    "compact": dict(requests=30, warmup=3, repeats=1, case_timeout=300),
    "full": dict(requests=1000, warmup=20, repeats=3, case_timeout=1800),
}


def settings(preset, requests=0, warmup=-1, repeats=0, case_timeout=0):
    if preset not in DEFAULTS:
        raise ValueError(f"Unknown preset: {preset}")
    if min(requests, repeats, case_timeout) < 0 or warmup < -1:
        raise ValueError("Counts/timeouts cannot be negative; warmup=-1 selects the preset default")
    result = dict(DEFAULTS[preset])
    for key, value in dict(requests=requests, warmup=warmup, repeats=repeats,
                           case_timeout=case_timeout).items():
        if (key == "warmup" and value >= 0) or (key != "warmup" and value > 0):
            result[key] = value
    return result


def cases(preset):
    if preset == "smoke":
        return [("short", 1, None, "poisson"), ("short", 4, None, "poisson")]
    if preset == "compact":
        # Single-user prefill/decode, the same decode workload under load, then
        # heterogeneous lengths. No imposed arrival-rate wait in this preset.
        return [("prefill_512", 1, None, "poisson"),
                ("decode_128", 1, None, "poisson"),
                ("decode_128", 4, None, "poisson"),
                ("mixed_small", 4, None, "poisson")]
    if preset != "full":
        raise ValueError(f"Unknown preset: {preset}")
    fixed = [(name, c, None, "poisson") for name in
             ("short", "prefill_512", "prefill_4096", "decode_128", "decode_512",
              "context_2048", "context_7680", "mixed") for c in (1, 2, 4, 8, 16, 32, 64)]
    rates = [("mixed", None, r, pattern) for r in (.25, .5, 1, 2, 4, 8, 16)
             for pattern in ("constant", "poisson", "gamma")]
    return fixed + rates
