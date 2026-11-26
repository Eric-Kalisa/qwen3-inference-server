"""Join server timings to measured AIPerf requests, ranked by client TTFT."""
import json
import math

PHASES = (
    "queue", "tokenize", "prefix_lookup", "prefill_kv_prepare", "prefill_forward",
    "scheduler_and_bookkeeping", "first_token_after_logits", "first_token_to_delta",
    "delta_to_stream_write", "decode_wall", "decode_forward",
    "mixed_prefill_forward", "mixed_decode_forward",
)


def records(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def ttft_ms(record):
    metric = record["metrics"].get("time_to_first_token")
    if not metric:
        return None
    factor = {"ns": 1e-6, "us": 1e-3, "µs": 1e-3, "ms": 1, "s": 1000}.get(metric["unit"])
    value = metric["value"]
    if factor is not None and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
        return value * factor
    return None


def describe(client_ms, request, write):
    row = {"client_ttft_ms": client_ms, "client_request_id": request["client_request_id"],
           "server_request_id": request["id"], "prompt_tokens": request["prompt_tokens"],
           "prefix_hit_tokens": request["prefix_hit_tokens"],
           "generated_tokens": request["generated_tokens"]}
    for phase in PHASES:
        value = request.get(phase + "_seconds")
        row[phase + "_ms"] = None if value is None else 1000 * value
    first = request["ttft_worker_seconds"]
    delta = request["first_delta_seconds"]
    write_time = write.get("server_ttft_seconds")
    phases_to_logits = sum(request[phase + "_seconds"] for phase in PHASES[:5])
    row.update(
        scheduler_and_bookkeeping_ms=1000 * (request["first_logits_seconds"] - phases_to_logits),
        first_token_ms=1000 * first,
        first_token_to_delta_ms=1000 * (delta-first) if delta is not None else None,
        delta_to_stream_write_ms=1000 * (write_time-delta) if write_time is not None and delta is not None else None,
        server_ttft_ms=1000 * write_time if write_time is not None else None,
        client_minus_server_ttft_ms=client_ms-1000 * write_time if write_time is not None else None,
    )
    return row


def write_breakdown(run_dir, export, telemetry):
    measured_path = export.with_name("profile_export.jsonl")
    if not measured_path.is_file():
        measured_path = run_dir / "profile_export.jsonl"
    report = {"status": "missing_per_request_export", "matched": 0}
    if measured_path.is_file() and not telemetry.is_file():
        report["status"] = "missing_server_telemetry"
    elif measured_path.is_file():
        sent, done = {}, {}
        for event in records(telemetry):
            if event["event"] == "first_stream_write":
                sent[event["id"]] = event
            if (event["event"] == "request_end" and not event.get("error")
                    and event.get("first_logits_seconds") is not None and event.get("client_request_id")):
                done[event["client_request_id"]] = event
        measured_successes = 0
        measured = []
        for record in records(measured_path):
            meta = record["metadata"]
            if meta.get("benchmark_phase") == "profiling" and not record.get("error"):
                measured_successes += 1
                value = ttft_ms(record)
                if value is not None:
                    measured.append((value, meta.get("x_request_id")))
        ordered = sorted(measured, key=lambda entry: entry[0])
        matched = {}
        for value, key in ordered:
            if key in done:
                request = done[key]
                matched[key] = describe(value, request, sent.get(request["id"], {}))

        report = {
            "status": "complete" if ordered and len(matched) == measured_successes else "incomplete_join",
            "measured_successes": measured_successes, "matched": len(matched),
            "first_stream_writes": sum(row["server_ttft_ms"] is not None for row in matched.values()),
            "tail_sample_note": "Nearest-rank selection by AIPerf client TTFT; smoke has too few requests for stable tail estimates.",
            "phase_note": "Forward times are shared batch wall time seen by the request. Mixed forward times are subsets, not additional time. Decode wall starts at the first generated token and ends at the last.",
            "tail": {},
            "requests": [matched[key] for _, key in reversed(ordered) if key in matched],
        }
        for percentile in (95, 99):
            if ordered:
                rank = math.ceil(len(ordered) * percentile / 100)
                value, key = ordered[rank - 1]
                report["tail"][f"p{percentile}"] = {
                    "rank": rank, "client_ttft_ms": value, "client_request_id": key,
                    "phases": matched.get(key),
                }
    target = run_dir / "latency-breakdown.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
