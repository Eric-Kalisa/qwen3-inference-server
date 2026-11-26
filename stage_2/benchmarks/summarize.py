"""Normalize AIPerf exports without inventing missing metrics or pooling tails.

One CSV row per measured run. Repetition summaries use mean/stdev throughput;
percentiles remain per-run because averaging p99s is not a pooled p99.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from benchmarks.artifacts import profile_export

METRICS={"time_to_first_token":("avg","p50","p95","p99","count"),
         "inter_token_latency":("avg","p50","p95","p99","count"),
         "request_latency":("avg","p50","p95","p99","count"),
         "request_throughput":("avg",),"output_token_throughput":("avg",),
         "request_count":("avg",),"error_request_count":("avg",),
         "input_sequence_length":("avg",),"output_sequence_length":("avg",)}


def summarize(root):
    root=Path(root); rows=[]
    for run in sorted(root.rglob("run.json")):
        metadata=json.loads(run.read_text())
        export=profile_export(run.parent)
        row={k:metadata.get(k) for k in ("variant","workload","concurrency","rate","arrival","repeat")}
        row["artifact"]=str(run.parent.relative_to(root))
        row["status"]="missing_export"
        if export is not None:
            row["metric_source"]=str(export.relative_to(run.parent))
            data=json.loads(export.read_text())
            row["status"]="exported"
            for metric,fields in METRICS.items():
                block=data.get(metric,{})
                for field in fields: row[f"{metric}.{field}"]=block.get(field)
                row[f"{metric}.unit"]=block.get("unit")
            # Unknown fields remain preserved in the raw JSON, never guessed.
            if row.get("error_request_count.avg",0): row["status"]="exported_with_errors"
            if not isinstance(row.get("request_count.avg"),(int,float)):
                row["status"]="missing_required_metrics"
            count=row.get("request_latency.count")
            row["tail_warning"]="fewer than 1000 observations" if count is None or count<1000 else ""
        rows.append(row)
    manifest=root/"manifest.json"
    if manifest.exists():
        keys=("variant","workload","concurrency","rate","arrival","repeat")
        present={tuple(row.get(k) for k in keys) for row in rows}
        for planned in json.loads(manifest.read_text()).get("plan",[]):
            if tuple(planned.get(k) for k in keys) not in present:
                rows.append(dict({k:planned.get(k) for k in keys},artifact=None,status="not_run"))
    baseline={tuple(r[k] for k in ("workload","concurrency","rate","arrival","repeat")):r
              for r in rows if r["variant"]=="uncached" and r["status"]=="exported"}
    for row in rows:
        key=tuple(row[k] for k in ("workload","concurrency","rate","arrival","repeat"))
        base=baseline.get(key,{}).get("output_token_throughput.avg")
        value=row.get("output_token_throughput.avg")
        row["output_throughput_speedup_vs_uncached"]=(value/base if row["status"]=="exported" and
                                                     isinstance(base,(float,int)) and base>0 and
                                                     isinstance(value,(float,int)) else None)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (root/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields or ["status"]); writer.writeheader(); writer.writerows(rows)
    groups={}
    for row in rows:
        key=tuple(row[k] for k in ("variant","workload","concurrency","rate","arrival"))
        groups.setdefault(key,[]).append(row)
    aggregates=[]
    for key,group in groups.items():
        values=[r.get("output_token_throughput.avg") for r in group if r["status"]=="exported"]
        values=[v for v in values if isinstance(v,(float,int))]
        aggregates.append(dict(zip(("variant","workload","concurrency","rate","arrival"),key),
                               successful_export_runs=len(values),runs=len(group),
                               output_tokens_per_second_mean=statistics.mean(values) if values else None,
                               output_tokens_per_second_stdev=statistics.stdev(values) if len(values)>1 else None))
    output=dict(runs=rows,repetitions=aggregates,
                notes=["Latency units are preserved from each AIPerf export.",
                       "Percentiles are per-run; no pooled percentile is claimed.",
                       "Missing exports and request errors are included, not dropped.",
                       "Smoke measurements are functional checks, not stable performance conclusions."])
    (root/"summary.json").write_text(json.dumps(output,indent=2))
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("root",type=Path)
    summarize(parser.parse_args().root)
