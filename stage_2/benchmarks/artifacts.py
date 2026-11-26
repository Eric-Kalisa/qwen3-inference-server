"""Select measured AIPerf results, excluding warmup and other phase exports."""
from pathlib import Path


def profile_export(directory):
    directory = Path(directory)
    # Phase-aware versions also write an aggregate at the root. Prefer the
    # measured phase explicitly; a recursive glob also finds warmup exports.
    if (directory / "phases").is_dir():
        candidate = directory / "phases/profiling/profile_export_aiperf.json"
    else:
        candidate = directory / "profile_export_aiperf.json"
    return candidate if candidate.is_file() else None
