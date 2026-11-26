"""Run the custom CUDA server's shared-prefix smoke benchmark."""
from benchmarks.shared_smoke import REVISION, app, run_smoke_and_download


@app.local_entrypoint()
def main(revision: str = REVISION):
    run_smoke_and_download("paged_shared", revision)
