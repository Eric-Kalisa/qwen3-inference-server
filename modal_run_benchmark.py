"""
Runs the week_01 matmul test suite + benchmark on a Modal container with
real multiple CPU cores, instead of the 1-core sandbox it was built in.

Usage:
    modal run modal_run_benchmark.py
    modal run modal_run_benchmark.py --matrix-size 2048
"""
import modal

app = modal.App("matmul-benchmark")

# Base image: Debian + gcc/make, with the week_01 project folder copied in.
# Adjust the local_path below if this script isn't sitting next to week_01/.
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("gcc", "make")
    .add_local_dir("week_01", remote_path="/root/week_01")
)


@app.function(image=image, cpu=64)  # request 64 physical cores
def run_benchmark(matrix_size: int = 1024) -> str:
    import subprocess

    def run(cmd):
        return subprocess.run(
            cmd, cwd="/root/week_01", capture_output=True, text=True
        )

    run(["make", "clean"])
    build = run(["make"])
    if build.returncode != 0:
        return f"BUILD FAILED:\n{build.stdout}\n{build.stderr}"

    nproc = subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip()
    test = run(["./test_matmul"])
    bench = run(["./benchmark_matmul", str(matrix_size)])

    return (
        f"Container reports {nproc} CPU cores available\n\n"
        f"--- Test suite ---\n{test.stdout}\n"
        f"--- Benchmark ---\n{bench.stdout}"
    )


@app.local_entrypoint()
def main(matrix_size: int = 1024):
    output = run_benchmark.remote(matrix_size)
    print(output)
