import csv
import statistics
import subprocess
import sys
import re
import argparse
from pathlib import Path



SOURCE_FILE = "blockedopenmp.cpp"          # path to .cpp file
BINARY_FILE = "./blockedopenmp"        # compiled output
SIZES = [64, 128, 256, 512, 1024, 2048]   # matrix sizes (N) to test
THREADS = [1, 2, 4, 8, 10, 12]
REPEATS = 30                          # runs per size
OUTPUT_CSV = "blockedopenmp_results.csv"
COMPILE_FLAGS = ["-O3", "-fopenmp"]  

TIME_RE = re.compile(r"([\d.]+)\s*ms")


def compile_program(source: str, binary: str, flags: list[str]) -> None:
    if not Path(source).exists():
        sys.exit(f"Error: source file '{source}' not found.")

    cmd = ["g++", *flags, source, "-o", binary]
    print(f"Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit("Compilation failed.")
    print("Compilation succeeded.\n")


def run_once(binary: str, n: int, t:int, timeout: float = 120.0) -> float:
    """Run the compiled binary once for size N, return elapsed ms."""
    proc = subprocess.run(
        [binary],
        input=f"{n}\n{t}\n",
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Program exited with code {proc.returncode}, stderr: {proc.stderr}"
        )

    match = TIME_RE.search(proc.stdout)
    if not match:
        raise RuntimeError(f"Could not parse timing from output: {proc.stdout!r}")

    return float(match.group(1))


def benchmark(binary: str, n: int, threads: list[int], repeats: int) -> list[dict]:
    rows = []
    for t in threads:    
        print(f"Benchmarking N={n}, Threads t={t} ({repeats} runs)...")
        samples = []
        for i in range(repeats):
            elapsed = run_once(binary, n, t)
            samples.append(elapsed)
            print(f"  run {i + 1}/{repeats}: {elapsed:.3f} ms")

        mean = statistics.mean(samples)
        std = statistics.stdev(samples) if len(samples) > 1 else 0.0
        rows.append(
            {
                "N": n,
                "threads": t,
                "repeats": repeats,
                "mean_ms": mean,
                "std_ms": std,
                "min_ms": min(samples),
                "max_ms": max(samples),
                "samples_ms": ";".join(f"{s:.3f}" for s in samples),
            }
        )
        print(f"  -> mean={mean:.3f} ms, std={std:.3f} ms\n")
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    fieldnames = ["N", "threads", "repeats", "mean_ms", "std_ms", "min_ms", "max_ms", "samples_ms"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark matmul.cpp")
    parser.add_argument("--source", default=SOURCE_FILE)
    parser.add_argument("--binary", default=BINARY_FILE)
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--threads", type=int, nargs="+", default=THREADS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--output", default=OUTPUT_CSV)
    parser.add_argument("--skip-compile", action="store_true",
                         help="Skip compilation and use existing binary")
    args = parser.parse_args()

    if not args.skip_compile:
        compile_program(args.source, args.binary, COMPILE_FLAGS)
    elif not Path(args.binary).exists():
        sys.exit(f"Binary '{args.binary}' not found and --skip-compile was set.")

    all_rows=[]
    for n in args.sizes:    
        all_rows.extend(benchmark(args.binary, n, args.threads, args.repeats))

    write_csv(all_rows, args.output)


if __name__ == "__main__":
    main()
