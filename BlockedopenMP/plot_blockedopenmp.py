import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def remove_outliers_iqr(values):
    values = np.asarray(values)

    if len(values) < 4:
        return values

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return values[(values >= lower) & (values <= upper)]


def load_csv(path: str):
    Ns, threads, means, stds = [], [], [], []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            samples = np.array(
                [float(x) for x in row["samples_ms"].split(";")]
            )

            # Remove outliers
            filtered = remove_outliers_iqr(samples)

            # If everything gets removed (unlikely), keep original data
            if len(filtered) == 0:
                filtered = samples

            Ns.append(int(row["N"]))
            threads.append(int(row["threads"]))
            means.append(np.mean(filtered))
            stds.append(np.std(filtered, ddof=1) if len(filtered) > 1 else 0.0)

    # sort by N
    order = np.lexsort((threads, Ns))

    Ns = np.array(Ns)[order]
    threads = np.array(threads)[order]
    means = np.array(means)[order]
    stds = np.array(stds)[order]
    removed = len(samples) - len(filtered)
    if removed > 0:
        print(f"N={row['N']}, threads={row['threads']}: removed {removed} outlier(s)")


    return Ns, threads, means, stds


def plot_linear(Ns, threads, means, stds, outpath):
    plt.figure(figsize=(8, 6))

    unique_Ns = np.unique(Ns)

    for N in unique_Ns:
        mask = Ns == N

        # Sort by thread count so the line is drawn correctly
        order = np.argsort(threads[mask])

        plt.errorbar(
            threads[mask][order],
            means[mask][order],
            yerr=stds[mask][order],
            fmt="o-",
            capsize=4,
            linewidth=2,
            markersize=6,
            label=f"N={N}",
        )

    plt.xlabel("Threads")
    plt.ylabel("Mean runtime (ms)")
    plt.title("Runtime vs Threads(Block Size=48)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Matrix Size")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    print(f"Saved {outpath}")


def plot_speedup(Ns, threads, means, outpath):
    plt.figure(figsize=(8, 6))

    unique_Ns = np.unique(Ns)

    for n in unique_Ns:
        mask = Ns == n

        t = threads[mask]
        m = means[mask]

        # Sort both arrays together
        order = np.argsort(t)
        t = t[order]
        m = m[order]

        # Runtime at 1 thread
        baseline = m[t == 1][0]

        speedup = baseline / m

        plt.plot(
            t,
            speedup,
            "o-",
            linewidth=2,
            markersize=6,
            label=f"N={n}",
        )

    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Threads(Block Size=48)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Matrix size")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    print(f"Saved {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark CSV results")
    parser.add_argument("--input", default="blockedopenmp_results.csv")
    parser.add_argument("--outdir", default="plots")
    args = parser.parse_args()

    if not Path(args.input).exists():
        sys.exit(f"Error: input file '{args.input}' not found.")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    Ns, threads, means, stds = load_csv(args.input)

    if len(Ns) == 0:
        sys.exit("No data found in CSV.")

    plot_linear(Ns, threads, means, stds, Path(args.outdir) / "runtime_linear.png")

    plot_speedup(Ns, threads, means, Path(args.outdir) / "runtime_speedup.png")

if __name__ == "__main__":
    main()