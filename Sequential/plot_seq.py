import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_csv(path: str):
    Ns, means, stds = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            Ns.append(int(row["N"]))
            means.append(float(row["mean_ms"]))
            stds.append(float(row["std_ms"]))

    # sort by N in case rows aren't ordered
    order = np.argsort(Ns)
    Ns = np.array(Ns)[order]
    means = np.array(means)[order]
    stds = np.array(stds)[order]
    return Ns, means, stds


def plot_linear(Ns, means, stds, outpath):
    plt.figure(figsize=(8, 6))
    plt.errorbar(Ns, means, yerr=stds, fmt="o-", capsize=4,
                 color="#2563eb", ecolor="#93c5fd", linewidth=2, markersize=6)
    plt.xlabel("Matrix size N")
    plt.ylabel("Mean runtime (ms)")
    plt.title("Matrix Multiplication Runtime vs N")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"Saved {outpath}")
    plt.close()



def main():
    parser = argparse.ArgumentParser(description="Plot benchmark CSV results")
    parser.add_argument("--input", default="seq_results.csv")
    parser.add_argument("--outdir", default="plots")
    args = parser.parse_args()

    if not Path(args.input).exists():
        sys.exit(f"Error: input file '{args.input}' not found.")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    Ns, means, stds = load_csv(args.input)

    if len(Ns) == 0:
        sys.exit("No data found in CSV.")

    plot_linear(Ns, means, stds, Path(args.outdir) / "runtime_linear.png")

    if len(Ns) > 1:
        plot_loglog(Ns, means, Path(args.outdir) / "runtime_loglog.png")
    else:
        print("Skipping log-log plot (need at least 2 data points).")


if __name__ == "__main__":
    main()
