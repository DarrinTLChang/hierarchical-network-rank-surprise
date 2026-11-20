"""Plot event-Jaccard summaries from scored synthetic outputs, not patient data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=Path("results/network_burst/recording_scores.csv"))
    parser.add_argument("--out", type=Path, required=True, help="new or empty output directory")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("choose a new or empty output directory")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    data = pd.read_csv(args.scores, float_precision="round_trip")
    required = {"method", "trial", "task", "jaccard"}
    if not required.issubset(data.columns):
        parser.error("scores must contain method, trial, task, and jaccard")
    recording = data.groupby(["method", "trial", "task"], as_index=False).jaccard.mean()
    methods = ["HNRS", "logIBEI", "Network-CMA", "meaRtools/Otsu"]
    colors = ["#7E57C2", "#3586B8", "#45995D", "#D65B57"]
    tasks = [("general_events", "General burst detection"),
             ("network_targets_native", "Network targets: native outputs"),
             ("regional_classification_on", "Regional-only: region-aware"),
             ("network_classification_on", "Network: region-aware")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    rng = np.random.default_rng(20261119)
    means = []
    for ax, (task, title) in zip(axes.flat, tasks):
        for index, (method, color) in enumerate(zip(methods, colors)):
            values = recording[(recording.method == method) & (recording.task == task)].jaccard.to_numpy()
            if not len(values):
                continue
            x = index + rng.uniform(-.14, .14, len(values))
            ax.scatter(x, values, s=15, color=color, alpha=.55, edgecolors="none")
            average = float(values.mean())
            ax.scatter([index], [average], color="white", edgecolors=color, linewidths=1.7,
                       s=70, zorder=4)
            ax.annotate(f"{average:.3f}", (index, average), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=10)
            means.append({"task": task, "method": method, "n_recordings": len(values),
                          "mean_jaccard": average})
        ax.set(xticks=range(4), xticklabels=["HNRS", "logIBEI", "Network-CMA", "meaRtools"],
               xlim=(-.5, 3.5), ylim=(-.04, 1.13), title=title, ylabel="Event Jaccard")
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)
    figure.tight_layout(pad=1.7)
    args.out.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out / "synthetic_jaccard_summary.png", dpi=220, facecolor="white")
    figure.savefig(args.out / "synthetic_jaccard_summary.svg", facecolor="white")
    pd.DataFrame(means).to_csv(args.out / "plotted_means.csv", index=False, float_format="%.17g")
    plt.close(figure)
    print("Saved PNG/SVG summary and plotted_means.csv. No detector was rerun.")


if __name__ == "__main__":
    main()
