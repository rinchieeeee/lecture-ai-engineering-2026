"""保存済み実験の指標・学習曲線を出力する（matplotlib が必要）。"""

import argparse
import csv
from pathlib import Path

from .data import read_json, write_json


def export_comparison(runs: list[Path], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    summaries, curves = [], []
    styles = [
        ("#2457A7", "o", "-"),
        ("#CE7417", "s", "--"),
        ("#857525", "^", "-."),
        ("#BA5487", "D", ":"),
    ]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        for index, run in enumerate(runs):
            result = read_json(run / "result.json")
            config = result["config"]
            history = [row for row in read_json(run / "training_history.json") if "loss" in row]
            if not history:
                raise ValueError(f"学習 loss のログがありません: {run}")
            history.sort(key=lambda row: row["step"])
            rank, alpha = config["lora"]["r"], config["lora"]["lora_alpha"]
            summaries.append({
                "run": run.name, "rank": rank, "alpha": alpha, "alpha_over_rank": alpha / rank,
                "format_rate": result["after"]["format_rate"],
                "content_rate": result["after"]["content_rate"],
                "last_logged_loss": result["training"]["last_logged_loss"],
                "train_loss": result["training"]["train_loss"],
                "training_seconds": result["training"]["seconds"],
                "global_step": result["training"]["global_step"],
                "seed": config["experiment"]["seed"], "test_n": result["after"]["n"],
            })
            for row in history:
                curves.append({"run": run.name, "rank": rank, "alpha": alpha,
                               "step": row["step"], "epoch": row["epoch"], "loss": row["loss"]})
            steps, losses = [row["step"] for row in history], [row["loss"] for row in history]
            color, marker, linestyle = styles[index % len(styles)]
            label = f"{config['experiment']['name']} (r={rank}, alpha={alpha})"
            for axis in axes:
                axis.plot(steps, losses, label=label, color=color, marker=marker,
                          linestyle=linestyle, linewidth=2, markersize=6)
                axis.set_xlabel("Optimizer step")
                axis.set_ylabel("Training loss")
                axis.set_xticks(sorted({point["step"] for point in curves}))
                axis.grid(axis="y", alpha=0.22)
            axes[1].annotate(f"{losses[-1]:.5f}", xy=(steps[-1], losses[-1]),
                             xytext=(-8, 12 if index % 2 == 0 else -20),
                             textcoords="offset points", color=color, ha="right")
        axes[0].set_title("Linear scale")
        axes[0].set_ylim(bottom=0)
        axes[1].set_title("Log scale")
        axes[1].set_yscale("log")
        axes[1].set_ylim(bottom=min(point["loss"] for point in curves) / 1.8)
        figure.suptitle("LoRA rank comparison: training loss", fontsize=16, y=0.98)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.915),
                      ncol=min(len(runs), 2), frameon=False)
        intervals = sorted({read_json(run / "config.json")["training"]["logging_steps"] for run in runs})
        figure.text(0.5, 0.015,
                    f"Logged interval means ({', '.join(map(str, intervals))} steps). "
                    "Markers show observed logs; validation loss was not recorded.",
                    ha="center", fontsize=10, color="#555555")
        figure.tight_layout(rect=(0, 0.05, 1, 0.83))
        figure.savefig(output / "learning_curve.png", dpi=180)
        figure.savefig(output / "learning_curve.svg")
        plt.close(figure)
    for name, rows in (("comparison.csv", summaries), ("learning_curve.csv", curves)):
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(output / "sources.json", {
        "runs": [str(run.resolve()) for run in runs],
        "metrics": "result.json: after.*, training.last_logged_loss, training.seconds",
        "curve": "training_history.json: loss entries at recorded optimizer steps",
    })
    print(f"比較表と学習曲線: {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="比較する実験ディレクトリ")
    parser.add_argument("--output-dir", type=Path, required=True, help="比較表・学習曲線の保存先")
    args = parser.parse_args()
    export_comparison(args.runs, args.output_dir)


if __name__ == "__main__":
    main()
