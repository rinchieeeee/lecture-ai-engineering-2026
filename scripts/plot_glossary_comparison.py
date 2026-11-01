"""保存した比較結果から、質問の種類別の正解件数を描く（matplotlibが必要）。"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = json.loads((args.run_dir / "comparison.json").read_text(encoding="utf-8"))
    categories = [c for c in ["known", "paraphrase", "unknown"] if c in result["by_category"]]
    labels = {"known": "Named terms", "paraphrase": "Paraphrased questions", "unknown": "Terms outside glossary"}
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    for phase, offset, color, hatch, label in [
        ("before", -0.19, "#56728D", "//", "Before LoRA"),
        ("after", 0.19, "#C87520", "", "After LoRA"),
    ]:
        counts = [sum(r[phase]["both_ok"] for r in result["rows"] if r["category"] == c) for c in categories]
        bars = ax.bar([i + offset for i in range(len(categories))], counts, width=0.34,
                      color=color, hatch=hatch, edgecolor="#283642", linewidth=0.6, label=label)
        totals = [result["by_category"][c][phase]["n"] for c in categories]
        ax.bar_label(bars, labels=[f"{count}/{total}" for count, total in zip(counts, totals)], padding=5, fontsize=11)
    nmax = max(result["by_category"][c]["before"]["n"] for c in categories)
    ax.set_ylim(0, nmax + 2)
    ax.set_yticks(range(0, nmax + 1, 2))
    ax.set_xticks(range(len(categories)), [labels[c] for c in categories])
    ax.set_ylabel("Correct answers (count)")
    ax.set_title("Glossary Q&A: before and after LoRA", loc="left", pad=18, fontsize=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#DFE4E8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncols=2)
    fig.supxlabel("Automatic grading: valid JSON, expected needs_review, and all required keywords.\n"
                  "Same fixed questions and generation settings before and after; one run.", fontsize=9)
    fig.savefig(args.run_dir / "comparison.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
