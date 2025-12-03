#!/usr/bin/env python3
"""
Extract rows from eagle/evaluation/eval_rp.tsv where both final columns are filled.

The output matches the format requested in the task description:
<first-number-of-second-column>, <second-to-last-column>, <last-column>
"""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
TSV_PATH = ROOT / "eval_rp.tsv"


def main() -> None:
    rows = []
    with TSV_PATH.open() as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
# resilient-paper_annealing100	12_10_100	1	greedy0_avg	167.266	138.84	167.27
            last_two = [cell.strip() for cell in row[-2:]]
            if not (last_two[0] and last_two[1]):
                continue
            parts = [p.strip() for p in row[1].split("_")]
            if not parts:
                continue
            first = parts[0]
            last = parts[-1] if len(parts) > 1 else ""
            rows.append((first, last, last_two[0], last_two[1]))

            print(first, last, last_two[0], last_two[1])

    ticks = [f'{first}_{second}' for first, second, col_before_last, last_col in rows]
    x = np.arange(len(ticks))
    values1 = [float(col_before_last) for first, second, col_before_last, last_col in rows]
    values2 = [float(last_col) for first, second, col_before_last, last_col in rows]

    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width/2, values1, width, label="Baseline")
    ax.bar(x + width/2, values2, width, label="Pondering")

    ax.set_title("EA3 vs. Pondering EA3")
    ax.set_xlabel("topk-total")
    ax.set_ylabel("speed")

    ax.set_xticks(x)
    ax.set_xticklabels(ticks, rotation=90, va="bottom")  # last char aligned

    # 🔑 extra vertical space between axis and labels
    ax.tick_params(axis="x", which="major", pad=36)

    # 🔑 more bottom margin so long labels fit
    fig.subplots_adjust(bottom=0.2)

    ax.legend()

    fig.savefig("bar.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
