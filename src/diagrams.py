"""Block diagram (system architecture) and flowchart drawn with matplotlib shapes."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

FILL = "#DCE4EC"
EDGE = "#3E5C76"


def _box(ax, cx, cy, w, h, text, fs=10, fill=FILL):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fill, edgecolor=EDGE, linewidth=1.4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color="#111111")


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.6))


def architecture(path: Path):
    stages = [
        "1. Dataset Input\n(CSV / XLSX)",
        "2. Inspection and\nPreprocessing",
        "3. Basket / One-Hot\nEncoding",
        "4. Frequent Itemset\nMining (Apriori,\nFP-Growth)",
        "5. Association Rule\nGeneration",
        "6. Rule Evaluation\nand Ranking",
        "7. Visualisation and\nResult Export",
    ]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
    # Top row: stages 1-4 left to right; bottom row: stages 5-7 right to left
    top = [(1.4 + i * 2.7, 3.4) for i in range(4)]
    bottom = [(9.6 - i * 2.7, 1.2) for i in range(3)]
    pos = top + bottom
    for (x, y), t in zip(pos, stages):
        _box(ax, x, y, 2.2, 1.2, t)
    for i in range(3):                                   # arrows along top row
        _arrow(ax, pos[i][0] + 1.12, 3.4, pos[i + 1][0] - 1.12, 3.4)
    _arrow(ax, pos[3][0], 3.4 - 0.62, pos[4][0], 1.2 + 0.62)   # drop to bottom row
    for i in (4, 5):                                     # arrows along bottom row
        _arrow(ax, pos[i][0] - 1.12, 1.2, pos[i + 1][0] + 1.12, 1.2)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    fig.savefig(Path(path).with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def flowchart(path: Path):
    fig, ax = plt.subplots(figsize=(7.4, 10))
    ax.set_xlim(0, 7.4); ax.set_ylim(-0.2, 10.2); ax.axis("off")
    cx, w, h = 2.8, 3.8, 0.62
    box_edge, dia_edge = h / 2 + 0.03, 0.5      # half-heights incl. patch padding

    def oval(y, text):
        ax.add_patch(FancyBboxPatch((cx - 1.0, y - 0.28), 2.0, 0.56,
                                    boxstyle="round,pad=0.02,rounding_size=0.28",
                                    facecolor="#B8C6D4", edgecolor=EDGE, linewidth=1.4))
        ax.text(cx, y, text, ha="center", va="center", fontsize=10)

    def diamond(y, text):
        ax.add_patch(Polygon([[cx, y + 0.5], [cx + 1.5, y], [cx, y - 0.5], [cx - 1.5, y]],
                             closed=True, facecolor="#EEF2F6", edgecolor=EDGE, linewidth=1.4))
        ax.text(cx, y, text, ha="center", va="center", fontsize=9.5)

    ys = [9.7, 8.7, 7.7, 6.6, 5.6, 4.6, 3.5, 2.5, 1.5, 0.5]
    kinds = ["oval", "box", "dia", "box", "box", "box", "dia", "box", "box", "oval"]
    texts = ["Start", "Load dataset and map columns", "Data valid?",
             "Clean data (7 logged rules)", "Build basket / one-hot matrix",
             "Mine itemsets: Apriori (+ FP-Growth)", "Itemsets found?",
             "Generate and filter rules", "Rank rules, plot figures, export CSV", "End"]
    half = {"oval": 0.3, "box": box_edge, "dia": dia_edge}
    for y, k, t in zip(ys, kinds, texts):
        if k == "oval":
            oval(y, t)
        elif k == "dia":
            diamond(y, t)
        else:
            _box(ax, cx, y, w, h, t, 10)
    for (y1, k1), (y2, k2) in zip(zip(ys, kinds), zip(ys[1:], kinds[1:])):
        _arrow(ax, cx, y1 - half[k1], cx, y2 + half[k2])
    # "Yes" labels sit beside the downward arrows below each decision
    for y in (ys[2], ys[6]):
        ax.text(cx + 0.1, y - 0.62, "Yes", fontsize=9, va="center")
    # "No" branches lead to a stop message
    for y, msg in ((ys[2], "Stop and report\nthe problem"),
                   (ys[6], "Report: no itemsets;\nlower min_support")):
        _arrow(ax, cx + 1.5, y, 5.0, y)
        ax.text(4.25, y + 0.17, "No", fontsize=9, ha="center")
        _box(ax, 6.1, y, 2.1, 0.8, msg, 8.5, fill="#F3F3F3")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    fig.savefig(Path(path).with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
