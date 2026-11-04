"""All charts. Plain 2-D charts, muted blue-grey palette, saved as PNG."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .mining import itemset_text, unique_rule_pairs

BLUE = "#3E5C76"
GREY = "#8D99A6"
LIGHT = "#C9D3DD"
DATA_LABEL = "Synthetic demonstration data"


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444444", "figure.dpi": 100, "savefig.dpi": 200,
    })


def _save(fig, path: Path, note: str = ""):
    if note:
        fig.text(0.99, 0.005, note, ha="right", va="bottom", fontsize=8, color="#666666")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_preprocessing(log: pd.DataFrame, path: Path, note=""):
    _style()
    steps = log.iloc[1:]
    steps = steps[steps["Rows removed"] > 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if steps.empty:
        ax.text(0.5, 0.5, "No rows were removed", ha="center", va="center")
        ax.axis("off")
    else:
        labels = steps["Step"].str.replace(r"^\w+\.\s*", "", regex=True)
        ax.barh(labels, steps["Rows removed"], color=BLUE)
        ax.invert_yaxis()
        for y, v in enumerate(steps["Rows removed"]):
            ax.text(v, y, f" {v}", va="center", fontsize=10)
        ax.set_xlabel("Rows removed")
        ax.set_title("Rows Removed by Each Cleaning Step")
    _save(fig, path, note)


def plot_basket_sizes(basket: pd.DataFrame, path: Path, note=""):
    _style()
    sizes = basket.sum(axis=1).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(sizes.index.astype(int), sizes.values, color=BLUE)
    ax.set_xlabel("Number of distinct products in basket")
    ax.set_ylabel("Number of baskets")
    ax.set_title("Distribution of Basket Sizes")
    _save(fig, path, note)


def plot_top_products(basket: pd.DataFrame, path: Path, top_n=15, note=""):
    _style()
    freq = basket.mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(freq.index, freq.values, color=BLUE)
    ax.invert_yaxis()
    ax.set_xlabel("Support (fraction of baskets containing the product)")
    ax.set_title(f"Top {len(freq)} Products by Support")
    _save(fig, path, note)


def plot_itemsets(itemsets: pd.DataFrame, path: Path, top_n=15, note=""):
    _style()
    multi = itemsets[itemsets["itemsets"].apply(len) >= 2].head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    if multi.empty:
        ax.text(0.5, 0.5, "No frequent itemsets with 2+ items", ha="center", va="center")
        ax.axis("off")
    else:
        labels = [itemset_text(s) for s in multi["itemsets"]]
        labels = [l if len(l) <= 55 else l[:52] + "..." for l in labels]
        colors = [BLUE if len(s) == 2 else GREY for s in multi["itemsets"]]
        ax.barh(labels, multi["support"], color=colors)
        ax.invert_yaxis()
        ax.set_xlabel("Support")
        ax.set_title(f"Top {len(multi)} Frequent Itemsets (2 or more items)")
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE),
                           plt.Rectangle((0, 0), 1, 1, color=GREY)],
                  labels=["2 items", "3 items"], loc="lower right", frameon=False)
    _save(fig, path, note)


def plot_top_rules(rules: pd.DataFrame, path: Path, top_n=10, note=""):
    _style()
    shown = unique_rule_pairs(rules).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    if shown.empty:
        ax.text(0.5, 0.5, "No rules met the thresholds", ha="center", va="center")
        ax.axis("off")
    else:
        labels = [f"{itemset_text(a)}  \u2192  {itemset_text(c)}"
                  for a, c in zip(shown["antecedents"], shown["consequents"])]
        labels = [l if len(l) <= 60 else l[:57] + "..." for l in labels]
        ax.barh(labels, shown["lift"], color=BLUE)
        ax.invert_yaxis()
        ax.axvline(1.0, color="#444444", linestyle="--", linewidth=1)
        ax.text(1.0, -0.7, " lift = 1 (independence)", fontsize=9, color="#444444")
        ax.set_xlabel("Lift")
        ax.set_title(f"Top {len(shown)} Association Rules by Lift")
    _save(fig, path, note)


def plot_support_confidence(rules: pd.DataFrame, path: Path, note=""):
    _style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if rules.empty:
        ax.text(0.5, 0.5, "No rules met the thresholds", ha="center", va="center")
        ax.axis("off")
    else:
        sc = ax.scatter(rules["support"], rules["confidence"], c=rules["lift"],
                        cmap="Blues", edgecolor="#333333", linewidth=0.5, s=60,
                        vmin=max(1.0, rules["lift"].min() * 0.9))
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Lift")
        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1.02)
        ax.set_title("Support vs Confidence of Association Rules (colour = lift)")
    _save(fig, path, note)


def plot_runtime(times: dict, path: Path, note=""):
    _style()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    names = list(times.keys())
    vals = [times[n] * 1000 for n in names]
    ax.bar(names, vals, color=[BLUE, GREY][:len(names)])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f} ms", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Median runtime (milliseconds)")
    ax.set_title("Apriori vs FP-Growth Runtime (same data)")
    _save(fig, path, note)
