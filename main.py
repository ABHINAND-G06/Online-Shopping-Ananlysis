"""Online Shopping Analysis Using Association Rule Mining - command-line entry point.

Run:   python main.py                       (uses the synthetic demo dataset)
       python main.py --input "data/Online Retail.xlsx" --min-support 0.02
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src import diagrams, visualization as viz
from src.config import (DEFAULT_DATASET, FIGURES_DIR, OUTPUT_DIR, MiningConfig)
from src.data_loader import load_transactions
from src.generate_synthetic_data import build_dataset
from src.mining import (flatten_for_csv, generate_rules, mine_frequent_itemsets,
                        time_algorithm, validate_rules)
from src.preprocessing import build_basket_matrix, clean_transactions, summarise


def parse_args():
    p = argparse.ArgumentParser(description="Association rule mining on retail transactions.")
    p.add_argument("--input", type=Path, default=DEFAULT_DATASET, help="CSV or XLSX file")
    d = MiningConfig()
    p.add_argument("--min-support", type=float, default=d.min_support)
    p.add_argument("--min-confidence", type=float, default=d.min_confidence)
    p.add_argument("--min-lift", type=float, default=d.min_lift)
    p.add_argument("--max-len", type=int, default=d.max_itemset_len)
    p.add_argument("--no-fpgrowth", action="store_true", help="skip FP-Growth comparison")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = MiningConfig(min_support=args.min_support, min_confidence=args.min_confidence,
                       min_lift=args.min_lift, max_itemset_len=args.max_len,
                       run_fpgrowth=not args.no_fpgrowth)
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Create the demo file if the default dataset is requested but not present yet.
    if args.input == DEFAULT_DATASET and not args.input.exists():
        args.input.parent.mkdir(exist_ok=True)
        build_dataset().to_csv(args.input, index=False)
        print("Default dataset missing: generated the SYNTHETIC demo file.")

    # Label every figure when the data is synthetic
    is_synthetic = "synthetic" in args.input.name.lower()
    note = "Synthetic demonstration data" if is_synthetic else ""

    # 1-2. Load and clean ------------------------------------------------------
    raw = load_transactions(args.input)
    clean, log = clean_transactions(raw)
    if clean.empty:
        sys.exit("No rows left after cleaning - check the dataset.")
    # 3. Basket encoding -------------------------------------------------------
    basket = build_basket_matrix(clean, cfg.min_basket_size)
    if basket.empty:
        sys.exit("No baskets available after encoding.")
    summary = summarise(raw, clean, basket)
    summary.update({"dataset_file": args.input.name, "synthetic": is_synthetic,
                    "min_support": cfg.min_support, "min_confidence": cfg.min_confidence,
                    "min_lift": cfg.min_lift, "max_itemset_len": cfg.max_itemset_len})
    log.to_csv(OUTPUT_DIR / "preprocessing_log.csv", index=False)
    print(f"Transactions: {summary['transactions']}, products: {summary['distinct_products']}")

    # 4. Frequent itemsets (Apriori) ----------------------------------------
    itemsets = mine_frequent_itemsets(basket, "apriori", cfg.min_support, cfg.max_itemset_len)
    summary["frequent_itemsets"] = int(len(itemsets))
    summary["frequent_itemsets_2plus"] = int((itemsets["itemsets"].apply(len) >= 2).sum()) if len(itemsets) else 0
    if itemsets.empty:
        print("No frequent itemsets found. Try a smaller --min-support.")
    flatten_for_csv(itemsets).to_csv(OUTPUT_DIR / "frequent_itemsets.csv", index=False)

    # Optional: FP-Growth comparison on the SAME prepared data
    if cfg.run_fpgrowth and not itemsets.empty:
        fp = mine_frequent_itemsets(basket, "fpgrowth", cfg.min_support, cfg.max_itemset_len)
        same = ({frozenset(s) for s in itemsets["itemsets"]} == {frozenset(s) for s in fp["itemsets"]})
        times = {"Apriori": time_algorithm(basket, "apriori", cfg.min_support, cfg.max_itemset_len, cfg.timing_repeats),
                 "FP-Growth": time_algorithm(basket, "fpgrowth", cfg.min_support, cfg.max_itemset_len, cfg.timing_repeats)}
        summary.update({"fpgrowth_itemsets": int(len(fp)), "same_itemsets": bool(same),
                        "apriori_seconds": times["Apriori"], "fpgrowth_seconds": times["FP-Growth"],
                        "timing_repeats": cfg.timing_repeats})
        pd.DataFrame([{"Algorithm": k, "Median runtime (s)": round(v, 5),
                       "Itemsets found": summary["frequent_itemsets"] if k == "Apriori" else len(fp)}
                      for k, v in times.items()]).to_csv(OUTPUT_DIR / "algorithm_comparison.csv", index=False)
        viz.plot_runtime(times, FIGURES_DIR / "fig_runtime_comparison.png", note)
        print(f"Apriori and FP-Growth return identical itemsets: {same}")

    # 5-6. Rules -------------------------------------------------------------------
    rules = generate_rules(itemsets, len(basket), cfg.min_confidence, cfg.min_lift) if not itemsets.empty else pd.DataFrame()
    summary["rules"] = int(len(rules))
    summary["rules_valid"] = validate_rules(rules)
    if rules.empty:
        print("No rules met the thresholds. Try lowering --min-confidence or --min-lift.")
    else:
        flatten_for_csv(rules).to_csv(OUTPUT_DIR / "association_rules.csv", index=False)
        flatten_for_csv(rules.head(cfg.top_n_rules)).to_csv(OUTPUT_DIR / "top_rules.csv", index=False)

    # 7. Figures and summary --------------------------------------------------
    viz.plot_preprocessing(log, FIGURES_DIR / "fig_preprocessing_summary.png", note)
    viz.plot_basket_sizes(basket, FIGURES_DIR / "fig_basket_sizes.png", note)
    viz.plot_top_products(basket, FIGURES_DIR / "fig_top_products.png", note=note)
    viz.plot_itemsets(itemsets, FIGURES_DIR / "fig_frequent_itemsets.png", note=note)
    viz.plot_top_rules(rules, FIGURES_DIR / "fig_top_rules.png", note=note)
    viz.plot_support_confidence(rules, FIGURES_DIR / "fig_support_confidence.png", note)
    diagrams.architecture(FIGURES_DIR / "fig_architecture.png")
    diagrams.flowchart(FIGURES_DIR / "fig_flowchart.png")

    top_products = basket.mean().sort_values(ascending=False).head(15).round(4)
    top_products.rename("support").to_csv(OUTPUT_DIR / "top_products.csv", header=True)
    (OUTPUT_DIR / "results_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
