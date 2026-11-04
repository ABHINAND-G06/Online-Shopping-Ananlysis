"""Frequent itemset mining (Apriori, FP-Growth) and association rule generation."""
import time
from statistics import median

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth


def mine_frequent_itemsets(basket: pd.DataFrame, algorithm: str,
                           min_support: float, max_len: int) -> pd.DataFrame:
    """Run Apriori or FP-Growth. Returns columns: support, itemsets (frozenset)."""
    if basket.empty:
        return pd.DataFrame(columns=["support", "itemsets"])
    func = apriori if algorithm == "apriori" else fpgrowth
    found = func(basket, min_support=min_support, use_colnames=True, max_len=max_len)
    return found.sort_values("support", ascending=False).reset_index(drop=True)


def time_algorithm(basket, algorithm, min_support, max_len, repeats=5) -> float:
    """Median wall-clock seconds over several runs on the same prepared data."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        mine_frequent_itemsets(basket, algorithm, min_support, max_len)
        times.append(time.perf_counter() - start)
    return median(times)


def generate_rules(itemsets: pd.DataFrame, n_transactions: int,
                   min_confidence: float, min_lift: float) -> pd.DataFrame:
    """Create rules from frequent itemsets, then keep those meeting the lift threshold.

    support    = P(A and B)          confidence = P(B | A)
    lift       = confidence / P(B)   (> 1 means B is more likely when A is present)
    """
    multi = itemsets[itemsets["itemsets"].apply(len) >= 2]
    if multi.empty:
        return pd.DataFrame()           # nothing to build rules from
    rules = association_rules(itemsets, num_itemsets=n_transactions,
                              metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return pd.DataFrame()
    rules = rules[rules["lift"] >= min_lift]
    if rules.empty:
        return pd.DataFrame()
    keep = ["antecedents", "consequents", "antecedent support",
            "consequent support", "support", "confidence", "lift"]
    rules = rules[keep].sort_values(["lift", "confidence"], ascending=False)
    return rules.reset_index(drop=True)


def validate_rules(rules: pd.DataFrame) -> bool:
    """Sanity check: support/confidence in [0, 1] and lift >= 0."""
    if rules.empty:
        return True
    ok = rules["support"].between(0, 1).all() and rules["confidence"].between(0, 1).all()
    return bool(ok and (rules["lift"] >= 0).all())


def itemset_text(items) -> str:
    """Turn a frozenset into 'A + B' text (sorted so output is stable)."""
    return " + ".join(sorted(items))


def flatten_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Replace frozensets by readable strings so the table can be saved/shown."""
    out = df.copy()
    for col in ("itemsets", "antecedents", "consequents"):
        if col in out.columns:
            out[col] = out[col].apply(itemset_text)
    if "itemsets" in out.columns:
        out.insert(1, "size", df["itemsets"].apply(len))
    return out


def unique_rule_pairs(rules: pd.DataFrame) -> pd.DataFrame:
    """For display: keep one direction per item combination (the higher confidence).

    A -> B and B -> A share the same support and lift, so showing both in a
    ranking chart would repeat the same association.
    """
    if rules.empty:
        return rules
    tmp = rules.copy()
    tmp["_key"] = tmp.apply(lambda r: frozenset(r["antecedents"] | r["consequents"]), axis=1)
    tmp = tmp.sort_values("confidence", ascending=False).drop_duplicates("_key")
    return tmp.drop(columns="_key").sort_values(["lift", "confidence"], ascending=False)
