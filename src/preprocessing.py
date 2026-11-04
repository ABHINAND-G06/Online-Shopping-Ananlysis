"""Data cleaning and basket (one-hot) encoding.

Every cleaning rule records how many rows it affected, so nothing is deleted
silently. The log is saved to output/preprocessing_log.csv and used in the report.
"""
import pandas as pd

# Stock codes that are services/fees rather than products (found in the UCI dataset).
NON_PRODUCT_CODES = {"POST", "DOT", "M", "BANK CHARGES", "D", "CRUK", "C2",
                     "AMAZONFEE", "S", "B", "PADS", "ADJUST", "ADJUST2"}
NON_PRODUCT_DESCRIPTIONS = {"POSTAGE", "DOTCOM POSTAGE", "MANUAL", "BANK CHARGES",
                            "DISCOUNT", "SAMPLES", "CARRIAGE", "AMAZON FEE"}


def clean_transactions(data: pd.DataFrame):
    """Apply cleaning rules in a fixed order. Returns (clean_df, log_df)."""
    log = []
    df = data.copy()

    def record(step, rule, before, after):
        log.append({"Step": step, "Rule applied": rule,
                    "Rows removed": before - after, "Rows remaining": after})

    log.append({"Step": "0. Raw data", "Rule applied": "Rows read from the file",
                "Rows removed": 0, "Rows remaining": len(df)})

    # 1. Missing invoice number: a line cannot be assigned to a basket.
    n = len(df)
    df = df[df["InvoiceNo"].notna() & (df["InvoiceNo"] != "nan") & (df["InvoiceNo"] != "")]
    record("1. Missing invoice", "Remove lines without an invoice number", n, len(df))

    # 2. Missing description: first try to recover it from the same StockCode.
    if "StockCode" in df.columns:
        lookup = (df.dropna(subset=["Description"]).groupby("StockCode")["Description"]
                  .agg(lambda s: s.mode().iloc[0]))
        fill_mask = df["Description"].isna() & df["StockCode"].isin(lookup.index)
        df.loc[fill_mask, "Description"] = df.loc[fill_mask, "StockCode"].map(lookup)
        log.append({"Step": "2a. Recover description",
                    "Rule applied": "Fill missing description from the same StockCode",
                    "Rows removed": 0, "Rows remaining": len(df)})
        recovered = int(fill_mask.sum())
    else:
        recovered = 0
    n = len(df)
    df = df[df["Description"].notna()]
    record("2b. Missing description", "Remove lines still lacking a description "
           f"({recovered} recovered in step 2a)", n, len(df))

    # 3. Normalise product names (strip spaces, unify case) so identical products match.
    df["Description"] = df["Description"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["Description"] = df["Description"].str.title()

    # 4. Cancelled invoices (number starts with 'C') and returns (negative quantity).
    n = len(df)
    df = df[~df["InvoiceNo"].str.upper().str.startswith("C")]
    record("3. Cancelled invoices", "Remove invoices whose number starts with 'C'", n, len(df))
    n = len(df)
    df = df[pd.to_numeric(df["Quantity"], errors="coerce") > 0]
    record("4. Non-positive quantity", "Remove lines with quantity <= 0 (returns)", n, len(df))

    # 5. Invalid price: a zero or negative price is not a normal sale.
    n = len(df)
    df = df[pd.to_numeric(df["UnitPrice"], errors="coerce") > 0]
    record("5. Invalid price", "Remove lines with unit price <= 0", n, len(df))

    # 6. Non-product service lines (postage, fees) are not items a customer 'buys together'.
    n = len(df)
    if "StockCode" in df.columns:
        df = df[~df["StockCode"].astype(str).str.upper().isin(NON_PRODUCT_CODES)]
    df = df[~df["Description"].str.upper().isin(NON_PRODUCT_DESCRIPTIONS)]
    record("6. Service lines", "Remove postage, fees and similar non-product lines", n, len(df))

    # 7. Repeated product on one invoice: a basket is a SET of products, so duplicates collapse.
    n = len(df)
    df = df.drop_duplicates(subset=["InvoiceNo", "Description"])
    record("7. Repeated lines", "Keep one line per (invoice, product) pair", n, len(df))

    return df.reset_index(drop=True), pd.DataFrame(log)


def build_basket_matrix(clean: pd.DataFrame, min_basket_size: int = 1) -> pd.DataFrame:
    """One-hot encode: rows = invoices, columns = products, values = True/False.

    Quantity is deliberately ignored: association rules only need to know
    whether a product is present in a basket.
    """
    basket = pd.crosstab(clean["InvoiceNo"], clean["Description"]).gt(0)
    sizes = basket.sum(axis=1)
    basket = basket[sizes >= min_basket_size]
    return basket.astype(bool)


def summarise(raw: pd.DataFrame, clean: pd.DataFrame, basket: pd.DataFrame) -> dict:
    """Headline numbers used in the report."""
    sizes = basket.sum(axis=1)
    summary = {
        "raw_rows": int(len(raw)),
        "raw_invoices": int(raw["InvoiceNo"].nunique()),
        "clean_rows": int(len(clean)),
        "transactions": int(basket.shape[0]),
        "distinct_products": int(basket.shape[1]),
        "avg_basket_size": round(float(sizes.mean()), 3) if len(sizes) else 0.0,
        "median_basket_size": float(sizes.median()) if len(sizes) else 0.0,
        "max_basket_size": int(sizes.max()) if len(sizes) else 0,
        "single_item_baskets": int((sizes == 1).sum()),
    }
    if "InvoiceDate" in clean.columns:
        dates = pd.to_datetime(clean["InvoiceDate"], errors="coerce").dropna()
        if len(dates):
            summary["date_from"] = str(dates.min().date())
            summary["date_to"] = str(dates.max().date())
    return summary
