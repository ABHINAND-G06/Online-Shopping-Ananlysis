"""Load a transaction file and map its columns onto standard names."""
from pathlib import Path

import pandas as pd

from .config import COLUMN_ALIASES, REQUIRED_COLUMNS


def load_transactions(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file and return a DataFrame with standard column names.

    Raises a clear error (instead of crashing later) when the file is missing,
    empty, or lacks a required column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place the file in the data/ folder or pass --input <file>.")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)          # e.g. the UCI Online Retail workbook
    elif suffix in {".csv", ".txt"}:
        raw = pd.read_csv(path, encoding="latin-1")
    else:
        raise ValueError(f"Unsupported file type: {suffix} (use .csv or .xlsx)")

    if raw.empty:
        raise ValueError("The dataset file contains no rows.")

    # Map whichever alias is present onto the standard column name.
    rename = {}
    for standard, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in raw.columns:
                rename[alias] = standard
                break
    data = raw.rename(columns=rename)

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. "
                         f"Columns found: {list(raw.columns)}")

    data["InvoiceNo"] = data["InvoiceNo"].astype(str).str.strip()
    return data
