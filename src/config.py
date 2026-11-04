"""Central configuration: folder paths and default mining thresholds.

All paths are relative to the project root (found from this file's location),
so the project runs from any machine without editing paths.
"""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
DEFAULT_DATASET = DATA_DIR / "synthetic_online_retail.csv"

# Alternative names that different retail datasets use for the same field.
# The loader maps whichever name it finds onto the standard name on the left.
COLUMN_ALIASES = {
    "InvoiceNo": ["InvoiceNo", "Invoice", "TransactionID", "Transaction_ID", "OrderID"],
    "StockCode": ["StockCode", "ProductID", "Product_ID", "SKU"],
    "Description": ["Description", "ProductName", "Product_Name", "Product"],
    "Quantity": ["Quantity", "Qty"],
    "InvoiceDate": ["InvoiceDate", "Date", "OrderDate", "InvoiceDateTime"],
    "UnitPrice": ["UnitPrice", "Price", "Unit_Price"],
    "CustomerID": ["CustomerID", "Customer ID", "Customer_ID", "CustID"],
}
# Columns that the pipeline cannot work without.
REQUIRED_COLUMNS = ["InvoiceNo", "Description", "Quantity", "UnitPrice"]


@dataclass
class MiningConfig:
    """User-adjustable parameters. Changing them changes how many patterns are found."""
    min_support: float = 0.01      # itemset must appear in >= 1 % of baskets
    min_confidence: float = 0.30   # rule must be right in >= 30 % of cases
    min_lift: float = 1.2          # keep only clearly positive associations
    max_itemset_len: int = 3       # limit itemset size (keeps the output readable)
    min_basket_size: int = 1       # baskets smaller than this are removed
    top_n_rules: int = 15          # how many rules to show in tables / charts
    run_fpgrowth: bool = True      # also run FP-Growth and compare with Apriori
    timing_repeats: int = 5        # repeats used for the runtime comparison
