# Online Shopping Analysis Using Association Rule Mining

## Project Overview

Online Shopping Analysis Using Association Rule Mining is a Data Mining project that identifies relationships between products frequently purchased together by customers.

The project analyses online retail transaction data using **Association Rule Mining** techniques to discover frequent itemsets and generate meaningful association rules. These patterns can help online retailers understand customer purchasing behaviour and improve product recommendations, cross-selling strategies, and product bundling.

The project primarily uses the **Apriori algorithm** to discover frequent itemsets and generate association rules based on support, confidence, and lift.

## Objectives

- Analyse online shopping transaction data.
- Identify frequently purchased products and product combinations.
- Apply the Apriori algorithm to discover frequent itemsets.
- Generate association rules to identify relationships between products.
- Evaluate association rules using support, confidence, and lift.
- Visualise frequent itemsets and discovered purchasing patterns.
- Demonstrate the application of Data Mining in e-commerce.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data preprocessing and manipulation |
| NumPy | Numerical operations |
| Mlxtend | Association Rule Mining |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| Jupyter Notebook / VS Code | Development environment |

## Data Mining Techniques

### 1. Association Rule Mining

Association Rule Mining is used to discover relationships between items in large transactional datasets.

For example:

`{Bread} → {Butter}`

This rule indicates that customers purchasing bread may also tend to purchase butter.

### 2. Apriori Algorithm

The Apriori algorithm identifies frequent itemsets by analysing transaction data using a minimum support threshold.

It uses the Apriori property:

*If an itemset is frequent, all its non-empty subsets must also be frequent.*

### 3. Evaluation Metrics

The generated association rules are evaluated using three important measures:

- **Support:** Measures how frequently an itemset occurs in the dataset.
- **Confidence:** Measures how often the consequent appears in transactions containing the antecedent.
- **Lift:** Measures the strength of an association compared with what would be expected if the items were independent.

## System Workflow

1. Load the online retail transaction dataset.
2. Inspect and preprocess the dataset.
3. Handle missing values, invalid transactions, and cancelled orders.
4. Group products based on transaction IDs.
5. Convert transactions into a one-hot encoded basket format.
6. Apply the Apriori algorithm.
7. Generate association rules.
8. Evaluate and rank rules using support, confidence, and lift.
9. Visualise the discovered patterns.
10. Export the results for further analysis.

## Project Structure

```text
Online-Shopping-Analysis/
│
├── data/
│   └── online_retail.csv
│
├── output/
│   ├── frequent_itemsets.csv
│   ├── association_rules.csv
│   └── figures/
│
├── src/
│   ├── preprocessing.py
│   ├── association_mining.py
│   └── visualization.py
│
├── main.py
├── requirements.txt
├── README.md
└── report/
    └── Project_Report.docx
```

*The structure may vary depending on the final implementation.*

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- pip package manager

### 1. Clone the Repository

```bash
git clone <repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd Online-Shopping-Analysis
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Place the online retail transaction dataset inside the `data/` directory using the filename expected by the implementation.

### 5. Run the Project

```bash
python main.py
```

## Expected Results

The project produces the following outputs:

- Frequently purchased individual products.
- Frequent product combinations (itemsets).
- Association rules between products.
- Support, confidence, and lift values for each rule.
- Visualisations of frequent itemsets and association rules.
- CSV files containing the mining results.

These results provide insights into customer purchasing patterns and demonstrate how association rule mining can support retail decision-making.

## Applications

- Product recommendation systems.
- Market basket analysis.
- Cross-selling strategies.
- Product bundling.
- Customer purchasing behaviour analysis.
- E-commerce catalogue optimisation.

## Limitations

- The quality of discovered rules depends on the dataset.
- Results are sensitive to minimum support and confidence thresholds.
- Association rules indicate relationships but do not establish causation.
- Frequently occurring products may dominate the discovered patterns.

## Future Enhancements

- Implement FP-Growth and compare it with Apriori.
- Develop an interactive dashboard using Streamlit.
- Explore customer-specific purchasing patterns.
- Integrate discovered rules into a product recommendation interface.
- Analyse purchasing trends over different time periods.

## References

1. R. Agrawal, T. Imieliński, and A. Swami, "Mining Association Rules between Sets of Items in Large Databases," ACM SIGMOD, 1993.
2. UCI Machine Learning Repository – Online Retail Dataset: https://archive.ics.uci.edu/dataset/352/online+retail
3. Mlxtend Documentation: https://rasbt.github.io/mlxtend/
4. Pandas Documentation: https://pandas.pydata.org/docs/

---

**Project Domain:** Data Mining

**Project Title:** Online Shopping Analysis Using Association Rule Mining

**Core Algorithm:** Apriori

**Programming Language:** Python
