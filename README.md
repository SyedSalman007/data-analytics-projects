# data-analytics-projects

## Structure

```text
data-analytics-projects
├── datasets
└── notebooks
```

## Project 01: Customer Churn Analytics

Implemented in:
- `notebooks/Customer Churn Analytics/customer_churn_analysis.py`
- `notebooks/Customer Churn Analytics/01_customer_churn_analytics.ipynb`

This project includes:
1. Cohort analysis (customer retention by signup month)
2. Behavior patterns before churn
3. Churn risk segmentation
4. Churn prediction model

### Setup and Run

```bash
scripts/bootstrap_venv.sh
source .venv/bin/activate
python notebooks/Customer\\ Churn\\ Analytics/customer_churn_analysis.py
```

Outputs are generated in `notebooks/Customer Churn Analytics/output/`.

## Project 02: E-Commerce Sales Analytics

Implemented in:
- `notebooks/02_ecommerce_sales/ecommerce_sales_analysis.py`
- `notebooks/02_ecommerce_sales/02_ecommerce_sales_analysis.ipynb`

Outputs are generated in `notebooks/02_ecommerce_sales/output/`.

## Project 03: A/B Analysis Testing

Implemented in:
- `notebooks/03_ab_testing/ab_testing_analysis.py`
- `notebooks/03_ab_testing/03_ab_testing_dashboard.ipynb`

Outputs are generated in `notebooks/03_ab_testing/output/`.

## Python Version

This repo targets Python 3.13 (see `.python-version` and `pyproject.toml`). Use `scripts/bootstrap_venv.sh` to recreate `.venv` with `python3.13`.
