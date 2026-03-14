# Project 02: E-Commerce Sales Dataset

## Dataset
Reads CSVs from `datasets/E-Commerce Sales Dataset/`:
- `Amazon Sale Report.csv`
- `International sale Report.csv`
- `Sale Report.csv` (inventory)
- `P  L March 2021.csv` (pricing snapshot)
- `May-2022.csv` (pricing snapshot)
- `Cloud Warehouse Compersion Chart.csv` (Shiprocket vs INCREFF)
- `Expense IIGF.csv`

## Run
From repo root:

```bash
scripts/bootstrap_venv.sh
source .venv/bin/activate
python notebooks/02_ecommerce_sales/ecommerce_sales_analysis.py
```

Or open the notebook:

```bash
jupyter notebook notebooks/02_ecommerce_sales/02_ecommerce_sales_analysis.ipynb
```

## Outputs
Generated in `notebooks/02_ecommerce_sales/output/`.
