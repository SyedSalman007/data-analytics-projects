from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

if sys.version_info < (3, 13):
    raise RuntimeError(
        f"Python 3.13+ required (business standard). Found: {sys.version.split()[0]}. "
        "Run: scripts/bootstrap_venv.sh"
    )

@dataclass(frozen=True)
class ProjectPaths:
    base_dir: Path
    dataset_dir: Path
    project_dir: Path
    output_dir: Path


def resolve_paths(base_dir: str | Path = ".") -> ProjectPaths:
    base = Path(base_dir).resolve()
    dataset_dir = base / "datasets" / "E-Commerce Sales Dataset"
    project_dir = base / "notebooks" / "02_ecommerce_sales"
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(
        base_dir=base,
        dataset_dir=dataset_dir,
        project_dir=project_dir,
        output_dir=output_dir,
    )


def _normalize_columns(columns: Iterable[str]) -> list[str]:
    out: list[str] = []
    for col in columns:
        col = str(col)
        col = col.strip()
        col = col.replace(" ", "_")
        col = col.replace("-", "_")
        col = col.replace("/", "_")
        col = col.replace("&", "and")
        out.append(col.lower())
    return out


def _drop_junk_columns(df: pd.DataFrame) -> pd.DataFrame:
    junk = []
    for c in df.columns:
        lc = str(c).lower().strip()
        if lc == "index":
            junk.append(c)
        if lc.startswith("unnamed"):
            junk.append(c)
    return df.drop(columns=sorted(set(junk)), errors="ignore")


def _to_number(series: pd.Series) -> pd.Series:
    if series.dtype.kind in {"i", "u", "f"}:
        return series
    s = series.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce")


def read_csv_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding_errors="ignore")
    df = _drop_junk_columns(df)
    df.columns = _normalize_columns(df.columns)
    return df


def load_amazon_sales(paths: ProjectPaths) -> pd.DataFrame:
    df = read_csv_clean(paths.dataset_dir / "Amazon Sale Report.csv")

    # Standardize key columns
    df = df.rename(
        columns={
            "order_id": "order_id",
            "date": "date",
            "sku": "sku",
            "qty": "qty",
            "amount": "amount",
            "currency": "currency",
            "category": "category",
            "status": "status",
            "fulfilment": "fulfilment",
            "fulfilled_by": "fulfilled_by",
            "ship_city": "ship_city",
            "ship_state": "ship_state",
            "ship_country": "ship_country",
        }
    )

    if "sales_channel" in df.columns and "sales_channel_" not in df.columns:
        # Some files have "Sales Channel " with trailing space which normalizes to sales_channel_
        pass
    if "sales_channel_" in df.columns and "sales_channel" not in df.columns:
        df = df.rename(columns={"sales_channel_": "sales_channel"})

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["qty"] = _to_number(df.get("qty")).fillna(0).astype(int)
    df["amount"] = _to_number(df.get("amount")).fillna(0.0)

    if "b2b" in df.columns:
        df["b2b"] = (
            df["b2b"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
        )
    else:
        df["b2b"] = False

    df["sku"] = df.get("sku").astype(str).str.strip()
    df["category"] = df.get("category").astype(str).str.strip().replace({"nan": np.nan})
    df["currency"] = df.get("currency").astype(str).str.upper().str.strip().replace({"NAN": np.nan})

    df["channel"] = "Amazon"
    df = df.dropna(subset=["date"]).copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df


def load_international_sales(paths: ProjectPaths) -> pd.DataFrame:
    df = read_csv_clean(paths.dataset_dir / "International sale Report.csv")

    # Expected cols after normalization: date, months, customer, style, sku, size, pcs, rate, gross_amt
    rename_map = {
        "date": "date",
        "months": "months",
        "customer": "customer",
        "sku": "sku",
        "pcs": "qty",
        "rate": "rate",
        "gross_amt": "amount",
        "gross_amt_": "amount",
        "gross_amt__": "amount",
    }
    for old, new in list(rename_map.items()):
        if old in df.columns:
            df = df.rename(columns={old: new})

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["qty"] = _to_number(df.get("qty")).fillna(0).astype(int)
    df["rate"] = _to_number(df.get("rate")).fillna(0.0)
    df["amount"] = _to_number(df.get("amount")).fillna(0.0)

    df["customer"] = df.get("customer").astype(str).str.strip().replace({"nan": np.nan})
    df["sku"] = df.get("sku").astype(str).str.strip()

    df["currency"] = "INTERNATIONAL"  # not provided in file
    df["channel"] = "International"
    df = df.dropna(subset=["date"]).copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df


def load_inventory(paths: ProjectPaths) -> pd.DataFrame:
    df = read_csv_clean(paths.dataset_dir / "Sale Report.csv")

    # Standardize
    if "sku_code" in df.columns and "sku" not in df.columns:
        df = df.rename(columns={"sku_code": "sku"})

    df["stock"] = _to_number(df.get("stock")).fillna(0).astype(int)
    df["sku"] = df.get("sku").astype(str).str.strip()
    df["category"] = df.get("category").astype(str).str.strip().replace({"nan": np.nan})
    df["size"] = df.get("size").astype(str).str.strip().replace({"nan": np.nan})
    df["color"] = df.get("color").astype(str).str.strip().replace({"nan": np.nan})

    if "design_no." in df.columns and "design_no" not in df.columns:
        df = df.rename(columns={"design_no.": "design_no"})

    return df


def load_pricing_table(paths: ProjectPaths, filename: str) -> pd.DataFrame:
    df = read_csv_clean(paths.dataset_dir / filename)

    df = df.rename(columns={"sku": "sku"})
    df["sku"] = df.get("sku").astype(str).str.strip()

    # Numeric columns: everything except identifiers
    id_cols = {"sku", "style_id", "catalog", "category"}
    for c in df.columns:
        if c not in id_cols:
            df[c] = _to_number(df[c])

    return df


def load_cloud_warehouse_comparison(paths: ProjectPaths) -> pd.DataFrame:
    df = read_csv_clean(paths.dataset_dir / "Cloud Warehouse Compersion Chart.csv")

    # Some files contain a stray unnamed column between shiprocket and increff
    keep = []
    for c in df.columns:
        if c in {"shiprocket", "increff"}:
            keep.append(c)
    df = df[keep].copy()

    for c in df.columns:
        df[c] = _to_number(df[c])

    return df


def load_expenses(paths: ProjectPaths) -> pd.DataFrame:
    df = read_csv_clean(paths.dataset_dir / "Expense IIGF.csv")

    df = df.rename(columns={"recived_amount": "received_amount", "expance": "expense"})

    df["received_amount"] = _to_number(df.get("received_amount")).fillna(0.0)
    df["expense"] = _to_number(df.get("expense")).fillna(0.0)

    df["net"] = df["received_amount"] - df["expense"]
    return df


def build_sales_facts(amazon: pd.DataFrame, intl: pd.DataFrame) -> pd.DataFrame:
    common_cols = [
        "channel",
        "date",
        "month",
        "sku",
        "category",
        "qty",
        "amount",
        "currency",
    ]

    amazon_fact = amazon.copy()
    if "category" not in amazon_fact.columns:
        amazon_fact["category"] = np.nan

    intl_fact = intl.copy()
    if "category" not in intl_fact.columns:
        intl_fact["category"] = np.nan

    amazon_fact = amazon_fact.reindex(columns=common_cols)
    intl_fact = intl_fact.reindex(columns=common_cols)

    return pd.concat([amazon_fact, intl_fact], ignore_index=True)


def plot_monthly_trends(sales: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    monthly = (
        sales.groupby(["channel", "month"], as_index=False)
        .agg(orders=("sku", "size"), units=("qty", "sum"), revenue=("amount", "sum"))
        .sort_values(["channel", "month"])
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly, x="month", y="revenue", hue="channel", marker="o")
    plt.title("Monthly Revenue Trend by Channel")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_revenue_trend.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly, x="month", y="units", hue="channel", marker="o")
    plt.title("Monthly Units Trend by Channel")
    plt.xlabel("Month")
    plt.ylabel("Units")
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_units_trend.png", dpi=160)
    plt.close()

    monthly.to_csv(output_dir / "monthly_trends.csv", index=False)
    return monthly


def plot_category_mix(amazon: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if "category" not in amazon.columns:
        return pd.DataFrame()

    by_cat = (
        amazon.groupby("category", dropna=False, as_index=False)
        .agg(orders=("order_id", "nunique"), units=("qty", "sum"), revenue=("amount", "sum"))
        .sort_values("revenue", ascending=False)
    )

    top = by_cat.head(15)
    plt.figure(figsize=(12, 7))
    sns.barplot(data=top, y="category", x="revenue", orient="h")
    plt.title("Top Categories by Revenue (Amazon)")
    plt.xlabel("Revenue")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(output_dir / "amazon_top_categories_revenue.png", dpi=160)
    plt.close()

    by_cat.to_csv(output_dir / "amazon_category_mix.csv", index=False)
    return by_cat


def plot_fulfilment_status_breakdown(amazon: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    cols = [c for c in ["status", "courier_status", "fulfilment", "fulfilled_by"] if c in amazon.columns]
    if not cols:
        return pd.DataFrame()

    results = {}
    for c in cols:
        vc = amazon[c].astype(str).str.strip().replace({"nan": "(missing)"}).value_counts().head(20)
        results[c] = vc

        plt.figure(figsize=(12, 6))
        sns.barplot(x=vc.values, y=vc.index, orient="h")
        plt.title(f"Top {min(20, len(vc))} values: {c} (Amazon)")
        plt.xlabel("Rows")
        plt.ylabel(c)
        plt.tight_layout()
        plt.savefig(output_dir / f"amazon_{c}_top_values.png", dpi=160)
        plt.close()

    # Persist a compact table
    long_rows = []
    for c, vc in results.items():
        for k, v in vc.items():
            long_rows.append({"field": c, "value": k, "rows": int(v)})
    out = pd.DataFrame(long_rows)
    out.to_csv(output_dir / "amazon_status_fulfilment_breakdown.csv", index=False)
    return out


def analyze_cloud_warehouse_profitability(comp: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = comp.agg(["count", "mean", "median", "min", "max"]).T.reset_index().rename(columns={"index": "channel"})

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=comp.melt(var_name="channel", value_name="profit"), x="channel", y="profit")
    plt.title("Profitability Comparison: Shiprocket vs INCREFF")
    plt.xlabel("Channel")
    plt.ylabel("Profit")
    plt.tight_layout()
    plt.savefig(output_dir / "profitability_shiprocket_vs_increff.png", dpi=160)
    plt.close()

    summary.to_csv(output_dir / "profitability_comparison_summary.csv", index=False)
    return summary


def _mrp_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_mrp") or c in {"mrp_old", "final_mrp_old"}]


def analyze_pricing(pricing_2021: pd.DataFrame, pricing_2022: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    p21 = pricing_2021.copy()
    p22 = pricing_2022.copy()
    p21["snapshot"] = "2021-03"
    p22["snapshot"] = "2022-05"

    combined = pd.concat([p21, p22], ignore_index=True, sort=False)

    # Cost proxy: average of tp columns if present
    if "tp_1" in combined.columns and "tp_2" in combined.columns:
        combined["tp_cost"] = combined[["tp_1", "tp_2"]].mean(axis=1)
    elif "tp" in combined.columns:
        combined["tp_cost"] = combined["tp"]
    else:
        combined["tp_cost"] = np.nan

    for c in _mrp_columns(combined):
        combined[c] = _to_number(combined[c])

    mrp_cols = [c for c in _mrp_columns(combined) if c not in {"mrp_old", "final_mrp_old"}]
    if mrp_cols:
        combined["best_mrp"] = combined[mrp_cols].max(axis=1, skipna=True)
        combined["worst_mrp"] = combined[mrp_cols].min(axis=1, skipna=True)
        combined["mrp_spread"] = combined["best_mrp"] - combined["worst_mrp"]

    combined["gross_margin_best"] = combined["best_mrp"] - combined["tp_cost"]

    snap_summary = (
        combined.groupby("snapshot", as_index=False)
        .agg(
            skus=("sku", "nunique"),
            avg_cost=("tp_cost", "mean"),
            avg_best_mrp=("best_mrp", "mean"),
            avg_margin_best=("gross_margin_best", "mean"),
            avg_mrp_spread=("mrp_spread", "mean"),
        )
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(data=snap_summary, x="snapshot", y="avg_margin_best")
    plt.title("Average Best-Channel Margin (MRP - TP cost)")
    plt.xlabel("Snapshot")
    plt.ylabel("Average Margin")
    plt.tight_layout()
    plt.savefig(output_dir / "pricing_avg_margin_by_snapshot.png", dpi=160)
    plt.close()

    # Channel-level analysis (which marketplace gives highest MRP most often)
    channel_wins = []
    for snap, part in combined.groupby("snapshot"):
        if not mrp_cols:
            continue
        best_channel = part[mrp_cols].idxmax(axis=1)
        vc = best_channel.value_counts()
        for ch, n in vc.items():
            channel_wins.append({"snapshot": snap, "channel_mrp_col": ch, "sku_rows": int(n)})

    wins_df = pd.DataFrame(channel_wins)
    if not wins_df.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=wins_df, x="channel_mrp_col", y="sku_rows", hue="snapshot")
        plt.title("Which Channel Has the Highest MRP (count of SKUs)")
        plt.xlabel("MRP Column")
        plt.ylabel("SKU Rows")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "pricing_channel_best_mrp_counts.png", dpi=160)
        plt.close()

        wins_df.to_csv(output_dir / "pricing_channel_best_mrp_counts.csv", index=False)

    snap_summary.to_csv(output_dir / "pricing_snapshot_summary.csv", index=False)
    combined.to_csv(output_dir / "pricing_combined_clean.csv", index=False)
    return snap_summary


def analyze_customer_sku_performance(intl: pd.DataFrame, output_dir: Path) -> None:
    if "customer" not in intl.columns:
        return

    top_customers = (
        intl.groupby("customer", as_index=False)
        .agg(orders=("sku", "size"), units=("qty", "sum"), revenue=("amount", "sum"))
        .sort_values("revenue", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(data=top_customers, y="customer", x="revenue", orient="h")
    plt.title("Top Customers by Revenue (International)")
    plt.xlabel("Revenue")
    plt.ylabel("Customer")
    plt.tight_layout()
    plt.savefig(output_dir / "international_top_customers_revenue.png", dpi=160)
    plt.close()

    top_customers.to_csv(output_dir / "international_top_customers.csv", index=False)

    top_skus = (
        intl.groupby("sku", as_index=False)
        .agg(orders=("customer", "size"), units=("qty", "sum"), revenue=("amount", "sum"))
        .sort_values("revenue", ascending=False)
        .head(25)
    )
    top_skus.to_csv(output_dir / "international_top_skus.csv", index=False)


def analyze_inventory_turnover(inventory: pd.DataFrame, amazon: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    # Sales velocity from the most recent 90 days available in the Amazon data.
    if amazon.empty:
        return pd.DataFrame()

    max_date = amazon["date"].max()
    window_start = max_date - pd.Timedelta(days=90)

    recent = amazon.loc[(amazon["date"] > window_start) & (amazon["date"] <= max_date)].copy()
    velocity = (
        recent.groupby("sku", as_index=False)
        .agg(units_90d=("qty", "sum"), revenue_90d=("amount", "sum"))
        .sort_values("units_90d", ascending=False)
    )

    inv = inventory[["sku", "stock", "category", "size", "color"]].copy()
    merged = inv.merge(velocity, on="sku", how="left").fillna({"units_90d": 0, "revenue_90d": 0})

    merged["units_per_day"] = merged["units_90d"] / 90.0
    merged["days_of_stock"] = np.where(
        merged["units_per_day"] > 0,
        merged["stock"] / merged["units_per_day"],
        np.inf,
    )

    merged = merged.sort_values(["days_of_stock"], ascending=False)
    merged.to_csv(output_dir / "inventory_turnover_90d.csv", index=False)

    # Visual: stock vs units sold
    sample = merged.replace([np.inf], np.nan).dropna(subset=["days_of_stock"]).copy()
    sample = sample[sample["days_of_stock"] < 3650]  # remove extreme outliers for plotting

    if not sample.empty:
        plt.figure(figsize=(9, 6))
        sns.scatterplot(data=sample, x="units_90d", y="stock")
        plt.title("Inventory vs 90-day Units Sold (Amazon)")
        plt.xlabel("Units Sold (90d)")
        plt.ylabel("Current Stock")
        plt.tight_layout()
        plt.savefig(output_dir / "inventory_vs_units_90d.png", dpi=160)
        plt.close()

    return merged


def run_pipeline(base_dir: str | Path = ".") -> dict:
    paths = resolve_paths(base_dir)

    amazon = load_amazon_sales(paths)
    intl = load_international_sales(paths)
    inventory = load_inventory(paths)
    pricing_2021 = load_pricing_table(paths, "P  L March 2021.csv")
    pricing_2022 = load_pricing_table(paths, "May-2022.csv")
    cloud_comp = load_cloud_warehouse_comparison(paths)
    expenses = load_expenses(paths)

    # Build consolidated sales
    sales = build_sales_facts(amazon, intl)
    sales.to_csv(paths.output_dir / "sales_facts_combined.csv", index=False)

    # General sales trends
    monthly = plot_monthly_trends(sales, paths.output_dir)

    # Amazon breakdowns
    category_mix = plot_category_mix(amazon, paths.output_dir)
    status_breakdown = plot_fulfilment_status_breakdown(amazon, paths.output_dir)

    # Warehouse profitability comparison
    profitability_summary = analyze_cloud_warehouse_profitability(cloud_comp, paths.output_dir)

    # Price comparisons across channels
    pricing_summary = analyze_pricing(pricing_2021, pricing_2022, paths.output_dir)

    # Customer and SKU performance (international)
    analyze_customer_sku_performance(intl, paths.output_dir)

    # Inventory optimization lens (stock vs velocity)
    inventory_turnover = analyze_inventory_turnover(inventory, amazon, paths.output_dir)

    # Expenses summary
    expenses.to_csv(paths.output_dir / "expenses_clean.csv", index=False)

    out = {
        "output_dir": str(paths.output_dir),
        "amazon_rows": int(len(amazon)),
        "international_rows": int(len(intl)),
        "inventory_rows": int(len(inventory)),
        "pricing_rows_2021": int(len(pricing_2021)),
        "pricing_rows_2022": int(len(pricing_2022)),
        "sales_rows": int(len(sales)),
        "date_min": str(pd.to_datetime(sales["date"]).min()),
        "date_max": str(pd.to_datetime(sales["date"]).max()),
        "expense_net_total": float(expenses["net"].sum()),
        "monthly_rows": int(len(monthly)),
        "amazon_category_rows": int(len(category_mix)) if isinstance(category_mix, pd.DataFrame) else 0,
        "amazon_status_rows": int(len(status_breakdown)) if isinstance(status_breakdown, pd.DataFrame) else 0,
        "inventory_turnover_rows": int(len(inventory_turnover)) if isinstance(inventory_turnover, pd.DataFrame) else 0,
        "profitability_summary_rows": int(len(profitability_summary)),
        "pricing_summary_rows": int(len(pricing_summary)),
    }

    (paths.output_dir / "pipeline_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    result = run_pipeline(Path(__file__).resolve().parents[2])
    print(json.dumps(result, indent=2))
