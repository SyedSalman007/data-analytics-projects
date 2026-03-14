from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")

if sys.version_info < (3, 13):
    raise RuntimeError(
        f"Python 3.13+ required (business standard). Found: {sys.version.split()[0]}. "
        "Run: scripts/bootstrap_venv.sh"
    )


@dataclass
class Paths:
    base_dir: Path
    dataset_dir: Path
    output_dir: Path


def resolve_paths(base_dir: str | Path = ".") -> Paths:
    base = Path(base_dir).resolve()
    dataset_dir = base / "datasets" / "Financial Transactions Dataset"
    output_dir = base / "notebooks" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return Paths(base_dir=base, dataset_dir=dataset_dir, output_dir=output_dir)


def parse_currency(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan})
        .astype(float)
    )


def month_diff(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)


def load_users_and_signup_month(paths: Paths) -> pd.DataFrame:
    users = pd.read_csv(paths.dataset_dir / "users_data.csv")
    cards = pd.read_csv(paths.dataset_dir / "cards_data.csv")

    cards["acct_open_date"] = pd.to_datetime(cards["acct_open_date"], format="%m/%Y", errors="coerce")
    signup = (
        cards.groupby("client_id", as_index=False)["acct_open_date"]
        .min()
        .rename(columns={"client_id": "id", "acct_open_date": "signup_date"})
    )

    users = users.merge(signup, on="id", how="left")

    for col in ["per_capita_income", "yearly_income", "total_debt"]:
        users[col] = parse_currency(users[col])

    users["gender"] = users["gender"].fillna("Unknown")
    return users


def first_pass_transactions(paths: Paths, chunk_size: int = 500_000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    tx_path = paths.dataset_dir / "transactions_data.csv"

    last_tx: Dict[int, pd.Timestamp] = {}
    first_tx: Dict[int, pd.Timestamp] = {}
    monthly_pairs: list[pd.DataFrame] = []
    reference_date: pd.Timestamp | None = None

    usecols = ["client_id", "date"]

    for chunk in pd.read_csv(tx_path, usecols=usecols, chunksize=chunk_size):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk.dropna(subset=["date", "client_id"])

        max_date = chunk["date"].max()
        if reference_date is None or max_date > reference_date:
            reference_date = max_date

        grp = chunk.groupby("client_id")["date"].agg(["min", "max"])
        for client_id, row in grp.iterrows():
            mn = row["min"]
            mx = row["max"]
            if client_id not in first_tx or mn < first_tx[client_id]:
                first_tx[client_id] = mn
            if client_id not in last_tx or mx > last_tx[client_id]:
                last_tx[client_id] = mx

        chunk["activity_month"] = chunk["date"].values.astype("datetime64[M]")
        monthly_pairs.append(chunk[["client_id", "activity_month"]].drop_duplicates())

    if reference_date is None:
        raise ValueError("No valid transaction dates found.")

    last_tx_df = pd.DataFrame(
        {
            "id": list(last_tx.keys()),
            "first_tx_date": [first_tx[k] for k in last_tx.keys()],
            "last_tx_date": [last_tx[k] for k in last_tx.keys()],
        }
    )
    activity = pd.concat(monthly_pairs, ignore_index=True).drop_duplicates()
    return last_tx_df, activity, reference_date


def build_cohort_retention(users: pd.DataFrame, activity: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    cohort_base = users[["id", "signup_date"]].copy()
    cohort_base = cohort_base.dropna(subset=["signup_date"])
    cohort_base["cohort_month"] = cohort_base["signup_date"].values.astype("datetime64[M]")

    active = activity.merge(cohort_base[["id", "cohort_month"]], left_on="client_id", right_on="id", how="inner")
    active["period_number"] = month_diff(active["activity_month"], active["cohort_month"])
    active = active[active["period_number"] >= 0]

    cohort_size = (
        cohort_base.groupby("cohort_month")["id"]
        .nunique()
        .rename("cohort_size")
        .reset_index()
    )

    retained = (
        active.groupby(["cohort_month", "period_number"])["client_id"]
        .nunique()
        .rename("retained_users")
        .reset_index()
    )

    cohort_retention = retained.merge(cohort_size, on="cohort_month", how="left")
    cohort_retention["retention_rate"] = cohort_retention["retained_users"] / cohort_retention["cohort_size"]

    pivot = cohort_retention.pivot(index="cohort_month", columns="period_number", values="retention_rate")
    plt.figure(figsize=(14, 7))
    sns.heatmap(pivot, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Customer Retention by Signup Cohort Month")
    plt.xlabel("Months Since Signup")
    plt.ylabel("Signup Cohort Month")
    plt.tight_layout()
    plt.savefig(output_dir / "cohort_retention_heatmap.png", dpi=160)
    plt.close()

    cohort_retention.to_csv(output_dir / "cohort_retention_table.csv", index=False)
    return cohort_retention


def aggregate_metrics(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[key_col, "txn_count", "spend_sum", "error_count", "chip_count"])

    grouped = (
        df.groupby(key_col)
        .agg(
            txn_count=(key_col, "size"),
            spend_sum=("amount_abs", "sum"),
            error_count=("error_flag", "sum"),
            chip_count=("chip_flag", "sum"),
        )
        .reset_index()
    )
    return grouped


def second_pass_features(
    paths: Paths,
    users: pd.DataFrame,
    churn_base: pd.DataFrame,
    reference_date: pd.Timestamp,
    chunk_size: int = 500_000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tx_path = paths.dataset_dir / "transactions_data.csv"

    pre_window = churn_base[["id", "window_end"]].copy()
    pre_window["window_start"] = pre_window["window_end"] - pd.Timedelta(days=90)

    recent_180_start = reference_date - pd.Timedelta(days=180)
    prev_180_start = reference_date - pd.Timedelta(days=360)

    pre_frames: list[pd.DataFrame] = []
    recent_frames: list[pd.DataFrame] = []
    prev_frames: list[pd.DataFrame] = []

    usecols = ["client_id", "date", "amount", "use_chip", "errors"]

    for chunk in pd.read_csv(tx_path, usecols=usecols, chunksize=chunk_size):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk.dropna(subset=["date", "client_id"])

        chunk["amount_num"] = parse_currency(chunk["amount"])
        chunk["amount_abs"] = chunk["amount_num"].abs()
        chunk["error_flag"] = chunk["errors"].astype(str).str.strip().ne("")
        chunk["chip_flag"] = chunk["use_chip"].astype(str).str.contains("Chip", case=False, na=False)

        merged = chunk.merge(pre_window, left_on="client_id", right_on="id", how="inner")
        pre_mask = (merged["date"] > merged["window_start"]) & (merged["date"] <= merged["window_end"])
        pre_frames.append(merged.loc[pre_mask, ["id", "amount_abs", "error_flag", "chip_flag"]])

        recent_mask = (chunk["date"] > recent_180_start) & (chunk["date"] <= reference_date)
        prev_mask = (chunk["date"] > prev_180_start) & (chunk["date"] <= recent_180_start)

        recent_subset = chunk.loc[recent_mask, ["client_id", "amount_abs", "error_flag", "chip_flag"]].copy()
        prev_subset = chunk.loc[prev_mask, ["client_id", "amount_abs", "error_flag", "chip_flag"]].copy()

        recent_frames.append(recent_subset)
        prev_frames.append(prev_subset)

    pre_all = pd.concat(pre_frames, ignore_index=True)
    recent_all = pd.concat(recent_frames, ignore_index=True)
    prev_all = pd.concat(prev_frames, ignore_index=True)

    pre_agg = aggregate_metrics(pre_all, "id")
    recent_agg = aggregate_metrics(recent_all, "client_id").rename(columns={"client_id": "id"})
    prev_agg = aggregate_metrics(prev_all, "client_id").rename(columns={"client_id": "id"})

    for frame in [pre_agg, recent_agg, prev_agg]:
        frame["avg_ticket"] = np.where(frame["txn_count"] > 0, frame["spend_sum"] / frame["txn_count"], 0)
        frame["error_rate"] = np.where(frame["txn_count"] > 0, frame["error_count"] / frame["txn_count"], 0)
        frame["chip_rate"] = np.where(frame["txn_count"] > 0, frame["chip_count"] / frame["txn_count"], 0)

    pre_behavior = churn_base[["id", "is_churned"]].merge(pre_agg, on="id", how="left").fillna(0)

    model_df = churn_base[["id", "is_churned", "signup_date", "last_tx_date"]].copy()
    model_df = model_df.merge(recent_agg.add_prefix("recent_"), left_on="id", right_on="recent_id", how="left")
    model_df = model_df.merge(prev_agg.add_prefix("prev_"), left_on="id", right_on="prev_id", how="left")

    model_df = model_df.drop(columns=["recent_id", "prev_id"], errors="ignore").fillna(0)

    model_df["recency_days"] = (reference_date - pd.to_datetime(model_df["last_tx_date"])).dt.days
    model_df["tenure_days"] = (reference_date - pd.to_datetime(model_df["signup_date"])).dt.days
    model_df["txn_trend_180"] = model_df["recent_txn_count"] - model_df["prev_txn_count"]
    model_df["spend_trend_180"] = model_df["recent_spend_sum"] - model_df["prev_spend_sum"]

    users_numeric = users[["id", "current_age", "credit_score", "num_credit_cards", "yearly_income", "total_debt"]].copy()
    model_df = model_df.merge(users_numeric, on="id", how="left")

    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = [c for c in model_df.columns if c not in ["id", "signup_date", "last_tx_date", "is_churned"]]
    model_df[numeric_cols] = model_df[numeric_cols].fillna(model_df[numeric_cols].median())

    return pre_behavior, model_df


def create_prechurn_outputs(pre_behavior: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = (
        pre_behavior.groupby("is_churned")
        .agg(
            customers=("id", "nunique"),
            avg_txn_count_90d=("txn_count", "mean"),
            avg_spend_90d=("spend_sum", "mean"),
            avg_ticket_90d=("avg_ticket", "mean"),
            avg_error_rate_90d=("error_rate", "mean"),
            avg_chip_rate_90d=("chip_rate", "mean"),
        )
        .reset_index()
    )
    summary["group"] = summary["is_churned"].map({1: "Churned", 0: "Active"})

    plot_df = summary.melt(
        id_vars=["group"],
        value_vars=[
            "avg_txn_count_90d",
            "avg_spend_90d",
            "avg_ticket_90d",
            "avg_error_rate_90d",
            "avg_chip_rate_90d",
        ],
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(14, 6))
    sns.barplot(data=plot_df, x="metric", y="value", hue="group")
    plt.title("Behavior in Last 90 Days Before Observation End")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "prechurn_behavior_patterns.png", dpi=160)
    plt.close()

    summary.to_csv(output_dir / "prechurn_behavior_summary.csv", index=False)
    return summary


def safe_qcut_rank(series: pd.Series, reverse: bool = False) -> pd.Series:
    ranked = series.rank(method="average")
    q = pd.qcut(ranked, q=5, labels=False, duplicates="drop")
    score = q.astype(float) + 1
    if reverse:
        score = score.max() + 1 - score
    return score.fillna(score.median())


def create_risk_segments(model_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    seg = model_df[["id", "is_churned", "recency_days", "recent_txn_count", "recent_spend_sum"]].copy()
    seg = seg.rename(columns={"recent_txn_count": "frequency_180d", "recent_spend_sum": "monetary_180d"})

    seg["r_score"] = safe_qcut_rank(seg["recency_days"], reverse=True)
    seg["f_score"] = safe_qcut_rank(seg["frequency_180d"], reverse=False)
    seg["m_score"] = safe_qcut_rank(seg["monetary_180d"], reverse=False)

    seg["risk_points"] = (6 - seg["r_score"]) + (6 - seg["f_score"]) + (6 - seg["m_score"])

    seg["risk_segment"] = pd.cut(
        seg["risk_points"],
        bins=[-np.inf, 6.5, 10.5, np.inf],
        labels=["Low", "Medium", "High"],
    )

    counts = seg["risk_segment"].value_counts().reindex(["Low", "Medium", "High"])
    plt.figure(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title("Customer Churn Risk Segments")
    plt.xlabel("Risk Segment")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(output_dir / "risk_segment_counts.png", dpi=160)
    plt.close()

    seg.to_csv(output_dir / "churn_risk_segments.csv", index=False)
    return seg


def train_churn_model(model_df: pd.DataFrame, output_dir: Path) -> dict:
    feature_cols = [
        "recency_days",
        "tenure_days",
        "recent_txn_count",
        "recent_spend_sum",
        "recent_avg_ticket",
        "recent_error_rate",
        "recent_chip_rate",
        "prev_txn_count",
        "prev_spend_sum",
        "prev_avg_ticket",
        "prev_error_rate",
        "prev_chip_rate",
        "txn_trend_180",
        "spend_trend_180",
        "current_age",
        "credit_score",
        "num_credit_cards",
        "yearly_income",
        "total_debt",
    ]

    X = model_df[feature_cols].copy()
    y = model_df["is_churned"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    lr_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    rf_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    models = {
        "logistic_regression": lr_model,
        "random_forest": rf_model,
    }

    results = {}
    best_name = None
    best_auc = -np.inf

    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        f1 = f1_score(y_test, pred)

        results[name] = {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "f1_at_0_5": float(f1),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "classification_report": classification_report(y_test, pred, output_dict=True),
        }

        if roc_auc > best_auc:
            best_auc = roc_auc
            best_name = name

    summary = {
        "best_model": best_name,
        "metrics": results,
        "n_customers": int(len(model_df)),
        "churn_rate": float(y.mean()),
    }

    with (output_dir / "churn_model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if best_name == "logistic_regression":
        best_model = lr_model.fit(X_train, y_train)
        coefs = best_model.named_steps["clf"].coef_[0]
        importance = pd.DataFrame({"feature": feature_cols, "importance": coefs})
        importance["abs_importance"] = importance["importance"].abs()
    else:
        best_model = rf_model.fit(X_train, y_train)
        importance = pd.DataFrame(
            {"feature": feature_cols, "importance": best_model.feature_importances_}
        )
        importance["abs_importance"] = importance["importance"].abs()

    importance = importance.sort_values("abs_importance", ascending=False).head(12)
    importance.to_csv(output_dir / "model_feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance, x="abs_importance", y="feature", orient="h")
    plt.title(f"Top Feature Importance ({best_name})")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "model_feature_importance.png", dpi=160)
    plt.close()

    return summary


def run_pipeline(base_dir: str | Path = ".") -> dict:
    paths = resolve_paths(base_dir)

    users = load_users_and_signup_month(paths)
    last_tx_df, activity, reference_date = first_pass_transactions(paths)

    users = users.merge(last_tx_df, on="id", how="left")
    users["signup_date"] = users["signup_date"].fillna(users["first_tx_date"])

    churn_cutoff = reference_date - pd.Timedelta(days=90)
    users["is_churned"] = (pd.to_datetime(users["last_tx_date"]) < churn_cutoff).astype(int)
    users["window_end"] = np.where(users["is_churned"] == 1, users["last_tx_date"], reference_date)
    users["window_end"] = pd.to_datetime(users["window_end"])

    cohort = build_cohort_retention(users, activity, paths.output_dir)
    pre_behavior, model_df = second_pass_features(paths, users, users, reference_date)
    pre_summary = create_prechurn_outputs(pre_behavior, paths.output_dir)
    segments = create_risk_segments(model_df, paths.output_dir)
    model_summary = train_churn_model(model_df, paths.output_dir)

    outputs = {
        "reference_date": str(reference_date),
        "cohort_rows": int(len(cohort)),
        "pre_behavior_groups": int(len(pre_summary)),
        "segment_counts": segments["risk_segment"].value_counts().to_dict(),
        "model_best": model_summary["best_model"],
        "output_dir": str(paths.output_dir),
    }

    with (paths.output_dir / "pipeline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    return outputs


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))
