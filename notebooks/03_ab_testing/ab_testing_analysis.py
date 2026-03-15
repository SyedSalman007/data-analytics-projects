from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")

if sys.version_info < (3, 13):
    raise RuntimeError(
        f"Python 3.13+ required (business standard). Found: {sys.version.split()[0]}. "
        "Run: scripts/bootstrap_venv.sh"
    )


@dataclass(frozen=True)
class ProjectPaths:
    base_dir: Path
    dataset_path: Path
    project_dir: Path
    output_dir: Path


def resolve_paths(base_dir: str | Path = ".") -> ProjectPaths:
    base = Path(base_dir).resolve()
    dataset_path = base / "datasets" / "A:B Analysis Testing" / "cleaned_speakers_data.csv"
    project_dir = base / "notebooks" / "03_ab_testing"
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(
        base_dir=base,
        dataset_path=dataset_path,
        project_dir=project_dir,
        output_dir=output_dir,
    )


def _coerce_bool01(series: pd.Series) -> pd.Series:
    # Accept 0/1 ints, strings, booleans
    s = series
    if s.dtype == bool:
        return s.astype(int)
    if s.dtype.kind in {"i", "u", "f"}:
        return s.fillna(0).astype(int).clip(0, 1)
    ss = s.astype(str).str.strip().str.lower()
    return ss.isin(["1", "true", "yes", "y"]).astype(int)


def _to_number(series: pd.Series) -> pd.Series:
    if series.dtype.kind in {"i", "u", "f"}:
        return series
    s = series.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "none": np.nan, "null": np.nan})
    return pd.to_numeric(s, errors="coerce")


def load_dataset(paths: ProjectPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.dataset_path, encoding_errors="ignore")

    # Canonicalize a couple of column names for easier analysis
    df = df.rename(columns={"revenue_$": "revenue_usd"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["demographic_age"] = _to_number(df["demographic_age"])
    df["time_spent"] = _to_number(df["time_spent"])  # minutes
    df["pages_visited"] = _to_number(df["pages_visited"]).fillna(0).astype(int)
    df["conversion_flag"] = _coerce_bool01(df["conversion_flag"])
    df["bounce_flag"] = _coerce_bool01(df["bounce_flag"])
    df["revenue_usd"] = _to_number(df["revenue_usd"]).fillna(0.0)

    # Clean some categorical strings
    for c in [
        "sign_in",
        "demographic_age_group",
        "demographic_gender",
        "location",
        "country",
        "device_type",
        "variant_group",
        "conversion_type",
        "traffic_source",
        "payment_type",
        "card_type",
        "coupon_applied",
    ]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].str.lower().isin(["nan", "none", "null", "" ]), c] = np.nan

    return df


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    # Wilson score interval for a binomial proportion.
    if n <= 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _holm_bonferroni(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Dict[str, Any]] = {}
    rejected_up_to = -1
    for i, (name, p) in enumerate(items):
        thresh = alpha / (m - i)
        reject = p <= thresh and (rejected_up_to == i - 1)
        if reject:
            rejected_up_to = i
        out[name] = {"p_value": float(p), "threshold": float(thresh), "reject": bool(reject)}
    return out


def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        missing = int(s.isna().sum())
        nunique = int(s.nunique(dropna=True))

        dtype = str(s.dtype)
        example = None
        if n > 0:
            nonnull = s.dropna()
            if not nonnull.empty:
                example = nonnull.iloc[0]

        row: Dict[str, Any] = {
            "column": c,
            "dtype": dtype,
            "missing": missing,
            "missing_pct": float(missing / n) if n else 0.0,
            "unique": nunique,
            "example": str(example) if example is not None else "",
        }

        if s.dtype.kind in {"i", "u", "f"}:
            row.update(
                {
                    "min": float(np.nanmin(s)) if missing < n else float("nan"),
                    "p25": float(np.nanpercentile(s, 25)) if missing < n else float("nan"),
                    "median": float(np.nanmedian(s)) if missing < n else float("nan"),
                    "p75": float(np.nanpercentile(s, 75)) if missing < n else float("nan"),
                    "max": float(np.nanmax(s)) if missing < n else float("nan"),
                    "mean": float(np.nanmean(s)) if missing < n else float("nan"),
                }
            )
        else:
            top = s.astype(str).value_counts(dropna=True).head(5)
            row["top_values"] = "; ".join([f"{k} ({v})" for k, v in top.items()])

        rows.append(row)

    return pd.DataFrame(rows)


def ab_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("variant_group", dropna=False)
    summary = (
        g.agg(
            sessions=("session_id", "nunique"),
            users=("user_id", "nunique"),
            conversions=("conversion_flag", "sum"),
            conversion_rate=("conversion_flag", "mean"),
            bounces=("bounce_flag", "sum"),
            bounce_rate=("bounce_flag", "mean"),
            avg_time_spent_min=("time_spent", "mean"),
            avg_pages=("pages_visited", "mean"),
            total_revenue=("revenue_usd", "sum"),
            revenue_per_session=("revenue_usd", "mean"),
        )
        .reset_index()
    )

    cis = []
    for _, r in summary.iterrows():
        ci_lo, ci_hi = _wilson_ci(int(r["conversions"]), int(r["sessions"]))
        cis.append((ci_lo, ci_hi))
    summary["conversion_ci_low"] = [x[0] for x in cis]
    summary["conversion_ci_high"] = [x[1] for x in cis]

    return summary.sort_values("variant_group")


def chi_square_conversion_test(df: pd.DataFrame) -> dict:
    tab = pd.crosstab(df["variant_group"], df["conversion_flag"]).reindex(columns=[0, 1], fill_value=0)
    chi2, p, dof, expected = stats.chi2_contingency(tab.values)
    return {
        "contingency_table": tab.to_dict(),
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "expected": expected.tolist(),
    }


def two_proportion_ztest(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, float, float]:
    # Returns (diff, z, p_two_sided)
    if n1 <= 0 or n2 <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (p1 - p2, float("inf"), 0.0)
    z = (p1 - p2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return (p1 - p2, float(z), float(p))


def pairwise_conversion_tests(summary: pd.DataFrame) -> dict:
    rows = summary.set_index("variant_group")
    variants = [v for v in rows.index if pd.notna(v)]
    results: Dict[str, Any] = {}
    pvals: Dict[str, float] = {}

    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            a = variants[i]
            b = variants[j]
            k1, n1 = int(rows.loc[a, "conversions"]), int(rows.loc[a, "sessions"])
            k2, n2 = int(rows.loc[b, "conversions"]), int(rows.loc[b, "sessions"])
            diff, z, p = two_proportion_ztest(k1, n1, k2, n2)
            name = f"{a}_vs_{b}"
            results[name] = {
                "a": a,
                "b": b,
                "diff_p": float(diff),
                "z": float(z),
                "p_value": float(p),
            }
            pvals[name] = float(p)

    adj = _holm_bonferroni(pvals)
    for name, info in results.items():
        info["holm_bonferroni"] = adj[name]

    return results


def ttest_revenue_by_variant(df: pd.DataFrame) -> dict:
    # T-test on revenue per session by variant; also provide a non-parametric check.
    out: Dict[str, Any] = {}

    variants = [v for v in sorted(df["variant_group"].dropna().unique())]
    series = {v: df.loc[df["variant_group"] == v, "revenue_usd"].astype(float).values for v in variants}

    pvals_t: Dict[str, float] = {}
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            a, b = variants[i], variants[j]
            x, y = series[a], series[b]
            # Welch's t-test
            t, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
            name = f"{a}_vs_{b}"
            out[name] = {
                "a": a,
                "b": b,
                "test": "welch_ttest",
                "t_stat": float(t),
                "p_value": float(p),
                "mean_a": float(np.nanmean(x)),
                "mean_b": float(np.nanmean(y)),
            }
            pvals_t[name] = float(p)

            # Mann-Whitney U as robustness check (skewed revenue)
            try:
                u, p_u = stats.mannwhitneyu(x, y, alternative="two-sided")
                out[name]["mannwhitneyu"] = {"u_stat": float(u), "p_value": float(p_u)}
            except ValueError:
                out[name]["mannwhitneyu"] = {"u_stat": float("nan"), "p_value": float("nan")}

    adj = _holm_bonferroni(pvals_t)
    for name in out:
        out[name]["holm_bonferroni"] = adj[name]

    return out


def plot_dashboard(summary: pd.DataFrame, df: pd.DataFrame, output_dir: Path) -> None:
    # Conversion rate with Wilson CI
    plt.figure(figsize=(10, 6))
    s = summary.sort_values("variant_group")
    x = np.arange(len(s))
    y = s["conversion_rate"].values
    yerr = np.vstack([y - s["conversion_ci_low"].values, s["conversion_ci_high"].values - y])
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=6)
    plt.xticks(x, s["variant_group"].astype(str).values)
    plt.title("Conversion Rate by Variant (Wilson 95% CI)")
    plt.ylabel("Conversion Rate")
    plt.tight_layout()
    plt.savefig(output_dir / "conversion_rate_by_variant.png", dpi=160)
    plt.close()

    # Revenue per session
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, x="variant_group", y="revenue_per_session")
    plt.title("Revenue per Session by Variant")
    plt.xlabel("Variant")
    plt.ylabel("Revenue (USD) per Session")
    plt.tight_layout()
    plt.savefig(output_dir / "revenue_per_session_by_variant.png", dpi=160)
    plt.close()

    # Engagement distributions
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="variant_group", y="time_spent")
    plt.title("Time Spent (minutes) by Variant")
    plt.xlabel("Variant")
    plt.ylabel("Time Spent (min)")
    plt.tight_layout()
    plt.savefig(output_dir / "time_spent_by_variant_box.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="variant_group", y="pages_visited")
    plt.title("Pages Visited by Variant")
    plt.xlabel("Variant")
    plt.ylabel("Pages Visited")
    plt.tight_layout()
    plt.savefig(output_dir / "pages_visited_by_variant_box.png", dpi=160)
    plt.close()

    # Segmentation heatmap: conversion by device x variant
    if "device_type" in df.columns:
        pivot = (
            df.pivot_table(
                index="device_type",
                columns="variant_group",
                values="conversion_flag",
                aggfunc="mean",
            )
            .sort_index()
        )
        plt.figure(figsize=(10, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1)
        plt.title("Conversion Rate by Device Type and Variant")
        plt.xlabel("Variant")
        plt.ylabel("Device Type")
        plt.tight_layout()
        plt.savefig(output_dir / "conversion_by_device_variant_heatmap.png", dpi=160)
        plt.close()


def _df_to_markdown_fallback(df: pd.DataFrame, floatfmt: str = ".6g") -> str:
    """
    Render a DataFrame as a Markdown table without requiring the optional `tabulate` dependency.

    Pandas' DataFrame.to_markdown() requires `tabulate`. This fallback keeps the project runnable
    even when `tabulate` isn't installed in the environment.
    """

    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and math.isnan(v):
            return ""
        if isinstance(v, (float, np.floating)):
            try:
                return format(float(v), floatfmt)
            except Exception:
                return str(v)
        return str(v)

    cols = [str(c) for c in df.columns.tolist()]
    rows = [[fmt(v) for v in r] for r in df.to_numpy().tolist()]
    widths = [len(c) for c in cols]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def line(cells: List[str]) -> str:
        return "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells))) + " |"

    header = line(cols)
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = "\n".join(line(r) for r in rows)
    return "\n".join([header, sep, body]) if body else "\n".join([header, sep])


def df_to_markdown(df: pd.DataFrame) -> str:
    # Prefer pandas native if available; fallback otherwise.
    try:
        return df.to_markdown(index=False)  # requires tabulate
    except Exception:
        return _df_to_markdown_fallback(df)


def write_markdown_report(
    paths: ProjectPaths,
    df: pd.DataFrame,
    schema: pd.DataFrame,
    summary: pd.DataFrame,
    chi2: dict,
    pairwise_conv: dict,
    revenue_tests: dict,
) -> None:
    ts_min = pd.to_datetime(df["timestamp"]).min()
    ts_max = pd.to_datetime(df["timestamp"]).max()

    overall_conv = float(df["conversion_flag"].mean())
    overall_bounce = float(df["bounce_flag"].mean())
    total_rev = float(df["revenue_usd"].sum())

    lines: List[str] = []
    lines.append("# A/B Testing Dataset: Data Description Report\n")
    lines.append("## Overview\n")
    lines.append(f"- Rows: **{len(df):,}**\n")
    lines.append(f"- Columns: **{df.shape[1]}**\n")
    lines.append(f"- Time range: **{ts_min}** to **{ts_max}**\n")
    lines.append(f"- Overall conversion rate: **{overall_conv:.4f}**\n")
    lines.append(f"- Overall bounce rate: **{overall_bounce:.4f}**\n")
    lines.append(f"- Total revenue (USD): **{total_rev:,.2f}**\n")

    lines.append("\n## Variant Summary\n")
    lines.append(df_to_markdown(summary))

    lines.append("\n\n## Statistical Tests\n")
    lines.append("### Conversion Rate (Chi-square)\n")
    lines.append(f"- chi2: **{chi2['chi2']:.4f}**\n")
    lines.append(f"- dof: **{chi2['dof']}**\n")
    lines.append(f"- p-value: **{chi2['p_value']:.6g}**\n")

    lines.append("\n### Pairwise Conversion Tests (Two-proportion z-test + Holm-Bonferroni)\n")
    pairwise_rows = []
    for name, r in pairwise_conv.items():
        pairwise_rows.append(
            {
                "comparison": name,
                "diff_p": r["diff_p"],
                "z": r["z"],
                "p_value": r["p_value"],
                "holm_reject": r["holm_bonferroni"]["reject"],
                "holm_threshold": r["holm_bonferroni"]["threshold"],
            }
        )
    pairwise_df = pd.DataFrame(pairwise_rows).sort_values("p_value")
    lines.append(df_to_markdown(pairwise_df))

    lines.append("\n\n### Revenue per Session (Welch t-test + Mann-Whitney U + Holm-Bonferroni)\n")
    rev_rows = []
    for name, r in revenue_tests.items():
        rev_rows.append(
            {
                "comparison": name,
                "mean_a": r["mean_a"],
                "mean_b": r["mean_b"],
                "t_p_value": r["p_value"],
                "u_p_value": r.get("mannwhitneyu", {}).get("p_value"),
                "holm_reject": r["holm_bonferroni"]["reject"],
            }
        )
    rev_df = pd.DataFrame(rev_rows).sort_values("t_p_value")
    lines.append(df_to_markdown(rev_df))

    lines.append("\n\n## Data Dictionary (Observed)\n")
    cols = [
        "column",
        "dtype",
        "missing",
        "missing_pct",
        "unique",
        "example",
    ]
    extra = [c for c in schema.columns if c not in cols]
    lines.append(df_to_markdown(schema[cols + extra]))

    out_path = paths.output_dir / "data_description_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(base_dir: str | Path = ".") -> dict:
    paths = resolve_paths(base_dir)

    df = load_dataset(paths)
    if df.empty:
        raise ValueError("Dataset loaded but contains 0 rows.")

    schema = describe_dataframe(df)
    schema.to_csv(paths.output_dir / "data_schema_profile.csv", index=False)

    summary = ab_summary(df)
    summary.to_csv(paths.output_dir / "variant_summary.csv", index=False)

    chi2 = chi_square_conversion_test(df)
    pairwise_conv = pairwise_conversion_tests(summary)
    revenue_tests = ttest_revenue_by_variant(df)

    plot_dashboard(summary, df, paths.output_dir)

    results = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "variant_groups": sorted([str(x) for x in df["variant_group"].dropna().unique()]),
        "overall_conversion_rate": float(df["conversion_flag"].mean()),
        "overall_bounce_rate": float(df["bounce_flag"].mean()),
        "total_revenue_usd": float(df["revenue_usd"].sum()),
        "chi_square_conversion": chi2,
        "pairwise_conversion": pairwise_conv,
        "revenue_tests": revenue_tests,
    }

    (paths.output_dir / "ab_test_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    write_markdown_report(paths, df, schema, summary, chi2, pairwise_conv, revenue_tests)

    summary_path = paths.output_dir / "pipeline_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "output_dir": str(paths.output_dir),
                "dataset_path": str(paths.dataset_path),
                "rows": int(len(df)),
                "generated": [
                    "data_description_report.md",
                    "data_schema_profile.csv",
                    "variant_summary.csv",
                    "ab_test_results.json",
                    "conversion_rate_by_variant.png",
                    "revenue_per_session_by_variant.png",
                    "time_spent_by_variant_box.png",
                    "pages_visited_by_variant_box.png",
                    "conversion_by_device_variant_heatmap.png",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {"output_dir": str(paths.output_dir), "rows": int(len(df)), "variants": results["variant_groups"]}


if __name__ == "__main__":
    # Run from repo root when called directly.
    base = Path(__file__).resolve().parents[2]
    out = run_pipeline(base)
    print(json.dumps(out, indent=2))
