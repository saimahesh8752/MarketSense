"""
MarketSense: Do Small AI Models Understand Financial News?
An Empirical Study of Compact Language Models and Stock Return Reactions

Author: Sai Mahesh Sandeboina
Affiliation: Pace University, USA
Date: October 2025

REPRODUCIBILITY NOTE:
---------------------
This script reads from saved CSV files (headlines_scored.csv, prices.csv)
to exactly reproduce all figures and statistics reported in the paper.
Do NOT re-scrape live data — results will differ due to news feed changes.

Usage:
    python marketsense_pipeline.py

Requirements:
    pip install pandas numpy scipy matplotlib seaborn transformers torch
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

TICKERS = ["AAPL", "AMZN", "GS", "JPM", "MS", "MSFT", "NVDA", "TSLA", "WMT"]

DATA_DIR = "data"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

FINBERT_COL = "finbert_sent"
DISTIL_COL  = "distil_sent"
RETURN_COL  = "return_open_close"
DATE_COL    = "date"
TICKER_COL  = "ticker"

PLOT_STYLE = {
    "finbert_color":  "#E07B00",   # orange
    "distil_color":   "#2166AC",   # blue
    "scatter_alpha":  0.55,
    "scatter_size":   35,
    "fig_dpi":        150,
}

# ── Load Data ─────────────────────────────────────────────────────────────────

def load_data():
    """Load saved CSV files. Reproduces exact paper results."""
    headlines = pd.read_csv(os.path.join(DATA_DIR, "headlines_scored.csv"), parse_dates=[DATE_COL])
    prices    = pd.read_csv(os.path.join(DATA_DIR, "prices.csv"),           parse_dates=[DATE_COL])

    # Compute open→close return if not already present
    if RETURN_COL not in prices.columns:
        prices[RETURN_COL] = (prices["Close"] - prices["Open"]) / prices["Open"]

    # Daily sentiment: average across headlines per (date, ticker)
    daily = (
        headlines
        .groupby([DATE_COL, TICKER_COL])[[FINBERT_COL, DISTIL_COL]]
        .mean()
        .reset_index()
    )

    # Inner join sentiment ↔ returns
    joined = daily.merge(
        prices[[DATE_COL, TICKER_COL, RETURN_COL]],
        on=[DATE_COL, TICKER_COL],
        how="inner"
    ).dropna(subset=[FINBERT_COL, DISTIL_COL, RETURN_COL])

    print(f"✓ Loaded {len(joined)} aligned (date, ticker) observations across {joined[TICKER_COL].nunique()} tickers.")
    return joined


# ── Statistics ────────────────────────────────────────────────────────────────

def pearson_with_ci(x, y, n_boot=2000, ci=95, seed=42):
    """Pearson r with bootstrapped confidence interval."""
    r, p = stats.pearsonr(x, y)
    rng = np.random.default_rng(seed)
    boot_r = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if xb.std() == 0 or yb.std() == 0:
            continue
        boot_r.append(stats.pearsonr(xb, yb)[0])
    lo = np.percentile(boot_r, (100 - ci) / 2)
    hi = np.percentile(boot_r, 100 - (100 - ci) / 2)
    return r, p, lo, hi


def compute_correlations(df):
    """Overall and per-ticker Pearson correlations for both models."""
    results = []

    # Overall pooled
    for model, col in [("FinBERT", FINBERT_COL), ("DistilBERT", DISTIL_COL)]:
        x = df[col].values
        y = df[RETURN_COL].values
        r, p, lo, hi = pearson_with_ci(x, y)
        results.append({
            "ticker": "ALL", "model": model,
            "r": r, "p": p, "ci_lo": lo, "ci_hi": hi, "n": len(x)
        })

    # Per-ticker
    for ticker in TICKERS:
        sub = df[df[TICKER_COL] == ticker]
        if len(sub) < 10:
            continue
        for model, col in [("FinBERT", FINBERT_COL), ("DistilBERT", DISTIL_COL)]:
            x = sub[col].values
            y = sub[RETURN_COL].values
            r, p, lo, hi = pearson_with_ci(x, y)
            results.append({
                "ticker": ticker, "model": model,
                "r": r, "p": p, "ci_lo": lo, "ci_hi": hi, "n": len(x)
            })

    return pd.DataFrame(results)


# ── Figure 1 & 2: Pooled Scatter ──────────────────────────────────────────────

def plot_pooled_scatter(df, corr_df):
    """Reproduce Figure 1 (FinBERT) and Figure 2 (DistilBERT) from paper."""
    for model, col, color, fig_num in [
        ("FinBERT",    FINBERT_COL, PLOT_STYLE["finbert_color"], 1),
        ("DistilBERT", DISTIL_COL,  PLOT_STYLE["distil_color"],  2),
    ]:
        row = corr_df[(corr_df["ticker"] == "ALL") & (corr_df["model"] == model)].iloc[0]
        r, p, n = row["r"], row["p"], int(row["n"])

        fig, ax = plt.subplots(figsize=(7, 5), dpi=PLOT_STYLE["fig_dpi"])
        x = df[col].values
        y = df[RETURN_COL].values

        ax.scatter(x, y,
                   alpha=PLOT_STYLE["scatter_alpha"],
                   s=PLOT_STYLE["scatter_size"],
                   color="#5BA4CF", edgecolors="none")

        # Regression line
        m, b = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 200)
        ax.plot(xline, m * xline + b, color=color, linewidth=2)

        ax.set_xlabel(f"{col} sentiment", fontsize=11)
        ax.set_ylabel("Same-day return (Open→Close)", fontsize=11)
        ax.set_title(
            f"Overall {model} Sentiment vs Same-Day Return\n"
            f"r={r:.3f}, p={p:.4f}, n={n}",
            fontsize=12
        )
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.6, linestyle="--")
        sns.despine(ax=ax)
        fig.tight_layout()
        path = os.path.join(FIGURES_DIR, f"fig{fig_num}_{model.lower()}_pooled_scatter.png")
        fig.savefig(path, dpi=PLOT_STYLE["fig_dpi"])
        plt.close()
        print(f"✓ Saved {path}  (r={r:.3f}, p={p:.4f}, n={n})")


# ── Figure 3: Per-Ticker Bar Chart ────────────────────────────────────────────

def plot_per_ticker_bar(corr_df):
    """Reproduce Figure 3: per-ticker Pearson r for both models."""
    sub = corr_df[corr_df["ticker"] != "ALL"].copy()
    pivot = sub.pivot(index="ticker", columns="model", values="r").reindex(TICKERS)

    x = np.arange(len(TICKERS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), dpi=PLOT_STYLE["fig_dpi"])
    ax.bar(x - width/2, pivot["DistilBERT"], width,
           label="DistilBERT", color=PLOT_STYLE["distil_color"], alpha=0.85)
    ax.bar(x + width/2, pivot["FinBERT"],    width,
           label="FinBERT",    color=PLOT_STYLE["finbert_color"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(TICKERS, fontsize=10)
    ax.set_ylabel("Pearson r", fontsize=11)
    ax.set_xlabel("Ticker", fontsize=11)
    ax.set_title("Per-Ticker Pearson Correlation (Sentiment vs Same-Day Return)", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_per_ticker_correlation.png")
    fig.savefig(path, dpi=PLOT_STYLE["fig_dpi"])
    plt.close()
    print(f"✓ Saved {path}")


# ── Figure 4: Rolling Sentiment ───────────────────────────────────────────────

def plot_rolling_sentiment(df):
    """Reproduce Figure 4: 3-day rolling sentiment by ticker (FinBERT)."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), dpi=PLOT_STYLE["fig_dpi"])
    axes = axes.flatten()

    for i, ticker in enumerate(TICKERS):
        sub = df[df[TICKER_COL] == ticker].sort_values(DATE_COL).copy()
        sub["finbert_roll"] = sub[FINBERT_COL].rolling(3, min_periods=1).mean()
        sub["distil_roll"]  = sub[DISTIL_COL].rolling(3, min_periods=1).mean()

        ax = axes[i]
        ax.plot(sub[DATE_COL], sub["finbert_roll"],
                color=PLOT_STYLE["finbert_color"], label="FinBERT (3d avg)", linewidth=1.5)
        ax.plot(sub[DATE_COL], sub["distil_roll"],
                color=PLOT_STYLE["distil_color"],  label="DistilBERT (3d avg)",
                linewidth=1.2, linestyle="--")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(f"3-Day Rolling Sentiment — {ticker}", fontsize=9)
        ax.tick_params(axis="x", labelsize=6, rotation=30)
        ax.tick_params(axis="y", labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle("FinBERT vs DistilBERT: 3-Day Rolling Sentiment by Ticker", fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_rolling_sentiment.png")
    fig.savefig(path, dpi=PLOT_STYLE["fig_dpi"])
    plt.close()
    print(f"✓ Saved {path}")


# ── Figure 5 & 6: Per-Ticker Scatter (AAPL, MSFT) ────────────────────────────

def plot_per_ticker_scatter(df, corr_df, ticker, fig_num):
    """Reproduce Figure 5 (AAPL) and Figure 6 (MSFT) per-ticker scatterplots."""
    sub = df[df[TICKER_COL] == ticker]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=PLOT_STYLE["fig_dpi"])

    for ax, (model, col, color) in zip(axes, [
        ("FinBERT",    FINBERT_COL, PLOT_STYLE["finbert_color"]),
        ("DistilBERT", DISTIL_COL,  PLOT_STYLE["distil_color"]),
    ]):
        row = corr_df[(corr_df["ticker"] == ticker) & (corr_df["model"] == model)].iloc[0]
        r, p, n = row["r"], row["p"], int(row["n"])

        x = sub[col].values
        y = sub[RETURN_COL].values

        ax.scatter(x, y,
                   alpha=PLOT_STYLE["scatter_alpha"],
                   s=PLOT_STYLE["scatter_size"],
                   color="#5BA4CF", edgecolors="none")
        m, b = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 200)
        ax.plot(xline, m * xline + b, color=color, linewidth=2)

        ax.set_xlabel("Daily sentiment", fontsize=10)
        ax.set_ylabel("Open→Close return", fontsize=10)
        ax.set_title(f"{ticker} — {model} Sentiment vs Return\nr={r:.3f}, p={p:.4f}, n={n}", fontsize=10)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        sns.despine(ax=ax)

    fig.suptitle(f"{ticker}: FinBERT (left) vs DistilBERT (right) sentiment vs return", fontsize=11)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, f"fig{fig_num}_{ticker.lower()}_scatter.png")
    fig.savefig(path, dpi=PLOT_STYLE["fig_dpi"])
    plt.close()
    print(f"✓ Saved {path}")


# ── Summary Table ─────────────────────────────────────────────────────────────

def print_summary_table(corr_df):
    """Print correlation summary matching paper Table."""
    print("\n" + "="*65)
    print("CORRELATION SUMMARY — MarketSense Paper Results")
    print("="*65)
    overall = corr_df[corr_df["ticker"] == "ALL"]
    for _, row in overall.iterrows():
        print(f"  {row['model']:12s}  r={row['r']:+.3f}  p={row['p']:.4f}  "
              f"95% CI=[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]  n={int(row['n'])}")
    print("-"*65)
    print(f"{'Ticker':<8}", end="")
    for model in ["FinBERT", "DistilBERT"]:
        print(f"  {model+' r':>12}  {model+' p':>12}", end="")
    print()
    for ticker in TICKERS:
        sub = corr_df[corr_df["ticker"] == ticker]
        if sub.empty:
            continue
        print(f"{ticker:<8}", end="")
        for model in ["FinBERT", "DistilBERT"]:
            row = sub[sub["model"] == model]
            if row.empty:
                print(f"  {'N/A':>12}  {'N/A':>12}", end="")
            else:
                r = row.iloc[0]["r"]
                p = row.iloc[0]["p"]
                print(f"  {r:>+12.3f}  {p:>12.4f}", end="")
        print()
    print("="*65 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n🔬 MarketSense Reproducibility Pipeline")
    print("   Reading from saved CSVs — results match paper exactly.\n")

    df      = load_data()
    corr_df = compute_correlations(df)

    print_summary_table(corr_df)

    print("Generating figures...")
    plot_pooled_scatter(df, corr_df)
    plot_per_ticker_bar(corr_df)
    plot_rolling_sentiment(df)
    plot_per_ticker_scatter(df, corr_df, "AAPL", fig_num=5)
    plot_per_ticker_scatter(df, corr_df, "MSFT", fig_num=6)

    print(f"\n✅ All figures saved to ./{FIGURES_DIR}/")
    print("   Push this repo to GitHub with your CSVs in ./data/ to enable full reproducibility.\n")


if __name__ == "__main__":
    main()
