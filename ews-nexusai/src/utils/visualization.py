"""Shared plotting helpers (bar charts with bootstrap error bars, etc.)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_metric_with_ci(
    df: pd.DataFrame,
    x: str,
    y: str,
    ci_lower_col: str,
    ci_upper_col: str,
    title: str,
    ylabel: str,
    order: list[str] | None = None,
    figsize: tuple[int, int] = (10, 6),
):
    """Bar plot of a metric with bootstrap error bars and value labels."""
    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)
    palette = sns.color_palette("tab20")[: len(df)]

    sns.barplot(data=df, x=x, y=y, order=order, palette=palette)

    means = df[y].values
    lowers = df[ci_lower_col].values
    uppers = df[ci_upper_col].values
    errors = [means - lowers, uppers - means]

    plt.errorbar(
        x=range(len(df)), y=means, yerr=errors, fmt="none", c="black", capsize=5, capthick=1.5, lw=1.5
    )

    for i, val in enumerate(means):
        plt.text(i, val + 0.02, f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

    plt.title(title, fontsize=15, fontweight="bold")
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("")
    plt.tight_layout()
    return plt.gcf()


def plot_feature_importance(df: pd.DataFrame, title: str, top_n: int = 15, figsize: tuple[int, int] = (12, 10)):
    """Horizontal bar plot of feature importances."""
    top_df = df.head(top_n).sort_values(by="Importance", ascending=True)
    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)
    colors = sns.color_palette("tab20", n_colors=len(top_df))
    plt.barh(top_df["Feature"], top_df["Importance"], color=colors)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    return plt.gcf()
