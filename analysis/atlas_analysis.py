"""Calibration analysis primitives for llm-tokens-atlas.

Single module that contains every computation and plotting routine used by
`analysis/notebooks/calibration.ipynb`. Keeping the logic in a `.py` makes
it diffable in git, importable from tests, and reproducible from the CLI
(`python analysis/atlas_analysis.py --parquet ... --out-dir analysis/`).

Design choices:

- **Deterministic.** ``numpy.random.seed(0)`` and ``matplotlib`` rc set in
  :func:`set_publication_style` so every figure rebuilds byte-identical.
- **Color-blind safe.** Provider colors are pulled from Wong (2011) and pass
  the Color Universal Design (CUD) test for protanopia, deuteranopia, and
  tritanopia.
- **Single source of truth.** Numeric findings funnel through
  :func:`summarize_results` which writes one dict + one JSON; the markdown
  tables and the figures both consume that dict so they cannot drift.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

# Provider display order — keeps figures consistent across the paper.
PROVIDER_ORDER: tuple[str, ...] = ("openai", "anthropic", "google", "mistral", "cohere")

# Format display order — short → long, matching how they appear in the paper.
FORMAT_ORDER: tuple[str, ...] = ("plain", "markdown", "json", "xml", "yaml")

# Domain display order — content-type clusters first, "other" last.
DOMAIN_ORDER: tuple[str, ...] = (
    "code",
    "prose",
    "chat",
    "structured",
    "multilingual",
    "other",
)

# Wong 2011 CUD palette (5 colors, distinguishable under common color-vision
# deficiencies). Provider order mirrors PROVIDER_ORDER so the legend is
# stable across all figures.
PROVIDER_COLORS: dict[str, str] = {
    "openai": "#0072B2",      # blue
    "anthropic": "#D55E00",   # vermillion
    "google": "#009E73",      # bluish-green
    "mistral": "#CC79A7",     # reddish-purple
    "cohere": "#F0E442",      # yellow
}


# --------------------------------------------------------------------------- #
# Style                                                                       #
# --------------------------------------------------------------------------- #


def set_publication_style() -> None:
    """Set matplotlib rcParams for publication-grade output.

    Called once at notebook startup. Idempotent.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 11,
            "font.family": "sans-serif",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.constrained_layout.use": True,
        }
    )
    np.random.seed(0)


# --------------------------------------------------------------------------- #
# Loading + sanity                                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SanityReport:
    """Summary of basic dataset health metrics."""

    n_rows: int
    n_nan_delta_pct: int
    n_zero_empirical: int
    providers: list[str]
    formats: list[str]
    domains: list[str]
    n_prompts: int


def load_atlas(parquet_path: Path) -> pd.DataFrame:
    """Load the processed atlas parquet.

    Categoricals get a stable display order so downstream groupby + plots
    are deterministic.
    """
    df = pd.read_parquet(parquet_path)
    df["provider"] = pd.Categorical(
        df["provider"], categories=list(PROVIDER_ORDER), ordered=True
    )
    df["format"] = pd.Categorical(
        df["format"], categories=list(FORMAT_ORDER), ordered=True
    )
    df["domain"] = pd.Categorical(
        df["domain"], categories=list(DOMAIN_ORDER), ordered=True
    )
    return df


def sanity_report(df: pd.DataFrame) -> SanityReport:
    """Compute headline sanity metrics."""
    return SanityReport(
        n_rows=len(df),
        n_nan_delta_pct=int(df["delta_pct"].isna().sum()),
        n_zero_empirical=int((df["empirical_count"] == 0).sum()),
        providers=[
            p for p in PROVIDER_ORDER if p in df["provider"].cat.categories
            and (df["provider"] == p).any()
        ],
        formats=[
            f for f in FORMAT_ORDER if f in df["format"].cat.categories
            and (df["format"] == f).any()
        ],
        domains=[
            d for d in DOMAIN_ORDER if d in df["domain"].cat.categories
            and (df["domain"] == d).any()
        ],
        n_prompts=df["prompt_id"].nunique(),
    )


# --------------------------------------------------------------------------- #
# Statistics                                                                  #
# --------------------------------------------------------------------------- #


def per_provider_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-provider delta_pct summary statistics.

    Columns: provider, n, median, p25, p75, p95, mean.
    Rows are sorted in PROVIDER_ORDER.
    """
    grouped = df.groupby("provider", observed=True)["delta_pct"]
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "mean": grouped.mean(),
            "median": grouped.median(),
            "p25": grouped.quantile(0.25),
            "p75": grouped.quantile(0.75),
            "p95": grouped.quantile(0.95),
        }
    )
    out = out.reindex([p for p in PROVIDER_ORDER if p in out.index])
    out.index.name = "provider"
    return out.reset_index()


def per_provider_format_median(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the median delta_pct for every (format, provider) cell.

    Returns a wide DataFrame with format as the row index (in FORMAT_ORDER)
    and provider as the columns (in PROVIDER_ORDER).
    """
    pivot = (
        df.groupby(["format", "provider"], observed=True)["delta_pct"]
        .median()
        .unstack("provider")
    )
    row_order = [f for f in FORMAT_ORDER if f in pivot.index]
    col_order = [p for p in PROVIDER_ORDER if p in pivot.columns]
    return pivot.loc[row_order, col_order]


def per_domain_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-domain delta_pct mean/std/n (any provider mixed in)."""
    grouped = df.groupby("domain", observed=True)["delta_pct"]
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "mean": grouped.mean(),
            "std": grouped.std(),
            "median": grouped.median(),
        }
    )
    out = out.reindex([d for d in DOMAIN_ORDER if d in out.index])
    out.index.name = "domain"
    return out.reset_index()


def per_domain_provider_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per (domain, provider) mean delta_pct + SEM for error-bar plots."""
    grouped = df.groupby(["domain", "provider"], observed=True)["delta_pct"]
    summary = pd.DataFrame(
        {
            "n": grouped.size(),
            "mean": grouped.mean(),
            "std": grouped.std(),
        }
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["n"].clip(lower=1))
    return summary.reset_index()


def per_provider_bias_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Per-provider rates of underestimate / exact / overestimate.

    Returns a wide DataFrame with provider as rows and direction as columns,
    values in [0,1]. Provider order is preserved.
    """
    counts = (
        df.groupby(["provider", "direction"], observed=True).size().unstack("direction").fillna(0)
    )
    # Ensure all three columns exist in a stable order even if a provider
    # never disagrees in some direction.
    for col in ("underestimate", "exact", "overestimate"):
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["underestimate", "exact", "overestimate"]]
    rates = counts.div(counts.sum(axis=1), axis=0)
    return rates.reindex([p for p in PROVIDER_ORDER if p in rates.index])


@dataclass(frozen=True)
class CalibrationFit:
    """Per-provider linear fit of empirical_count = a * offline_count + b."""

    provider: str
    n: int
    slope: float
    intercept: float
    r_squared: float


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Closed-form linear regression returning (slope, intercept, r2).

    Uses normal equations with a manual R² computation so we don't pull in
    scipy for one fit. ``slope`` is the per-provider calibration multiplier
    a practitioner could apply to the offline tokenizer's output.
    """
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    dx = x - x_mean
    dy = y - y_mean
    denom = float(np.dot(dx, dx))
    if denom == 0.0:
        return float("nan"), float("nan"), float("nan")
    slope = float(np.dot(dx, dy) / denom)
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum(dy * dy))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot else float("nan")
    return slope, intercept, r2


def per_provider_calibration_fits(df: pd.DataFrame) -> list[CalibrationFit]:
    """Fit a linear model per provider; offline → empirical."""
    fits: list[CalibrationFit] = []
    for provider in PROVIDER_ORDER:
        sub = df[df["provider"] == provider]
        if sub.empty:
            continue
        x = sub["offline_count"].to_numpy(dtype=float)
        y = sub["empirical_count"].to_numpy(dtype=float)
        slope, intercept, r2 = linear_fit(x, y)
        fits.append(
            CalibrationFit(
                provider=provider,
                n=int(len(sub)),
                slope=slope,
                intercept=intercept,
                r_squared=r2,
            )
        )
    return fits


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #


def _provider_palette(providers: list[str]) -> list[str]:
    return [PROVIDER_COLORS[p] for p in providers]


def plot_violin_delta(df: pd.DataFrame, out_stem: Path) -> Path:
    """Violin plot of delta_pct per provider; saves .png + .pdf.

    Clips the y-axis to [-50, 200] for legibility — outliers from
    pathologically short prompts otherwise dominate the visual scale.
    """
    providers = [p for p in PROVIDER_ORDER if (df["provider"] == p).any()]
    data = [df.loc[df["provider"] == p, "delta_pct"].dropna().to_numpy() for p in providers]

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(
        data,
        positions=np.arange(len(providers)),
        widths=0.7,
        showmedians=True,
        showextrema=False,
    )
    palette = _provider_palette(providers)
    # parts["bodies"] is a list of PolyCollections at runtime, but mypy
    # only sees Collection. Cast through Any to keep the iteration typable.
    bodies: list[Any] = list(parts["bodies"])  # type: ignore[call-overload]
    for body, color in zip(bodies, palette, strict=True):
        body.set_facecolor(color)
        body.set_alpha(0.6)
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.2)

    ax.set_xticks(np.arange(len(providers)))
    ax.set_xticklabels(providers)
    ax.set_ylabel(r"$\Delta\%$ = (empirical $-$ offline) / empirical $\times$ 100")
    ax.set_xlabel("Provider")
    ax.set_title("Per-provider offline vs empirical token-count drift")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_ylim(-50, 200)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    return _save(fig, out_stem)


def plot_format_heatmap(df: pd.DataFrame, out_stem: Path) -> Path:
    """Heatmap of median delta_pct over (format × provider)."""
    pivot = per_provider_format_median(df)
    providers = list(pivot.columns)
    formats = list(pivot.index)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = LinearSegmentedColormap.from_list(
        "calibration_div", ["#2166AC", "#F7F7F7", "#B2182B"], N=256
    )
    vmax = float(np.nanmax(np.abs(pivot.to_numpy())))
    vmax = max(vmax, 5.0)  # don't squash near-zero matrices
    im = ax.imshow(pivot.to_numpy(), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(providers)))
    ax.set_xticklabels(providers, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(formats)))
    ax.set_yticklabels(formats)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            color = "white" if abs(val) > 0.6 * vmax else "black"
            ax.text(
                j,
                i,
                f"{val:+.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Median $\Delta\%$")
    ax.set_xlabel("Provider")
    ax.set_ylabel("Format")
    ax.set_title("Median calibration drift by format × provider")
    ax.grid(False)

    return _save(fig, out_stem)


def plot_domain_effect(df: pd.DataFrame, out_stem: Path) -> Path:
    """Grouped bar chart of mean delta_pct by (domain, provider) with SEM bars."""
    summary = per_domain_provider_stats(df)
    domains = [d for d in DOMAIN_ORDER if d in summary["domain"].unique()]
    providers = [p for p in PROVIDER_ORDER if p in summary["provider"].unique()]
    palette = _provider_palette(providers)

    fig, ax = plt.subplots(figsize=(10, 5))
    n_providers = len(providers)
    width = 0.8 / max(n_providers, 1)
    positions = np.arange(len(domains))

    for i, provider in enumerate(providers):
        sub = summary[summary["provider"] == provider].set_index("domain").reindex(domains)
        means = sub["mean"].to_numpy()
        sems = sub["sem"].to_numpy()
        offset = (i - (n_providers - 1) / 2) * width
        ax.bar(
            positions + offset,
            np.nan_to_num(means, nan=0.0),
            width=width,
            color=palette[i],
            edgecolor="black",
            linewidth=0.4,
            label=provider,
            yerr=np.nan_to_num(sems, nan=0.0),
            capsize=2,
            error_kw={"elinewidth": 0.6, "alpha": 0.8},
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(domains)
    ax.set_xlabel("Prompt domain")
    ax.set_ylabel(r"Mean $\Delta\%$ (error bars: SEM)")
    ax.set_title("Calibration drift by prompt domain")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.legend(title="Provider", ncols=min(n_providers, 5), loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    return _save(fig, out_stem)


def plot_bias_direction(df: pd.DataFrame, out_stem: Path) -> Path:
    """Stacked horizontal bar of bias direction per provider."""
    rates = per_provider_bias_direction(df)
    providers = list(rates.index)
    directions = ["underestimate", "exact", "overestimate"]
    # Three colors that are CUD-safe + semantically meaningful: red for
    # underestimate (i.e. you'll be billed more than you thought), grey for
    # exact, blue for overestimate.
    direction_colors = {
        "underestimate": "#D55E00",
        "exact": "#999999",
        "overestimate": "#0072B2",
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(providers))
    left = np.zeros(len(providers))
    for direction in directions:
        widths = rates[direction].to_numpy() * 100.0
        ax.barh(
            y,
            widths,
            left=left,
            color=direction_colors[direction],
            edgecolor="black",
            linewidth=0.4,
            label=direction,
        )
        # In-bar percentage label when the slice is wide enough to fit.
        for i, w in enumerate(widths):
            if w >= 6:
                ax.text(
                    left[i] + w / 2,
                    y[i],
                    f"{w:.0f}%",
                    ha="center",
                    va="center",
                    color="white" if direction != "exact" else "black",
                    fontsize=9,
                )
        left = left + widths

    ax.set_yticks(y)
    ax.set_yticklabels(providers)
    ax.invert_yaxis()  # first provider at top
    ax.set_xlabel("Share of rows (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Direction of offline-tokenizer bias by provider")
    ax.legend(title="Direction (offline vs. empirical)", ncols=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    return _save(fig, out_stem)


def plot_calibration_regression(df: pd.DataFrame, out_stem: Path) -> Path:
    """Per-provider scatter + regression line of empirical vs offline counts."""
    fits = per_provider_calibration_fits(df)
    providers = [f.provider for f in fits]
    n = len(providers)
    cols = min(3, n) if n else 1
    rows = int(np.ceil(n / cols)) if n else 1

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows), squeeze=False)
    axes_flat = axes.flatten()
    for ax in axes_flat:
        ax.axis("off")

    rng = np.random.default_rng(0)

    for ax, fit in zip(axes_flat[: len(fits)], fits, strict=True):
        ax.axis("on")
        sub = df[df["provider"] == fit.provider]
        x = sub["offline_count"].to_numpy(dtype=float)
        y = sub["empirical_count"].to_numpy(dtype=float)
        # Subsample to keep the rendered file small + scannable.
        if len(x) > 1500:
            idx = rng.choice(len(x), size=1500, replace=False)
            x_plot, y_plot = x[idx], y[idx]
        else:
            x_plot, y_plot = x, y

        ax.scatter(
            x_plot,
            y_plot,
            s=6,
            color=PROVIDER_COLORS[fit.provider],
            alpha=0.35,
            edgecolors="none",
        )
        if x.size:
            xs = np.linspace(float(x.min()), float(x.max()), 200)
            ax.plot(
                xs,
                fit.slope * xs + fit.intercept,
                color="black",
                linewidth=1.4,
                label=(
                    f"slope={fit.slope:.3f}\n"
                    f"intercept={fit.intercept:.1f}\n"
                    f"$R^2$={fit.r_squared:.3f}"
                ),
            )
            ax.plot(
                xs,
                xs,
                color="grey",
                linewidth=0.8,
                linestyle=":",
                label="y = x",
            )
        ax.set_title(f"{fit.provider} (n={fit.n})")
        ax.set_xlabel("Offline token count")
        ax.set_ylabel("Empirical token count")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Per-provider calibration regressions  "
        "(empirical $\\approx$ slope $\\cdot$ offline + intercept)",
        fontsize=12,
    )
    return _save(fig, out_stem)


def _save(fig: plt.Figure, out_stem: Path) -> Path:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


# --------------------------------------------------------------------------- #
# Tables (markdown)                                                           #
# --------------------------------------------------------------------------- #


def render_provider_stats_md(df: pd.DataFrame) -> str:
    stats = per_provider_stats(df)
    rows = ["| provider | n | median | p25 | p75 | p95 | mean |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in stats.iterrows():
        rows.append(
            "| {p} | {n} | {med:+.2f} | {p25:+.2f} | {p75:+.2f} | "
            "{p95:+.2f} | {mean:+.2f} |".format(
                p=r["provider"],
                n=int(r["n"]),
                med=r["median"],
                p25=r["p25"],
                p75=r["p75"],
                p95=r["p95"],
                mean=r["mean"],
            )
        )
    return "\n".join(rows) + "\n"


def render_format_heatmap_md(df: pd.DataFrame) -> str:
    pivot = per_provider_format_median(df)
    providers = list(pivot.columns)
    header = "| format | " + " | ".join(providers) + " |"
    align = "|---|" + "---:|" * len(providers)
    lines = [header, align]
    for fmt in pivot.index:
        values = " | ".join(f"{pivot.at[fmt, p]:+.2f}" for p in providers)
        lines.append(f"| {fmt} | {values} |")
    return "\n".join(lines) + "\n"


def render_calibration_factors_md(df: pd.DataFrame) -> str:
    fits = per_provider_calibration_fits(df)
    rows = [
        "| provider | n | slope | intercept | R² |",
        "|---|---:|---:|---:|---:|",
    ]
    for f in fits:
        rows.append(
            f"| {f.provider} | {f.n} | {f.slope:.4f} | {f.intercept:+.2f} "
            f"| {f.r_squared:.4f} |"
        )
    return "\n".join(rows) + "\n"


# --------------------------------------------------------------------------- #
# Results JSON                                                                #
# --------------------------------------------------------------------------- #


def summarize_results(df: pd.DataFrame) -> dict[str, Any]:
    """Build the canonical machine-readable findings payload.

    All numeric fields are JSON-friendly Python floats so downstream paper
    agents can ``json.load(...)`` and inline values into LaTeX without
    additional conversion.
    """
    rep = sanity_report(df)
    stats = per_provider_stats(df)
    fmt_pivot = per_provider_format_median(df)
    domain_stats = per_domain_stats(df)
    domain_provider_stats = per_domain_provider_stats(df)
    bias = per_provider_bias_direction(df)
    fits = per_provider_calibration_fits(df)

    payload: dict[str, Any] = OrderedDict()
    payload["dataset"] = OrderedDict(
        n_rows=rep.n_rows,
        n_prompts=rep.n_prompts,
        providers=rep.providers,
        formats=rep.formats,
        domains=rep.domains,
        n_nan_delta_pct=rep.n_nan_delta_pct,
        n_zero_empirical=rep.n_zero_empirical,
    )

    payload["per_provider"] = OrderedDict()
    for _, row in stats.iterrows():
        payload["per_provider"][row["provider"]] = OrderedDict(
            n=int(row["n"]),
            median_delta_pct=float(row["median"]),
            p25_delta_pct=float(row["p25"]),
            p75_delta_pct=float(row["p75"]),
            p95_delta_pct=float(row["p95"]),
            mean_delta_pct=float(row["mean"]),
        )

    payload["format_x_provider_median_delta_pct"] = {
        fmt: {p: float(fmt_pivot.at[fmt, p]) for p in fmt_pivot.columns}
        for fmt in fmt_pivot.index
    }

    payload["per_domain"] = OrderedDict()
    for _, row in domain_stats.iterrows():
        payload["per_domain"][row["domain"]] = OrderedDict(
            n=int(row["n"]),
            mean_delta_pct=float(row["mean"]),
            std_delta_pct=(
                float(row["std"]) if not pd.isna(row["std"]) else None
            ),
            median_delta_pct=float(row["median"]),
        )

    payload["per_domain_provider"] = OrderedDict()
    for _, row in domain_provider_stats.iterrows():
        key = f"{row['domain']}::{row['provider']}"
        payload["per_domain_provider"][key] = OrderedDict(
            n=int(row["n"]),
            mean_delta_pct=float(row["mean"]),
            sem_delta_pct=(
                float(row["sem"]) if not pd.isna(row["sem"]) else None
            ),
        )

    payload["bias_direction_share"] = {
        provider: {
            direction: float(bias.at[provider, direction])
            for direction in ("underestimate", "exact", "overestimate")
        }
        for provider in bias.index
    }

    payload["calibration_fits"] = OrderedDict()
    for f in fits:
        payload["calibration_fits"][f.provider] = OrderedDict(
            n=f.n,
            slope=float(f.slope),
            intercept=float(f.intercept),
            r_squared=float(f.r_squared),
        )

    return payload


def write_results_json(df: pd.DataFrame, out_path: Path) -> Path:
    payload = summarize_results(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")
    return out_path


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def build_all_artifacts(
    parquet_path: Path,
    figures_dir: Path,
    tables_dir: Path,
    results_path: Path,
) -> dict[str, Path]:
    """Top-level orchestrator: load data, write every artifact, return paths."""
    set_publication_style()
    df = load_atlas(parquet_path)

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    outputs["fig01"] = plot_violin_delta(df, figures_dir / "fig01_violin_delta_per_provider")
    outputs["fig02"] = plot_format_heatmap(df, figures_dir / "fig02_heatmap_format_x_provider")
    outputs["fig03"] = plot_domain_effect(df, figures_dir / "fig03_domain_effect")
    outputs["fig04"] = plot_bias_direction(df, figures_dir / "fig04_bias_direction")
    outputs["fig05"] = plot_calibration_regression(df, figures_dir / "fig05_calibration_regression")

    (tables_dir / "tab01_summary_stats.md").write_text(
        render_provider_stats_md(df), encoding="utf-8"
    )
    (tables_dir / "tab02_format_heatmap.md").write_text(
        render_format_heatmap_md(df), encoding="utf-8"
    )
    (tables_dir / "tab03_calibration_factors.md").write_text(
        render_calibration_factors_md(df), encoding="utf-8"
    )
    outputs["tab01"] = tables_dir / "tab01_summary_stats.md"
    outputs["tab02"] = tables_dir / "tab02_format_heatmap.md"
    outputs["tab03"] = tables_dir / "tab03_calibration_factors.md"

    outputs["results"] = write_results_json(df, results_path)
    return outputs


def _argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build all atlas-analysis artifacts (figures + tables + results.json)."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/processed/atlas.parquet"),
        help="Processed parquet path (default: data/processed/atlas.parquet)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("analysis/figures"),
        help="Output directory for figure PNG/PDF (default: analysis/figures)",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("analysis/tables"),
        help="Output directory for markdown tables (default: analysis/tables)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("analysis/results.json"),
        help="Output path for results.json (default: analysis/results.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argparser().parse_args(argv)
    outputs = build_all_artifacts(
        parquet_path=args.parquet,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
        results_path=args.results,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
