"""Build analysis/notebooks/calibration.ipynb from a paired .py.

This file is intentionally small: it owns the notebook *structure*
(markdown + code cells) and delegates all heavy lifting to
``analysis/atlas_analysis.py``. Re-running this script regenerates the
notebook deterministically. Cells are then executed via
``jupyter nbconvert --execute --inplace``.

Why this pattern (instead of editing an .ipynb directly):
  - the .py file diffs cleanly in code review
  - cell ordering is explicit and trivially editable
  - reproducible builds: nbconvert produces the same outputs every time
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "calibration.ipynb"


def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def _code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        },
        "language_info": {"name": "python"},
    }

    cells: list[nbf.NotebookNode] = []

    cells.append(_md(
        "# llm-tokens-atlas — calibration analysis\n"
        "\n"
        "Open notebook for the paper *On the Calibration of Offline LLM "
        "Tokenizers: A 5-Provider Empirical Study*. Every figure and table "
        "in the paper is regenerated from this notebook.\n"
        "\n"
        "All computations and plotting routines live in "
        "`analysis/atlas_analysis.py`; this notebook is a thin orchestrator "
        "so the logic stays diffable in git and importable from tests.\n"
        "\n"
        "Inputs: `data/processed/atlas.parquet` (built by `make build`).\n"
        "\n"
        "Outputs: `analysis/figures/*.png|*.pdf`, `analysis/tables/*.md`, "
        "`analysis/results.json`."
    ))

    cells.append(_code(
        "import sys  # noqa: E402\n"
        "from pathlib import Path  # noqa: E402\n"
        "\n"
        "# Make analysis/ importable when the notebook is opened directly.\n"
        "NB_DIR = Path.cwd()\n"
        "ANALYSIS_DIR = (\n"
        "    NB_DIR if (NB_DIR / 'atlas_analysis.py').exists() else NB_DIR.parent\n"
        ")\n"
        "REPO_ROOT = ANALYSIS_DIR.parent\n"
        "sys.path.insert(0, str(ANALYSIS_DIR))\n"
        "\n"
        "import atlas_analysis as aa  # noqa: E402\n"
        "import pandas as pd  # noqa: E402,F401\n"
        "from IPython.display import Image  # noqa: E402\n"
        "\n"
        "aa.set_publication_style()\n"
        "\n"
        "PARQUET_PATH = REPO_ROOT / 'data' / 'processed' / 'atlas.parquet'\n"
        "FIGURES_DIR = REPO_ROOT / 'analysis' / 'figures'\n"
        "TABLES_DIR = REPO_ROOT / 'analysis' / 'tables'\n"
        "RESULTS_PATH = REPO_ROOT / 'analysis' / 'results.json'\n"
        "\n"
        "FIGURES_DIR.mkdir(parents=True, exist_ok=True)\n"
        "TABLES_DIR.mkdir(parents=True, exist_ok=True)\n"
        "# Redact absolute paths from output to keep the notebook portable.\n"
        "print('repo root:', REPO_ROOT.name)\n"
        "print('parquet path:', PARQUET_PATH.relative_to(REPO_ROOT))\n"
        "print('parquet exists:', PARQUET_PATH.exists())"
    ))

    cells.append(_md(
        "## 1. Load + sanity\n"
        "\n"
        "Load the joined parquet and confirm the schema is as expected. We "
        "flag any rows with `delta_pct == NaN` (the edge case is "
        "`empirical_count == 0`, which is unusual but legal — short prompts "
        "in extreme tokenizers can collapse to 0)."
    ))

    cells.append(_code(
        "df = aa.load_atlas(PARQUET_PATH)\n"
        "print('rows:', len(df))\n"
        "print('schema:')\n"
        "print(df.dtypes)\n"
        "df.head(3)"
    ))

    cells.append(_code(
        "report = aa.sanity_report(df)\n"
        "print('n_rows                ', report.n_rows)\n"
        "print('n_prompts             ', report.n_prompts)\n"
        "print('providers (present)   ', report.providers)\n"
        "print('formats (present)     ', report.formats)\n"
        "print('domains (present)     ', report.domains)\n"
        "print('n_nan_delta_pct       ', report.n_nan_delta_pct)\n"
        "print('n_zero_empirical      ', report.n_zero_empirical)"
    ))

    cells.append(_code(
        "# Summary statistics across the entire frame.\n"
        "df[['offline_count', 'empirical_count', 'delta', 'delta_pct']].describe()"
    ))

    cells.append(_md(
        "**Note on missing providers.** The atlas measures five providers, "
        "but rows for Anthropic / Google / Cohere only exist when the "
        "corresponding `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` / "
        "`COHERE_API_KEY` was set during `make empirical`. OpenAI and "
        "Mistral are always present because their tokenizers ship as Python "
        "libraries (tiktoken, mistral-common). Providers absent from the "
        "parquet are silently dropped by every routine below — `n` per "
        "provider in the tables shows which ran."
    ))

    cells.append(_md(
        "## 2. Per-provider calibration distribution\n"
        "\n"
        "Violin plot of `delta_pct = (empirical - offline) / empirical * "
        "100` per provider, accompanied by a summary table with median, "
        "p25, p75, and p95.  For Anthropic specifically, we expect to see "
        "the cl100k_base / `claude-opus-4-7` underestimate from the "
        "tokenometer foundation work, which two 2026 blog posts have "
        "informally measured at +40 to +47%. The violin shape is the "
        "paper's headline figure."
    ))

    cells.append(_code(
        "stats = aa.per_provider_stats(df)\n"
        "stats"
    ))

    cells.append(_code(
        "fig01 = aa.plot_violin_delta(\n"
        "    df, FIGURES_DIR / 'fig01_violin_delta_per_provider'\n"
        ")\n"
        "print('saved', fig01.relative_to(REPO_ROOT))\n"
        "Image(filename=str(fig01))"
    ))

    cells.append(_md(
        "## 3. Per-format breakdown — load-bearing\n"
        "\n"
        "Prior art (the 2026 blog posts, AgentOps `tokencost`) reports "
        "per-provider drift but **does not break drift down by serialization "
        "format**. This is the centerpiece novelty of the analysis: "
        "a heatmap of median `delta_pct` over "
        "`format × provider`."
    ))

    cells.append(_code(
        "fmt_pivot = aa.per_provider_format_median(df)\n"
        "fmt_pivot"
    ))

    cells.append(_code(
        "fig02 = aa.plot_format_heatmap(\n"
        "    df, FIGURES_DIR / 'fig02_heatmap_format_x_provider'\n"
        ")\n"
        "print('saved', fig02.relative_to(REPO_ROOT))\n"
        "Image(filename=str(fig02))"
    ))

    cells.append(_md(
        "## 4. Domain effect\n"
        "\n"
        "We tagged each prompt with a domain — `code`, `prose`, `chat`, "
        "`structured`, `multilingual`, `other`. The grouped bar chart "
        "shows how mean `delta_pct` varies by domain × provider, with SEM "
        "error bars for each cell. Practitioners can read this as: *\"if "
        "my prompt is `code`, this is how much more or fewer tokens it "
        "will cost than my offline tokenizer claims.\"*"
    ))

    cells.append(_code(
        "aa.per_domain_stats(df)"
    ))

    cells.append(_code(
        "fig03 = aa.plot_domain_effect(\n"
        "    df, FIGURES_DIR / 'fig03_domain_effect'\n"
        ")\n"
        "print('saved', fig03.relative_to(REPO_ROOT))\n"
        "Image(filename=str(fig03))"
    ))

    cells.append(_md(
        "## 5. Direction of bias\n"
        "\n"
        "How often does the offline tokenizer underestimate, exactly "
        "match, or overestimate the empirical count? A stacked horizontal "
        "bar gives one row per provider with the three shares."
    ))

    cells.append(_code(
        "aa.per_provider_bias_direction(df).mul(100).round(1).rename(columns=lambda c: c + ' %')"
    ))

    cells.append(_code(
        "fig04 = aa.plot_bias_direction(\n"
        "    df, FIGURES_DIR / 'fig04_bias_direction'\n"
        ")\n"
        "print('saved', fig04.relative_to(REPO_ROOT))\n"
        "Image(filename=str(fig04))"
    ))

    cells.append(_md(
        "## 6. Calibration model — per-provider correction factors\n"
        "\n"
        "We fit a simple linear regression `empirical = slope * offline + "
        "intercept` per provider and report slope, intercept, and R². The "
        "slope is the per-provider correction factor a practitioner could "
        "apply to their existing offline-tokenizer pipeline to get a "
        "first-order calibrated estimate of true token cost. R² near 1.0 "
        "indicates the residuals are small relative to the signal — the "
        "offline tokenizer is wrong, but consistently wrong."
    ))

    cells.append(_code(
        "fits = aa.per_provider_calibration_fits(df)\n"
        "pd.DataFrame([{\n"
        "    'provider': f.provider,\n"
        "    'n': f.n,\n"
        "    'slope': round(f.slope, 4),\n"
        "    'intercept': round(f.intercept, 2),\n"
        "    'R^2': round(f.r_squared, 4),\n"
        "} for f in fits])"
    ))

    cells.append(_code(
        "fig05 = aa.plot_calibration_regression(\n"
        "    df, FIGURES_DIR / 'fig05_calibration_regression'\n"
        ")\n"
        "print('saved', fig05.relative_to(REPO_ROOT))\n"
        "Image(filename=str(fig05))"
    ))

    cells.append(_md(
        "## 7. Persist tables + machine-readable results\n"
        "\n"
        "All numerical findings are written to `analysis/results.json` "
        "for the paper-results agent to inline. Markdown tables for paper "
        "inlining are written to `analysis/tables/`."
    ))

    cells.append(_code(
        "(TABLES_DIR / 'tab01_summary_stats.md').write_text(\n"
        "    aa.render_provider_stats_md(df), encoding='utf-8'\n"
        ")\n"
        "(TABLES_DIR / 'tab02_format_heatmap.md').write_text(\n"
        "    aa.render_format_heatmap_md(df), encoding='utf-8'\n"
        ")\n"
        "(TABLES_DIR / 'tab03_calibration_factors.md').write_text(\n"
        "    aa.render_calibration_factors_md(df), encoding='utf-8'\n"
        ")\n"
        "results_path = aa.write_results_json(df, RESULTS_PATH)\n"
        "print('tables ->', TABLES_DIR.relative_to(REPO_ROOT))\n"
        "print('results.json ->', results_path.relative_to(REPO_ROOT))"
    ))

    cells.append(_code(
        "import json  # noqa: E402\n"
        "\n"
        "with results_path.open() as f:\n"
        "    summary = json.load(f)\n"
        "print('dataset:', summary['dataset'])\n"
        "print()\n"
        "for prov, vals in summary['per_provider'].items():\n"
        "    median = vals['median_delta_pct']\n"
        "    n = vals['n']\n"
        "    print(f'  {prov:>10} n={n:>6}  median delta_pct = {median:+7.2f}%')\n"
        "print()\n"
        "print('calibration_fits:')\n"
        "for prov, vals in summary['calibration_fits'].items():\n"
        "    slope = vals['slope']\n"
        "    r2 = vals['r_squared']\n"
        "    print(f'  {prov:>10} slope={slope:.4f}  R^2={r2:.4f}')"
    ))

    cells.append(_md(
        "---\n"
        "\n"
        "*End of analysis notebook. Re-run via* "
        "`jupyter nbconvert --to notebook --execute --inplace "
        "analysis/notebooks/calibration.ipynb`."
    ))

    nb["cells"] = cells
    return nb


def main() -> int:
    nb = build_notebook()
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NB_PATH.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote notebook -> {NB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
