"""Tests for analysis/atlas_analysis.py.

These tests verify that the analysis primitives behave correctly on a
hand-crafted miniature dataset. They protect the load-bearing transforms
(per-provider stats, format pivot, bias direction, linear fit) from
regression and ensure the public ``build_all_artifacts`` entrypoint
returns every expected file path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the analysis package importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

import atlas_analysis as aa  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def tiny_frame() -> pd.DataFrame:
    """A handful of rows spanning 2 providers × 2 formats × 2 domains.

    Constructed so the headline statistics are predictable by hand:
      - openai: empirical == offline (slope 1.0, R^2 1.0, all 'exact')
      - anthropic: empirical = 1.5 * offline (slope 1.5, R^2 1.0,
                                              all 'underestimate')
    """
    rows: list[dict] = []
    for prompt_id, domain in [("p1", "code"), ("p2", "prose"), ("p3", "chat")]:
        for fmt in ("plain", "markdown"):
            for provider, multiplier in (("openai", 1.0), ("anthropic", 1.5)):
                offline = 100
                empirical = int(offline * multiplier)
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "source": "synthetic",
                        "text": "x" * 50,
                        "text_len_chars": 50,
                        "text_len_words": 1,
                        "language": "en",
                        "domain": domain,
                        "collected_at": "2026-01-01T00:00:00Z",
                        "provider": provider,
                        "format": fmt,
                        "model": f"{provider}-test",
                        "offline_count": offline,
                        "tokenizer_version": "test",
                        "offline_ts": "2026-01-01T00:00:00Z",
                        "empirical_count": empirical,
                        "is_oracle": True,
                        "empirical_source": "api",
                        "endpoint": "test://",
                        "empirical_ts": "2026-01-01T00:00:00Z",
                        "delta": empirical - offline,
                        "delta_pct": (
                            (empirical - offline) / empirical * 100.0
                            if empirical
                            else float("nan")
                        ),
                        "abs_delta": abs(empirical - offline),
                        "direction": (
                            "underestimate"
                            if empirical > offline
                            else "exact"
                            if empirical == offline
                            else "overestimate"
                        ),
                    }
                )
    df = pd.DataFrame(rows)
    df["provider"] = pd.Categorical(
        df["provider"], categories=list(aa.PROVIDER_ORDER), ordered=True
    )
    df["format"] = pd.Categorical(
        df["format"], categories=list(aa.FORMAT_ORDER), ordered=True
    )
    df["domain"] = pd.Categorical(
        df["domain"], categories=list(aa.DOMAIN_ORDER), ordered=True
    )
    return df


@pytest.fixture
def tiny_parquet(tmp_path: Path, tiny_frame: pd.DataFrame) -> Path:
    """Persist tiny_frame as parquet so load_atlas can roundtrip it."""
    p = tmp_path / "atlas.parquet"
    tiny_frame.to_parquet(p, engine="pyarrow", index=False)
    return p


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_set_publication_style_is_idempotent() -> None:
    """Style setter should be safe to call twice and seed numpy reproducibly."""
    aa.set_publication_style()
    a = np.random.rand(4).tolist()
    aa.set_publication_style()
    b = np.random.rand(4).tolist()
    assert a == b


def test_load_atlas_roundtrip(tiny_parquet: Path) -> None:
    df = aa.load_atlas(tiny_parquet)
    assert len(df) == 12
    assert set(df["provider"].cat.categories) == set(aa.PROVIDER_ORDER)
    assert df["provider"].cat.ordered


def test_sanity_report_counts(tiny_frame: pd.DataFrame) -> None:
    report = aa.sanity_report(tiny_frame)
    assert report.n_rows == 12
    assert report.n_prompts == 3
    assert report.providers == ["openai", "anthropic"]
    assert report.formats == ["plain", "markdown"]
    assert report.n_nan_delta_pct == 0
    assert report.n_zero_empirical == 0


def test_per_provider_stats_correctness(tiny_frame: pd.DataFrame) -> None:
    stats = aa.per_provider_stats(tiny_frame)
    by_provider = stats.set_index("provider")
    # openai = empirical == offline, so delta_pct = 0 everywhere.
    assert by_provider.loc["openai", "median"] == pytest.approx(0.0)
    # anthropic = empirical = 1.5 * offline, so delta_pct = 50/150 * 100 ≈ 33.33.
    assert by_provider.loc["anthropic", "median"] == pytest.approx(33.333333, abs=1e-4)


def test_per_provider_format_median_pivot_shape(tiny_frame: pd.DataFrame) -> None:
    pivot = aa.per_provider_format_median(tiny_frame)
    assert list(pivot.index) == ["plain", "markdown"]
    assert list(pivot.columns) == ["openai", "anthropic"]
    assert pivot.loc["plain", "openai"] == pytest.approx(0.0)
    assert pivot.loc["markdown", "anthropic"] == pytest.approx(33.333333, abs=1e-4)


def test_per_provider_bias_direction(tiny_frame: pd.DataFrame) -> None:
    rates = aa.per_provider_bias_direction(tiny_frame)
    # openai: all rows are exact agreement.
    assert rates.loc["openai", "exact"] == pytest.approx(1.0)
    assert rates.loc["openai", "underestimate"] == pytest.approx(0.0)
    # anthropic: all rows are underestimate (offline < empirical).
    assert rates.loc["anthropic", "underestimate"] == pytest.approx(1.0)
    assert rates.loc["anthropic", "exact"] == pytest.approx(0.0)


def test_linear_fit_recovers_exact_relationship() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 1.5 * x + 0.0
    slope, intercept, r2 = aa.linear_fit(x, y)
    assert slope == pytest.approx(1.5)
    assert intercept == pytest.approx(0.0, abs=1e-9)
    assert r2 == pytest.approx(1.0)


def test_per_provider_calibration_fits_anthropic_slope(tiny_frame: pd.DataFrame) -> None:
    fits = {f.provider: f for f in aa.per_provider_calibration_fits(tiny_frame)}
    assert "openai" in fits and "anthropic" in fits
    # All openai rows are (100, 100), so the closed-form fit returns NaNs for
    # slope/intercept (zero variance in x). The dataset-level fits do not hit
    # this case because real offline counts vary.
    assert np.isnan(fits["openai"].slope) or fits["openai"].r_squared == pytest.approx(1.0)
    assert np.isnan(fits["anthropic"].slope) or fits["anthropic"].slope == pytest.approx(1.5)


def test_summarize_results_round_trip_via_json(
    tmp_path: Path, tiny_frame: pd.DataFrame
) -> None:
    out = tmp_path / "results.json"
    aa.write_results_json(tiny_frame, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["dataset"]["n_rows"] == 12
    assert "openai" in payload["per_provider"]
    assert "anthropic" in payload["per_provider"]
    # Sanity: the format×provider matrix has 2 formats × 2 providers = 4 entries.
    fmt_pivot = payload["format_x_provider_median_delta_pct"]
    assert set(fmt_pivot) == {"plain", "markdown"}
    assert set(fmt_pivot["plain"]) == {"openai", "anthropic"}


def test_markdown_renderers_are_nonempty(tiny_frame: pd.DataFrame) -> None:
    for renderer in (
        aa.render_provider_stats_md,
        aa.render_format_heatmap_md,
        aa.render_calibration_factors_md,
    ):
        md = renderer(tiny_frame)
        assert "|" in md
        assert "anthropic" in md or "openai" in md


def test_build_all_artifacts_writes_every_expected_file(
    tmp_path: Path, tiny_parquet: Path
) -> None:
    figures_dir = tmp_path / "figures"
    tables_dir = tmp_path / "tables"
    results = tmp_path / "results.json"

    outputs = aa.build_all_artifacts(
        parquet_path=tiny_parquet,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        results_path=results,
    )
    # All 5 figures land as both PNG and PDF.
    for i in range(1, 6):
        png = next(figures_dir.glob(f"fig0{i}_*.png"))
        pdf = png.with_suffix(".pdf")
        assert png.exists()
        assert pdf.exists()
    # Tables + results.
    assert (tables_dir / "tab01_summary_stats.md").exists()
    assert (tables_dir / "tab02_format_heatmap.md").exists()
    assert (tables_dir / "tab03_calibration_factors.md").exists()
    assert results.exists()
    # Output dict carries the same paths.
    assert set(outputs) == {"fig01", "fig02", "fig03", "fig04", "fig05",
                            "tab01", "tab02", "tab03", "results"}
