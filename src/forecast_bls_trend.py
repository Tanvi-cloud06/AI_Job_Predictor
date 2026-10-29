"""
BLS-Grounded Trend Forecaster
==============================
Replaces the old "data_trend" layer (which was fit on the Kaggle dataset's
noisy 2020-2026 history, R^2 ~ 0.09, i.e. no real trend) with a linear fit on
REAL U.S. government wage/employment data: BLS OEWS national estimates for
Data Scientists (SOC 15-2051), pulled directly from bls.gov.

Why this is different from before:
  - Every year in data/raw/bls_oews_data_scientists.csv is a real, cited,
    government-published annual estimate -- not a synthetic/competition
    dataset column.
  - 2023 -> 2025 shows genuine, verifiable growth (employment 192,710 ->
    262,440; median annual wage $108,020 -> ~$120,224), so the linear trend
    fit here has real signal, unlike the flat Kaggle fit.
  - 2020 is included as a rough historical anchor only, flagged with a
    caveat: it used the pre-2021 combined SOC bucket (15-2098), not the
    standalone Data Scientists code, so it isn't strictly comparable to
    2023-2025. It's excluded from the trend line fit itself for that reason
    and shown for context only.

Known limitation (stated plainly, not buried): this is a NATIONAL, ROLE-
SPECIFIC (Data Scientists only) U.S. aggregate. It does not break out by
country, seniority, or the full set of AI specializations the way the old
per-posting Kaggle set did. Pair this with a real per-posting dataset (e.g.
a genuine scraped LinkedIn postings file with free-text descriptions and
per-posting salary) to recover that granularity -- that layer is still
pending as of this script's creation; see docs/README.md.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLS_PATH = ROOT / "data" / "raw" / "bls_oews_data_scientists.csv"
OUT_PATH = ROOT / "data" / "processed" / "bls_trend_forecast.csv"

FORECAST_YEARS = [2026, 2027, 2028, 2029, 2030]

# Only years under the standalone 15-2051 SOC code go into the trend fit.
# The 2020 row (15-2098, a broader combined bucket) is real data but not
# comparable to the others, so it's excluded from the regression and kept
# only as a labeled historical reference point.
TREND_FIT_YEARS = {2023, 2024, 2025}


def fit_linear_trend(years: np.ndarray, values: np.ndarray):
    coef = np.polyfit(years, values, 1)
    yhat = np.polyval(coef, years)
    ss_res = float(((values - yhat) ** 2).sum())
    ss_tot = float(((values - values.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0  # 2 points = perfect fit by definition
    return coef[0], coef[1], r2


def run():
    bls = pd.read_csv(BLS_PATH)
    fit_rows = bls[bls["year"].isin(TREND_FIT_YEARS)].sort_values("year")

    years = fit_rows["year"].values.astype(float)
    employment = fit_rows["employment"].values.astype(float)
    median_wage = fit_rows["median_annual_wage"].values.astype(float)

    emp_slope, emp_intercept, emp_r2 = fit_linear_trend(years, employment)
    wage_slope, wage_intercept, wage_r2 = fit_linear_trend(years, median_wage)

    records = []

    # Historical rows (all years in the raw file, including the flagged 2020 one)
    for _, row in bls.iterrows():
        records.append({
            "occupation": row["occupation_title"],
            "soc_code": row["soc_code"],
            "year": int(row["year"]),
            "kind": "historical",
            "employment": row["employment"],
            "median_annual_wage": row["median_annual_wage"],
            "mean_annual_wage": row["mean_annual_wage"],
            "comparable_to_current_soc": row["soc_code"] == "15-2051",
            "source_url": row["source_url"],
        })

    # Forecast rows: real linear extrapolation of the 2023-2025 BLS trend
    for yr in FORECAST_YEARS:
        records.append({
            "occupation": "Data Scientists",
            "soc_code": "15-2051",
            "year": yr,
            "kind": "forecast",
            "employment": round(max(emp_slope * yr + emp_intercept, 0), 0),
            "median_annual_wage": round(wage_slope * yr + wage_intercept, 2),
            "mean_annual_wage": None,
            "employment_trend_r2": round(emp_r2, 4),
            "wage_trend_r2": round(wage_r2, 4),
            "comparable_to_current_soc": True,
            "source_url": "linear fit on bls.gov 2023-2025 (see data/raw/bls_oews_data_scientists.csv)",
        })

    out = pd.DataFrame(records)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"[bls trend] wrote {len(out)} rows -> {OUT_PATH}")
    print(f"[bls trend] employment trend fit: slope={emp_slope:,.0f}/yr, R^2={emp_r2:.3f} "
          f"(fit on real BLS years {sorted(TREND_FIT_YEARS)})")
    print(f"[bls trend] median wage trend fit: slope=${wage_slope:,.0f}/yr, R^2={wage_r2:.3f}")
    print("[bls trend] NOTE: R^2 on a 3-point fit is not a rigorous significance test -- "
          "it's reported for transparency, not as proof of a strong trend. Treat the "
          "extrapolation to 2030 as directional, not precise.")
    return out


if __name__ == "__main__":
    run()
