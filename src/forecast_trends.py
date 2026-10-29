"""
Job Market Trend Forecaster
Produces future (2027-2030) salary and demand forecasts by role,
specialization, and country.

METHODOLOGY (see docs/README.md "Limitations" for full discussion):
The Kaggle dataset has no genuine year-over-year trend -- fitting a plain
linear trend to 2020-2026 history gives R^2 < 0.25 for every role (mostly
near 0), i.e. statistically flat/noise. Rather than either (a) silently
forecast a flat line and call it a "trend predictor", or (b) fabricate a
trend with no basis, we produce BOTH, clearly separated:

  1. "data_trend_*"   -- an honest linear extrapolation of this dataset's
                          actual history (will be close to flat; reported
                          alongside its R^2 so the weak fit is visible)
  2. "market_trend_*" -- a projection using documented, cited 2026 AI-hiring
                          growth-rate ranges (LinkedIn Workforce Report, PwC
                          Global AI Jobs Barometer, Second Talent/HeroHunt/
                          Acceler8 Talent market analyses) applied as
                          compound annual growth from the most recent
                          (2026) observed baseline, tiered by the same
                          demand-tier classification used elsewhere in this
                          project (High/Medium/Low -> higher/lower assumed
                          CAGR). This is a projection grounded in outside
                          research, not a pattern mined from this dataset.

WITHIN-TIER DIFFERENTIATION (added after an honest bug report): demand_tier
is assigned per SPECIALIZATION only, so every role sharing a specialization
(e.g. all 8 roles x "LLM") got an IDENTICAL market-informed CAGR -- the
leaderboard's top 10 were exact ties, which read as fake/unhandled. Fixed
by adding a small per-combo (role x specialization) adjustment on top of
the tier CAGR, drawn from that combo's own real historical openings/salary
slope (weak overall, R^2~0.09, but genuinely different combo-to-combo --
see docs/README.md). The tier still sets the primary band (a Low-tier combo
cannot leapfrog a High-tier one); the adjustment only breaks ties within a
tier using real, if noisy, per-combo history rather than a flat constant.
The two per-combo means used to source this (hiring_difficulty_score,
ai_adoption_score) were checked and rejected first -- they vary by <1% across
combos in this dataset (near-random per-posting noise averaging out), so
using them would have added fake-looking noise rather than real signal.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "clean_jobs_with_demand_tier.csv"
OUT_PATH = ROOT / "data" / "processed" / "trend_forecast.csv"

FORECAST_YEARS = [2027, 2028, 2029, 2030]

# Approximate composite annualized growth-rate assumptions by demand tier,
# derived from 2026 AI-hiring market reports (see docs/README.md for
# citations). These are intentionally conservative composites: raw posting
# growth reported in some sources runs far higher (e.g. 100%+ YoY for
# LLM-specialist postings off a small base) but compensation growth is
# structurally slower than posting-count growth, so salary CAGR assumptions
# here are well below the raw demand-growth figures reported in the press.
SALARY_CAGR = {"High": 0.07, "Medium": 0.04, "Low": 0.015}
DEMAND_CAGR = {"High": 0.22, "Medium": 0.10, "Low": 0.03}

# Caps on the per-combo real-history adjustment, so it can only nudge a
# combo within its tier's neighborhood, never invert High vs. Low ordering
# outright.
DEMAND_ADJ_CAP = 0.05   # +/- 5 pts/yr on top of the tier's demand CAGR
SALARY_ADJ_CAP = 0.02   # +/- 2 pts/yr on top of the tier's salary CAGR


def fit_linear_trend(years: np.ndarray, values: np.ndarray):
    """Return (slope, intercept, r2) for a simple linear fit."""
    coef = np.polyfit(years, values, 1)
    yhat = np.polyval(coef, years)
    ss_res = float(((values - yhat) ** 2).sum())
    ss_tot = float(((values - values.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return coef[0], coef[1], r2


def run():
    df = pd.read_csv(DATA_PATH)
    records = []

    group_cols = ["job_role", "ai_specialization"]
    for keys, g in df.groupby(group_cols):
        role, spec = keys
        hist = g.groupby("year").agg(
            salary=("salary_usd", "mean"),
            job_openings=("job_openings", "sum"),
        ).reset_index()
        if len(hist) < 3:
            continue

        years = hist["year"].values.astype(float)
        salaries = hist["salary"].values.astype(float)
        openings = hist["job_openings"].values.astype(float)

        s_slope, s_intercept, s_r2 = fit_linear_trend(years, salaries)
        o_slope, o_intercept, o_r2 = fit_linear_trend(years, openings)

        tier = g["demand_tier"].mode()[0]
        base_year = int(years.max())
        base_salary = float(salaries[years.argmax()])
        base_openings = float(openings[years.argmax()])
        tier_salary_cagr = SALARY_CAGR[tier]
        tier_demand_cagr = DEMAND_CAGR[tier]

        # Per-combo real-history adjustment: this combo's own historical
        # slope as a fraction of its own baseline, capped so it can only
        # nudge within the tier's neighborhood, never override it.
        demand_adj = float(np.clip(o_slope / base_openings, -DEMAND_ADJ_CAP, DEMAND_ADJ_CAP))
        salary_adj = float(np.clip(s_slope / base_salary, -SALARY_ADJ_CAP, SALARY_ADJ_CAP))
        salary_cagr = tier_salary_cagr + salary_adj
        demand_cagr = tier_demand_cagr + demand_adj

        for yr in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
            row = hist[hist["year"] == yr]
            if len(row):
                records.append({
                    "job_role": role, "ai_specialization": spec, "demand_tier": tier,
                    "year": yr, "kind": "historical",
                    "salary": float(row["salary"].iloc[0]),
                    "job_openings": float(row["job_openings"].iloc[0]),
                })

        for yr in FORECAST_YEARS:
            dy = yr - base_year
            data_trend_salary = s_slope * yr + s_intercept
            data_trend_openings = max(o_slope * yr + o_intercept, 0)
            market_salary = base_salary * ((1 + salary_cagr) ** dy)
            market_openings = base_openings * ((1 + demand_cagr) ** dy)
            records.append({
                "job_role": role, "ai_specialization": spec, "demand_tier": tier,
                "year": yr, "kind": "forecast",
                "data_trend_salary": round(data_trend_salary, 2),
                "data_trend_openings": round(data_trend_openings, 2),
                "data_trend_r2": round(s_r2, 4),
                "market_informed_salary": round(market_salary, 2),
                "market_informed_openings": round(market_openings, 2),
                "salary_cagr_assumed": round(salary_cagr, 4),
                "demand_cagr_assumed": round(demand_cagr, 4),
                "tier_salary_cagr": tier_salary_cagr,
                "tier_demand_cagr": tier_demand_cagr,
            })

    out = pd.DataFrame(records)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"[trend forecast] wrote {len(out):,} rows -> {OUT_PATH}")

    # summary of trend fit quality, for the docs/limitations discussion
    fit_quality = out[out["kind"] == "forecast"][
        ["job_role", "ai_specialization", "data_trend_r2"]
    ].drop_duplicates()
    print("[trend forecast] historical linear-fit R^2 by role x specialization "
          f"(median={fit_quality['data_trend_r2'].median():.3f}, "
          f"max={fit_quality['data_trend_r2'].max():.3f}) "
          "-- low values confirm the raw data has no strong trend.")

    # sanity check: confirm the fix actually broke the ties
    dup_check = out[out["kind"] == "forecast"][
        ["job_role", "ai_specialization", "demand_cagr_assumed"]
    ].drop_duplicates()
    n_unique_cagrs = dup_check["demand_cagr_assumed"].nunique()
    print(f"[trend forecast] {n_unique_cagrs} distinct demand CAGR values across "
          f"{len(dup_check)} combos (was 3 before the per-combo adjustment).")
    return out


if __name__ == "__main__":
    run()
