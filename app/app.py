"""
AI Job Market Trend Predictor: Interactive Dashboard
Run locally:  streamlit run app/app.py
Deployed via Docker on Render/Railway (see Dockerfile).
"""
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "clean_jobs_with_demand_tier.csv"
FORECAST_PATH = ROOT / "data" / "processed" / "trend_forecast.csv"
BLS_FORECAST_PATH = ROOT / "data" / "processed" / "bls_trend_forecast.csv"
DEMAND_MODEL_PATH = ROOT / "models" / "demand_model.joblib"
DEMAND_ENCODER_PATH = ROOT / "models" / "demand_encoders.joblib"
DEMAND_METRICS_PATH = ROOT / "models" / "demand_model_metrics.json"

# Real-salary layer: trained on genuine LinkedIn postings (April 2024
# snapshot, US-only, disclosed salary). This is the ONLY salary model wired
# into the app -- the earlier synthetic-data salary model (R^2=0.93) is
# deliberately not used for prediction; its accuracy came from a dataset with
# baked-in relationships, not real signal. See docs/README.md.
LI_DATA_PATH = ROOT / "data" / "processed" / "linkedin_ai_jobs_clean.csv"
LI_MODEL_PATH = ROOT / "models" / "linkedin_salary_model.joblib"
LI_META_PATH = ROOT / "models" / "linkedin_salary_meta.joblib"
LI_METRICS_PATH = ROOT / "models" / "linkedin_salary_model_metrics.json"

st.set_page_config(
    page_title="AI Job Market Trend Predictor",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Visual identity: a job board you'd actually want to browse. Cool light
# canvas, white result cards, one cobalt action color, and a green / amber /
# grey traffic-light for demand tiers (High / Medium / Low) so the meaning
# of a color never changes from tab to tab. Bricolage Grotesque gives
# headlines some character; Figtree keeps numbers and body text easy to read.
# ---------------------------------------------------------------------------
BG = "#F3F5F9"
SURFACE = "#FFFFFF"
LINE = "#DDE2EC"
INK = "#1B2238"
MUTED = "#5D667C"
COBALT = "#2F55D4"
COBALT_SOFT = "#E6ECFD"
GREEN = "#13896B"
AMBER = "#E39B1B"
SLATE = "#8A93A6"
# aliases used by the chart code below
PAPER, TEAL = INK, GREEN

TIER_COLORS = {"High": GREEN, "Medium": AMBER, "Low": SLATE}
SERIES_COLORS = {
    "historical": COBALT, "forecast": GREEN,
    "Past data (actual)": COBALT,
    "Straight-line trend from past data": SLATE,
    "Market-research projection": GREEN,
}

_template = go.layout.Template()
_template.layout = go.Layout(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family="Figtree, sans-serif", color=INK, size=13),
    title=dict(font=dict(family="Bricolage Grotesque, sans-serif", size=17, color=INK)),
    colorway=[COBALT, GREEN, AMBER, SLATE, "#8E5CD9"],
    xaxis=dict(gridcolor="#EAEEF5", zerolinecolor=LINE, linecolor=LINE),
    yaxis=dict(gridcolor="#EAEEF5", zerolinecolor=LINE, linecolor=LINE),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=52, l=10, r=10, b=10),
)
pio.templates["jobs"] = _template
pio.templates.default = "jobs"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=Figtree:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Figtree', sans-serif; color: {INK}; }}
.stApp {{ background: {BG}; }}
h1, h2, h3, h4 {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700;
                  letter-spacing: -0.02em; color: {INK}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}

/* hero */
.hero {{ background: {INK}; color: #fff; border-radius: 22px; padding: 1.9rem 2.2rem 1.6rem;
         margin-bottom: 1.6rem; }}
.hero-title {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800;
               font-size: 2.6rem; line-height: 1.02; letter-spacing: -0.03em; margin: 0 0 0.6rem; }}
.hero-sub {{ color: #B9C2DA; font-size: 1.05rem; max-width: 46rem; line-height: 1.5; margin: 0 0 1.2rem; }}
.hero-stats {{ display: flex; flex-wrap: wrap; gap: 0.6rem; }}
.hero-stat {{ background: rgba(255,255,255,0.09); border-radius: 999px; padding: 0.35rem 0.95rem;
              font-size: 0.92rem; color: #E8ECF8; font-variant-numeric: tabular-nums; }}
.hero-stat b {{ color: #fff; font-weight: 700; }}

/* tabs */
div[data-baseweb="tab-list"] {{ gap: 0.4rem; border-bottom: 1px solid {LINE}; }}
button[data-baseweb="tab"] {{ font-weight: 600; color: {MUTED}; padding: 0.7rem 1rem; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {COBALT}; }}
div[data-baseweb="tab-highlight"] {{ background: {COBALT}; height: 3px; }}
div[data-baseweb="tab-border"] {{ display: none; }}

/* metrics as result cards */
[data-testid="stMetric"] {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 14px;
                            padding: 0.9rem 1.1rem; }}
[data-testid="stMetricValue"] {{ font-weight: 700; color: {INK}; font-variant-numeric: tabular-nums; }}
[data-testid="stMetricLabel"] {{ color: {MUTED}; }}

/* charts, tables, expanders sit on white cards */
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {{ background: {SURFACE};
        border: 1px solid {LINE}; border-radius: 14px; padding: 0.4rem; overflow: hidden; }}
[data-testid="stExpander"] {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 14px; }}

/* inputs and buttons */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{ border-radius: 10px; }}
.stButton > button {{ border-radius: 999px; font-weight: 600; }}
span[data-baseweb="tag"] {{ background: {COBALT_SOFT}; color: {COBALT}; border-radius: 999px; }}

section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {LINE}; }}
hr {{ border-color: {LINE} !important; }}

@media (max-width: 640px) {{
  .hero {{ padding: 1.3rem 1.2rem; border-radius: 16px; }}
  .hero-title {{ font-size: 1.9rem; }}
}}
:focus-visible {{ outline: 2px solid {COBALT}; outline-offset: 2px; }}
</style>
""", unsafe_allow_html=True)

SALARY_CAT = ["country", "job_role", "ai_specialization", "experience_level",
              "education_required", "industry", "company_size", "work_mode"]
SALARY_NUM = ["experience_years", "year", "weekly_hours", "company_rating",
              "job_openings", "hiring_difficulty_score", "ai_adoption_score",
              "company_funding_billion", "economic_index", "ai_maturity_years",
              "cost_of_living_index"]

DEMAND_CAT = SALARY_CAT
DEMAND_NUM = ["year", "job_openings", "hiring_difficulty_score", "ai_adoption_score",
              "career_growth_score", "company_funding_billion", "automation_risk"]
TIER_ORDER = ["Low", "Medium", "High"]

COUNTRY_ISO3 = {
    "USA": "USA", "UK": "GBR", "Germany": "DEU", "France": "FRA", "Canada": "CAN",
    "Australia": "AUS", "Singapore": "SGP", "Switzerland": "CHE", "UAE": "ARE",
    "Japan": "JPN", "Netherlands": "NLD", "China": "CHN", "India": "IND",
    "Brazil": "BRA",
}


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_forecast():
    return pd.read_csv(FORECAST_PATH)


@st.cache_data
def load_bls_forecast():
    return pd.read_csv(BLS_FORECAST_PATH)


def _retrain(script_name: str):
    """Re-run a training script with THIS environment's own numpy/sklearn,
    so the resulting .joblib is pickled in a format this environment can
    read back. The shipped .joblib files were pickled in a different numpy
    version and can raise errors like
    "numpy.random._mt19937.MT19937 is not a known BitGenerator module" when
    unpickled elsewhere -- that's a pickle/numpy ABI mismatch, not a data or
    logic bug, and retraining locally is the fix."""
    script_path = ROOT / "src" / script_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Retraining {script_name} failed:\n{result.stdout}\n{result.stderr}"
        )


@st.cache_resource
def load_models():
    try:
        demand_model = joblib.load(DEMAND_MODEL_PATH)
        demand_encoder = joblib.load(DEMAND_ENCODER_PATH)
    except Exception:
        with st.spinner(
            "First run on this machine: retraining the demand model "
            "locally (one-time, ~30s)..."
        ):
            _retrain("train_demand_model.py")
        demand_model = joblib.load(DEMAND_MODEL_PATH)
        demand_encoder = joblib.load(DEMAND_ENCODER_PATH)

    try:
        li_model = joblib.load(LI_MODEL_PATH)
        li_meta = joblib.load(LI_META_PATH)
    except Exception:
        with st.spinner(
            "First run on this machine: retraining the salary model "
            "locally (one-time, ~30s)..."
        ):
            _retrain("train_salary_model_linkedin.py")
        li_model = joblib.load(LI_MODEL_PATH)
        li_meta = joblib.load(LI_META_PATH)

    return demand_model, demand_encoder, li_model, li_meta


@st.cache_data
def load_metrics():
    import json
    with open(DEMAND_METRICS_PATH) as f:
        demand_metrics = json.load(f)
    with open(LI_METRICS_PATH) as f:
        li_metrics = json.load(f)
    return demand_metrics, li_metrics


@st.cache_data
def load_li_data():
    return pd.read_csv(LI_DATA_PATH)


df = load_data()
forecast_df = load_forecast()
bls_df = load_bls_forecast()
demand_model, demand_encoder, li_model, li_meta = load_models()
demand_metrics, li_metrics = load_metrics()
li_df = load_li_data()

EXP_LEVELS = list(li_meta["exp_level_order"].keys())
SKILL_KEYWORDS = li_meta["skill_keywords"]
SKILL_COLS = li_meta["skill_cols"]
LI_FEATURE_COLS = li_meta["feature_cols"]
LI_TOP_STATES = li_meta["top_states"]
LI_STATE_ENC = li_meta["state_encoder"]
LI_IMPORTANCES = pd.Series(li_meta["importances"]).sort_values(ascending=False)


def predict_salary_range(exp_level: str, remote_allowed: int, state: str, skills: list):
    """Real-model prediction -> (point estimate, low, high) using ±1 MAE as
    the displayed range, since a bare point number overstates what an
    R^2=0.24 model can actually promise."""
    state_grouped = state if state in LI_TOP_STATES else "Other"
    state_code = int(LI_STATE_ENC.transform([state_grouped])[0])
    row = {
        "experience_level_ord": li_meta["exp_level_order"][exp_level],
        "remote_allowed": remote_allowed,
        "n_skills": len(skills),
        "state_code": state_code,
    }
    for col, kw in zip(SKILL_COLS, SKILL_KEYWORDS):
        row[col] = int(kw in skills)
    X = pd.DataFrame([row])[LI_FEATURE_COLS]
    pred = float(li_model.predict(X)[0])
    mae = li_metrics["mae_usd"]
    return pred, max(0, pred - mae), pred + mae


st.markdown(f"""
<div class="hero">
  <div class="hero-title">AI jobs: what is growing and what it pays</div>
  <div class="hero-sub">Find out which AI and machine learning jobs are growing, roughly what they pay, and which skills show up most often in job ads.</div>
  <div class="hero-stats">
    <span class="hero-stat"><b>{len(df):,}</b> postings</span>
    <span class="hero-stat"><b>{df['country'].nunique()}</b> countries</span>
    <span class="hero-stat"><b>{df['job_role'].nunique()}</b> roles</span>
    <span class="hero-stat"><b>{df['year'].min()}–{df['year'].max()}</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("New here? Start with this"):
    st.markdown(
        "1. **Growing jobs**: which AI jobs are projected to grow fastest by 2030.\n"
        "2. **Career planner**: which skill to learn next, based on your profile.\n"
        "3. **Salary estimator**: a likely pay range for a job you describe.\n"
        "4. **Pay by country**, **Skill gaps** and **In-demand skills**: comparisons "
        "across countries, specializations and demand tiers.\n\n"
        "**Words used in this app**\n\n"
        "- **Salary range**: the model's best guess, plus or minus its typical error.\n"
        "- **R²**: how much of the pay differences the model explains. 1.0 is perfect; "
        "0 means no better than guessing the average.\n"
        "- **Demand tier**: High, Medium or Low, based on how fast a specialization is "
        "expected to grow.\n"
        "- **Projection**: an estimate of the future, not a guarantee.\n"
        "- **Specialization**: the AI focus of a job, such as NLP (language) or Computer Vision (images).\n"
        "- **Job openings**: the number of open positions in the data."
    )

with st.sidebar:
    st.header("How reliable are these numbers?")
    st.caption(f"Salary estimates use {li_metrics['n_rows_total']:,} real US job ads from "
               "LinkedIn. Pay data is messy, so read every salary as a rough range.")
    st.metric("Typical salary error", f"±${li_metrics['mae_usd']:,.0f}",
              help="On average, the salary model is off by about this much on job ads "
                   "it was not trained on.")
    st.metric("Demand tier accuracy", f"{demand_metrics['accuracy']*100:.1f}%",
              help="How often the demand model picks the right tier on test data. The tiers "
                   "come from market research, so this shows how well the model learns that rule.")
    with st.expander("Technical details"):
        st.write(f"Salary model R²: **{li_metrics['r2']}**. Skills account for "
                 f"{li_metrics['total_importance_share_from_skill_features']*100:.0f}% "
                 "of what the model uses.")
        st.write(f"Demand tier F1 score: **{demand_metrics['f1_macro']*100:.1f}%** "
                 "(a balanced score across the three tiers).")
        st.caption("The demand and trend tabs use an older dataset plus market research "
                   "assumptions. Each tab says which parts are real data and which are illustrative.")

tabF, tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "Growing jobs", "Career planner", "Salary estimator", "Pay by country",
    "Skill gaps", "In-demand skills",
])

# ---------------------------------------------------------------- TAB F ----
with tabF:
    st.subheader("Which AI jobs are growing fastest?")
    st.caption(
        "Each job and specialization, ranked by projected growth in job openings through 2030. "
        "The projections come from AI hiring research, applied to each job's most recent figure "
        "in the data. The explanation is further down the page."
    )

    latest_fc = forecast_df[forecast_df["kind"] == "forecast"]
    base_hist = (forecast_df[forecast_df["kind"] == "historical"]
                 .sort_values("year")
                 .groupby(["job_role", "ai_specialization"])
                 .tail(1)
                 .rename(columns={"job_openings": "base_openings", "salary": "base_salary", "year": "base_year"})
                 [["job_role", "ai_specialization", "base_year", "base_openings", "base_salary"]])
    fc_2030 = latest_fc[latest_fc["year"] == 2030][
        ["job_role", "ai_specialization", "demand_tier", "market_informed_openings",
         "market_informed_salary", "demand_cagr_assumed", "salary_cagr_assumed"]
    ]
    board = base_hist.merge(fc_2030, on=["job_role", "ai_specialization"])
    board["openings_growth_pct"] = (board["market_informed_openings"] / board["base_openings"] - 1) * 100
    board["salary_growth_pct"] = (board["market_informed_salary"] / board["base_salary"] - 1) * 100
    board = board.sort_values("openings_growth_pct", ascending=False).reset_index(drop=True)
    board["job"] = board["job_role"] + " · " + board["ai_specialization"]

    top_n = 10
    top_board = board.head(top_n)
    fig_board = px.bar(
        top_board.sort_values("openings_growth_pct"), x="openings_growth_pct", y="job",
        orientation="h", color="demand_tier", color_discrete_map=TIER_COLORS,
        title=f"Top {top_n} fastest-growing jobs: projected growth in openings by 2030",
        labels={"openings_growth_pct": "Projected openings growth to 2030 (%)", "job": ""},
    )
    st.plotly_chart(fig_board, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    fastest = board.iloc[0]
    m1.metric("Fastest-growing job", fastest["job"],
              f"+{fastest['openings_growth_pct']:.0f}% openings by 2030")
    m2.metric("Projected 2030 salary", f"${fastest['market_informed_salary']:,.0f}",
              f"+{fastest['salary_growth_pct']:.0f}% vs {int(fastest['base_year'])}")
    m3.metric("Demand tier", fastest["demand_tier"])

    st.caption(
        "**Why the ranking is not just the three tiers:** the demand tier (High, Medium or Low) sets "
        "each job's main growth rate, based on market research. Within a tier, jobs are ordered by their "
        "own past job-openings trend. That trend is weak, so it only shifts the order a little, and it "
        "never moves a Low-tier job above a High-tier one."
    )

    with st.expander("See the full ranking (all jobs)"):
        show = board[["job_role", "ai_specialization", "demand_tier", "base_year", "base_openings",
                      "market_informed_openings", "openings_growth_pct", "base_salary",
                      "market_informed_salary", "salary_growth_pct"]].rename(columns={
            "job_role": "Role", "ai_specialization": "Specialization", "demand_tier": "Demand tier",
            "base_year": "Baseline year", "base_openings": "Baseline openings",
            "market_informed_openings": "2030 projected openings", "openings_growth_pct": "Openings growth %",
            "base_salary": "Baseline salary", "market_informed_salary": "2030 projected salary",
            "salary_growth_pct": "Salary growth %",
        })
        st.dataframe(
            show.style.format({
                "Baseline openings": "{:,.0f}", "2030 projected openings": "{:,.0f}",
                "Openings growth %": "{:+.0f}%", "Baseline salary": "${:,.0f}",
                "2030 projected salary": "${:,.0f}", "Salary growth %": "{:+.0f}%",
            }),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")
    st.markdown("**Real-world check: US government data for Data Scientists**")
    st.caption(
        "The ranking above is a projection. For Data Scientists, the US Bureau of Labor Statistics (BLS) "
        "publishes yearly employment and wage figures, so we can check the direction against real numbers."
    )
    bls_hist = bls_df[bls_df["kind"] == "historical"]
    bls_fut = bls_df[bls_df["kind"] == "forecast"]
    bc1, bc2 = st.columns(2)
    with bc1:
        fig_bls_emp = px.line(
            pd.concat([bls_hist, bls_fut]), x="year", y="employment", markers=True,
            color="kind", title="Data Scientists in the US: number employed (BLS data)",
            color_discrete_map=SERIES_COLORS,
        )
        st.plotly_chart(fig_bls_emp, use_container_width=True)
    with bc2:
        fig_bls_wage = px.line(
            pd.concat([bls_hist, bls_fut]), x="year", y="median_annual_wage", markers=True,
            color="kind", title="Data Scientists in the US: median yearly wage (BLS data)",
            color_discrete_map=SERIES_COLORS,
        )
        st.plotly_chart(fig_bls_wage, use_container_width=True)
    r2_emp = bls_fut["employment_trend_r2"].iloc[0]
    r2_wage = bls_fut["wage_trend_r2"].iloc[0]
    st.caption(
        f"A straight line fitted to the 2023–2025 BLS figures fits well (employment R² = {r2_emp:.2f}, "
        f"wage R² = {r2_wage:.2f}; 1.0 is a perfect fit). Source: bls.gov, Occupational Employment and "
        "Wage Statistics. Full sources are in the documentation at the bottom of the page."
    )

    st.markdown("---")
    st.markdown("### Look at one job in detail")
    fc1, fc2 = st.columns(2)
    with fc1:
        f_role = st.selectbox("Role", sorted(forecast_df["job_role"].unique()), key="f_role")
    with fc2:
        f_spec_options = sorted(forecast_df[forecast_df["job_role"] == f_role]["ai_specialization"].unique())
        f_spec = st.selectbox("Specialization", f_spec_options, key="f_spec")

    fdata = forecast_df[(forecast_df["job_role"] == f_role) & (forecast_df["ai_specialization"] == f_spec)]
    hist = fdata[fdata["kind"] == "historical"].sort_values("year")
    fut = fdata[fdata["kind"] == "forecast"].sort_values("year")
    tier = fdata["demand_tier"].iloc[0]
    r2 = fut["data_trend_r2"].iloc[0] if len(fut) else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Demand tier", tier)
    m2.metric("Assumed yearly salary growth (2027+)", f"{fut['salary_cagr_assumed'].iloc[0]*100:.1f}%/yr" if len(fut) else "n/a")
    m3.metric("Past trend fit (R²)", f"{r2:.3f}", help="How well a straight line fits this job's 2020–2026 history. "
              "Near 0 means the history is mostly noise; 1 means a perfect line.")

    # build a single chart: historical actuals + two forecast lines
    plot_rows = []
    for _, r in hist.iterrows():
        plot_rows.append({"year": r["year"], "salary": r["salary"], "series": "Past data (actual)"})
    if len(fut):
        last_hist_year = hist["year"].max()
        last_hist_salary = hist[hist["year"] == last_hist_year]["salary"].iloc[0]
        # connect the lines to the last historical point
        for series_name, col in [("Straight-line trend from past data", "data_trend_salary"),
                                  ("Market-research projection", "market_informed_salary")]:
            plot_rows.append({"year": last_hist_year, "salary": last_hist_salary, "series": series_name})
            for _, r in fut.iterrows():
                plot_rows.append({"year": r["year"], "salary": r[col], "series": series_name})

    plot_df = pd.DataFrame(plot_rows)
    fig = px.line(plot_df, x="year", y="salary", color="series", markers=True,
                 title=f"Salary: past data and projection, {f_role} · {f_spec}",
                 color_discrete_map=SERIES_COLORS)
    fig.update_layout(yaxis_title="Avg salary (USD)", legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Projected numbers (2027–2030)"):
        show_cols = ["year", "data_trend_salary", "market_informed_salary",
                     "data_trend_openings", "market_informed_openings"]
        st.dataframe(
            fut[show_cols].rename(columns={
                "year": "Year", "data_trend_salary": "Trend-line salary",
                "market_informed_salary": "Projected salary (market research)",
                "data_trend_openings": "Trend-line openings",
                "market_informed_openings": "Projected openings (market research)",
            }).style.format({
                "Trend-line salary": "${:,.0f}", "Projected salary (market research)": "${:,.0f}",
                "Trend-line openings": "{:,.0f}", "Projected openings (market research)": "{:,.0f}",
            }),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")
    st.markdown("**Which specializations are growing fastest?**")
    latest = forecast_df[forecast_df["kind"] == "forecast"].drop_duplicates(
        subset=["job_role", "ai_specialization"]
    )
    trend_rank = latest.groupby("ai_specialization").agg(
        demand_tier=("demand_tier", lambda x: x.mode()[0]),
        avg_demand_cagr=("demand_cagr_assumed", "mean"),
        avg_salary_cagr=("salary_cagr_assumed", "mean"),
    ).reset_index().sort_values("avg_demand_cagr", ascending=False)
    fig2 = px.bar(trend_rank, x="ai_specialization", y="avg_demand_cagr", color="demand_tier",
                 color_discrete_map=TIER_COLORS,
                 title="Projected yearly growth in job openings, by specialization")
    fig2.update_layout(yaxis_title="Assumed yearly growth rate", yaxis_tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### How to read these projections")
    st.markdown(
        "The main dataset has no real year-to-year trend. A straight line through 2020–2026 explains very "
        "little for most jobs (median R² of about 0.09). The grey line shows this: it is almost flat.\n\n"
        "The green line is a projection instead. It applies a yearly growth rate that depends on the "
        "specialization's demand tier. High tier: about 7% a year for salaries and 22% a year for job "
        "openings. Medium tier: about 4% and 10%. Low tier: about 1.5% and 3%. These rates come from recent "
        "AI hiring research (sources are listed in the documentation at the bottom of the page).\n\n"
        "Treat the green line as a research-based estimate. The data itself does not prove it."
    )

# ---------------------------------------------------------------- TAB 0 ----
with tab0:
    st.subheader("Career planner: what to learn next")
    st.caption(
        "Enter your current profile. The planner tests each skill you do not have yet and ranks them by "
        "how much the LinkedIn salary model's estimate would change if you added it."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        cp_exp_level = st.selectbox("Experience level", EXP_LEVELS, index=3, key="cp_exp_level")
    with c2:
        cp_state = st.selectbox("US state", sorted(LI_TOP_STATES) + ["Other"], key="cp_state")
    with c3:
        cp_remote = st.checkbox("Remote allowed", value=False, key="cp_remote")

    cp_skills = st.multiselect(
        "Your current skills", sorted(SKILL_KEYWORDS),
        default=["python", "sql"], key="cp_skills",
    )

    base_pred, base_lo, base_hi = predict_salary_range(cp_exp_level, int(cp_remote), cp_state, cp_skills)

    st.markdown("---")
    st.metric("Your estimated salary range now", f"${base_lo:,.0f} – ${base_hi:,.0f}")

    # ---- Skill recommendation: try adding each candidate skill one at a
    # time, rank by predicted uplift to the central estimate ----
    KW_TO_COL = dict(zip(SKILL_KEYWORDS, SKILL_COLS))
    candidates = [s for s in SKILL_KEYWORDS if s not in cp_skills]
    skill_rows = []
    for cand in candidates:
        pred, lo, hi = predict_salary_range(cp_exp_level, int(cp_remote), cp_state, cp_skills + [cand])
        skill_rows.append({
            "Skill": cand, "Predicted range": f"${lo:,.0f} – ${hi:,.0f}",
            "Change vs current": pred - base_pred,
            "Model importance": float(LI_IMPORTANCES.get(KW_TO_COL[cand], 0.0)),
        })
    skill_df = pd.DataFrame(skill_rows).sort_values("Change vs current", ascending=False)

    # ---- Experience-level scenario: same skills, next level up ----
    level_rows = []
    for lvl in EXP_LEVELS:
        pred, lo, hi = predict_salary_range(lvl, int(cp_remote), cp_state, cp_skills)
        level_rows.append({"Experience level": lvl, "Predicted range": f"${lo:,.0f} – ${hi:,.0f}",
                            "vs current": pred - base_pred})
    level_df = pd.DataFrame(level_rows)

    st.markdown("### Top recommendation")
    if len(skill_df):
        top_skill = skill_df.iloc[0]
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Skill to learn next**")
            if top_skill["Change vs current"] <= 0:
                st.write("None of the remaining skills on this list would raise your predicted salary. "
                         "You already cover this list well.")
            else:
                st.metric("Skill", top_skill["Skill"],
                          f"+${top_skill['Change vs current']:,.0f} predicted salary")
                st.write(f"New predicted range with this skill: {top_skill['Predicted range']}")
            st.caption("Ranked by how much the salary estimate rises when each skill is added, one at a time.")
        with r2:
            st.markdown("**Next experience level**")
            next_idx = min(EXP_LEVELS.index(cp_exp_level) + 1, len(EXP_LEVELS) - 1)
            nxt = level_df.iloc[next_idx]
            if nxt["Experience level"] == cp_exp_level:
                st.write(f"{cp_exp_level} is already the highest level in the model.")
            else:
                v = nxt["vs current"]
                st.metric("Level", nxt["Experience level"],
                          f"{'+' if v >= 0 else '-'}${abs(v):,.0f} predicted salary")
                st.write(f"New predicted range: {nxt['Predicted range']}")
            st.caption("Experience level is the strongest factor in this model.")

    with st.expander("See predicted change for every other skill"):
        st.dataframe(
            skill_df[["Skill", "Predicted range", "Change vs current"]]
            .style.format({"Change vs current": "{:+,.0f}"}),
            use_container_width=True, hide_index=True,
        )
    with st.expander("See predicted range at each experience level"):
        st.dataframe(
            level_df.style.format({"vs current": "{:+,.0f}"}),
            use_container_width=True, hide_index=True,
        )

    st.caption(
        "Note: this model explains about 24% of the differences in salary (R² = 0.24) and was trained on "
        "1,891 job ads. Treat the dollar amounts as rough estimates. The ranking of skills is more reliable "
        "than the exact figures."
    )

# ---------------------------------------------------------------- TAB 1 ----
with tab1:
    st.subheader("Salary estimator")
    st.caption(
        f"Based on {li_metrics['n_rows_total']:,} US tech, AI and data job ads from LinkedIn that list a "
        "salary (April 2024). The model is only moderately accurate (R² = 0.24), so it shows a range "
        "instead of one number. The skill breakdown on the right is the more reliable part of the result."
    )
    col1, col2 = st.columns([1, 1.4])

    with col1:
        exp_level = st.selectbox("Experience level", EXP_LEVELS, index=3)
        state = st.selectbox("US state", sorted(li_meta["top_states"]) + ["Other"],
                              index=0, help="Top-15 most common states in the data; anything "
                                            "else is grouped as 'Other'.")
        remote = st.checkbox("Remote allowed", value=False)
        skills = st.multiselect(
            "Skills (pick the ones you have)",
            sorted(SKILL_KEYWORDS),
            default=["python", "sql"],
        )

        pred, lo, hi = predict_salary_range(exp_level, int(remote), state, skills)
        st.metric("Likely salary range (USD)", f"${lo:,.0f} – ${hi:,.0f}")
        st.caption(f"Middle estimate: ${pred:,.0f}. The range is that estimate plus or minus "
                   f"${li_metrics['mae_usd']:,.0f}, the model's typical error.")

    with col2:
        st.markdown("**What drives this estimate**")
        st.caption(
            f"Skills make up {li_metrics['total_importance_share_from_skill_features']*100:.0f}% of what "
            "the model uses to estimate pay. The skills were picked out of the text of real job ads."
        )
        fi_df = LI_IMPORTANCES.head(10).reset_index()
        fi_df.columns = ["Feature", "Importance"]
        fig = px.bar(fi_df.sort_values("Importance"), x="Importance", y="Feature",
                     orientation="h", title="Top 10 factors by importance")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Salaries in similar job ads**")
        similar = li_df[li_df["experience_level_ord"] == li_meta["exp_level_order"][exp_level]]
        if len(similar) > 15:
            fig2 = px.histogram(similar, x="normalized_salary", nbins=30,
                                title=f"Listed salaries for {exp_level}-level job ads (n={len(similar):,})")
            fig2.add_vline(x=pred, line_color=COBALT, line_width=3,
                           annotation_text="Your estimate")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("Not enough postings at this experience level to plot a distribution.")

# ---------------------------------------------------------------- TAB 2 ----
with tab2:
    st.subheader("Pay by country")
    st.caption(
        "Note: this tab uses the older illustrative dataset, not the LinkedIn salary model. Use it to compare "
        "countries and roles with each other, not as exact pay figures. The Salary estimator tab uses real, "
        "listed US salaries."
    )
    metric_choice = st.selectbox("Metric", ["Median salary", "Mean salary", "Job openings"])
    geo = df.groupby("country").agg(
        median_salary=("salary_usd", "median"),
        mean_salary=("salary_usd", "mean"),
        job_openings=("job_openings", "sum"),
    ).reset_index()
    geo["iso3"] = geo["country"].map(COUNTRY_ISO3)
    col_map = {"Median salary": "median_salary", "Mean salary": "mean_salary", "Job openings": "job_openings"}
    z = col_map[metric_choice]

    fig = px.choropleth(geo, locations="iso3", color=z, hover_name="country",
                        color_continuous_scale=[SURFACE, "#B7C6F5", COBALT],
                        title=f"{metric_choice} by country")
    fig.update_layout(geo=dict(bgcolor=SURFACE, lakecolor=SURFACE,
                                landcolor=BG, showframe=False))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        top = geo.sort_values(z, ascending=False).head(6)
        fig_bar = px.bar(top, x="country", y=z, title=f"Top countries by {metric_choice.lower()}")
        st.plotly_chart(fig_bar, use_container_width=True)
    with c2:
        role_geo = df.groupby(["country", "job_role"])["salary_usd"].median().reset_index()
        fig_heat = px.density_heatmap(role_geo, x="job_role", y="country", z="salary_usd",
                                      histfunc="avg", color_continuous_scale=[SURFACE, "#B7C6F5", COBALT],
                                      title="Median salary by role and country")
        st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------------- TAB 3 ----
with tab3:
    st.subheader("Skill gaps: which specializations pay more")
    st.caption(
        "Compare how much each AI specialization pays against the typical pay for the chosen role. "
        "This uses the older illustrative dataset, where the specialization is a fixed label rather than "
        "something taken from job ad text, so treat the differences as rough."
    )
    role_gap = st.selectbox("Role to analyze", sorted(df["job_role"].unique()), key="gap_role")
    subset = df[df["job_role"] == role_gap]

    gap = subset.groupby("ai_specialization").agg(
        median_salary=("salary_usd", "median"),
        postings=("id", "count"),
        avg_hiring_difficulty=("hiring_difficulty_score", "mean"),
    ).reset_index().sort_values("median_salary", ascending=False)
    gap["salary_premium_vs_role_median"] = gap["median_salary"] - subset["salary_usd"].median()

    fig = px.bar(gap, x="ai_specialization", y="salary_premium_vs_role_median",
                color="salary_premium_vs_role_median", color_continuous_scale=[SLATE, LINE, GREEN],
                title=f"Pay compared with the {role_gap} median, by specialization")
    fig.update_layout(yaxis_title="USD vs role median")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Details by specialization**")
    st.dataframe(
        gap.rename(columns={
            "ai_specialization": "Specialization", "median_salary": "Median salary",
            "postings": "# postings", "avg_hiring_difficulty": "Avg hiring difficulty",
            "salary_premium_vs_role_median": "Premium vs role median",
        }).style.format({"Median salary": "${:,.0f}", "Premium vs role median": "${:,.0f}",
                          "Avg hiring difficulty": "{:.1f}"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown(
        "**How to read this:** a specialization that pays a premium and is also hard to hire for is one "
        "employers struggle to fill. Those may be worth learning."
    )

# ---------------------------------------------------------------- TAB 4 ----
with tab4:
    st.subheader("Demand tiers: how in-demand each specialization is")
    st.caption(
        "The tiers were set from 2025–2026 AI hiring research. LLM, Generative AI and MLOps are High "
        "(fastest-growing). NLP and Computer Vision are Medium (strong but more established). Analytics, "
        "Forecasting and Reinforcement Learning are Low (stable). The model learns to predict these tiers "
        "from the older dataset. Full sources are in the documentation at the bottom of the page."
    )

    tier_counts = df["demand_tier"].value_counts().reindex(TIER_ORDER).reset_index()
    tier_counts.columns = ["Tier", "Postings"]
    fig = px.bar(tier_counts, x="Tier", y="Postings", color="Tier",
                color_discrete_map=TIER_COLORS,
                title="Postings by demand tier")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Check the tier for a job profile**")
    c1, c2, c3 = st.columns(3)
    with c1:
        q_role = st.selectbox("Role", sorted(df["job_role"].unique()), key="q_role")
        q_spec = st.selectbox("Specialization", sorted(df["ai_specialization"].unique()), key="q_spec")
    with c2:
        q_country = st.selectbox("Country", sorted(df["country"].unique()), key="q_country")
        q_industry = st.selectbox("Industry", sorted(df["industry"].unique()), key="q_industry")
    with c3:
        q_exp = st.selectbox("Experience level", ["Entry", "Mid", "Senior", "Lead"], key="q_exp")
        q_size = st.selectbox("Company size", sorted(df["company_size"].unique()), key="q_size")

    qrow = pd.DataFrame([{
        "country": q_country, "job_role": q_role, "ai_specialization": q_spec,
        "experience_level": q_exp, "education_required": df["education_required"].mode()[0],
        "industry": q_industry, "company_size": q_size, "work_mode": df["work_mode"].mode()[0],
        "year": 2026,
        "job_openings": df["job_openings"].median(),
        "hiring_difficulty_score": df["hiring_difficulty_score"].median(),
        "ai_adoption_score": df["ai_adoption_score"].median(),
        "career_growth_score": df["career_growth_score"].median(),
        "company_funding_billion": df["company_funding_billion"].median(),
        "automation_risk": df["automation_risk"].median(),
    }])
    Xq = qrow[DEMAND_CAT + DEMAND_NUM].copy()
    Xq[DEMAND_CAT] = demand_encoder.transform(Xq[DEMAND_CAT])
    proba = demand_model.predict_proba(Xq)[0]
    pred_tier = TIER_ORDER[int(np.argmax(proba))]

    st.metric("Predicted demand tier", pred_tier)
    proba_df = pd.DataFrame({"Tier": TIER_ORDER, "Probability": proba})
    st.plotly_chart(px.bar(proba_df, x="Tier", y="Probability", color="Tier",
                           color_discrete_map=TIER_COLORS),
                     use_container_width=True)

st.markdown("---")
with st.expander("Documentation, data sources and limitations"):
    st.markdown((ROOT / "docs" / "README.md").read_text())
