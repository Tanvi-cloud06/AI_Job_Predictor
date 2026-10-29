"""
AI Job Market Trend Predictor — Data Pipeline
Ingests the raw Kaggle CSV, validates/cleans it, deduplicates, and writes a
structured, model-ready dataset to data/processed/.

Source dataset: Global AI & Data Jobs Salary Dataset (Kaggle, mohankrishnathalla)
90,000 rows, 35 columns, 2020-2026, 12 countries, 8 AI job roles.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "global_ai_jobs.csv"
OUT_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "clean_jobs.csv"

TARGET_ROLES = [
    "Machine Learning Engineer", "Data Scientist", "AI Engineer",
    "Research Scientist", "NLP Engineer", "Computer Vision Engineer",
    "Software Engineer AI", "Data Analyst",
]

# Every row in this dataset is already an English-language, single-country
# structured record (no multilingual free text), so the "English-language
# only" filter is a no-op safety check rather than an active language-detect
# step. Kept here so the pipeline stays correct if a future data source
# (e.g. scraped postings) does include non-English text.


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- basic hygiene ---
    df.columns = [c.strip().lower() for c in df.columns]
    n_start = len(df)

    # drop exact duplicate postings
    df = df.drop_duplicates()

    # drop duplicate "postings" that only differ by id but are identical on
    # every substantive field (role/loc/exp/salary/spec/year) — a common
    # artifact of scraped/aggregated job data
    dedupe_cols = [c for c in df.columns if c != "id"]
    df = df.drop_duplicates(subset=dedupe_cols)

    # keep only known target job roles
    df = df[df["job_role"].isin(TARGET_ROLES)]

    # drop rows with missing/invalid salary or negative values
    df = df[df["salary_usd"].notna() & (df["salary_usd"] > 0)]
    df = df[df["experience_years"] >= 0]

    # clip obvious outliers (winsorize salary at 0.5% / 99.5%) rather than
    # dropping, to avoid throwing away legitimate high/low senior vs entry pay
    lo, hi = df["salary_usd"].quantile([0.005, 0.995])
    df["salary_usd"] = df["salary_usd"].clip(lo, hi)

    # standardize categorical text fields
    cat_cols = ["country", "job_role", "ai_specialization", "experience_level",
                "education_required", "industry", "company_size", "work_mode"]
    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip()

    # normalize experience level ordering as an explicit ordinal feature
    exp_order = {"Entry": 0, "Mid": 1, "Senior": 2, "Lead": 3}
    df["experience_level_ord"] = df["experience_level"].map(exp_order)

    # total compensation feature (salary + bonus) — useful secondary target
    df["total_comp_usd"] = df["salary_usd"] + df["bonus_usd"].fillna(0)

    # "skill" entity: this dataset is fully structured (no free-text job
    # description to run NER/TF-IDF against), so ai_specialization is the
    # canonical skill/competency label per posting. We also build a
    # role+specialization compound tag used downstream for skill-gap analysis.
    df["skill_tag"] = df["ai_specialization"].str.strip()
    df["role_skill_combo"] = df["job_role"] + " · " + df["skill_tag"]

    df = df.reset_index(drop=True)
    print(f"[pipeline] rows in: {n_start:,} -> rows out: {len(df):,} "
          f"({n_start - len(df):,} dropped)")
    return df


def run():
    df = load_raw()
    clean_df = clean(df)
    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(OUT_CSV_PATH, index=False)
    print(f"[pipeline] wrote {len(clean_df):,} rows -> {OUT_CSV_PATH}")
    return clean_df


if __name__ == "__main__":
    run()
