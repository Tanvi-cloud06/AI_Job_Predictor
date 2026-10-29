"""
LinkedIn Postings Pipeline (real per-posting skill/salary layer)
==================================================================
Source: Kaggle "LinkedIn Job Postings (2023-2024)" by arshkon
(https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

IMPORTANT HONEST NOTE ON DATE RANGE: despite the dataset's name, the actual
`original_listed_time` values in the uploaded file run from 2023-12-05 to
2024-04-20, with 99%+ of postings concentrated in April 2024 alone. This is
effectively a single-month snapshot, not a genuine 2023-2024 span. It is
therefore used here ONLY as a cross-sectional (skill/salary at a point in
time) source, never as a trend source. The real multi-year trend comes from
the separate BLS OEWS layer (see src/forecast_bls_trend.py).

What this file actually fixes vs. the original Kaggle salary dataset:
  - `description` is genuine free-text -> real keyword-based skill extraction
    (Python, SQL, AWS, TensorFlow, etc.) instead of a pre-baked category
    column.
  - `normalized_salary` is a real per-posting disclosed salary, not an
    averaged/synthetic figure.

What it does NOT fix:
  - Sample size after filtering to tech/data/AI roles + disclosed salary is
    small (~2,300-2,400 rows) -- real but thin. Metrics from any model
    trained on this should be read with that caveat, not treated as
    equivalent to a model trained on tens of thousands of rows.
  - The bundled `job_skills.csv` / `mappings/skills.csv` files are NOT
    fine-grained tech skills -- they're 35 broad LinkedIn job-function
    categories (e.g. "Engineering", "IT", "Science"). They are not used
    here for that reason; skills are instead mined directly from
    `description` text.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_POSTINGS = ROOT / "data" / "raw" / "linkedin" / "postings.csv"
OUT_PATH = ROOT / "data" / "processed" / "linkedin_ai_jobs_clean.csv"

INCLUDE_KW = (
    r"data scientist|machine learning|\bml engineer|\bai engineer|"
    r"artificial intelligence|deep learning|data analyst|data engineer|"
    r"\bnlp\b|computer vision|research scientist|software engineer|"
    r"software developer|\bdeveloper\b|information technology|"
    r"\bprogrammer\b|full stack|backend|back end|frontend|front end|"
    r"devops|cloud engineer|systems engineer|\bdatabase\b|"
    r"business intelligence|statistician|\bqa engineer\b|data architect|"
    r"\bit \b|\bit\b analyst|network engineer|security engineer"
)
EXCLUDE_KW = (
    r"behavior analyst|financial analyst|business analyst|policy analyst|"
    r"legal|nurse|clinical|hairdress|driver|warehouse|forklift"
)

# Real technical skills mined directly from posting text. Not exhaustive --
# a deliberately curated, checkable list rather than an open-ended NER model,
# so every "skill" in the output is auditable back to a literal keyword.
SKILL_KEYWORDS = [
    "python", "sql", "r programming", "java", "scala", "c++", "javascript",
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "tensorflow", "pytorch", "keras", "scikit-learn", "spark", "hadoop",
    "tableau", "power bi", "excel", "nlp", "natural language processing",
    "computer vision", "deep learning", "machine learning", "airflow",
    "snowflake", "databricks", "etl", "rest api", "git", "linux",
    "statistics", "a/b testing", "agile", "scrum",
]


def extract_skills(text: str) -> list:
    if not isinstance(text, str):
        return []
    t = text.lower()
    return [kw for kw in SKILL_KEYWORDS if kw in t]


def extract_state(location: str) -> str:
    if not isinstance(location, str):
        return "Unknown"
    parts = [p.strip() for p in location.split(",")]
    return parts[-1] if len(parts) > 1 else "Unknown"


def run(raw_path: Path = RAW_POSTINGS, out_path: Path = OUT_PATH):
    usecols = [
        "job_id", "title", "description", "normalized_salary", "pay_period",
        "formatted_experience_level", "location", "remote_allowed",
    ]
    chunks = pd.read_csv(raw_path, usecols=usecols, chunksize=200_000, low_memory=False)

    kept = []
    n_total = 0
    for chunk in chunks:
        n_total += len(chunk)
        mask = (
            chunk["title"].str.contains(INCLUDE_KW, case=False, na=False, regex=True)
            & ~chunk["title"].str.contains(EXCLUDE_KW, case=False, na=False, regex=True)
            & chunk["normalized_salary"].notna()
        )
        kept.append(chunk[mask])

    df = pd.concat(kept, ignore_index=True)
    print(f"[linkedin pipeline] scanned {n_total:,} postings -> "
          f"{len(df):,} tech/AI/data postings with disclosed salary")

    # winsorize extreme salary entries (real data has some clear entry errors,
    # e.g. $535M "annual salary" outliers)
    lo, hi = df["normalized_salary"].quantile([0.01, 0.99])
    df["normalized_salary"] = df["normalized_salary"].clip(lo, hi)

    df["skills"] = df["description"].apply(extract_skills)
    df["n_skills"] = df["skills"].apply(len)
    df["state"] = df["location"].apply(extract_state)
    df["remote_allowed"] = df["remote_allowed"].fillna(0).astype(int)

    exp_order = {
        "Internship": 0, "Entry level": 1, "Associate": 2,
        "Mid-Senior level": 3, "Director": 4, "Executive": 5,
    }
    df["experience_level_ord"] = df["formatted_experience_level"].map(exp_order)

    # one-hot flags for each tracked skill (model-ready)
    for kw in SKILL_KEYWORDS:
        col = "skill_" + re.sub(r"[^a-z0-9]+", "_", kw)
        df[col] = df["skills"].apply(lambda s, kw=kw: int(kw in s))

    df = df.drop(columns=["description"])  # keep output lean; skills already extracted
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[linkedin pipeline] wrote {len(df):,} rows -> {out_path}")
    print(f"[linkedin pipeline] median disclosed salary: ${df['normalized_salary'].median():,.0f}, "
          f"median skills tagged per posting: {df['n_skills'].median():.0f}")
    return df


if __name__ == "__main__":
    run()
