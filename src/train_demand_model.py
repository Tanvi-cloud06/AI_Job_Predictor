"""
Skill Demand Tier Classifier
Tags postings' AI specialization into High / Medium / Low growth demand tiers.

IMPORTANT METHODOLOGY NOTE (see docs/README.md "Limitations" for full
discussion): the raw Kaggle dataset's `skill_demand_score` column was found
during EDA to be statistically independent of every other field (near-zero
correlation with job_openings, ai_adoption_score, career_growth_score,
salary, etc. -- consistent with it being randomly generated rather than a
genuine market signal). Rather than train a classifier on that noise column
(which would be unlearnable and would silently produce a useless model), we
construct the ground-truth demand tier from documented, current (2026)
industry research on AI hiring trends -- LLM/Generative AI and MLOps are
consistently reported as the fastest-growing AI specializations, NLP and
Computer Vision as strong-but-more-mature, and Analytics/Forecasting/
Reinforcement Learning as stable/slower-growth (sources cited in docs).
A small amount of context-dependent variation (driven by real per-posting
signals: job openings, hiring difficulty, career growth score, industry) is
layered on top so the label isn't a pure lookup table and the classifier has
a genuine (if easier than average) learning task, evaluated on a held-out
test set in the normal way.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import OrdinalEncoder

try:
    from xgboost import XGBClassifier
    BACKEND = "xgboost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    BACKEND = "sklearn-hgb"

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "clean_jobs.csv"
MODEL_PATH = ROOT / "models" / "demand_model.joblib"
ENCODER_PATH = ROOT / "models" / "demand_encoders.joblib"
METRICS_PATH = ROOT / "models" / "demand_model_metrics.json"
LABELED_DATA_PATH = ROOT / "data" / "processed" / "clean_jobs_with_demand_tier.csv"

# Base tier grounded in 2026 AI hiring trend reporting (LinkedIn Workforce
# Report, PwC Global AI Jobs Barometer, Second Talent / HeroHunt / Acceler8
# Talent market analyses -- see docs/README.md for citations).
BASE_TIER = {
    "LLM": "High",
    "Generative AI": "High",
    "MLOps": "High",
    "NLP": "Medium",
    "Computer Vision": "Medium",
    "Analytics": "Low",
    "Forecasting": "Low",
    "Reinforcement Learning": "Low",
}
TIER_ORDER = ["Low", "Medium", "High"]
TIER_TO_INT = {t: i for i, t in enumerate(TIER_ORDER)}

CATEGORICAL_FEATURES = [
    "country", "job_role", "ai_specialization", "experience_level",
    "education_required", "industry", "company_size", "work_mode",
]
NUMERIC_FEATURES = [
    "year", "job_openings", "hiring_difficulty_score", "ai_adoption_score",
    "career_growth_score", "company_funding_billion", "automation_risk",
]
TARGET = "demand_tier"


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    base = df["ai_specialization"].map(BASE_TIER).map(TIER_TO_INT)

    # context-driven nudge: postings with unusually high hiring difficulty,
    # career-growth score, and job openings for their row get nudged a tier
    # up; unusually low get nudged down. Keeps the label from being a pure
    # specialization lookup while remaining grounded in real per-row signal.
    z = lambda s: (s - s.mean()) / s.std()
    momentum = (
        0.4 * z(df["hiring_difficulty_score"])
        + 0.3 * z(df["career_growth_score"])
        + 0.3 * z(df["job_openings"])
    )
    nudge = pd.cut(momentum, bins=[-np.inf, -0.6, 0.6, np.inf], labels=[-1, 0, 1]).astype(int)

    tier_int = (base + nudge).clip(0, 2)
    df[TARGET] = tier_int.map({v: k for k, v in TIER_TO_INT.items()})
    return df


def build_model():
    if BACKEND == "xgboost":
        return XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            num_class=3,
            random_state=42,
            n_jobs=-1,
        )
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=400, max_depth=5, learning_rate=0.08, random_state=42
        )


def run():
    df = pd.read_csv(DATA_PATH)
    df = build_labels(df)
    df.to_csv(LABELED_DATA_PATH, index=False)

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    y = df[TARGET].map(TIER_TO_INT)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[CATEGORICAL_FEATURES] = encoder.fit_transform(X[CATEGORICAL_FEATURES])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    f1_weighted = f1_score(y_test, preds, average="weighted")
    report = classification_report(
        y_test, preds, target_names=TIER_ORDER, output_dict=True
    )

    metrics = {
        "backend": BACKEND,
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_report": report,
        "class_distribution": df[TARGET].value_counts().to_dict(),
    }
    print("[demand model]", json.dumps(
        {k: v for k, v in metrics.items() if k != "class_report"}, indent=2
    ))
    print(classification_report(y_test, preds, target_names=TIER_ORDER))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[demand model] saved -> {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    run()
