"""
Salary model trained on the real LinkedIn cross-section
(data/processed/linkedin_ai_jobs_clean.csv).

This is the direct test of the question that motivated switching datasets:
does skill data carry real salary signal once it comes from genuine posting
text instead of a pre-baked category column? Reported honestly either way.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from linkedin_pipeline import SKILL_KEYWORDS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "linkedin_ai_jobs_clean.csv"
MODEL_DIR = ROOT / "models"
EXP_LEVEL_ORDER = {
    "Internship": 0, "Entry level": 1, "Associate": 2,
    "Mid-Senior level": 3, "Director": 4, "Executive": 5,
}


def run():
    df = pd.read_csv(DATA_PATH)
    df = df[df["experience_level_ord"].notna()].copy()

    skill_cols = [c for c in df.columns if c.startswith("skill_")]

    # top-15 state buckets, rest as "Other" (avoid one-hot explosion on noisy geo)
    top_states = df["state"].value_counts().head(15).index
    df["state_grouped"] = df["state"].where(df["state"].isin(top_states), "Other")
    state_enc = LabelEncoder()
    df["state_code"] = state_enc.fit_transform(df["state_grouped"])

    feature_cols = ["experience_level_ord", "remote_allowed", "n_skills",
                     "state_code"] + skill_cols
    X = df[feature_cols]
    y = df["normalized_salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    # feature importance -- the actual test of "do real skills matter"
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    skill_importance_share = importances[skill_cols].sum()

    metrics = {
        "n_rows_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "r2": round(float(r2), 4),
        "mae_usd": round(float(mae), 2),
        "top_10_features_by_importance": importances.head(10).round(4).to_dict(),
        "total_importance_share_from_skill_features": round(float(skill_importance_share), 4),
        "caveat": (
            f"Trained on {len(df):,} real LinkedIn postings (April 2024 "
            "snapshot) with disclosed salary and a tech/AI/data-relevant "
            "title. This is a genuine but small sample -- read R^2 here as "
            "indicative, not as strong as a model trained on tens of "
            "thousands of rows would be."
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "linkedin_salary_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # persist everything app.py needs to run inference on a fresh profile:
    # the model itself, the state encoder + the exact top-15 state list used
    # at train time (so a new prediction buckets an unseen state into
    # "Other" the same way), the feature column order, and the full
    # (unsorted-by-value) importances so the UI can show a real breakdown.
    joblib.dump(model, MODEL_DIR / "linkedin_salary_model.joblib")
    joblib.dump(
        {
            "state_encoder": state_enc,
            "top_states": list(top_states),
            "feature_cols": feature_cols,
            "skill_cols": skill_cols,
            "skill_keywords": SKILL_KEYWORDS,
            "exp_level_order": EXP_LEVEL_ORDER,
            "importances": importances.round(4).to_dict(),
        },
        MODEL_DIR / "linkedin_salary_meta.joblib",
    )

    print(f"[linkedin salary model] n={len(df):,} rows, R^2={r2:.3f}, MAE=${mae:,.0f}")
    print(f"[linkedin salary model] combined importance share from real skill "
          f"features: {skill_importance_share:.1%}")
    print("[linkedin salary model] top 10 features by importance:")
    for feat, imp in importances.head(10).items():
        print(f"    {feat}: {imp:.3f}")

    return model, metrics


if __name__ == "__main__":
    run()
