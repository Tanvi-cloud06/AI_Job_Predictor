# AI Job Market Trend Predictor

A dashboard for exploring AI and machine learning jobs: which ones are growing,
what they pay, and which skills employers ask for.

The full guide, including where the data comes from and what the numbers mean,
is in [docs/README.md](docs/README.md). It is also shown at the bottom of the dashboard.

## Run it on your computer

1. Install Python 3.10 or newer.
2. Install the required packages:

   ```bash
   pip install streamlit pandas numpy plotly scikit-learn joblib
   ```

3. Start the dashboard from this folder:

   ```bash
   streamlit run app/app.py
   ```

4. Your browser opens the dashboard. If it does not, go to http://localhost:8501.

The trained models are included, so there is nothing else to set up. If a model
fails to load because of a version mismatch on your computer, the app retrains it
automatically the first time (about 30 seconds).

## Put it online for free

[Streamlit Community Cloud](https://streamlit.io/cloud) can host the dashboard
for free:

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, choose **New app**, select the repository, and set the main file to `app/app.py`.
3. Click **Deploy**.

## Folder layout

```
app/          the dashboard
src/          scripts that clean the data and train the models
data/raw/     original data files
data/processed/  cleaned data used by the dashboard
models/       trained models and their scores
docs/         the full guide
```

## Results at a glance

| Model | What it does | Score |
|---|---|---|
| Salary model (LinkedIn job ads) | Estimates pay from experience, location and skills | R² = 0.24, typical error about $35,000 |
| Demand tier model | Predicts High, Medium or Low demand for a job profile | 98.45% accuracy on test data |

The salary model is only moderately accurate, so the dashboard shows pay as a range.
The demand tier model learns tiers that were set from research, so its accuracy shows
how well it follows those rules, not how well it predicts the real market.
