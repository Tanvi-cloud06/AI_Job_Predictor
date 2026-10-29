# About this project

This dashboard helps you explore AI and machine learning jobs. It answers four questions:

- Which AI jobs are expected to grow the most?
- About how much do these jobs pay?
- Which skills show up most often in job ads?
- What should I learn next, based on my current skills and experience?

Some parts of the dashboard use real job data. Others use projections or an older illustrative dataset. Each tab says which is which.

## Quick guide to the tabs

| Tab | What it shows | Data used |
|---|---|---|
| Growing jobs | The AI jobs with the fastest projected growth in openings by 2030, plus a check against US government figures for Data Scientists | Market research projections and US BLS data |
| Career planner | Which skill to learn next, and which experience level would raise pay the most, for a profile you enter | Real LinkedIn job ads |
| Salary estimator | A likely pay range for a job you describe, and which factors drive the estimate | Real LinkedIn job ads |
| Pay by country | Median pay and job openings by country and role | Older illustrative dataset |
| Skill gaps | How much each AI specialization pays compared with the typical pay for a role | Older illustrative dataset |
| In-demand skills | Demand tiers (High, Medium, Low) and a tool that checks the tier for a job profile | Older illustrative dataset plus market research |

## Where the data comes from

**1. Global AI and data jobs (Kaggle)**
- About 90,000 job postings from 2020 to 2026.
- 12 countries, 8 AI and ML job roles, 8 AI specializations.
- Used for the Growing jobs, Pay by country, Skill gaps and In-demand skills tabs.
- The data is structured (no job ad text), so the specialization is a label, not a skill found in the text.

**2. LinkedIn job postings, 2023–2024 (Kaggle, by arshkon)**
- 123,849 postings in total.
- 2,376 of them are tech, AI or data jobs that list a salary.
- 1,891 of those have a clear experience level, and these are used to train the salary model.
- Almost all postings are from April 2024, so this is a snapshot of one month, not a multi-year trend.
- Skills were found by searching the text of each job ad for known keywords such as Python, SQL, AWS and TensorFlow.
- Used for the Salary estimator and Career planner tabs.

**3. US Bureau of Labor Statistics (BLS), Occupational Employment and Wage Statistics**
- Yearly figures for Data Scientists in the US (SOC code 15-2051).

| Year | Employed | Median yearly wage |
|---|---|---|
| 2023 | 192,710 | $108,020 |
| 2024 | 245,900 | $112,590 |
| 2025 | 262,440 | $120,224 |

- The 2026 to 2030 values on the Growing jobs tab extend a straight line through these three years. They are a simple extrapolation, not an official forecast.

## How the salary estimator works

- The model is a gradient-boosted decision tree. It is a common method that combines many small decision trees into one prediction.
- Inputs: experience level, US state, whether remote work is allowed, and which skills you have.
- Output: a middle estimate, shown as a range (middle estimate plus or minus the typical error).

Key results from the saved model:

| Measure | Value | What it means |
|---|---|---|
| R² | 0.24 | The model explains about a quarter of the differences in pay between job ads. A perfect model would score 1.0. |
| Typical error | about $34,800 | On average, the estimate is this far from the actual listed salary on job ads the model did not see during testing. |
| Training and test split | 1,512 and 379 job ads | The model learned from 1,512 ads and was checked on the other 379. |

The most important factors, in order: experience level (about 28% of the model's weight), US state (about 11%), number of skills listed (about 11%), Python (about 10%) and machine learning (about 7%). Skills together make up about 48% of the weight.

**How to use it:** the exact dollar figures are rough. The ranking of which skill matters most is more reliable than the dollar amount next to it.

## How the demand tiers work

The demand tiers are High, Medium and Low. They were set using AI hiring research, not learned from the data:

- **High:** LLM, Generative AI and MLOps. These are expected to grow fastest.
- **Medium:** NLP and Computer Vision. Strong demand, but more established.
- **Low:** Analytics, Forecasting and Reinforcement Learning. Stable, slower growth.

Each job posting's tier was then adjusted up or down by one level based on its hiring difficulty, career growth and job openings, so the model has something to learn. The demand model predicts the tier with 98.45% accuracy on test data. Because the tiers were built from the same kind of rules the model learns, this number shows how well the model follows that rule. It does not show that the tiers match real-world demand.

Class sizes in the dataset: High 32,185, Low 31,969, Medium 25,846.

## How the growth projections work

The main dataset has no real year-to-year trend. A straight line through 2020 to 2026 explains very little for most jobs (median R² of about 0.09). The dashboard shows this with a grey line that is almost flat.

The green line is a projection. It applies a yearly growth rate based on each specialization's demand tier:

| Tier | Yearly salary growth | Yearly growth in job openings |
|---|---|---|
| High (LLM, Generative AI, MLOps) | about 7% | about 22% |
| Medium (NLP, Computer Vision) | about 4% | about 10% |
| Low (Analytics, Forecasting, Reinforcement Learning) | about 1.5% | about 3% |

Within each tier, each job's own past openings trend nudges its growth rate slightly, by at most 5 percentage points a year for openings and 2 points a year for salaries. This stops the ranking from showing the three tiers only, and it never moves a Low-tier job above a High-tier job.

The salary growth rates are set below the growth rates reported for job postings. Pay usually rises more slowly than the number of postings, so the salary assumptions are more conservative.

## Limitations

- **The salary model is only moderately accurate.** It explains about a quarter of pay differences, and its typical error is about $35,000. Use the range, not the single number.
- **The LinkedIn salary data is a single-month snapshot.** It covers US job ads from April 2024 only. It may not reflect current pay.
- **The LinkedIn sample is small.** Only 1,891 ads were usable for training.
- **Skills are found by keyword.** A skill written in a different way, or not on the keyword list, will be missed.
- **The main dataset looks partly random.** Apart from salary, most of its numeric columns have very weak links to each other. Charts that use it are illustrative, meaning they show how the method works rather than real market conditions.
- **Demand tiers come from research opinion.** They are not discovered from the data.
- **Growth rates are assumptions.** The projections on the Growing jobs tab are estimates of the future, not guarantees.
- **Pay is in US dollars.** Figures from other countries are converted to a common currency in the source data. No adjustment for local purchasing power is made beyond the cost-of-living figure in the dataset.

An earlier salary model was trained on the main Kaggle dataset and scored R² of 0.93. That score came from patterns built into the data, not from real-world signal, so the app no longer uses that model for predictions. It is kept in the project for reference only.

## Sources

Cited by the original project. Please check them before relying on any figure:

- Kaggle: "Global AI & Data Jobs Salary Dataset" (mohankrishnathalla)
- Kaggle: "LinkedIn Job Postings (2023-2024)" (arshkon)
- US Bureau of Labor Statistics, Occupational Employment and Wage Statistics: bls.gov/oes (May 2023 estimates, and the 2024 and 2025 figures from bls.gov/ooh and bls.gov/news.release/ocwage.t01.htm)
- LinkedIn Workforce Report (2025 and 2026 editions)
- PwC Global AI Jobs Barometer
- Market analyses from Second Talent, HeroHunt.ai and Acceler8 Talent (2026)

## Running the dashboard

Install the Python packages:

```bash
pip install streamlit pandas numpy plotly scikit-learn joblib
```

Start the dashboard from the project folder:

```bash
streamlit run app/app.py
```

The trained models are included. If they fail to load on your computer (this can happen when numpy or scikit-learn versions differ), the app retrains them automatically the first time. This takes about 30 seconds and happens only once.

To rebuild everything from the raw data:

```bash
python src/data_pipeline.py                 # clean the main dataset
python src/train_demand_model.py            # train the demand tier model
python src/train_salary_model_linkedin.py   # train the salary model on LinkedIn data
python src/forecast_trends.py               # build the trend projections
python src/forecast_bls_trend.py            # build the BLS trend for Data Scientists
```

## Project files

```
app/app.py                                 the dashboard
src/data_pipeline.py                       cleans the main Kaggle dataset
src/linkedin_pipeline.py                   extracts skills and salaries from LinkedIn job ads
src/train_demand_model.py                  trains the demand tier model
src/train_salary_model_linkedin.py         trains the salary model
src/forecast_trends.py                     builds the growth projections
src/forecast_bls_trend.py                  builds the BLS trend for Data Scientists
data/raw/                                  original data files
data/processed/                            cleaned data and forecasts used by the app
models/                                    trained models and their scores
docs/README.md                             this page
```
