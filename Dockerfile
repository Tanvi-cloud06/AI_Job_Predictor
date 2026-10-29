# AI Job Market Trend Predictor — deployment image (Render / Railway)
FROM python:3.11-slim

WORKDIR /app

# system deps for xgboost / scientific python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Retrain both models inside the image so the pickles match this image's
# numpy/scikit-learn versions. Data + processed CSVs already ship in the repo.
RUN python src/train_demand_model.py \
    && python src/train_salary_model_linkedin.py \
    && python src/forecast_trends.py

# Render/Railway inject $PORT at runtime; default to 8501 for local docker run.
ENV PORT=8501
EXPOSE 8501

CMD ["sh", "-c", "streamlit run app/app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]
