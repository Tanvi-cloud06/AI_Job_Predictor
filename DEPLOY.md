# Deploying this app for free

**Streamlit Community Cloud** is the right fit here — it's built specifically
for Streamlit apps, it's free with no card required, and this repo already
has everything it needs (`requirements.txt`, `.streamlit/config.toml` for
the theme, self-healing model loading). Total repo size is well under its
limits, so no Git LFS needed.

## Steps

1. **Push this project to a public GitHub repo.**
   ```bash
   cd ai_job_predictor
   git init
   git add .
   git commit -m "AI job market predictor"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
   (Create the empty repo on GitHub first if you haven't.)

2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in
   with GitHub.

3. Click **"Create app"** → **"Yup, I have an app"** → fill in:
   - Repository: `<your-username>/<repo-name>`
   - Branch: `main`
   - Main file path: `app/app.py`

4. (Optional) Under **"Advanced settings"**, pick a Python version matching
   what you tested locally (3.10+).

5. Click **Deploy**. First build takes a few minutes — it installs
   `requirements.txt`, and on first load `app.py` will retrain any model
   artifact that doesn't unpickle cleanly in that environment (the
   self-healing fallback described in `README.md`), which adds ~30–60s
   once.

Your app is live at `https://<your-chosen-subdomain>.streamlit.app`.
Every push to `main` redeploys it automatically.

## Free-tier limits worth knowing

- **~1 GB RAM.** This app's data + model loading fits comfortably, but if
  you add more datasets, watch memory.
- **Apps sleep after ~12 hours idle** and wake on the next visit (takes a
  few seconds).
- **Unlimited public apps, one private app** per account.
- No custom domain on the free tier — you get `*.streamlit.app`.

## If you outgrow the free tier

- **Render / Railway** — both already have config in this repo
  (`render.yaml`, `railway.json`, `Dockerfile`); push the same repo and
  connect it there. No sleep, but typically not free past a trial/starter
  credit.
- **Docker on any VPS** — `docker build -t ai-job-predictor . && docker run
  -p 8501:8501 ai-job-predictor` works anywhere Docker runs.
