# 🧭 WorkBalance AI
### AI insights into work stress and work-addiction patterns across 85 cultures

A data science project on a 31,352-respondent, 85-culture survey measuring work-addiction
behaviors (16-item scale), job stress, job satisfaction, and self-esteem.

Instead of manufacturing a "burnout score" the data was never designed to support, this
project **discovers** data-driven work-behavior profiles with clustering, then trains a
neural network to classify new respondents into those profiles in real time — deployed as
an interactive Streamlit app.

## Environment

This project is set up for **Miniconda + Jupyter Notebook, Python 3.13**, with data files at:

```
C:\Users\yassin\Desktop\Bootcamp\Final Project\Data\Data.csv
C:\Users\yassin\Desktop\Bootcamp\Final Project\Data\Codebook.csv
```

Set up the environment once:

```bash
conda create -n myenv13 python=3.13
conda activate myenv13
pip install -r requirements.txt
python -m ipykernel install --user --name myenv13 --display-name "Python 3.13 (myenv13)"
```

(Or use `conda env create -f environment.yml` instead of the three commands above.)

## What's inside

```
WorkBalanceAI_Analysis.ipynb   ← full analysis notebook (EDA, PCA, K-Means, models, SHAP)
app.py                         ← Streamlit app (3 pages + about)
requirements.txt
environment.yml                ← optional one-shot conda environment file
Data.csv, Codebook.csv         ← a copy of the original data (put your own in Data\ next to
                                   the notebook per the paths above — these are just for reference)
models/                        ← trained artifacts used by the Streamlit app
app_data/                      ← employees_sample.csv, country_summary.csv used by the app
```

> **Why `app_data/` and not `data/`?** Windows filesystems are case-insensitive, so a
> folder named `data` would collide with the `Data` folder that holds the original survey
> files. The notebook and `app.py` both use `app_data/` for the processed CSVs the app
> reads, to avoid that collision.

## The four profiles (discovered, not assumed)

| Profile | Meaning |
|---|---|
| 🟢 Healthy & Balanced | Low work addiction, manageable stress, good satisfaction |
| 🟡 Engaged but Strained | Moderate involvement with some strain |
| 🟠 Overworked | High stress and workload, satisfaction dropping |
| 🔴 Work-Addicted | High compulsive-work behaviors, high stress, low satisfaction |

## Run the notebook (Jupyter Notebook, local)

1. Make sure `Data.csv` and `Codebook.csv` are at
   `C:\Users\yassin\Desktop\Bootcamp\Final Project\Data\` (create the `Data` folder if it
   doesn't exist yet). If you keep your files somewhere else, just edit the `PROJECT_ROOT`
   line in the notebook's **Setup** cell.
2. Activate the `myenv13` conda environment (see above) and launch Jupyter:
   ```bash
   conda activate myenv13
   jupyter notebook
   ```
3. Open `myenv13_Analysis.ipynb`, select the **Python 3.13 (myenv13)** kernel,
   and run all cells top to bottom. It will (re)generate everything in `models/` and
   `app_data/` next to the notebook.

## Run the Streamlit app

The app already ships with pre-trained artifacts in `models/` and `app_data/`, so you can
run it immediately without re-running the notebook:

```bash
conda activate myenv13
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

### App pages
- **🧠 Profile Predictor** — answer the 19 questions, get your profile + confidence + a
  SHAP explanation of what drove the result.
- **🌍 Global Work Map** — choropleth + country lookup, always shown with sample size.
- **🧩 Profile Explorer** — PCA scatter of respondents, filterable by country/profile; if
  you've used the Predictor, your own point shows up as a ⭐.
- **ℹ️ About** — methodology summary and honest limitations.

