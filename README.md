# 📱 Used Mobile Phone Price Predictor — Sri Lanka

> **ML Assignment 3 · University of Moratuwa · 2026**

Predicts the resale price (LKR) of used smartphones in Sri Lanka using a **LightGBM** gradient boosting model trained on data from ikman.lk, with **SHAP** explainability and a **Streamlit** front-end.

---

## 📁 Project Structure

```
Assignment3/
├── data/
│   ├── raw_listings.csv      ← scraped/generated listings
│   └── cleaned_data.csv      ← preprocessed features
├── models/
│   ├── lgbm_model.pkl        ← trained LightGBM model
│   ├── feature_cols.pkl      ← ordered feature list
│   ├── te_brand.pkl          ← brand target encoder
│   └── te_district.pkl       ← district target encoder
├── plots/                    ← all output charts
├── scraping.py               ← Selenium scraper (ikman.lk)
├── generate_dataset.py       ← synthetic data fallback
├── preprocessing.py          ← cleaning & feature engineering
├── train_model.py            ← LightGBM + Optuna tuning
├── explainability.py         ← SHAP, PDP, feature importance
├── app.py                    ← Streamlit web application
└── requirements.txt
```

---

## ⚡ Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Collect data

**Option A — Real scraping (requires Chrome)**
```bash
python scraping.py
```

**Option B — Synthetic data (if scraping is blocked)**
```bash
python generate_dataset.py
```

### Step 3 — Preprocess
```bash
python preprocessing.py
```

### Step 4 — Train model
```bash
python train_model.py
```

### Step 5 — Generate XAI plots
```bash
python explainability.py
```

### Step 6 — Launch web app
```bash
streamlit run app.py
```

---

## 🤖 Algorithm: LightGBM

| Property | Details |
|---|---|
| Type | Gradient Boosted Decision Trees |
| Growth strategy | Leaf-wise (not level-wise) |
| Tuning method | Optuna Bayesian search (50 trials) |
| Target transform | `log1p(price)` → reduces skew |
| CV strategy | 5-fold stratified by price quantile |

---

## 📊 Explainability Methods

| Method | Tool | Purpose |
|---|---|---|
| SHAP Beeswarm | `shap.TreeExplainer` | Global feature importance & direction |
| SHAP Waterfall | `shap.Explanation` | Per-prediction explanation |
| PDP | `sklearn.inspection` | Marginal effect per feature |
| Gain Importance | LightGBM built-in | Overall feature ranking |

---

## 📋 Features Used

| Feature | Type | Description |
|---|---|---|
| `storage` | Numeric | Internal storage in GB |
| `ram` | Numeric | RAM in GB |
| `warranty` | Binary | 1=warranty available |
| `days_since_posted` | Numeric | Recency of listing |
| `brand_enc` | Target-encoded | Brand name (target encoding) |
| `district_enc` | Target-encoded | Sri Lankan district |
| `cond_New/Like New/Used` | One-hot | Listing condition |

---

## ⚖️ Ethics & Data

- Data collected from public listings only
- No personal or user-identifiable information
- `scraping.py` uses 2–5 second delays to avoid server load
- Synthetic dataset clearly documented in report
