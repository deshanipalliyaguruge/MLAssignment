"""
app.py
------
Streamlit front-end for the Used Mobile Phone Price Predictor.

Features:
  - User enters phone specs via sidebar controls
  - Model predicts resale price in LKR
  - SHAP waterfall plot explains which features drove that specific prediction
  - Confidence interval displayed based on model variance

Run:
    streamlit run app.py
"""

import os
import logging
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📱 Sri Lanka Used Phone Price Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #a8b2c1; font-size: 1rem; margin-top: 0.5rem; }

    .price-card {
        background: linear-gradient(135deg, #0f3460, #e94560);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(233,69,96,0.3);
    }
    .price-card .label { font-size: 0.9rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 1px; }
    .price-card .value { font-size: 2.8rem; font-weight: 700; margin: 0.2rem 0; }
    .price-card .sub   { font-size: 0.85rem; opacity: 0.7; }

    .info-box {
        background: rgba(15, 52, 96, 0.15);
        border-left: 4px solid #e94560;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #e94560, #c62a47);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(233,69,96,0.4); }
</style>
""", unsafe_allow_html=True)


# ── Load Model and Artifacts ───────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(__file__)
MODEL_PATH        = os.path.join(BASE_DIR, "models", "lgbm_model.pkl")
FEATURES_PATH     = os.path.join(BASE_DIR, "models", "feature_cols.pkl")
TE_BRAND_PATH     = os.path.join(BASE_DIR, "models", "te_brand.pkl")
TE_MODEL_PATH     = os.path.join(BASE_DIR, "models", "te_model.pkl")
TE_DISTRICT_PATH  = os.path.join(BASE_DIR, "models", "te_district.pkl")
CLEAN_CSV         = os.path.join(BASE_DIR, "data", "cleaned_data.csv")

@st.cache_resource
def load_model():
    model        = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    te_brand     = joblib.load(TE_BRAND_PATH)
    te_model     = joblib.load(TE_MODEL_PATH)
    te_district  = joblib.load(TE_DISTRICT_PATH)
    df_ref       = pd.read_csv(CLEAN_CSV)
    explainer    = shap.TreeExplainer(model)
    return model, feature_cols, te_brand, te_model, te_district, df_ref, explainer


FEATURE_LABELS = {
    "storage":           "Storage (GB)",
    "ram":               "RAM (GB)",
    "warranty":          "Warranty",
    "days_since_posted": "Days Since Posted",
    "brand_enc":         "Brand",
    "model_enc":         "Model",
    "district_enc":      "District",
    "cond_Like New":     "Condition: Like New",
    "cond_New":          "Condition: New",
    "cond_Used":         "Condition: Used",
}

BRANDS = [
    "Samsung", "Apple", "Huawei", "Xiaomi", "Oppo", "Vivo",
    "Realme", "Nokia", "Motorola", "Sony", "Oneplus", "Redmi", "Other"
]

BRAND_MODELS = {
    "Apple":    ["iPhone 11", "iPhone 12", "iPhone 12 Pro", "iPhone 13",
                 "iPhone 13 Pro", "iPhone 14", "iPhone 14 Pro", "iPhone SE",
                 "iPhone X", "iPhone XR", "iPhone XS Max", "iPhone 15"],
    "Samsung":  ["Galaxy A12", "Galaxy A32", "Galaxy A52", "Galaxy A53",
                 "Galaxy A72", "Galaxy S21", "Galaxy S22", "Galaxy S23",
                 "Galaxy A23", "Galaxy A13", "Galaxy M12", "Galaxy M32"],
    "Xiaomi":   ["Redmi Note 10", "Redmi Note 11", "Redmi 10C", "Poco X3",
                 "Poco M4 Pro", "Mi 11 Lite", "Redmi Note 12", "Redmi 12C"],
    "Redmi":    ["Redmi 9A", "Redmi 10", "Redmi Note 9", "Redmi 9C", "Redmi 12"],
    "Oppo":     ["Oppo A54", "Oppo A74", "Oppo A92", "Oppo Reno 5",
                 "Oppo A16", "Oppo A57", "Oppo F19"],
    "Vivo":     ["Vivo Y20", "Vivo Y21s", "Vivo Y33s", "Vivo Y53s",
                 "Vivo Y75", "Vivo V21"],
    "Realme":   ["Realme 8", "Realme 9", "Realme C25", "Realme C30",
                 "Realme Narzo 50", "Realme GT Neo 2"],
    "Huawei":   ["Huawei P30", "Huawei P30 Pro", "Huawei Y7a", "Huawei Nova 5T",
                 "Huawei P40 Lite", "Huawei Mate 30"],
    "Nokia":    ["Nokia G20", "Nokia G21", "Nokia 5.4", "Nokia 3.4", "Nokia G10"],
    "Motorola": ["Moto G30", "Moto G60", "Moto G62", "Moto E40"],
    "Sony":     ["Sony Xperia 10", "Sony Xperia 1 III", "Sony Xperia 5 III"],
    "Oneplus":  ["OnePlus 9", "OnePlus 9R", "OnePlus Nord CE", "OnePlus 10 Pro"],
    "Other":    ["Tecno Pop 5", "Infinix Hot 12", "Itel A49", "Tecno Spark 8"],
}

DISTRICTS = [
    "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "Nuwara Eliya",
    "Galle", "Matara", "Hambantota", "Jaffna", "Kilinochchi", "Mannar",
    "Vavuniya", "Mullaitivu", "Batticaloa", "Ampara", "Trincomalee",
    "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa", "Badulla",
    "Monaragala", "Ratnapura", "Kegalle", "Unknown"
]
CONDITIONS  = ["Used", "Like New", "New"]
STORAGE_OPT = [8, 16, 32, 64, 128, 256, 512]
RAM_OPT     = [1, 2, 3, 4, 6, 8, 12, 16]


# ── Build Feature Vector ───────────────────────────────────────────────────────
def build_input(brand, phone_model, district, condition, storage, ram, warranty,
                days_since, feature_cols, te_brand, te_model, te_district) -> pd.DataFrame:
    """Construct a single-row DataFrame matching the training feature schema."""
    row = {c: 0.0 for c in feature_cols}
    # Use transform (not fit_transform) on a temp DF
    tmp_brand    = pd.DataFrame({"brand": [brand]})
    tmp_model    = pd.DataFrame({"model": [phone_model]})
    tmp_district = pd.DataFrame({"district": [district]})
    row["brand_enc"]    = float(te_brand.transform(tmp_brand)["brand"].iloc[0])
    row["model_enc"]    = float(te_model.transform(tmp_model)["model"].iloc[0])
    row["district_enc"] = float(te_district.transform(tmp_district)["district"].iloc[0])
    row["storage"]           = float(storage)
    row["ram"]               = float(ram)
    row["warranty"]          = 1.0 if warranty else 0.0
    row["days_since_posted"] = float(days_since)
    # One-hot condition
    for cond in CONDITIONS:
        col = f"cond_{cond}"
        if col in row:
            row[col] = 1.0 if cond == condition else 0.0
    return pd.DataFrame([row])


# ── SHAP Explainability Helpers ───────────────────────────────────────────────
def compute_shap_impacts(explainer, input_df: pd.DataFrame,
                         brand, phone_model, district, condition,
                         storage, ram, warranty, days_since, price):
    """
    Returns a list of (readable_label, rs_impact) sorted by absolute impact,
    the baseline price, and the raw shap Explanation object.
    """
    sv          = explainer(input_df)
    shap_obj    = sv[0]
    shap_vals   = shap_obj.values
    base_val    = float(shap_obj.base_values)
    base_price  = float(np.expm1(base_val))

    READABLE = {
        "brand_enc":         brand,
        "model_enc":         phone_model,
        "district_enc":      district,
        "storage":           f"{int(storage)} GB Storage",
        "ram":               f"{int(ram)} GB RAM",
        "warranty":          "Warranty: Yes" if warranty else "Warranty: No",
        "days_since_posted": f"Listed {int(days_since)} days ago",
        "cond_Like New":     "Condition: Like New",
        "cond_New":          "Condition: New",
        "cond_Used":         "Condition: Used",
    }

    impacts = []
    for i, col in enumerate(input_df.columns):
        # Convert log-space SHAP value → approximate Rs. contribution
        rs_impact = float(np.expm1(base_val + shap_vals[i]) - base_price)
        label = READABLE.get(col, FEATURE_LABELS.get(col, col))
        impacts.append((label, rs_impact))

    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    return impacts, base_price, shap_obj


def plot_impact_bars(impacts: list, price: float) -> plt.Figure:
    """User-friendly horizontal bar chart of Rs. price contributions."""
    labels  = [x[0] for x in impacts]
    values  = [x[1] for x in impacts]
    colors  = ["#27ae60" if v >= 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.55 + 1)))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                   height=0.55, edgecolor="none")

    # Value labels on bars
    for bar, val in zip(bars, values[::-1]):
        sign  = "+" if val >= 0 else ""
        xpos  = bar.get_width() + (max(abs(v) for v in values) * 0.02)
        xpos  = bar.get_width() + abs(bar.get_width()) * 0.03 + 200
        color = "#27ae60" if val >= 0 else "#e74c3c"
        ax.text(xpos if val >= 0 else bar.get_width() - abs(bar.get_width()) * 0.03 - 200,
                bar.get_y() + bar.get_height() / 2,
                f"{sign}Rs. {val:,.0f}",
                va="center", ha="left" if val >= 0 else "right",
                fontsize=9, color=color, fontweight="bold")

    ax.axvline(0, color="#888888", linewidth=1.2, linestyle="--")
    ax.set_title("Price Impact by Feature", fontsize=13,
                 fontweight="bold", color="#1a1a2e", pad=10)
    ax.set_xlabel("Price Contribution (Rs.)", color="#444444", fontsize=10)
    ax.tick_params(axis="y", labelsize=10, colors="#1a1a2e")
    ax.tick_params(axis="x", labelsize=9,  colors="#555555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#dddddd")
    ax.xaxis.label.set_color("#444444")
    plt.tight_layout()
    return fig


def waterfall_plot(shap_obj, input_df: pd.DataFrame) -> plt.Figure:
    """Technical SHAP waterfall on light background."""
    labels = [FEATURE_LABELS.get(c, c) for c in input_df.columns]
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_obj.values,
            base_values   = shap_obj.base_values,
            data          = input_df.values[0],
            feature_names = labels,
        ),
        max_display=10,
        show=False,
    )
    ax = plt.gca()
    ax.set_title("SHAP Waterfall (Technical View)",
                 color="#1a1a2e", fontsize=12, fontweight="bold", pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#d0d0d0")
    ax.tick_params(colors="#1a1a2e", labelsize=9)
    ax.xaxis.label.set_color("#444444")
    ax.yaxis.label.set_color("#444444")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color("#1a1a2e")
    plt.tight_layout()
    return fig


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>📱 Sri Lanka Used Phone Price Predictor</h1>
  <p>Powered by LightGBM · Explained by SHAP</p>
</div>
""", unsafe_allow_html=True)

# ── Check model exists ─────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.error(
        "⚠️ Trained model not found. Please run the pipeline first:\n\n"
        "```\npython scraping.py\npython preprocessing.py\npython train_model.py\n```"
    )
    st.stop()

model, feature_cols, te_brand, te_model, te_district, df_ref, explainer = load_model()

# ── Sidebar Inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Phone Specifications")
    st.markdown("---")

    brand     = st.selectbox("🏷️ Brand",        BRANDS,         index=0)
    # Filter models by selected brand
    available_models = BRAND_MODELS.get(brand, ["Unknown"])
    phone_model = st.selectbox("📱 Model", available_models, index=0)
    condition = st.selectbox("⭐ Condition",     CONDITIONS,     index=0)
    storage   = st.selectbox("💾 Storage (GB)", STORAGE_OPT,    index=4)    # default 128
    ram       = st.selectbox("🧠 RAM (GB)",     RAM_OPT,        index=3)    # default 4
    warranty  = st.checkbox("🛡️ Warranty Available", value=False)
    district  = st.selectbox("📍 District",     DISTRICTS,      index=0)
    days      = st.slider("📅 Days Since Posted", min_value=0, max_value=365, value=7, step=1)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Price")

# ── Main Panel ─────────────────────────────────────────────────────────────────
if predict_btn:
    try:
        input_df = build_input(
            brand, phone_model, district, condition, storage, ram, warranty, days,
            feature_cols, te_brand, te_model, te_district
        )
        log_pred = model.predict(input_df)[0]
        price    = np.expm1(log_pred)

        # Confidence range ± 15% typical for tree models
        low  = price * 0.85
        high = price * 1.15

        st.markdown(f"""
<div class='price-card'>
  <div class='label'>Estimated Resale Price</div>
  <div class='value'>Rs. {price:,.0f}</div>
  <div class='sub'>Range: Rs. {low:,.0f} – Rs. {high:,.0f}</div>
</div>
""", unsafe_allow_html=True)

        # ── Compute SHAP impacts ───────────────────────────────────────
        impacts, base_price, shap_obj = compute_shap_impacts(
            explainer, input_df, brand, phone_model, district, condition,
            storage, ram, warranty, days, price
        )

        # ── Layer 1: Natural Language Summary ─────────────────────────
        st.markdown("### 💡 Why This Price?")
        boosters  = [(l, v) for l, v in impacts if v > 0][:3]
        reducers  = [(l, v) for l, v in impacts if v < 0][:2]

        bullets_up   = "".join(
            f"<li>✅ <b>{l}</b> adds approximately <b>Rs. {v:,.0f}</b> to the value</li>"
            for l, v in boosters
        )
        bullets_down = "".join(
            f"<li>🔻 <b>{l}</b> reduces value by approximately <b>Rs. {abs(v):,.0f}</b></li>"
            for l, v in reducers
        ) if reducers else ""

        st.markdown(f"""
<div class='info-box'>
<b>Baseline market price:</b> Rs. {base_price:,.0f}&nbsp;&nbsp;→&nbsp;&nbsp;
<b>Predicted:</b> Rs. {price:,.0f}<br><br>
<ul style='margin:0.4rem 0 0 0; padding-left:1.2rem; line-height:2'>
{bullets_up}{bullets_down}
</ul>
</div>
""", unsafe_allow_html=True)

        # ── Layer 2: Impact Bar Chart ──────────────────────────────────
        st.markdown("### 📊 Price Factor Breakdown")
        st.caption("Green bars = features that **increase** the price · "
                   "Red bars = features that **decrease** the price")
        fig_bars = plot_impact_bars(impacts, price)
        st.pyplot(fig_bars)
        plt.close()

        # ── Layer 3: Technical SHAP (expander) ────────────────────────
        with st.expander("🔬 Technical SHAP Details (for advanced users)"):
            st.caption(
                "SHAP (SHapley Additive exPlanations) waterfall showing "
                "each feature's exact contribution in log-price space."
            )
            fig_wf = waterfall_plot(shap_obj, input_df)
            st.pyplot(fig_wf)
            plt.close()

    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.markdown("### 🔍 Prediction")
    st.info("👈  Fill in the phone specs in the sidebar and click **Predict Price**.")

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><sub>Academic project — University of Moratuwa · Machine Learning Assignment  · 2026</sub></center>",
    unsafe_allow_html=True,
)
