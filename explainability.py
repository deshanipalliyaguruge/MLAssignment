"""
explainability.py
-----------------
Applies XAI techniques to the trained LightGBM model:

  1. SHAP TreeExplainer  → summary (beeswarm) plot
  2. Feature Importance  → gain-based bar chart
  3. Partial Dependence Plots (PDPs) for top features

Outputs saved to plots/:
  - shap_summary.png
  - feature_importance.png
  - pdp_storage.png
  - pdp_brand.png
  - pdp_district.png
"""

import os
import logging
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb

from sklearn.inspection import PartialDependenceDisplay

warnings.filterwarnings("ignore")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
CLEAN_CSV = os.path.join(BASE_DIR, "data", "cleaned_data.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Feature name mapping (for human-readable plot labels) ─────────────────────
FEATURE_LABELS = {
    "storage":           "Storage (GB)",
    "ram":               "RAM (GB)",
    "warranty":          "Warranty (1=Yes)",
    "days_since_posted": "Days Since Posted",
    "brand_enc":         "Brand (Encoded)",
    "district_enc":      "District (Encoded)",
    "cond_Like New":     "Condition: Like New",
    "cond_New":          "Condition: New",
    "cond_Used":         "Condition: Used",
}


def load_artifacts():
    """Load trained model, feature list, and cleaned dataset."""
    model        = joblib.load(os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    df           = pd.read_csv(CLEAN_CSV)
    X            = df[[c for c in feature_cols if c in df.columns]]
    y            = df["price"]
    return model, feature_cols, X, y


# ── 1. SHAP Summary Plot ───────────────────────────────────────────────────────
def plot_shap_summary(model, X: pd.DataFrame, max_display: int = 10) -> None:
    """
    SHAP TreeExplainer computes exact Shapley values for tree-based models.
    The beeswarm plot shows:
      - Feature importance (vertical axis order by mean |SHAP|)
      - Effect direction (positive SHAP → higher price, negative → lower price)
      - Value range (color: blue=low feature value, red=high feature value)
    """
    logger.info("Computing SHAP values (this may take 30–60 seconds)…")
    explainer   = shap.TreeExplainer(model)
    # SHAP values are in log-price space (undo with expm1 for interpretation)
    shap_values = explainer(X)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=[FEATURE_LABELS.get(c, c) for c in X.columns],
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    ax = plt.gca()
    ax.set_title("SHAP Summary Plot — Feature Impact on Predicted Price",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ SHAP summary saved → {path}")
    return explainer, shap_values


# ── 2. Feature Importance ──────────────────────────────────────────────────────
def plot_feature_importance(model, feature_cols: list[str]) -> None:
    """
    LightGBM gain-based importance: the total gain accumulated by all splits
    using a given feature. Provides a model-level view of feature influence.
    """
    importances = model.feature_importances_
    labels      = [FEATURE_LABELS.get(c, c) for c in feature_cols]

    sorted_idx  = np.argsort(importances)
    colors      = plt.cm.Blues(np.linspace(0.4, 0.9, len(importances)))

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        [labels[i] for i in sorted_idx],
        importances[sorted_idx],
        color=[colors[i] for i in range(len(sorted_idx))],
        edgecolor="white", linewidth=0.5
    )
    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title("LightGBM Feature Importance — Gain-Based",
                 fontsize=13, fontweight="bold")
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"✓ Feature importance saved → {path}")


# ── 3. Partial Dependence Plots ────────────────────────────────────────────────
def plot_pdp(model, X: pd.DataFrame, feature: str, label: str) -> None:
    """
    A PDP shows the marginal effect of one feature on the predicted outcome
    while averaging out all other features. This reveals the overall
    relationship between a feature and the target, regardless of correlations.
    Predictions are in log-price space; we exponentiate for interpretability.
    """
    if feature not in X.columns:
        logger.warning(f"Feature '{feature}' not found in dataset. Skipping PDP.")
        return

    feature_idx = list(X.columns).index(feature)

    fig, ax = plt.subplots(figsize=(8, 5))
    disp = PartialDependenceDisplay.from_estimator(
        model, X, features=[feature],
        ax=ax, grid_resolution=30
    )
    plt.title(f"PDP: {FEATURE_LABELS.get(feature, feature)}", fontsize=14)
    # Convert log-price y-axis ticks to actual LKR values
    y_ticks = ax.get_yticks()
    ax.set_yticklabels([f"Rs. {np.expm1(v):,.0f}" for v in y_ticks])
    ax.set_xlabel(FEATURE_LABELS.get(feature, feature), fontsize=11)
    ax.set_ylabel("Predicted Price (LKR)", fontsize=11)
    ax.set_title(f"Partial Dependence Plot — {FEATURE_LABELS.get(feature, feature)}",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    safe_name = feature.replace(" ", "_").replace(":", "")
    path = os.path.join(PLOTS_DIR, f"pdp_{safe_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"✓ PDP saved → {path}")


# ── 4. SHAP Dependence Plot ────────────────────────────────────────────────────
def plot_shap_dependence(shap_values, X: pd.DataFrame,
                         feature: str, interaction: str | None = None) -> None:
    """SHAP dependence plot shows SHAP value vs feature value for one feature."""
    if feature not in X.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feature,
        shap_values.values,
        X,
        feature_names=[FEATURE_LABELS.get(c, c) for c in X.columns],
        interaction_index=interaction,
        ax=ax,
        show=False,
    )
    ax.set_title(
        f"SHAP Dependence — {FEATURE_LABELS.get(feature, feature)}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    safe = feature.replace(" ", "_")
    path = os.path.join(PLOTS_DIR, f"shap_dep_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ SHAP dependence saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    model, feature_cols, X, y = load_artifacts()

    # Only use a sample for speed (SHAP is O(n * features))
    X_sample = X.sample(min(500, len(X)), random_state=42)

    # 1. SHAP summary
    explainer, shap_values = plot_shap_summary(model, X_sample)

    # 2. Feature importance
    plot_feature_importance(model, list(X.columns))

    # 3. PDPs for top features
    for feat in ["storage", "ram", "brand_enc", "district_enc"]:
        plot_pdp(model, X, feat, FEATURE_LABELS.get(feat, feat))

    # 4. SHAP dependence plots
    for feat in ["storage", "brand_enc"]:
        plot_shap_dependence(shap_values, X_sample, feat)

    print("\n✓ All explainability plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
