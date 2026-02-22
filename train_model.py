"""
train_model.py
--------------
Trains a LightGBM Regressor to predict used mobile phone resale prices (LKR).

Why LightGBM?
  - Gradient Boosted Decision Trees (GBDT) variant with leaf-wise growth
  - Significantly faster than XGBoost for large tabular datasets
  - Built-in handling of categorical features
  - State-of-the-art on structured/tabular regression tasks
  - NOT covered in standard ML lectures (which focus on basic boosting)

Pipeline:
  1. Load cleaned_data.csv
  2. 80/20 stratified train/test split
  3. Hyperparameter optimisation with Optuna (Bayesian search, 50 trials)
  4. Final model trained on full training set with best params
  5. Evaluation: RMSE, MAE, R²
  6. Model saved to models/lgbm_model.pkl
"""

import os
import logging
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
CLEAN_CSV  = os.path.join(BASE_DIR, "data", "cleaned_data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model.pkl")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Feature columns (must match preprocessing.py output) ─────────────────────
FEATURE_COLS = [
    "storage", "ram", "warranty", "days_since_posted",
    "brand_enc", "district_enc",
    "cond_Like New", "cond_New", "cond_Used",
]
TARGET_COL = "price"


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Loading {CLEAN_CSV}")
    df = pd.read_csv(CLEAN_CSV)

    # Keep only columns that actually exist (guards against missing cond_ columns)
    existing_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[existing_features]
    y = df[TARGET_COL]
    logger.info(f"Feature matrix: {X.shape}, Target: {y.shape}")
    logger.info(f"Price range: Rs. {y.min():,.0f} – Rs. {y.max():,.0f}")
    logger.info(f"Price median: Rs. {y.median():,.0f}")
    return X, y


# ── Train/Test Split ───────────────────────────────────────────────────────────
def split_data(X: pd.DataFrame, y: pd.Series):
    """
    80/20 split using price quantile bins to ensure representative distribution
    in both train and test sets (stratified by price range).
    """
    price_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=price_bins
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ── Optuna Hyperparameter Tuning ───────────────────────────────────────────────
def objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """Optuna objective: minimise 5-fold CV RMSE on log-transformed target."""
    params = {
        "verbosity":        -1,
        "objective":        "regression",
        "metric":           "rmse",
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 20, 300),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state":     42,
    }
    model  = lgb.LGBMRegressor(**params)
    # Use log-price to reduce sensitivity to outliers; convert RMSE back after
    log_y  = np.log1p(y_train)
    scores = cross_val_score(model, X_train, log_y, cv=5,
                             scoring="neg_root_mean_squared_error", n_jobs=-1)
    return -scores.mean()


def tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series,
                         n_trials: int = 50) -> dict:
    logger.info(f"Starting Optuna hyperparameter search ({n_trials} trials)…")
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X_train, y_train),
                   n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info(f"Best params: {best}")
    logger.info(f"Best CV log-RMSE: {study.best_value:.4f}")
    return best


# ── Model Training ─────────────────────────────────────────────────────────────
def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                best_params: dict) -> lgb.LGBMRegressor:
    """Train final LightGBM model on log-transformed target."""
    params = {
        "verbosity":  -1,
        "objective":  "regression",
        "metric":     "rmse",
        "random_state": 42,
        **best_params,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, np.log1p(y_train))
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model: lgb.LGBMRegressor,
             X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Predict (expm1 to reverse log transform) and compute metrics."""
    y_pred = np.expm1(model.predict(X_test))
    y_pred = np.clip(y_pred, 0, None)          # no negative prices

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"  RMSE  : Rs. {rmse:>12,.2f}")
    print(f"  MAE   : Rs. {mae:>12,.2f}")
    print(f"  R²    : {r2:>15.4f}")
    print("=" * 50 + "\n")

    return {"rmse": rmse, "mae": mae, "r2": r2, "y_pred": y_pred}


# ── Evaluation Plots ───────────────────────────────────────────────────────────
def plot_actual_vs_predicted(y_test: pd.Series, y_pred: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("LightGBM — Actual vs Predicted Price", fontsize=14, fontweight="bold")

    # Scatter
    ax = axes[0]
    ax.scatter(y_test / 1000, y_pred / 1000, alpha=0.4, s=20, color="#4C72B0")
    lim = max(y_test.max(), y_pred.max()) / 1000
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Price (Rs. '000)")
    ax.set_ylabel("Predicted Price (Rs. '000)")
    ax.set_title("Actual vs Predicted")
    ax.legend()

    # Residuals
    ax = axes[1]
    residuals = (y_pred - y_test) / 1000
    ax.hist(residuals, bins=40, color="#DD8452", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (Rs. '000)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "actual_vs_predicted.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"✓ Plot saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Hyperparameter tuning
    best_params = tune_hyperparameters(X_train, y_train, n_trials=50)

    # Final training
    logger.info("Training final model with best hyperparameters…")
    model = train_model(X_train, y_train, best_params)

    # Evaluate
    metrics = evaluate(model, X_test, y_test)
    plot_actual_vs_predicted(y_test, metrics["y_pred"])

    # Save model and feature list
    joblib.dump(model, MODEL_PATH)
    # Save the list of features so app.py can reconstruct input correctly
    feature_list = list(X_train.columns)
    joblib.dump(feature_list, os.path.join(MODEL_DIR, "feature_cols.pkl"))
    logger.info(f"✓ Model saved → {MODEL_PATH}")

    return model, X_train, X_test, y_train, y_test, metrics


if __name__ == "__main__":
    main()
