"""
preprocessing.py
----------------
Cleans the raw scraped CSV and engineers features for model training.

Steps:
  1. Load raw_listings.csv
  2. Remove duplicates
  3. Parse price → numeric (LKR)
  4. Parse storage → numeric (GB)
  5. Parse RAM → numeric (GB)
  6. Normalize condition labels
  7. Compute days_since_posted
  8. Drop low-quality rows
  9. Encode categoricals
  10. Save cleaned_data.csv
"""

import os
import re
import logging
from datetime import datetime, date

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
RAW_CSV     = os.path.join(BASE_DIR, "data", "raw_listings.csv")
CLEAN_CSV   = os.path.join(BASE_DIR, "data", "cleaned_data.csv")
ENCODER_DIR = os.path.join(BASE_DIR, "models")


# ── Parsing Utilities ──────────────────────────────────────────────────────────
def parse_price(raw: str) -> float | None:
    """Convert 'Rs. 45,000' or '45000' → float."""
    if pd.isna(raw) or str(raw).strip() == "":
        return None
    cleaned = re.sub(r"[Rsr\s\.,]", "", str(raw).replace(",", ""))
    cleaned = re.sub(r"[^\d]", "", cleaned)
    return float(cleaned) if cleaned else None


def parse_storage(raw: str) -> float | None:
    """Convert '128GB' / '128 GB' → 128.0"""
    if pd.isna(raw) or str(raw).strip() == "":
        return None
    match = re.search(r"(\d+)", str(raw))
    return float(match.group(1)) if match else None


def parse_ram(raw: str) -> float | None:
    """Convert '8GB' / '8 GB RAM' → 8.0"""
    if pd.isna(raw) or str(raw).strip() == "":
        return None
    match = re.search(r"(\d+)", str(raw))
    return float(match.group(1)) if match else None


def parse_days_since_posted(raw: str) -> int | None:
    """
    ikman.lk uses relative dates like:
      '2 days ago', '1 week ago', 'Today', 'Yesterday', 'Jan 15, 2024'
    Returns integer number of days since posted.
    """
    if pd.isna(raw) or str(raw).strip() == "":
        return None
    raw = str(raw).strip().lower()
    today = date.today()

    if "today" in raw:
        return 0
    if "yesterday" in raw:
        return 1
    m = re.search(r"(\d+)\s*day", raw)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*week", raw)
    if m:
        return int(m.group(1)) * 7
    m = re.search(r"(\d+)\s*month", raw)
    if m:
        return int(m.group(1)) * 30
    m = re.search(r"(\d+)\s*year", raw)
    if m:
        return int(m.group(1)) * 365
    # Try explicit date e.g. "Jan 15, 2024"
    for fmt in ("%b %d, %Y", "%d %b %Y", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(raw, fmt).date()
            return (today - parsed).days
        except ValueError:
            continue
    return None


def normalize_condition(raw: str) -> str:
    """Standardise varied condition strings."""
    if pd.isna(raw):
        return "Used"
    raw = str(raw).strip().lower()
    if any(k in raw for k in ["new", "brand new", "sealed"]):
        return "New"
    if any(k in raw for k in ["like new", "excellent", "mint"]):
        return "Like New"
    return "Used"


def normalize_warranty(raw: str) -> int:
    """1 if warranty available, 0 otherwise."""
    if pd.isna(raw):
        return 0
    raw = str(raw).strip().lower()
    return int(any(k in raw for k in ["yes", "available", "warranty", "month", "year"]))


def normalize_brand(brand: str) -> str:
    """Collapse rare brands and normalize casing."""
    if pd.isna(brand) or str(brand).strip() == "":
        return "Other"
    brand = str(brand).strip().title()
    # Merge iPhone → Apple
    if brand.lower() in ["iphone"]:
        return "Apple"
    MAJOR = {
        "Samsung", "Apple", "Huawei", "Xiaomi", "Oppo", "Vivo",
        "Realme", "Nokia", "Motorola", "Sony", "Oneplus", "Lg",
        "Google", "Asus", "Tecno", "Infinix", "Honor"
    }
    return brand if brand in MAJOR else "Other"


# ── Main Preprocessing Pipeline ────────────────────────────────────────────────
def preprocess(target_encode: bool = True) -> pd.DataFrame:
    logger.info(f"Loading raw data from {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    logger.info(f"Raw shape: {df.shape}")

    # ── 1. Drop exact duplicates ───────────────────────────────────────────────
    df.drop_duplicates(inplace=True)
    logger.info(f"After dedup: {df.shape}")

    # ── 2. Parse numeric columns ───────────────────────────────────────────────
    df["price"]   = df["price_raw"].apply(parse_price)
    df["storage"] = df["storage_raw"].apply(parse_storage)
    df["ram"]     = df["ram_raw"].apply(parse_ram)
    df["days_since_posted"] = df["posted_date"].apply(parse_days_since_posted)

    # ── 3. Normalise categoricals ──────────────────────────────────────────────
    df["condition"] = df["condition"].apply(normalize_condition)
    df["warranty"]  = df["warranty"].apply(normalize_warranty)
    df["brand"]     = df["brand"].apply(normalize_brand)
    df["district"]  = df["district"].str.strip().str.title().fillna("Unknown")

    # ── 4. Remove rows with missing target or key features ────────────────────
    before = len(df)
    df.dropna(subset=["price"], inplace=True)
    df = df[df["price"] > 0]
    df = df[df["price"] < 5_000_000]   # upper sanity bound (Rs. 5M)
    logger.info(f"Removed {before - len(df)} rows with invalid price. Remaining: {len(df)}")

    # ── 5. Impute missing numeric features ─────────────────────────────────────
    df["storage"] = df["storage"].fillna(df["storage"].median())
    df["ram"]     = df["ram"].fillna(df["ram"].median())
    df["days_since_posted"] = df["days_since_posted"].fillna(
        df["days_since_posted"].median()
    )

    # ── 6. One-Hot encode condition ────────────────────────────────────────────
    condition_dummies = pd.get_dummies(df["condition"], prefix="cond", drop_first=False)
    df = pd.concat([df, condition_dummies], axis=1)

    # ── 7. Target-encode brand, model, and district (high cardinality) ────────
    os.makedirs(ENCODER_DIR, exist_ok=True)
    # Normalize model names
    df["model"] = df["model"].str.strip().fillna("Unknown")
    if target_encode:
        te_brand    = TargetEncoder(cols=["brand"])
        te_model    = TargetEncoder(cols=["model"])
        te_district = TargetEncoder(cols=["district"])
        df["brand_enc"]    = te_brand.fit_transform(df["brand"],    df["price"])
        df["model_enc"]    = te_model.fit_transform(df["model"],    df["price"])
        df["district_enc"] = te_district.fit_transform(df["district"], df["price"])
        import joblib
        joblib.dump(te_brand,    os.path.join(ENCODER_DIR, "te_brand.pkl"))
        joblib.dump(te_model,    os.path.join(ENCODER_DIR, "te_model.pkl"))
        joblib.dump(te_district, os.path.join(ENCODER_DIR, "te_district.pkl"))
        logger.info("✓ Target encoders saved (brand, model, district).")
    else:
        le = LabelEncoder()
        df["brand_enc"]    = le.fit_transform(df["brand"].astype(str))
        df["model_enc"]    = le.fit_transform(df["model"].astype(str))
        df["district_enc"] = le.fit_transform(df["district"].astype(str))

    # ── 8. Select final ML columns ─────────────────────────────────────────────
    feature_cols = (
        ["storage", "ram", "warranty", "days_since_posted",
         "brand_enc", "model_enc", "district_enc"]
        + [c for c in df.columns if c.startswith("cond_")]
    )
    df_clean = df[feature_cols + ["price", "brand", "model", "district", "condition"]].copy()
    df_clean.dropna(subset=feature_cols, inplace=True)

    logger.info(f"Final clean shape: {df_clean.shape}")
    df_clean.to_csv(CLEAN_CSV, index=False)
    logger.info(f"✓ Saved cleaned data → {CLEAN_CSV}")
    return df_clean


if __name__ == "__main__":
    preprocess()
