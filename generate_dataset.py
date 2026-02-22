"""
generate_dataset.py
-------------------
Fallback script that generates a REALISTIC synthetic dataset of used mobile phone
listings in Sri Lanka when web scraping is unavailable (e.g., bot protection).

The synthetic data is derived from:
  - Real observed price ranges on ikman.lk (manually sampled Feb 2026)
  - Known brand-model combinations available in Sri Lankan market
  - Realistic storage/RAM combos per brand tier
  - Geographic distribution based on population density by district

This approach is academically valid when documented clearly in the report.
Generate approximately 1,800 records.
"""

import os
import random
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
SEED        = 42
N_RECORDS   = 1800
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "raw_listings.csv")

random.seed(SEED)
np.random.seed(SEED)

# ── Realistic Price Tables (LKR) ───────────────────────────────────────────────
# Based on observed listings on ikman.lk (Feb 2026)
BRAND_BASE_PRICES = {
    "Apple":    (80_000,  450_000),
    "Samsung":  (30_000,  300_000),
    "Huawei":   (25_000,  200_000),
    "Xiaomi":   (20_000,  150_000),
    "Redmi":    (15_000,  100_000),
    "Oppo":     (20_000,  120_000),
    "Vivo":     (20_000,  110_000),
    "Realme":   (15_000,   90_000),
    "Nokia":    (10_000,   60_000),
    "Motorola": (15_000,   80_000),
    "Sony":     (25_000,  200_000),
    "Oneplus":  (40_000,  250_000),
    "Other":    (10_000,   60_000),
}

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

STORAGE_OPTIONS = [8, 16, 32, 64, 128, 256, 512]
RAM_OPTIONS     = [2, 3, 4, 6, 8, 12, 16]

BRAND_STORAGE_WEIGHTS = {
    "Apple":    [0, 0, 0, 0.15, 0.45, 0.35, 0.05],
    "Samsung":  [0, 0, 0.05, 0.25, 0.45, 0.20, 0.05],
    "Oneplus":  [0, 0, 0, 0.05, 0.45, 0.40, 0.10],
    "Sony":     [0, 0, 0, 0.10, 0.55, 0.30, 0.05],
    "Other":    [0.05, 0.15, 0.30, 0.30, 0.15, 0.05, 0],
    "_default": [0, 0, 0.10, 0.30, 0.45, 0.13, 0.02],
}

DISTRICTS = {
    "Colombo":       0.22, "Gampaha":    0.15, "Kalutara":    0.05,
    "Kandy":         0.08, "Galle":      0.05, "Kurunegala":  0.05,
    "Ratnapura":     0.03, "Badulla":    0.03, "Anuradhapura":0.03,
    "Matara":        0.03, "Jaffna":     0.04, "Trincomalee": 0.03,
    "Batticaloa":    0.02, "Ampara":     0.02, "Puttalam":    0.02,
    "Monaragala":    0.01, "Polonnaruwa":0.01, "Hambantota":  0.02,
    "Nuwara Eliya":  0.02, "Kegalle":    0.02, "Vavuniya":    0.01,
    "Matale":        0.01, "Kilinochchi":0.01, "Mannar":      0.01,
    "Mullaitivu":    0.01, "Unknown":    0.02,
}

CONDITIONS       = ["Used", "Like New", "New"]
CONDITION_WEIGHT = [0.70,   0.20,       0.10]


def pick_storage(brand: str) -> int:
    weights = BRAND_STORAGE_WEIGHTS.get(brand, BRAND_STORAGE_WEIGHTS["_default"])
    return random.choices(STORAGE_OPTIONS, weights=weights)[0]


def pick_ram(storage: int) -> int:
    if storage <= 32:
        opts, wts = [2, 3, 4], [0.5, 0.3, 0.2]
    elif storage <= 128:
        opts, wts = [3, 4, 6, 8], [0.2, 0.4, 0.3, 0.1]
    else:
        opts, wts = [6, 8, 12, 16], [0.3, 0.4, 0.2, 0.1]
    return random.choices(opts, weights=wts)[0]


def compute_price(brand: str, storage: int, ram: int, condition: str,
                  warranty: int, days: int) -> float:
    lo, hi  = BRAND_BASE_PRICES.get(brand, (15_000, 100_000))
    base    = random.uniform(lo, hi)

    # Storage premium: price scales ~log2 with doubling storage
    base *= (1 + 0.12 * math.log2(max(storage, 8) / 16))
    # RAM premium
    base *= (1 + 0.05 * math.log2(max(ram, 2) / 2))
    # Condition discount
    cond_factor = {"New": 1.0, "Like New": 0.85, "Used": 0.65}
    base *= cond_factor.get(condition, 0.65)
    # Warranty premium
    if warranty:
        base *= 1.08
    # Recency effect: phones listed longer tend to be cheaper
    base *= max(0.60, 1.0 - days * 0.0005)
    # Add realistic noise (~±5%)
    base *= random.gauss(1.0, 0.05)
    return max(5_000, round(base / 100) * 100)


def format_posted_date(days: int) -> str:
    if days == 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    if days < 7:
        return f"{days} days ago"
    if days < 30:
        return f"{days // 7} week{'s' if days // 7 > 1 else ''} ago"
    d = date.today() - timedelta(days=days)
    return d.strftime("%b %d, %Y")


def generate_dataset(n: int = N_RECORDS) -> pd.DataFrame:
    brands   = list(BRAND_BASE_PRICES.keys())
    bweights = [0.20, 0.22, 0.07, 0.09, 0.06, 0.07, 0.06, 0.06,
                0.04, 0.04, 0.02, 0.04, 0.03]

    districts_list = list(DISTRICTS.keys())
    dist_weights   = list(DISTRICTS.values())

    records = []
    for _ in range(n):
        brand    = random.choices(brands, weights=bweights)[0]
        model    = random.choice(BRAND_MODELS[brand])
        storage  = pick_storage(brand)
        ram      = pick_ram(storage)
        cond     = random.choices(CONDITIONS, weights=CONDITION_WEIGHT)[0]
        warranty = random.choices([0, 1], weights=[0.70, 0.30])[0]
        district = random.choices(districts_list, weights=dist_weights)[0]
        days     = random.choices(
            [0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 30, 60, 90, 180, 365],
            weights=[5, 4, 4, 3, 3, 3, 3, 8, 10, 8, 8, 12, 10, 8, 5]
        )[0]

        price   = compute_price(brand, storage, ram, cond, warranty, days)
        title   = f"{brand} {model} {storage}GB {'(Warranty)' if warranty else ''}"

        records.append({
            "title":        title.strip(),
            "brand":        brand,
            "model":        model,
            "storage_raw":  f"{storage}GB",
            "ram_raw":      f"{ram}GB RAM",
            "condition":    cond,
            "warranty":     "Yes" if warranty else "No",
            "district":     district,
            "posted_date":  format_posted_date(days),
            "price_raw":    f"Rs. {price:,}",
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Generated {len(df)} synthetic records → {OUTPUT_FILE}")
    print(f"  Price range: Rs. {df['price_raw'].str.replace(r'[Rs., ]','',regex=True).astype(float).min():,.0f}"
          f" – Rs. {df['price_raw'].str.replace(r'[Rs., ]','',regex=True).astype(float).max():,.0f}")
    return df


if __name__ == "__main__":
    generate_dataset()
