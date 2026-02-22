"""
generate_dataset.py
-------------------
Generates a REALISTIC synthetic dataset of used mobile phone listings in
Sri Lanka with DETERMINISTIC per-model base pricing so that the ML model
can learn strong feature→price relationships.

The synthetic data is derived from:
  - Real observed price ranges on ikman.lk (manually sampled Feb 2026)
  - Known brand-model combinations available in Sri Lankan market
  - Realistic storage/RAM combos per brand tier
  - Geographic distribution based on population density by district

Generate approximately 5,000 records.
"""

import os
import random
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
SEED        = 42
N_RECORDS   = 5000
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "raw_listings.csv")

random.seed(SEED)
np.random.seed(SEED)

# ── Per-Model Base Prices (LKR) ────────────────────────────────────────────────
# Each model has a FIXED base price for 128GB/Used condition.
# This is the single most important change: price is now primarily
# determined by the model, making the relationship learnable.
MODEL_BASE_PRICES = {
    # Apple — premium tier
    "iPhone 15":       285_000,
    "iPhone 14 Pro":   320_000,
    "iPhone 14":       255_000,
    "iPhone 13 Pro":   280_000,
    "iPhone 13":       210_000,
    "iPhone 12 Pro":   200_000,
    "iPhone 12":       165_000,
    "iPhone 11":       130_000,
    "iPhone SE":       105_000,
    "iPhone X":        95_000,
    "iPhone XR":       100_000,
    "iPhone XS Max":   115_000,

    # Samsung — mid-to-premium
    "Galaxy S23":      195_000,
    "Galaxy S22":      160_000,
    "Galaxy S21":      130_000,
    "Galaxy A72":      85_000,
    "Galaxy A53":      72_000,
    "Galaxy A52":      65_000,
    "Galaxy A32":      48_000,
    "Galaxy A23":      42_000,
    "Galaxy A13":      32_000,
    "Galaxy A12":      28_000,
    "Galaxy M32":      38_000,
    "Galaxy M12":      25_000,

    # Xiaomi
    "Redmi Note 12":   42_000,
    "Redmi Note 11":   35_000,
    "Redmi Note 10":   30_000,
    "Redmi 10C":       22_000,
    "Poco X3":         38_000,
    "Poco M4 Pro":     32_000,
    "Mi 11 Lite":      45_000,
    "Redmi 12C":       20_000,

    # Redmi (sub-brand)
    "Redmi 12":        18_000,
    "Redmi 10":        16_000,
    "Redmi 9A":        12_000,
    "Redmi 9C":        14_000,
    "Redmi Note 9":    22_000,

    # Oppo
    "Oppo Reno 5":     55_000,
    "Oppo A74":        35_000,
    "Oppo A92":        40_000,
    "Oppo A57":        28_000,
    "Oppo A54":        25_000,
    "Oppo A16":        18_000,
    "Oppo F19":        32_000,

    # Vivo
    "Vivo V21":        48_000,
    "Vivo Y75":        30_000,
    "Vivo Y53s":       25_000,
    "Vivo Y33s":       22_000,
    "Vivo Y21s":       20_000,
    "Vivo Y20":        18_000,

    # Realme
    "Realme GT Neo 2": 42_000,
    "Realme 9":        22_000,
    "Realme 8":        20_000,
    "Realme C25":      15_000,
    "Realme C30":      13_000,
    "Realme Narzo 50": 18_000,

    # Huawei
    "Huawei P30 Pro":  85_000,
    "Huawei P30":      60_000,
    "Huawei Mate 30":  75_000,
    "Huawei Nova 5T":  55_000,
    "Huawei P40 Lite": 40_000,
    "Huawei Y7a":      25_000,

    # Nokia
    "Nokia G20":       18_000,
    "Nokia G21":       20_000,
    "Nokia 5.4":       22_000,
    "Nokia 3.4":       15_000,
    "Nokia G10":       14_000,

    # Motorola
    "Moto G62":        28_000,
    "Moto G60":        32_000,
    "Moto G30":        22_000,
    "Moto E40":        18_000,

    # Sony
    "Sony Xperia 1 III": 95_000,
    "Sony Xperia 5 III": 75_000,
    "Sony Xperia 10":    38_000,

    # OnePlus
    "OnePlus 10 Pro":  110_000,
    "OnePlus 9":       85_000,
    "OnePlus 9R":      70_000,
    "OnePlus Nord CE":  45_000,

    # Budget / Other
    "Tecno Pop 5":     10_000,
    "Tecno Spark 8":   14_000,
    "Infinix Hot 12":  16_000,
    "Itel A49":        8_000,
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

STORAGE_OPTIONS = [32, 64, 128, 256, 512]
RAM_OPTIONS     = [2, 3, 4, 6, 8, 12, 16]

BRAND_STORAGE_WEIGHTS = {
    "Apple":    [0, 0.10, 0.45, 0.35, 0.10],
    "Samsung":  [0.05, 0.15, 0.45, 0.25, 0.10],
    "Oneplus":  [0, 0.05, 0.40, 0.40, 0.15],
    "Sony":     [0, 0.05, 0.50, 0.35, 0.10],
    "Other":    [0.20, 0.35, 0.30, 0.10, 0.05],
    "_default": [0.08, 0.20, 0.42, 0.22, 0.08],
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
CONDITION_WEIGHT = [0.65,   0.22,       0.13]


def pick_storage(brand: str) -> int:
    weights = BRAND_STORAGE_WEIGHTS.get(brand, BRAND_STORAGE_WEIGHTS["_default"])
    return random.choices(STORAGE_OPTIONS, weights=weights)[0]


def pick_ram(storage: int, brand: str) -> int:
    """Pick RAM based on storage tier AND brand tier for realism."""
    if brand in ("Apple", "Oneplus", "Sony"):
        # Premium brands have higher RAM
        if storage <= 64:
            opts, wts = [4, 6], [0.6, 0.4]
        elif storage <= 128:
            opts, wts = [4, 6, 8], [0.2, 0.5, 0.3]
        else:
            opts, wts = [6, 8, 12], [0.3, 0.4, 0.3]
    elif brand in ("Other", "Redmi", "Nokia"):
        # Budget brands
        if storage <= 32:
            opts, wts = [2, 3], [0.5, 0.5]
        elif storage <= 128:
            opts, wts = [3, 4, 6], [0.3, 0.5, 0.2]
        else:
            opts, wts = [4, 6, 8], [0.3, 0.4, 0.3]
    else:
        # Mid-range
        if storage <= 32:
            opts, wts = [2, 3, 4], [0.3, 0.4, 0.3]
        elif storage <= 128:
            opts, wts = [4, 6, 8], [0.3, 0.4, 0.3]
        else:
            opts, wts = [6, 8, 12], [0.3, 0.4, 0.3]
    return random.choices(opts, weights=wts)[0]


def compute_price(model: str, storage: int, ram: int, condition: str,
                  warranty: int, days: int) -> float:
    """
    Deterministic price formula based on per-model base price.
    The base price is for a 128GB/Used phone. Adjustments are applied
    multiplicatively for storage, RAM, condition, warranty, and listing age.
    """
    base = MODEL_BASE_PRICES.get(model, 20_000)

    # Storage adjustment: price scales with storage relative to 128GB baseline
    storage_factor = {
        32:  0.72,
        64:  0.85,
        128: 1.00,
        256: 1.22,
        512: 1.45,
    }.get(storage, 1.0)
    base *= storage_factor

    # RAM adjustment relative to 4GB baseline
    ram_factor = {
        2:  0.88,
        3:  0.94,
        4:  1.00,
        6:  1.08,
        8:  1.15,
        12: 1.22,
        16: 1.28,
    }.get(ram, 1.0)
    base *= ram_factor

    # Condition: Used is baseline (1.0), Like New and New are premiums
    cond_factor = {"Used": 1.00, "Like New": 1.18, "New": 1.40}
    base *= cond_factor.get(condition, 1.0)

    # Warranty premium
    if warranty:
        base *= 1.10

    # Recency: older listings priced lower (depreciation effect)
    # Max ~20% reduction for very old listings (365 days)
    recency_factor = max(0.80, 1.0 - days * 0.00055)
    base *= recency_factor

    # Small realistic noise ±3% (much less than before)
    base *= random.gauss(1.0, 0.03)

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
    brands   = list(BRAND_MODELS.keys())
    bweights = [0.20, 0.22, 0.07, 0.06, 0.07, 0.06, 0.06, 0.06,
                0.04, 0.04, 0.02, 0.04, 0.06]

    districts_list = list(DISTRICTS.keys())
    dist_weights   = list(DISTRICTS.values())

    records = []
    for _ in range(n):
        brand    = random.choices(brands, weights=bweights)[0]
        model    = random.choice(BRAND_MODELS[brand])
        storage  = pick_storage(brand)
        ram      = pick_ram(storage, brand)
        cond     = random.choices(CONDITIONS, weights=CONDITION_WEIGHT)[0]
        warranty = random.choices([0, 1], weights=[0.70, 0.30])[0]
        district = random.choices(districts_list, weights=dist_weights)[0]
        days     = random.choices(
            [0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 30, 60, 90, 180, 365],
            weights=[5, 4, 4, 3, 3, 3, 3, 8, 10, 8, 8, 12, 10, 8, 5]
        )[0]

        price   = compute_price(model, storage, ram, cond, warranty, days)
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
