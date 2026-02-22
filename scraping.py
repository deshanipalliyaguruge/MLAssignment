"""
scraping.py
-----------
Scrapes used mobile phone listings from ikman.lk using Selenium + BeautifulSoup.
Targets the Mobile Phones category and collects ~1500-2000 listings.

Ethical scraping practices:
  - Public data only (no login required)
  - Random delays between requests (2-5 seconds) to avoid server overload
  - No personal or sensitive user data is collected
  - Data is used solely for academic research
"""

import time
import random
import csv
import os
import re
import logging
from datetime import datetime

import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_URL   = "https://ikman.lk/en/ads/sri-lanka/mobile-phones"
MAX_PAGES  = 80          # ~20-25 listings per page → ~1600-2000 records
MIN_DELAY  = 2.5         # seconds between page loads (polite scraping)
MAX_DELAY  = 5.5
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "raw_listings.csv")

CSV_FIELDS = [
    "title", "brand", "model", "storage_raw", "ram_raw",
    "condition", "warranty", "district", "posted_date", "price_raw"
]

# ── Selenium Setup ─────────────────────────────────────────────────────────────
def create_driver() -> webdriver.Chrome:
    """Create a headless Chrome driver with anti-detection tweaks."""
    chromedriver_autoinstaller.install()
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1920,1080")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=opts)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


# ── Parsing Helpers ────────────────────────────────────────────────────────────
def extract_brand(title: str) -> str:
    """Heuristically extract brand from listing title."""
    known_brands = [
        "Samsung", "Apple", "iPhone", "Huawei", "Xiaomi", "Redmi", "OPPO",
        "Vivo", "Realme", "Nokia", "Motorola", "Sony", "OnePlus", "LG",
        "Google", "Asus", "Tecno", "Infinix", "Honor", "HTC"
    ]
    title_lower = title.lower()
    for brand in known_brands:
        if brand.lower() in title_lower:
            return "Apple" if brand == "iPhone" else brand
    return "Other"


def extract_storage(text: str) -> str:
    """Pull storage size string from title/description e.g. '128GB'."""
    match = re.search(r"(\d+)\s*(?:gb|GB|Gb)", text)
    return match.group(0).upper().replace(" ", "") if match else ""


def extract_ram(text: str) -> str:
    """Pull RAM size string from title/description e.g. '8GB RAM'."""
    match = re.search(r"(\d+)\s*(?:gb|GB)\s*(?:ram|RAM)", text, re.IGNORECASE)
    if match:
        return match.group(1) + "GB"
    return ""


def parse_listing_page(driver: webdriver.Chrome, url: str) -> dict | None:
    """
    Visit an individual listing page and scrape detailed attributes.
    Returns a dict of fields or None on failure.
    """
    try:
        driver.get(url)
        time.sleep(random.uniform(1.5, 3.0))
        soup = BeautifulSoup(driver.page_source, "lxml")

        # ── Price ──────────────────────────────────────────────────────────────
        price_el = soup.select_one("span.money-amount--3NTpl, .price--3NTpl")
        price_raw = price_el.get_text(strip=True) if price_el else ""

        # ── Title ──────────────────────────────────────────────────────────────
        title_el = soup.select_one("h1.title--3Pfc3, h1[class*='title']")
        title = title_el.get_text(strip=True) if title_el else ""

        # ── District & Date ────────────────────────────────────────────────────
        location_el = soup.select_one("span[class*='location'], .location--items")
        district = location_el.get_text(strip=True).split(",")[-1].strip() if location_el else ""

        date_el = soup.select_one("span[class*='date'], .date--3hBKY")
        posted_date = date_el.get_text(strip=True) if date_el else ""

        # ── Item Details (key-value pairs on the listing page) ─────────────────
        details = {}
        for row in soup.select("li[class*='detail']"):
            label_el = row.select_one("span[class*='label']")
            value_el = row.select_one("span[class*='value']")
            if label_el and value_el:
                details[label_el.get_text(strip=True).lower()] = value_el.get_text(strip=True)

        # ── Fallback attribute extraction from title ───────────────────────────
        storage_raw = (
            details.get("storage", "")
            or details.get("internal storage", "")
            or extract_storage(title)
        )
        ram_raw = (
            details.get("ram", "")
            or details.get("memory", "")
            or extract_ram(title)
        )
        condition = details.get("condition", details.get("item condition", "Used"))
        warranty  = details.get("warranty", "No")
        brand     = details.get("brand", extract_brand(title))
        # Model = title minus brand prefix
        model = re.sub(rf"(?i){re.escape(brand)}", "", title).strip()

        return {
            "title":        title,
            "brand":        brand,
            "model":        model,
            "storage_raw":  storage_raw,
            "ram_raw":      ram_raw,
            "condition":    condition,
            "warranty":     warranty,
            "district":     district,
            "posted_date":  posted_date,
            "price_raw":    price_raw,
        }
    except Exception as exc:
        logger.warning(f"Failed to parse listing {url}: {exc}")
        return None


def get_listing_urls(driver: webdriver.Chrome, page: int) -> list[str]:
    """Fetch listing URLs from a search results page."""
    url = f"{BASE_URL}?sort_by=date&page={page}"
    try:
        driver.get(url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a[class*='card']"))
        )
        time.sleep(random.uniform(1.0, 2.0))
        soup  = BeautifulSoup(driver.page_source, "lxml")
        links = soup.select("a[class*='card--3wd']")
        if not links:
            # Fallback selector
            links = soup.select("ul.items--3SWxi li a, a[href*='/en/ad/']")
        urls  = [
            "https://ikman.lk" + a["href"] if a["href"].startswith("/") else a["href"]
            for a in links if a.get("href") and "/en/ad/" in a.get("href", "")
        ]
        return list(dict.fromkeys(urls))   # deduplicate
    except Exception as exc:
        logger.warning(f"Could not load page {page}: {exc}")
        return []


# ── Main Scraping Loop ─────────────────────────────────────────────────────────
def scrape(max_pages: int = MAX_PAGES) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    driver  = create_driver()
    records = []
    seen    = set()

    try:
        for page in range(1, max_pages + 1):
            logger.info(f"── Scraping page {page}/{max_pages} ──")
            listing_urls = get_listing_urls(driver, page)

            if not listing_urls:
                logger.info("No listings found on this page. Stopping.")
                break

            for idx, url in enumerate(listing_urls, 1):
                if url in seen:
                    continue
                seen.add(url)

                logger.info(f"  [{idx}/{len(listing_urls)}] {url}")
                data = parse_listing_page(driver, url)
                if data and data.get("price_raw"):
                    records.append(data)

                # Polite delay between individual listings
                time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

            logger.info(f"  Collected so far: {len(records)} listings")

            # If we have enough records, stop early
            if len(records) >= 2000:
                logger.info("Reached 2000 records target.")
                break

            # Delay between pages
            time.sleep(random.uniform(MIN_DELAY + 1, MAX_DELAY + 1))

    finally:
        driver.quit()

    # ── Write CSV ──────────────────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"\n✓ Saved {len(records)} listings → {OUTPUT_CSV}")


if __name__ == "__main__":
    scrape()
