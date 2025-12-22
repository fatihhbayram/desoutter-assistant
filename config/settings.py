"""
Configuration settings for Desoutter scraper
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# MongoDB Settings
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
MONGO_DATABASE = os.getenv("MONGO_DATABASE", "desoutter")
MONGO_URI = os.getenv("MONGO_URI", f"mongodb://{MONGO_HOST}:{MONGO_PORT}/")

# Scraper Settings
BASE_URL = os.getenv("BASE_URL", "https://www.desouttertools.com")
USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; DesoutterScraper/1.0)")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "1"))  # Tek tek çek - rate limit önleme
DELAY_BETWEEN_REQUESTS = float(os.getenv("DELAY_BETWEEN_REQUESTS", "5"))  # Her istek arası 5 saniye

# Logging Settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / os.getenv("LOG_FILE", "data/logs/scraper.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Data Directories
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
EXPORTS_DIR = DATA_DIR / "exports"
CACHE_DIR = DATA_DIR / "cache"

# Create directories
for directory in [LOGS_DIR, EXPORTS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Scraping Targets
CATEGORIES = {
    "battery_tightening": {
        "name": "Battery Tightening Tools",
        "url": "https://www.desouttertools.com/en/c/battery-tightening-tools",
        "series": [
            "https://www.desouttertools.com/en/p/eab-transducerized-battery-tool-27367",
            "https://www.desouttertools.com/en/p/pulse-battery-electric-pistol-tool-27364",
            "https://www.desouttertools.com/en/p/pulse-battery-angle-head-707172",
            "https://www.desouttertools.com/en/p/epb-transducerized-pistol-battery-tool-27374",
            "https://www.desouttertools.com/en/p/eabs-eibs-wireless-transducerized-battery-tool-27369",
            "https://www.desouttertools.com/en/p/q-shield-c-smart-connected-wrench-192333",
            "https://www.desouttertools.com/en/p/e-lit-battery-clutch-tools-27370"
        ]
    },
    "cable_tightening": {
        "name": "Cable Tightening Tools",
        "url": "https://www.desouttertools.com/en/c/cable-tightening-tools",
        "series": [
            # Electric Nutrunners
            "https://www.desouttertools.com/en/p/epd-electric-pistol-direct-continuous-697261",
            "https://www.desouttertools.com/en/p/ead-transducerized-angle-head-electric-nutrunner-27328",
            "https://www.desouttertools.com/en/p/eid-transducerized-in-line-electric-nutrunner-27349",
            "https://www.desouttertools.com/en/p/erp-tranducerized-pistol-grip-nutrunner-27355",
            "https://www.desouttertools.com/en/p/erp-ht-handheld-high-torque-pistol-tool-147507",
            # Electric Screwdrivers
            "https://www.desouttertools.com/en/p/ers-electric-transducerized-screwdriver-27322",
            "https://www.desouttertools.com/en/p/ecs-current-controlled-screwdriver-27321",
            "https://www.desouttertools.com/en/p/erxs-transducerized-in-line-low-torque-screwdriver-147523",
            "https://www.desouttertools.com/en/p/slc-low-voltage-current-control-screwdriver-27327",
            "https://www.desouttertools.com/en/p/slbn-low-voltage-screwdriver-with-clutch-shut-off-27324",
            # Electric Pulse Tools
            "https://www.desouttertools.com/en/p/e-pulse-electric-pulse-pistol-corded-transducerized-nutrunner-27350",
            # Fixtured Spindles
            "https://www.desouttertools.com/en/p/efd-electric-fixtured-direct-nutrunner-130856",
            "https://www.desouttertools.com/en/p/efm-electric-fixtured-multi-nutrunner-191845",
            "https://www.desouttertools.com/en/p/erf-fixtured-electric-spindles-326679",
            "https://www.desouttertools.com/en/p/efma-transducerized-angle-head-spindle-718240",
            # Fast Integration Spindles
            "https://www.desouttertools.com/en/p/efbci-fast-integration-spindles-straight-718237",
            "https://www.desouttertools.com/en/p/efbcit-fast-integration-spindles-straight-telescopic-718238",
            "https://www.desouttertools.com/en/p/efbca-fast-integration-spindles-angled-715011"
        ]
    },
    "electric_drilling": {
        "name": "Electric Drilling Tools",
        "url": "https://www.desouttertools.com/en/c/electric-drilling-tools",
        "series": [
            "https://www.desouttertools.com/en/p/xpb-modular-164687",
            "https://www.desouttertools.com/en/p/xpb-one-164685",
            "https://www.desouttertools.com/en/p/tightening-head-679250",
            "https://www.desouttertools.com/en/p/drilling-head-679249"
        ]
    }
}

# Part Number Patterns - Extended for all product categories
PART_NUMBER_PATTERNS = [
    r'\b(6151[0-9]{6})\b',  # 6151 series (common)
    r'\b(8920[0-9]{6})\b',  # 8920 series
    r'\b(6153[0-9]{6})\b',  # 6153 series
    r'\b(6154[0-9]{6})\b',  # 6154 series
    r'\b(6159[0-9]{6})\b',  # 6159 series
    r'\b(8922[0-9]{6})\b',  # 8922 series
    r'\b(1462[0-9]{6})\b',  # 1462 series (drilling)
    r'\b(1465[0-9]{6})\b',  # 1465 series
]
