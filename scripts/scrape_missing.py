#!/usr/bin/env python3
"""
Scrape missing series with rate limit handling.
Waits between each request to avoid HTTP 429.
"""

import asyncio
import logging
import sys
from datetime import datetime

sys.path.insert(0, '/app')

from src.scraper.desoutter_scraper import DesoutterScraper
from src.database.mongo_client import MongoDBClient

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Atlanan seriler (rate limit nedeniyle)
MISSING_SERIES = {
    "cable_tightening": [
        ("SLBN", "https://www.desouttertools.com/en/p/slbn-low-voltage-screwdriver-with-clutch-shut-off-27324"),
        ("E-Pulse", "https://www.desouttertools.com/en/p/e-pulse-electric-pulse-pistol-corded-transducerized-nutrunner-27350"),
        ("EFD", "https://www.desouttertools.com/en/p/efd-electric-fixtured-direct-nutrunner-130856"),
        ("EFM", "https://www.desouttertools.com/en/p/efm-electric-fixtured-multi-nutrunner-191845"),
        ("ERF", "https://www.desouttertools.com/en/p/erf-fixtured-electric-spindles-326679"),
        ("EFMA", "https://www.desouttertools.com/en/p/efma-transducerized-angle-head-spindle-718240"),
        ("EFBCI", "https://www.desouttertools.com/en/p/efbci-fast-integration-spindles-straight-718237"),
        ("EFBCIT", "https://www.desouttertools.com/en/p/efbcit-fast-integration-spindles-straight-telescopic-718238"),
        ("EFBCA", "https://www.desouttertools.com/en/p/efbca-fast-integration-spindles-angled-715011"),
    ],
    "electric_drilling": [
        ("XPB Modular", "https://www.desouttertools.com/en/p/xpb-modular-164687"),
        ("XPB One", "https://www.desouttertools.com/en/p/xpb-one-164685"),
        ("Tightening Head", "https://www.desouttertools.com/en/p/tightening-head-679250"),
        ("Drilling Head", "https://www.desouttertools.com/en/p/drilling-head-679249"),
    ],
}

# Her istek arasında bekleme (saniye)
DELAY_BETWEEN_SERIES = 30  # 30 saniye
DELAY_BETWEEN_PRODUCTS = 3  # 3 saniye


async def scrape_single_series(scraper: DesoutterScraper, name: str, url: str, category: str) -> list:
    """Tek bir seriyi scrape et."""
    logger.info(f"\n{'='*60}")
    logger.info(f"📦 Scraping: {name}")
    logger.info(f"   URL: {url}")
    logger.info(f"   Category: {category}")
    logger.info(f"{'='*60}")
    
    try:
        products = await scraper.scrape_series(url, category, f"https://www.desouttertools.com/en/c/{category.replace('_', '-')}-tools")
        logger.info(f"✅ {name}: {len(products)} ürün bulundu")
        return products
    except Exception as e:
        logger.error(f"❌ {name}: Hata - {e}")
        return []


async def main():
    logger.info("="*60)
    logger.info("🚀 Missing Series Scraper (Rate Limit Safe)")
    logger.info(f"   Delay between series: {DELAY_BETWEEN_SERIES}s")
    logger.info("="*60)
    
    all_products = []
    scraper = DesoutterScraper()
    
    total_series = sum(len(series) for series in MISSING_SERIES.values())
    current = 0
    
    for category, series_list in MISSING_SERIES.items():
        logger.info(f"\n📂 Category: {category}")
        
        for name, url in series_list:
            current += 1
            logger.info(f"\n[{current}/{total_series}] Processing {name}...")
            
            products = await scrape_single_series(scraper, name, url, category)
            all_products.extend(products)
            
            if current < total_series:
                logger.info(f"⏳ Waiting {DELAY_BETWEEN_SERIES}s before next series...")
                await asyncio.sleep(DELAY_BETWEEN_SERIES)
    
    # MongoDB'ye kaydet
    if all_products:
        logger.info(f"\n💾 Saving {len(all_products)} products to MongoDB...")
        mongo = MongoDBClient()
        result = await mongo.bulk_smart_upsert(all_products)
        
        logger.info(f"✅ MongoDB save completed:")
        logger.info(f"   - New products: {result['inserted']}")
        logger.info(f"   - Updated products: {result['updated']}")
        logger.info(f"   - Errors: {result['errors']}")
        
        # Toplam sayı
        total = await mongo.get_product_count()
        logger.info(f"   - Total in DB: {total}")
        
        await mongo.close()
    else:
        logger.warning("⚠️ No products found!")
    
    # Özet
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Summary:")
    logger.info(f"   Total series attempted: {total_series}")
    logger.info(f"   Total products scraped: {len(all_products)}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
