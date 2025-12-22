#!/usr/bin/env python3
"""
Sequential Scraper - Rate limit safe version.
Fetches products one by one with delays to avoid HTTP 429.
"""

import asyncio
import logging
import sys
from datetime import datetime
import aiohttp

sys.path.insert(0, '/app')

from config import BASE_URL, USER_AGENT
from src.scraper.parsers import ProductParser
from src.scraper.product_categorizer import categorize_product, detect_tool_category
from src.database.models import ProductModel
from src.database.mongo_client import MongoDBClient

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
DELAY_BETWEEN_PRODUCTS = 2.0  # 2 seconds between each product
DELAY_BETWEEN_SERIES = 30.0   # 30 seconds between series
REQUEST_TIMEOUT = 30


# Missing series
MISSING_SERIES = {
    "cable_tightening": [
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


async def fetch_with_retry(session: aiohttp.ClientSession, url: str, max_retries: int = 5) -> str:
    """Fetch URL with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                elif response.status == 429:
                    wait_time = 10 * (attempt + 1)  # 10, 20, 30, 40, 50 seconds
                    logger.warning(f"⏳ Rate limited (429). Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return ""
        except Exception as e:
            logger.error(f"Error: {e}, attempt {attempt + 1}")
            await asyncio.sleep(5 * (attempt + 1))
    
    return ""


async def scrape_series_sequential(
    session: aiohttp.ClientSession,
    parser: ProductParser,
    series_name: str,
    series_url: str,
    category: str
) -> list:
    """Scrape a series by fetching products one by one."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📦 {series_name}")
    logger.info(f"   URL: {series_url}")
    logger.info(f"{'='*60}")
    
    # Step 1: Fetch series page
    series_html = await fetch_with_retry(session, series_url)
    if not series_html:
        logger.error(f"❌ Could not fetch series page")
        return []
    
    # Step 2: Extract part numbers
    series_title = parser.extract_series_name(series_html)
    part_numbers = parser.extract_part_numbers(series_html)
    
    if not part_numbers:
        logger.warning(f"⚠️ No products found in series")
        return []
    
    logger.info(f"📋 Found {len(part_numbers)} products")
    
    # Step 3: Fetch products ONE BY ONE with delay
    products = []
    
    for i, pn in enumerate(part_numbers, 1):
        product_url = f"{BASE_URL}/en/products/{pn}"
        
        # Wait between products
        if i > 1:
            await asyncio.sleep(DELAY_BETWEEN_PRODUCTS)
        
        # Fetch product
        product_html = await fetch_with_retry(session, product_url)
        
        if not product_html:
            logger.warning(f"[{i}/{len(part_numbers)}] ⚠️ Failed: {pn}")
            continue
        
        # Parse product
        try:
            parsed = parser.parse_product_page(
                product_html,
                series_name=series_title,
                series_url=series_url
            )
            
            if parsed:
                # Apply categorization
                categorized = categorize_product(parsed, category)
                
                product = ProductModel(
                    part_number=parsed.get('part_number', pn),
                    name=parsed.get('name', ''),
                    description=parsed.get('description', ''),
                    product_category=categorized.product_category,
                    tool_type=categorized.tool_type,
                    is_wireless=categorized.is_wireless,
                    platform_connection=categorized.platform_connection,
                    modular_system=categorized.modular_system,
                    wireless_info=categorized.wireless_info,
                    series_name=series_title,
                    series_url=series_url,
                    product_url=product_url,
                    specifications=parsed.get('specifications', {}),
                    features=parsed.get('features', []),
                    images=parsed.get('images', []),
                    documents=parsed.get('documents', []),
                    related_products=parsed.get('related_products', []),
                    scraped_at=datetime.utcnow()
                )
                products.append(product)
                
                # Show progress
                status = "🔋" if product.is_wireless else "🔌"
                logger.info(f"[{i}/{len(part_numbers)}] ✅ {product.name} {status}")
            else:
                logger.warning(f"[{i}/{len(part_numbers)}] ⚠️ Parse failed: {pn}")
                
        except Exception as e:
            logger.error(f"[{i}/{len(part_numbers)}] ❌ Error parsing {pn}: {e}")
    
    logger.info(f"✅ {series_name}: {len(products)}/{len(part_numbers)} products scraped")
    return products


async def main():
    logger.info("="*60)
    logger.info("🚀 Sequential Scraper (Rate Limit Safe)")
    logger.info(f"   Product delay: {DELAY_BETWEEN_PRODUCTS}s")
    logger.info(f"   Series delay: {DELAY_BETWEEN_SERIES}s")
    logger.info("="*60)
    
    parser = ProductParser()
    all_products = []
    
    total_series = sum(len(s) for s in MISSING_SERIES.values())
    current = 0
    
    headers = {'User-Agent': USER_AGENT}
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        for category, series_list in MISSING_SERIES.items():
            logger.info(f"\n📂 Category: {category}")
            
            for series_name, series_url in series_list:
                current += 1
                logger.info(f"\n[{current}/{total_series}] Starting {series_name}...")
                
                products = await scrape_series_sequential(
                    session, parser, series_name, series_url, category
                )
                all_products.extend(products)
                
                # Wait between series
                if current < total_series:
                    logger.info(f"\n⏳ Waiting {DELAY_BETWEEN_SERIES}s before next series...")
                    await asyncio.sleep(DELAY_BETWEEN_SERIES)
    
    # Save to MongoDB
    if all_products:
        logger.info(f"\n💾 Saving {len(all_products)} products to MongoDB...")
        mongo = MongoDBClient()
        
        # Convert to dict for bulk upsert
        product_dicts = [p.model_dump() for p in all_products]
        result = await mongo.bulk_smart_upsert(product_dicts)
        
        logger.info(f"✅ MongoDB save completed:")
        logger.info(f"   - New: {result.get('inserted', 0)}")
        logger.info(f"   - Updated: {result.get('updated', 0)}")
        logger.info(f"   - Errors: {result.get('errors', 0)}")
        
        total = await mongo.get_product_count()
        logger.info(f"   - Total in DB: {total}")
        
        await mongo.close()
    else:
        logger.warning("⚠️ No products scraped!")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Summary:")
    logger.info(f"   Total series: {total_series}")
    logger.info(f"   Total products: {len(all_products)}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
