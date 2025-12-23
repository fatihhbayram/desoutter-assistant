#!/usr/bin/env python3
"""
Scrape only missing products (not in MongoDB) for all configured categories/series.
"""
import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CATEGORIES

from src.scraper import DesoutterScraper
from src.database import MongoDBClient
from src.utils.logger import setup_logger
from src.scraper.product_categorizer import categorize_product
from src.database.models import ProductModel, WirelessInfo, PlatformConnection, ModularSystem

logger = setup_logger(__name__)

async def main():
    logger.info("=" * 80)
    logger.info("🚀 Starting scrape for missing products only!")
    logger.info("=" * 80)

    with MongoDBClient() as db:
        async with DesoutterScraper() as scraper:
            total_new = 0
            for category_key, category_config in CATEGORIES.items():
                logger.info(f"\n📂 Category: {category_config['name']}")
                for series_url in category_config.get('series', []):
                    logger.info(f"  🔎 Series: {series_url}")
                    # 1. Fetch part numbers from series page
                    series_html = await scraper.http_client.fetch(series_url)
                    if not series_html:
                        logger.warning(f"    ⚠️  Failed to fetch series page: {series_url}")
                        continue
                    part_numbers = scraper.parser.extract_part_numbers(series_html)
                    logger.info(f"    Found {len(part_numbers)} part numbers")
                    if not part_numbers:
                        continue
                    # 2. Check which part_numbers are missing in DB
                    existing = set(
                        p['part_number'] for p in db.collection.find({
                            'part_number': {'$in': part_numbers}
                        }, {'part_number': 1})
                    )
                    missing = [pn for pn in part_numbers if pn not in existing]
                    logger.info(f"    Missing in DB: {len(missing)}")
                    if not missing:
                        continue
                    # 3. Scrape only missing products
                    product_urls = [f"{scraper.base_url}/en/products/{pn}" for pn in missing]
                    product_htmls = await scraper.http_client.fetch_all(product_urls)
                    products = []
                    for i, (part_number, product_html, product_url) in enumerate(zip(missing, product_htmls, product_urls), 1):
                        if not product_html:
                            logger.warning(f"      [{i}/{len(missing)}] No HTML for {part_number}")
                            continue
                        try:
                            details = scraper.parser.parse_product_details(product_html, part_number)
                            categorization = categorize_product(
                                model_name=details.get("model_name", part_number),
                                part_number=part_number,
                                series_name=category_config['name'],
                                category_url=category_config['url'],
                                legacy_category=category_config['name'],
                                description=details.get("description", ""),
                                html_content=product_html,
                                legacy_wireless=details.get("wireless_communication", "No")
                            )
                            wireless = None
                            platform_connection = None
                            modular_system = None
                            if categorization.get("wireless"):
                                wireless = WirelessInfo(**categorization["wireless"])
                            if categorization.get("platform_connection"):
                                platform_connection = PlatformConnection(**categorization["platform_connection"])
                            if categorization.get("modular_system"):
                                modular_system = ModularSystem(**categorization["modular_system"])
                            product = ProductModel(
                                product_id=part_number,
                                part_number=part_number,
                                series_name=category_config['name'],
                                category=category_config['name'],
                                product_url=product_url,
                                tool_category=categorization["tool_category"],
                                tool_type=categorization.get("tool_type"),
                                product_family=categorization["product_family"],
                                wireless=wireless,
                                platform_connection=platform_connection,
                                modular_system=modular_system,
                                schema_version=2,
                                **details
                            )
                            products.append(product)
                            logger.info(f"      [{i}/{len(missing)}] ✓ {product.model_name}")
                        except Exception as e:
                            logger.error(f"      [{i}/{len(missing)}] ✗ Error parsing {part_number}: {e}")
                    # 4. Upsert missing products
                    if products:
                        stats = db.bulk_smart_upsert([p.to_dict() for p in products])
                        logger.info(f"    ➕ Inserted: {stats['inserted']}, Updated: {stats['updated']}, Errors: {stats['errors']}")
                        total_new += stats['inserted']
            logger.info(f"\n✅ Scraping completed. Total new products inserted: {total_new}")

if __name__ == "__main__":
    asyncio.run(main())
