#!/usr/bin/env python3
"""
Scrape all configured categories and save to MongoDB - Schema v2 Support
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CATEGORIES
from src.scraper import DesoutterScraper
from src.database import MongoDBClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def main():
    """Main execution with Schema v2 support"""
    
    start_time = datetime.now()
    
    try:
        logger.info("=" * 80)
        logger.info("🚀 Starting Desoutter scraper (ALL CATEGORIES) - Schema v2")
        logger.info(f"Categories to scrape: {len(CATEGORIES)}")
        logger.info("=" * 80)
        
        all_products = []
        category_stats = {}
        
        # Scrape each category
        async with DesoutterScraper() as scraper:
            for category_key, category_config in CATEGORIES.items():
                logger.info(f"\n📂 Category: {category_config['name']}")
                
                products = await scraper.scrape_category(category_config)
                all_products.extend(products)
                
                # Track per-category stats
                wifi_count = sum(1 for p in products if p.wireless and p.wireless.capable)
                category_stats[category_key] = {
                    "name": category_config["name"],
                    "count": len(products),
                    "wifi_capable": wifi_count
                }
                
                logger.info(f"   Products from this category: {len(products)}")
                if wifi_count > 0:
                    logger.info(f"   WiFi capable: {wifi_count}")
        
        if not all_products:
            logger.warning("⚠️  No products found in any category")
            return
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"📊 Scraping Summary (Schema v2):")
        logger.info(f"   Total products scraped: {len(all_products)}")
        logger.info(f"   Categories processed: {len(CATEGORIES)}")
        for key, stats in category_stats.items():
            logger.info(f"   - {stats['name']}: {stats['count']} products")
        logger.info(f"{'=' * 80}")
        
        # Save to MongoDB using smart upsert
        logger.info(f"\n💾 Saving {len(all_products)} products to MongoDB (Smart Upsert)...")
        
        with MongoDBClient() as db:
            # Create indexes (including Schema v2)
            db.create_indexes()
            
            # Smart bulk upsert - preserves existing non-empty values
            stats = db.bulk_smart_upsert([p.to_dict() for p in all_products])
            
            logger.info(f"\n✅ MongoDB save completed (Schema v2):")
            logger.info(f"   - New products: {stats['inserted']}")
            logger.info(f"   - Updated products: {stats['updated']}")
            logger.info(f"   - Errors: {stats['errors']}")
            logger.info(f"   - Total in DB: {db.count_products()}")
        
        # Final summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n{'=' * 80}")
        logger.info(f"✅ ALL SCRAPING COMPLETED SUCCESSFULLY! (Schema v2)")
        logger.info(f"   Total time: {elapsed:.2f} seconds")
        logger.info(f"   Average: {elapsed/len(all_products):.2f} sec/product")
        logger.info(f"{'=' * 80}")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
