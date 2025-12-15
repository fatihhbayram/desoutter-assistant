#!/usr/bin/env python3
"""
Scrape a single category and save to MongoDB
Usage: python scripts/scrape_single_category.py <category_key>
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


async def main(category_key: str):
    """Main execution for single category"""
    
    if category_key not in CATEGORIES:
        logger.error(f"Unknown category: {category_key}")
        logger.info(f"Available categories: {list(CATEGORIES.keys())}")
        return
    
    category_config = CATEGORIES[category_key]
    start_time = datetime.now()
    
    try:
        logger.info("=" * 80)
        logger.info(f"🚀 Starting Desoutter scraper for: {category_config['name']}")
        logger.info(f"Series count: {len(category_config.get('series', []))}")
        logger.info("=" * 80)
        
        # Scrape category
        async with DesoutterScraper() as scraper:
            products = await scraper.scrape_category(category_config)
        
        if not products:
            logger.warning("⚠️  No products found")
            return
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"📊 Scraping Summary:")
        logger.info(f"   Total products scraped: {len(products)}")
        logger.info(f"{'=' * 80}")
        
        # Save to MongoDB
        logger.info(f"\n💾 Saving {len(products)} products to MongoDB...")
        
        with MongoDBClient() as db:
            db.create_indexes()
            stats = db.bulk_upsert([p.to_dict() for p in products])
            
            logger.info(f"\n✅ MongoDB save completed:")
            logger.info(f"   - New products: {stats['inserted']}")
            logger.info(f"   - Updated products: {stats['modified']}")
            logger.info(f"   - Total in DB: {db.count_products()}")
        
        # Final summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n{'=' * 80}")
        logger.info(f"✅ SCRAPING COMPLETED SUCCESSFULLY!")
        logger.info(f"   Total time: {elapsed:.2f} seconds")
        logger.info(f"{'=' * 80}")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/scrape_single_category.py <category_key>")
        print(f"Available categories: {list(CATEGORIES.keys())}")
        sys.exit(1)
    
    asyncio.run(main(sys.argv[1]))
