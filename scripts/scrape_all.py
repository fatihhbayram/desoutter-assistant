#!/usr/bin/env python3
"""
Scrape all configured categories and save to MongoDB
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
    """Main execution"""
    
    start_time = datetime.now()
    
    try:
        logger.info("=" * 80)
        logger.info("🚀 Starting Desoutter scraper (ALL CATEGORIES)")
        logger.info(f"Categories to scrape: {len(CATEGORIES)}")
        logger.info("=" * 80)
        
        all_products = []
        
        # Scrape each category
        async with DesoutterScraper() as scraper:
            for category_key, category_config in CATEGORIES.items():
                logger.info(f"\n📂 Category: {category_config['name']}")
                
                products = await scraper.scrape_category(category_config)
                all_products.extend(products)
                
                logger.info(f"   Products from this category: {len(products)}")
        
        if not all_products:
            logger.warning("⚠️  No products found in any category")
            return
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"📊 Scraping Summary:")
        logger.info(f"   Total products scraped: {len(all_products)}")
        logger.info(f"   Categories processed: {len(CATEGORIES)}")
        logger.info(f"{'=' * 80}")
        
        # Save to MongoDB
        logger.info(f"\n💾 Saving {len(all_products)} products to MongoDB...")
        
        with MongoDBClient() as db:
            # Optional: Clear old data (uncomment if needed)
            # logger.info("🗑️  Clearing old data...")
            # db.clear_collection()
            
            # Create indexes
            db.create_indexes()
            
            # Bulk upsert
            stats = db.bulk_upsert([p.to_dict() for p in all_products])
            
            logger.info(f"\n✅ MongoDB save completed:")
            logger.info(f"   - New products: {stats['inserted']}")
            logger.info(f"   - Updated products: {stats['modified']}")
            logger.info(f"   - Total in DB: {db.count_products()}")
        
        # Final summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n{'=' * 80}")
        logger.info(f"✅ ALL SCRAPING COMPLETED SUCCESSFULLY!")
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
