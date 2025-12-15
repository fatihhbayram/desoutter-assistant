#!/usr/bin/env python3
"""
Scrape single series and save to MongoDB
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scraper import DesoutterScraper
from src.database import MongoDBClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def main(print_product_id: str = None):
    """Main execution"""
    
    # Configuration
    SERIES_URL = "https://www.desouttertools.com/en/p/epb-transducerized-pistol-battery-tool-27374"
    CATEGORY = "Battery Tightening Tools"
    
    try:
        # Scrape products
        logger.info("🚀 Starting Desoutter scraper (single series mode)")
        
        async with DesoutterScraper() as scraper:
            products = await scraper.scrape_series(SERIES_URL, CATEGORY)
        
        if not products:
            logger.warning("⚠️  No products found")
            return
        
        # Save to MongoDB
        logger.info(f"\n💾 Saving {len(products)} products to MongoDB...")
        
        with MongoDBClient() as db:
            # Optional: Clear old data
            # db.clear_collection()
            
            # Create indexes
            db.create_indexes()
            
            # Prepare product dicts and log image_url presence for debugging
            product_dicts = [p.to_dict() for p in products]
            print("\n" + "="*80)
            print("🔍 DEBUG: Checking image_url in scraped products:")
            print("="*80)
            for pd in product_dicts:
                print(f"   product_id={pd.get('product_id')} image_url={pd.get('image_url')}")

            # Bulk upsert
            stats = db.bulk_upsert(product_dicts)
            
            print("\n" + "="*80)
            print("🔍 DEBUG: Verifying image_url in MongoDB:")
            print("="*80)
            for pd in product_dicts[:5]:  # Check first 5
                pid = pd.get('product_id')
                docs = db.get_products({'product_id': pid}, limit=1)
                if docs:
                    print(f"   DB product_id={pid} stored_image_url={docs[0].get('image_url')}")

            logger.info(f"\n✅ MongoDB save completed:")
            logger.info(f"   - New products: {stats['inserted']}")
            logger.info(f"   - Updated products: {stats['modified']}")
            logger.info(f"   - Total in DB: {db.count_products()}")
            # If requested, fetch and pretty-print the full document for a specific product_id
            if print_product_id:
                try:
                    docs = db.get_products({'product_id': print_product_id}, limit=1)
                    if docs:
                        doc = docs[0]
                        pretty = json.dumps(doc, default=str, indent=2)
                        logger.info(f"\nFull DB document for product_id={print_product_id}:\n{pretty}")
                    else:
                        logger.warning(f"Requested product_id={print_product_id} not found in DB")
                except Exception as e:
                    logger.error(f"Error fetching full document for {print_product_id}: {e}")
        
        logger.info("\n✅ Scraping completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape single series and optionally print a product from MongoDB')
    parser.add_argument('--print-product', dest='print_product', default=os.getenv('PRINT_PRODUCT_ID'),
                        help='product_id to fetch and pretty-print from MongoDB after saving')
    args = parser.parse_args()

    asyncio.run(main(print_product_id=args.print_product))
