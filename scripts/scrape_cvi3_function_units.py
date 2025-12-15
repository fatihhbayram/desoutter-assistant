#!/usr/bin/env python3
"""
Scrape CVI3 Function Controller Units (615-series)
These are functional units that can be connected to tools via cables.
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import MongoDBClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# CVI3 Function Controller Unit URLs (615-series product IDs)
CVI3_UNITS_URLS = {
    "6159326910": "https://www.desouttertools.com/products/cvi3-multi-purpose-industrial-screwdriver-cvi3-0-3-brushless-6159326910",
    "6159326940": "https://www.desouttertools.com/products/cvi3-multi-purpose-industrial-screwdriver-cvi3-0-3-brushless-6159326940",
    "6159326970": "https://www.desouttertools.com/products/cvi3-multi-purpose-industrial-screwdriver-cvi3-0-3-brushless-6159326970",
    "6159326960": "https://www.desouttertools.com/products/cvi3-multi-purpose-industrial-screwdriver-cvi3-0-3-brushless-6159326960",
    "6159326950": "https://www.desouttertools.com/products/cvi3-multi-purpose-industrial-screwdriver-cvi3-0-3-brushless-6159326950",
    "6159327000": "https://www.desouttertools.com/products/cvi3-multi-purpose-industrial-screwdriver-cvi3-0-3-brushless-6159327000",
    "6159327010": "https://www.desouttertools.com/products/cvi3-multi-purpose-industrial-screwdriver-cvi3-0-3-brushless-6159327010",
}


async def scrape_unit(session, product_id, url):
    """Scrape a single CVI3 unit"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status != 200:
                logger.warning(f"  ❌ {product_id}: HTTP {response.status}")
                return None
            
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title_tag = soup.find('h1')
            title = title_tag.get_text(strip=True) if title_tag else f"CVI3 Unit {product_id}"
            
            # Extract description/overview
            description = ""
            overview_section = soup.find('div', class_='overview')
            if overview_section:
                description = overview_section.get_text(strip=True)
            
            # Extract product images
            images = []
            img_container = soup.find('div', class_='product-images')
            if img_container:
                img_tags = img_container.find_all('img')
                images = [img.get('src', '') for img in img_tags if img.get('src')]
            
            # Extract specifications
            specs = {}
            spec_section = soup.find('div', class_='specifications')
            if spec_section:
                spec_items = spec_section.find_all('div', class_='spec-item')
                for item in spec_items:
                    key = item.find('span', class_='spec-key')
                    value = item.find('span', class_='spec-value')
                    if key and value:
                        specs[key.get_text(strip=True)] = value.get_text(strip=True)
            
            unit_data = {
                "product_id": product_id,
                "title": title,
                "description": description,
                "images": images,
                "specifications": specs,
                "category": "CVI3 Function Controller",
                "url": url,
                "scraped_at": datetime.now().isoformat(),
            }
            
            logger.info(f"  ✅ {product_id}: {title}")
            return unit_data
            
    except asyncio.TimeoutError:
        logger.warning(f"  ⏱️  {product_id}: Timeout")
        return None
    except Exception as e:
        logger.warning(f"  ❌ {product_id}: {str(e)}")
        return None


async def main():
    """Main execution"""
    
    start_time = datetime.now()
    
    try:
        logger.info("=" * 80)
        logger.info("🚀 Starting CVI3 Function Controller Units Scraper")
        logger.info(f"Units to scrape: {len(CVI3_UNITS_URLS)}")
        logger.info("=" * 80)
        
        units = []
        
        # Create session and scrape all units concurrently
        async with aiohttp.ClientSession() as session:
            logger.info("\n📡 Scraping units...")
            
            tasks = [
                scrape_unit(session, product_id, url)
                for product_id, url in CVI3_UNITS_URLS.items()
            ]
            
            results = await asyncio.gather(*tasks)
            units = [u for u in results if u is not None]
        
        if not units:
            logger.warning("⚠️  No units scraped successfully")
            return
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"📊 Scraping Summary:")
        logger.info(f"   Units scraped: {len(units)}/{len(CVI3_UNITS_URLS)}")
        logger.info(f"{'=' * 80}")
        
        # Save to MongoDB in tool_units collection
        logger.info(f"\n💾 Saving {len(units)} units to MongoDB...")
        
        with MongoDBClient(collection_name="tool_units") as db:
            # Create indexes
            db.create_indexes()
            
            # Bulk upsert
            stats = db.bulk_upsert([u for u in units])
            
            logger.info(f"\n✅ MongoDB save completed:")
            logger.info(f"   - New units: {stats['inserted']}")
            logger.info(f"   - Updated units: {stats['modified']}")
            logger.info(f"   - Total units in DB: {db.count_products()}")
        
        # Final summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n{'=' * 80}")
        logger.info(f"✅ CVI3 UNITS SCRAPING COMPLETED!")
        logger.info(f"   Total time: {elapsed:.2f} seconds")
        logger.info(f"{'=' * 80}")
        
    except Exception as e:
        logger.error(f"❌ Script failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
