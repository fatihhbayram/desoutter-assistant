#!/usr/bin/env python3
"""
Scrape Controller Units (CVIR, CVIL, Connect, ESP, etc.)
Based on scrape_cvi3_function_units.py
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

# Controller Unit URLs
CONTROLLER_URLS = {
    # Axon Terminal
    "697650": "https://www.desouttertools.com/en/p/axon-terminal-697650",
    
    # CVIR II
    "110918": "https://www.desouttertools.com/en/p/cvir-ii-110918",
    
    # CVIxS
    "147522": "https://www.desouttertools.com/en/p/cvixs-147522",
    
    # ESPC Controller
    "167639": "https://www.desouttertools.com/en/p/espc-controller-167639",
    
    # CVIC II H2
    "110917": "https://www.desouttertools.com/en/p/cvic-ii-h2-110917",
    
    # CVIC II L2
    "110916": "https://www.desouttertools.com/en/p/cvic-ii-l2-110916",
    
    # ESP Low Voltage
    "110928": "http://desouttertools.com/en/p/esp-low-voltage-screwdriver-controller-range-110928",
    
    # ESP2A Low Voltage
    "110927": "https://www.desouttertools.com/en/p/esp2a-low-voltage-screwdriver-controller-110927",
    
    # CVIL II
    "110919": "https://www.desouttertools.com/en/p/cvil-ii-110919",
    
    # Connect Industrial Smart Hub
    "110912": "https://www.desouttertools.com/en/p/connect-industrial-smart-hub-110912",
}


async def scrape_controller(session, product_id, url):
    """Scrape a single controller unit"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status != 200:
                logger.warning(f"  ❌ {product_id}: HTTP {response.status}")
                return None
            
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title (h1)
            title_tag = soup.find('h1')
            title = title_tag.get_text(strip=True) if title_tag else f"Controller {product_id}"
            
            # Extract description
            description = ""
            # Try multiple selectors for description
            desc_selectors = [
                ('div', {'class': 'overview'}),
                ('div', {'class': 'description'}),
                ('div', {'class': 'product-description'}),
            ]
            for tag, attrs in desc_selectors:
                desc_section = soup.find(tag, attrs)
                if desc_section:
                    description = desc_section.get_text(strip=True)
                    break
            
            # Extract product images
            images = []
            img_selectors = [
                ('div', {'class': 'product-images'}),
                ('div', {'class': 'gallery'}),
                ('div', {'class': 'image-gallery'}),
            ]
            for tag, attrs in img_selectors:
                img_container = soup.find(tag, attrs)
                if img_container:
                    img_tags = img_container.find_all('img')
                    images = [img.get('src', '') for img in img_tags if img.get('src')]
                    break
            
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
            
            # Extract variants (for pages with multiple models)
            variants = []
            variants_section = soup.find(id='variants')
            if variants_section:
                variant_items = variants_section.find_all(['li', 'div'], class_=lambda x: x and ('variant' in x.lower() or 'model' in x.lower()))
                for item in variant_items:
                    variant_text = item.get_text(strip=True)
                    if variant_text:
                        variants.append(variant_text)
            
            controller_data = {
                "product_id": product_id,
                "title": title,
                "description": description,
                "images": images,
                "specifications": specs,
                "variants": variants,
                "category": "Controllers",
                "url": url,
                "scraped_at": datetime.now().isoformat(),
            }
            
            variant_info = f" ({len(variants)} variants)" if variants else ""
            logger.info(f"  ✅ {product_id}: {title}{variant_info}")
            return controller_data
            
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
        logger.info("🚀 Starting Controller Units Scraper")
        logger.info(f"Units to scrape: {len(CONTROLLER_URLS)}")
        logger.info("=" * 80)
        
        controllers = []
        
        # Create session and scrape all controllers concurrently
        async with aiohttp.ClientSession() as session:
            logger.info("\n📡 Scraping controllers...")
            
            tasks = [
                scrape_controller(session, product_id, url)
                for product_id, url in CONTROLLER_URLS.items()
            ]
            
            results = await asyncio.gather(*tasks)
            controllers = [c for c in results if c is not None]
        
        if not controllers:
            logger.warning("⚠️  No controllers scraped successfully")
            return
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"📊 Scraping Summary:")
        logger.info(f"   Controllers scraped: {len(controllers)}/{len(CONTROLLER_URLS)}")
        logger.info(f"{'=' * 80}")
        
        # Save to MongoDB in tool_units collection
        logger.info(f"\n💾 Saving {len(controllers)} controllers to MongoDB...")
        
        with MongoDBClient(collection_name="tool_units") as db:
            # Create indexes
            db.create_indexes()
            
            # Bulk upsert
            stats = db.bulk_upsert(controllers)
            
            logger.info(f"\n✅ MongoDB save completed:")
            logger.info(f"   - New controllers: {stats['inserted']}")
            logger.info(f"   - Updated controllers: {stats['modified']}")
            logger.info(f"   - Total units in DB: {db.count_products()}")
        
        # Final summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n{'=' * 80}")
        logger.info(f"✅ CONTROLLER SCRAPING COMPLETED!")
        logger.info(f"   Total time: {elapsed:.2f} seconds")
        logger.info(f"{'=' * 80}")
        
    except Exception as e:
        logger.error(f"❌ Script failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
