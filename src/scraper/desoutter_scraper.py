"""
Main Desoutter scraper implementation
"""
import asyncio
from datetime import datetime
from typing import List, Optional
from config import BASE_URL
from src.database.models import ProductModel
from src.scraper.parsers import ProductParser
from src.utils.http_client import AsyncHTTPClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DesoutterScraper:
    """Main scraper for Desoutter Tools website"""
    
    def __init__(self):
        """Initialize scraper"""
        self.base_url = BASE_URL
        self.parser = ProductParser()
        self.http_client: Optional[AsyncHTTPClient] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.http_client = AsyncHTTPClient()
        await self.http_client.__aenter__()
        return self
    
    async def __aexit__(self, *args):
        """Async context manager exit"""
        if self.http_client:
            await self.http_client.__aexit__(*args)
    
    async def scrape_series(
        self,
        series_url: str,
        category: str = ""
    ) -> List[ProductModel]:
        """
        Scrape all products from a series page
        
        Args:
            series_url: URL of the series page
            category: Category name
            
        Returns:
            List of ProductModel objects
        """
        logger.info("=" * 80)
        logger.info(f"Starting scrape for series: {series_url}")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Step 1: Fetch series page
        logger.info("Step 1/4: Fetching series page...")
        series_html = await self.http_client.fetch(series_url)
        
        if not series_html:
            logger.error("Failed to fetch series page")
            return []
        
        # Step 2: Extract series info and part numbers
        logger.info("Step 2/4: Extracting product part numbers...")
        series_name = self.parser.extract_series_name(series_html)
        part_numbers = self.parser.extract_part_numbers(series_html)
        
        logger.info(f"Series: {series_name}")
        logger.info(f"Found {len(part_numbers)} products")
        
        if not part_numbers:
            logger.warning("No part numbers found")
            return []
        
        # Step 3: Fetch all product pages in parallel
        logger.info("Step 3/4: Fetching product pages in parallel...")
        product_urls = [
            f"{self.base_url}/en/products/{pn}" 
            for pn in part_numbers
        ]
        product_htmls = await self.http_client.fetch_all(product_urls)
        
        # Step 4: Parse product data
        logger.info("Step 4/4: Parsing product data...")
        products = []
        
        for i, (part_number, product_html, product_url) in enumerate(
            zip(part_numbers, product_htmls, product_urls), 1
        ):
            if not product_html:
                logger.warning(f"[{i}/{len(part_numbers)}] No HTML for {part_number}")
                continue
            
            try:
                # Parse HTML details
                details = self.parser.parse_product_details(product_html, part_number)
                
                # Create product model
                product = ProductModel(
                    product_id=part_number,
                    part_number=part_number,
                    series_name=series_name,
                    category=category,
                    product_url=product_url,
                    **details
                )
                
                products.append(product)
                logger.info(f"[{i}/{len(part_numbers)}] ✓ {product.model_name}")
                
            except Exception as e:
                logger.error(f"[{i}/{len(part_numbers)}] ✗ Error parsing {part_number}: {e}")
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 80)
        logger.info(f"✅ Scraping completed!")
        logger.info(f"   Series: {series_name}")
        logger.info(f"   Products found: {len(part_numbers)}")
        logger.info(f"   Products parsed: {len(products)}")
        logger.info(f"   Time elapsed: {elapsed:.2f} seconds")
        logger.info("=" * 80)
        
        return products
    
    async def scrape_multiple_series(
        self,
        series_urls: List[str],
        category: str = ""
    ) -> List[ProductModel]:
        """
        Scrape multiple series
        
        Args:
            series_urls: List of series URLs
            category: Category name
            
        Returns:
            Combined list of ProductModel objects
        """
        logger.info(f"Scraping {len(series_urls)} series...")
        
        all_products = []
        for i, url in enumerate(series_urls, 1):
            logger.info(f"\n[Series {i}/{len(series_urls)}]")
            products = await self.scrape_series(url, category)
            all_products.extend(products)
        
        logger.info(f"\n✅ Total products from all series: {len(all_products)}")
        return all_products
    
    async def scrape_category(self, category_config: dict) -> List[ProductModel]:
        """
        Scrape entire category
        
        Args:
            category_config: Category configuration dictionary
                {
                    'name': 'Category Name',
                    'url': 'category_url',
                    'series': ['series_url1', 'series_url2', ...]
                }
                
        Returns:
            List of ProductModel objects
        """
        category_name = category_config.get('name', '')
        series_urls = category_config.get('series', [])
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Scraping category: {category_name}")
        logger.info(f"Series count: {len(series_urls)}")
        logger.info(f"{'=' * 80}\n")
        
        return await self.scrape_multiple_series(series_urls, category_name)
