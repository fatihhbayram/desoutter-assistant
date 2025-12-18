"""
Main Desoutter scraper implementation - Schema v2 Support
"""
import asyncio
from datetime import datetime
from typing import List, Optional
from config import BASE_URL
from src.database.models import ProductModel, WirelessInfo, PlatformConnection, ModularSystem
from src.scraper.parsers import ProductParser
from src.scraper.product_categorizer import categorize_product, detect_tool_category
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
        category: str = "",
        category_url: str = ""
    ) -> List[ProductModel]:
        """
        Scrape all products from a series page with Schema v2 categorization.
        
        Args:
            series_url: URL of the series page
            category: Category name (legacy)
            category_url: Category page URL for tool_category detection
            
        Returns:
            List of ProductModel objects (Schema v2)
        """
        logger.info("=" * 80)
        logger.info(f"Starting scrape for series: {series_url}")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Detect tool category from URL
        tool_category = detect_tool_category(category_url, category)
        logger.info(f"Tool category: {tool_category}")
        
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
        
        # Step 4: Parse product data with Schema v2 categorization
        logger.info("Step 4/4: Parsing product data (Schema v2)...")
        products = []
        
        for i, (part_number, product_html, product_url) in enumerate(
            zip(part_numbers, product_htmls, product_urls), 1
        ):
            if not product_html:
                logger.warning(f"[{i}/{len(part_numbers)}] No HTML for {part_number}")
                continue
            
            try:
                # Parse HTML details (legacy)
                details = self.parser.parse_product_details(product_html, part_number)
                
                # Apply Schema v2 categorization
                categorization = categorize_product(
                    model_name=details.get("model_name", part_number),
                    part_number=part_number,
                    series_name=series_name,
                    category_url=category_url,
                    legacy_category=category,
                    description=details.get("description", ""),
                    html_content=product_html,
                    legacy_wireless=details.get("wireless_communication", "No")
                )
                
                # Build sub-models if applicable
                wireless = None
                platform_connection = None
                modular_system = None
                
                if categorization.get("wireless"):
                    wireless = WirelessInfo(**categorization["wireless"])
                
                if categorization.get("platform_connection"):
                    platform_connection = PlatformConnection(**categorization["platform_connection"])
                
                if categorization.get("modular_system"):
                    modular_system = ModularSystem(**categorization["modular_system"])
                
                # Create product model (Schema v2)
                product = ProductModel(
                    product_id=part_number,
                    part_number=part_number,
                    series_name=series_name,
                    category=category,
                    product_url=product_url,
                    # Schema v2 fields
                    tool_category=categorization["tool_category"],
                    tool_type=categorization.get("tool_type"),
                    product_family=categorization["product_family"],
                    wireless=wireless,
                    platform_connection=platform_connection,
                    modular_system=modular_system,
                    schema_version=2,
                    # Legacy fields from parser
                    **details
                )
                
                products.append(product)
                
                # Log with category-specific info
                wifi_status = ""
                if wireless and wireless.capable:
                    wifi_status = " 📶"
                elif platform_connection:
                    wifi_status = f" 🔌 [{','.join(platform_connection.compatible_platforms[:2])}]"
                
                logger.info(f"[{i}/{len(part_numbers)}] ✓ {product.model_name} ({categorization['tool_category']}){wifi_status}")
                
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
        category: str = "",
        category_url: str = ""
    ) -> List[ProductModel]:
        """
        Scrape multiple series with Schema v2 support.
        
        Args:
            series_urls: List of series URLs
            category: Category name (legacy)
            category_url: Category page URL for tool_category detection
            
        Returns:
            Combined list of ProductModel objects
        """
        logger.info(f"Scraping {len(series_urls)} series...")
        
        all_products = []
        for i, url in enumerate(series_urls, 1):
            logger.info(f"\n[Series {i}/{len(series_urls)}]")
            products = await self.scrape_series(url, category, category_url)
            all_products.extend(products)
        
        logger.info(f"\n✅ Total products from all series: {len(all_products)}")
        return all_products
    
    async def scrape_category(self, category_config: dict) -> List[ProductModel]:
        """
        Scrape entire category with Schema v2 support.
        
        Args:
            category_config: Category configuration dictionary
                {
                    'name': 'Category Name',
                    'url': 'category_url',
                    'series': ['series_url1', 'series_url2', ...]
                }
                
        Returns:
            List of ProductModel objects (Schema v2)
        """
        category_name = category_config.get('name', '')
        category_url = category_config.get('url', '')
        series_urls = category_config.get('series', [])
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Scraping category: {category_name}")
        logger.info(f"Category URL: {category_url}")
        logger.info(f"Series count: {len(series_urls)}")
        logger.info(f"{'=' * 80}\n")
        
        return await self.scrape_multiple_series(series_urls, category_name, category_url)
