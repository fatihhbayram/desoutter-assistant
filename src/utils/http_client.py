"""
Async HTTP client utilities
"""
import asyncio
from typing import Optional, List
import aiohttp
from config import USER_AGENT, REQUEST_TIMEOUT, MAX_CONCURRENT_REQUESTS, DELAY_BETWEEN_REQUESTS
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AsyncHTTPClient:
    """Async HTTP client with rate limiting and error handling"""
    
    def __init__(
        self,
        timeout: int = REQUEST_TIMEOUT,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
        user_agent: str = USER_AGENT
    ):
        """
        Initialize HTTP client
        
        Args:
            timeout: Request timeout in seconds
            max_concurrent: Maximum concurrent requests
            user_agent: User agent string
        """
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.headers = {'User-Agent': user_agent}
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            headers=self.headers
        )
        return self
    
    async def __aexit__(self, *args):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            logger.info(f"HTTP Stats: {self.stats['success']} success, {self.stats['failed']} failed, {self.stats['total']} total")
    
    async def fetch(self, url: str, retry: int = 3) -> str:
        """
        Fetch HTML content from URL with retry logic
        
        Args:
            url: URL to fetch
            retry: Number of retries on failure
            
        Returns:
            HTML content as string
        """
        async with self.semaphore:
            self.stats['total'] += 1
            
            for attempt in range(retry):
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            self.stats['success'] += 1
                            return await response.text()
                        elif response.status == 429:
                            # Rate limited - wait much longer
                            wait_time = 60 * (attempt + 1)  # 60, 120, 180 saniye
                            logger.warning(f"HTTP 429 (Rate Limited) for {url}, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.warning(f"HTTP {response.status} for {url}")
                            await asyncio.sleep(10)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout fetching {url} (attempt {attempt + 1}/{retry})")
                    if attempt < retry - 1:
                        await asyncio.sleep(30)
                except Exception as e:
                    logger.error(f"Error fetching {url}: {e}")
                    if attempt < retry - 1:
                        await asyncio.sleep(30)
            
            self.stats['failed'] += 1
            logger.error(f"Failed to fetch {url} after {retry} attempts")
            return ""
    
    async def fetch_all(self, urls: List[str]) -> List[str]:
        """
        Fetch multiple URLs SEQUENTIALLY with delay between requests
        
        Args:
            urls: List of URLs to fetch
            
        Returns:
            List of HTML contents
        """
        logger.info(f"Fetching {len(urls)} URLs sequentially (delay: {DELAY_BETWEEN_REQUESTS}s)...")
        html_results = []
        
        for i, url in enumerate(urls):
            if i > 0:
                await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
            
            result = await self.fetch(url)
            html_results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i + 1}/{len(urls)}")
        
        return html_results
