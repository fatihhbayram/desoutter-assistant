"""
Async HTTP client utilities
"""
import asyncio
from typing import Optional, List
import aiohttp
from config import USER_AGENT, REQUEST_TIMEOUT, MAX_CONCURRENT_REQUESTS
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
                            # Rate limited - wait longer
                            wait_time = 5 * (attempt + 1)
                            logger.warning(f"HTTP 429 (Rate Limited) for {url}, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.warning(f"HTTP {response.status} for {url}")
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout fetching {url} (attempt {attempt + 1}/{retry})")
                    if attempt < retry - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                except Exception as e:
                    logger.error(f"Error fetching {url}: {e}")
                    if attempt < retry - 1:
                        await asyncio.sleep(2 ** attempt)
            
            self.stats['failed'] += 1
            logger.error(f"Failed to fetch {url} after {retry} attempts")
            return ""
    
    async def fetch_all(self, urls: List[str]) -> List[str]:
        """
        Fetch multiple URLs concurrently
        
        Args:
            urls: List of URLs to fetch
            
        Returns:
            List of HTML contents
        """
        logger.info(f"Fetching {len(urls)} URLs in parallel...")
        tasks = [self.fetch(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        html_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception for {urls[i]}: {result}")
                html_results.append("")
            else:
                html_results.append(result)
        
        return html_results
