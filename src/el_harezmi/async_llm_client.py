"""
Async LLM Client Wrapper

Provides async interface for LLM clients used in El-Harezmi pipeline.
Wraps sync Ollama client for async usage.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AsyncLLMClient:
    """
    Async wrapper for LLM clients.
    
    Used by El-Harezmi Stage 3 for structured information extraction.
    """
    
    def __init__(self, ollama_client=None):
        """
        Initialize async LLM client.
        
        Args:
            ollama_client: Sync OllamaClient instance (optional, lazy-loaded)
        """
        self._ollama = ollama_client
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialize Ollama client"""
        if not self._initialized:
            if self._ollama is None:
                try:
                    from src.llm.ollama_client import OllamaClient
                    self._ollama = OllamaClient()
                    logger.info("Ollama client initialized for El-Harezmi")
                except Exception as e:
                    logger.error(f"Failed to initialize Ollama: {e}")
                    self._ollama = None
            self._initialized = True
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        system: Optional[str] = None
    ) -> str:
        """
        Generate LLM response asynchronously.
        
        Args:
            prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system: Optional system prompt
            
        Returns:
            Generated text
        """
        self._ensure_initialized()
        
        if not self._ollama:
            logger.warning("No LLM client available")
            return ""
        
        try:
            # Run sync Ollama call in thread pool
            response = await asyncio.to_thread(
                self._ollama.generate,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system=system
            )
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""
    
    async def is_available(self) -> bool:
        """Check if LLM is available"""
        self._ensure_initialized()
        return self._ollama is not None


# Singleton instance
_async_llm_client: Optional[AsyncLLMClient] = None


def get_async_llm_client() -> AsyncLLMClient:
    """Get or create async LLM client singleton"""
    global _async_llm_client
    
    if _async_llm_client is None:
        _async_llm_client = AsyncLLMClient()
    
    return _async_llm_client
