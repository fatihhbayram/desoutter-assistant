"""
Ollama LLM Client
Interface with local Ollama instance
"""
import httpx
from typing import Dict, List, Optional, Generator
from config.ai_settings import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT,
    OLLAMA_MAX_TOKENS
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OllamaClient:
    """Client for Ollama LLM API"""
    
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        temperature: float = OLLAMA_TEMPERATURE,
        timeout: int = OLLAMA_TIMEOUT
    ):
        """
        Initialize Ollama client
        
        Args:
            base_url: Ollama API base URL
            model: Model name
            temperature: Sampling temperature
            timeout: Request timeout
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        
        logger.info(f"Ollama client initialized: {base_url}")
        logger.info(f"Model: {model}")
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama"""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                logger.info(f"✅ Connected to Ollama")
                logger.info(f"   Available models: {model_names}")
                
                if self.model not in model_names:
                    logger.warning(f"⚠️  Model '{self.model}' not found. Available: {model_names}")
            else:
                logger.warning(f"⚠️  Ollama connection test failed: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Cannot connect to Ollama: {e}")
            logger.error(f"   Make sure Ollama is running at {self.base_url}")
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate completion from Ollama
        
        Args:
            prompt: User prompt
            system: System prompt
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature or self.temperature,
                }
            }
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            if system:
                payload["system"] = system
            
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Ollama error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating from Ollama: {e}")
            return ""
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Chat completion (multi-turn conversation)
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature or self.temperature,
                }
            }
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                logger.error(f"Ollama chat error: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return ""
    
    def stream_generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Generator[str, None, None]:
        """
        Stream generation from Ollama
        
        Args:
            prompt: User prompt
            system: System prompt
            temperature: Override default temperature
            
        Yields:
            Generated text chunks
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature or self.temperature,
                }
            }
            
            if system:
                payload["system"] = system
            
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            ) as response:
                for line in response.iter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                            
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield ""
