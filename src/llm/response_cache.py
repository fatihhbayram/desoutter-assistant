"""
Response Caching Module - Phase 2.3
LRU cache for RAG responses to improve performance on repeated queries

Features:
- LRU (Least Recently Used) eviction policy
- TTL (Time To Live) support for cache entries
- Query similarity matching for near-duplicate detection
- Cache statistics and monitoring
- Thread-safe operations
"""
import hashlib
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from datetime import datetime, timedelta

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    key: str
    query: str
    response: Dict[str, Any]
    created_at: float
    last_accessed: float
    hit_count: int = 0
    ttl_seconds: int = 3600  # Default 1 hour
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL"""
        return time.time() - self.created_at > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds"""
        return time.time() - self.created_at
    
    def touch(self):
        """Update last accessed time and increment hit count"""
        self.last_accessed = time.time()
        self.hit_count += 1


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    total_queries: int = 0
    cache_size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_queries == 0:
            return 0.0
        return self.hits / self.total_queries
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate"""
        if self.total_queries == 0:
            return 0.0
        return self.misses / self.total_queries
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "total_queries": self.total_queries,
            "hit_rate": round(self.hit_rate * 100, 2),
            "miss_rate": round(self.miss_rate * 100, 2),
            "cache_size": self.cache_size,
            "max_size": self.max_size
        }


class ResponseCache:
    """
    LRU Cache for RAG responses
    
    Features:
    - Thread-safe operations
    - TTL-based expiration
    - Query normalization for better hit rate
    - Statistics tracking
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,
        enable_similarity_matching: bool = True,
        similarity_threshold: float = 0.95
    ):
        """
        Initialize response cache
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds (1 hour)
            enable_similarity_matching: Enable fuzzy query matching
            similarity_threshold: Threshold for similarity matching
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.enable_similarity_matching = enable_similarity_matching
        self.similarity_threshold = similarity_threshold
        
        # LRU cache using OrderedDict
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = CacheStats(max_size=max_size)
        
        # Query index for similarity matching
        self._query_index: Dict[str, str] = {}  # normalized_query -> cache_key
        
        logger.info(f"✅ ResponseCache initialized: max_size={max_size}, ttl={default_ttl}s")
    
    def _generate_cache_key(self, query: str, part_number: Optional[str] = None, language: str = "en") -> str:
        """
        Generate unique cache key from query parameters
        
        Args:
            query: The fault description query
            part_number: Optional product part number
            language: Response language
            
        Returns:
            MD5 hash as cache key
        """
        # Normalize query
        normalized = self._normalize_query(query)
        
        # Combine with other parameters
        key_string = f"{normalized}|{part_number or 'none'}|{language}"
        
        # Generate hash
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for better cache hits
        
        - Lowercase
        - Remove extra whitespace
        - Sort words (optional, for word-order invariance)
        """
        # Lowercase and strip
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def get(
        self,
        query: str,
        part_number: Optional[str] = None,
        language: str = "en"
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response for query
        
        Args:
            query: Fault description query
            part_number: Optional product part number
            language: Response language
            
        Returns:
            Cached response or None
        """
        with self._lock:
            self.stats.total_queries += 1
            
            # Generate cache key
            cache_key = self._generate_cache_key(query, part_number, language)
            
            # Check direct hit
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                # Check expiration
                if entry.is_expired:
                    self._remove_entry(cache_key)
                    self.stats.expirations += 1
                    self.stats.misses += 1
                    logger.debug(f"Cache EXPIRED: {query[:50]}...")
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(cache_key)
                entry.touch()
                
                self.stats.hits += 1
                logger.info(f"Cache HIT: {query[:50]}... (hits: {entry.hit_count})")
                return entry.response
            
            # Cache miss
            self.stats.misses += 1
            logger.debug(f"Cache MISS: {query[:50]}...")
            return None
    
    def set(
        self,
        query: str,
        response: Dict[str, Any],
        part_number: Optional[str] = None,
        language: str = "en",
        ttl: Optional[int] = None
    ) -> str:
        """
        Store response in cache
        
        Args:
            query: Fault description query
            response: RAG response to cache
            part_number: Optional product part number
            language: Response language
            ttl: Optional custom TTL (uses default if None)
            
        Returns:
            Cache key
        """
        with self._lock:
            # Generate cache key
            cache_key = self._generate_cache_key(query, part_number, language)
            
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            # Create entry
            now = time.time()
            entry = CacheEntry(
                key=cache_key,
                query=query,
                response=response,
                created_at=now,
                last_accessed=now,
                ttl_seconds=ttl or self.default_ttl
            )
            
            # Store in cache
            self._cache[cache_key] = entry
            self._query_index[self._normalize_query(query)] = cache_key
            
            self.stats.cache_size = len(self._cache)
            logger.debug(f"Cache SET: {query[:50]}... (key: {cache_key[:8]})")
            
            return cache_key
    
    def _evict_oldest(self):
        """Evict least recently used entry"""
        if self._cache:
            # Pop first item (oldest)
            oldest_key, oldest_entry = self._cache.popitem(last=False)
            
            # Remove from query index
            normalized = self._normalize_query(oldest_entry.query)
            if normalized in self._query_index:
                del self._query_index[normalized]
            
            self.stats.evictions += 1
            logger.debug(f"Cache EVICT: {oldest_entry.query[:50]}...")
    
    def _remove_entry(self, cache_key: str):
        """Remove specific entry from cache"""
        if cache_key in self._cache:
            entry = self._cache.pop(cache_key)
            
            # Remove from query index
            normalized = self._normalize_query(entry.query)
            if normalized in self._query_index:
                del self._query_index[normalized]
            
            self.stats.cache_size = len(self._cache)
    
    def invalidate(self, cache_key: Optional[str] = None, query: Optional[str] = None):
        """
        Invalidate cache entry by key or query
        
        Args:
            cache_key: Direct cache key
            query: Query string to invalidate
        """
        with self._lock:
            if cache_key:
                self._remove_entry(cache_key)
                logger.info(f"Cache INVALIDATE by key: {cache_key[:8]}")
            elif query:
                normalized = self._normalize_query(query)
                if normalized in self._query_index:
                    key = self._query_index[normalized]
                    self._remove_entry(key)
                    logger.info(f"Cache INVALIDATE by query: {query[:50]}...")
    
    def invalidate_all(self):
        """Clear entire cache"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._query_index.clear()
            self.stats.cache_size = 0
            logger.info(f"Cache CLEAR: Removed {count} entries")
    
    def clear(self):
        """Alias for invalidate_all - clear entire cache"""
        self.invalidate_all()
    
    def delete(self, cache_key: str) -> bool:
        """
        Delete a specific cache entry by key
        
        Args:
            cache_key: The cache key to delete
            
        Returns:
            True if entry was deleted, False if not found
        """
        with self._lock:
            if cache_key in self._cache:
                self._remove_entry(cache_key)
                return True
            return False
    
    def invalidate_by_pattern(self, pattern: str):
        """
        Invalidate entries matching a pattern in query
        
        Args:
            pattern: Substring to match in queries
        """
        with self._lock:
            pattern_lower = pattern.lower()
            keys_to_remove = []
            
            for key, entry in self._cache.items():
                if pattern_lower in entry.query.lower():
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            logger.info(f"Cache INVALIDATE by pattern '{pattern}': Removed {len(keys_to_remove)} entries")
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = []
            
            for key, entry in self._cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
                self.stats.expirations += 1
            
            if expired_keys:
                logger.info(f"Cache CLEANUP: Removed {len(expired_keys)} expired entries")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            self.stats.cache_size = len(self._cache)
            return self.stats.to_dict()
    
    def get_entries_info(self, limit: int = 10) -> List[Dict]:
        """
        Get info about cache entries
        
        Args:
            limit: Maximum entries to return
            
        Returns:
            List of entry info dictionaries
        """
        with self._lock:
            entries = []
            
            for key, entry in list(self._cache.items())[-limit:]:
                entries.append({
                    "key": key[:8] + "...",
                    "query": entry.query[:50] + "..." if len(entry.query) > 50 else entry.query,
                    "hit_count": entry.hit_count,
                    "age_seconds": round(entry.age_seconds, 1),
                    "is_expired": entry.is_expired
                })
            
            return entries


class QuerySimilarityCache(ResponseCache):
    """
    Extended cache with similarity-based matching
    Uses simple token overlap for similarity
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_index: Dict[str, set] = {}  # token -> set of cache keys
    
    def _tokenize(self, query: str) -> set:
        """Tokenize query into word set"""
        normalized = self._normalize_query(query)
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
                      'by', 'from', 'or', 'and', 'not', 'but', 'if', 'then', 'else',
                      'when', 'up', 'out', 'no', 'so', 'what', 'which', 'who', 'how'}
        tokens = set(normalized.split()) - stop_words
        return tokens
    
    def _calculate_similarity(self, tokens1: set, tokens2: set) -> float:
        """Calculate Jaccard similarity between token sets"""
        if not tokens1 or not tokens2:
            return 0.0
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        return intersection / union if union > 0 else 0.0
    
    def get(
        self,
        query: str,
        part_number: Optional[str] = None,
        language: str = "en"
    ) -> Optional[Dict[str, Any]]:
        """Get with similarity matching fallback"""
        # Try exact match first
        result = super().get(query, part_number, language)
        if result:
            return result
        
        # Try similarity matching if enabled
        if self.enable_similarity_matching:
            with self._lock:
                query_tokens = self._tokenize(query)
                best_match = None
                best_similarity = 0.0
                
                for key, entry in self._cache.items():
                    if entry.is_expired:
                        continue
                    
                    entry_tokens = self._tokenize(entry.query)
                    similarity = self._calculate_similarity(query_tokens, entry_tokens)
                    
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_match = entry
                        best_similarity = similarity
                
                if best_match:
                    best_match.touch()
                    self._cache.move_to_end(best_match.key)
                    self.stats.hits += 1
                    logger.info(f"Cache SIMILAR HIT ({best_similarity:.2f}): {query[:40]}... -> {best_match.query[:40]}...")
                    return best_match.response
        
        return None
    
    def get_similar(
        self,
        query: str,
        part_number: Optional[str] = None,
        language: str = "en"
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response using similarity matching only
        
        Args:
            query: Fault description query
            part_number: Optional product part number
            language: Response language
            
        Returns:
            Cached response with similarity info or None
        """
        with self._lock:
            query_tokens = self._tokenize(query)
            best_match = None
            best_similarity = 0.0
            
            for key, entry in self._cache.items():
                if entry.is_expired:
                    continue
                
                entry_tokens = self._tokenize(entry.query)
                similarity = self._calculate_similarity(query_tokens, entry_tokens)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_match = entry
                    best_similarity = similarity
            
            if best_match:
                best_match.touch()
                self._cache.move_to_end(best_match.key)
                self.stats.hits += 1
                logger.info(f"Cache SIMILAR HIT ({best_similarity:.2f}): {query[:40]}...")
                result = best_match.response.copy() if isinstance(best_match.response, dict) else best_match.response
                if isinstance(result, dict):
                    result["_similarity"] = best_similarity
                return result
            
            self.stats.misses += 1
            return None
    
    def clear(self):
        """Clear all cached entries"""
        self.invalidate_all()
    
    def delete(self, cache_key: str) -> bool:
        """
        Delete a specific cache entry by key
        
        Args:
            cache_key: The cache key to delete
            
        Returns:
            True if entry was deleted, False if not found
        """
        with self._lock:
            # Try direct key match first
            if cache_key in self._cache:
                self._remove_entry(cache_key)
                return True
            
            # Try matching by query string (cache_key might be the original key format)
            for key, entry in list(self._cache.items()):
                if key.startswith(cache_key[:8]) or entry.query.startswith(cache_key.split(':')[0]):
                    self._remove_entry(key)
                    return True
            
            return False


# Singleton instance for global access
_cache_instance: Optional[ResponseCache] = None


def get_response_cache(
    max_size: int = 1000,
    default_ttl: int = 3600,
    enable_similarity: bool = True
) -> ResponseCache:
    """
    Get or create singleton cache instance
    
    Args:
        max_size: Maximum cache entries
        default_ttl: Default TTL in seconds
        enable_similarity: Enable similarity matching
        
    Returns:
        ResponseCache instance
    """
    global _cache_instance
    
    if _cache_instance is None:
        if enable_similarity:
            _cache_instance = QuerySimilarityCache(
                max_size=max_size,
                default_ttl=default_ttl,
                enable_similarity_matching=True
            )
        else:
            _cache_instance = ResponseCache(
                max_size=max_size,
                default_ttl=default_ttl
            )
    
    return _cache_instance


def reset_cache():
    """Reset singleton cache instance"""
    global _cache_instance
    if _cache_instance:
        _cache_instance.invalidate_all()
    _cache_instance = None
