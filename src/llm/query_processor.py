"""
Query Processor - Centralized Query Enhancement and Processing
Phase 2.2: Extracted from rag_engine.py for better maintainability

Responsibilities:
1. Query normalization and cleaning
2. Keyword extraction (Turkish + English)
3. Domain-specific query enhancement  
4. Intent detection for routing
5. Query expansion with synonyms

PRODUCTION-SAFE:
- Backward compatible with existing RAG engine
- Optional enhancements (graceful degradation)
- Comprehensive logging for debugging
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import re
from collections import Counter
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ProcessedQuery:
    """Result of query processing"""
    original: str
    normalized: str
    enhanced: str
    keywords: List[str]
    intent: Optional[str] = None
    domain_terms: List[str] = field(default_factory=list)
    expansions: List[str] = field(default_factory=list)
    language: str = "en"
    
    def to_dict(self) -> Dict:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "enhanced": self.enhanced,
            "keywords": self.keywords,
            "intent": self.intent,
            "domain_terms": self.domain_terms,
            "expansions": self.expansions,
            "language": self.language
        }


class QueryProcessor:
    """
    Centralized query processing for RAG system.
    
    Consolidates all query enhancement logic:
    - Normalization
    - Keyword extraction
    - Domain enhancement (via domain_embeddings)
    - Intent detection (via relevance_filter)
    """
    
    # Stop words for keyword extraction (English + Turkish)
    STOP_WORDS_EN = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'must', 'shall', 'need', 'not',
        'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 'by', 'for',
        'with', 'about', 'against', 'between', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
        'once', 'here', 'there', 'all', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'how', 'what', 'which', 'who', 'why'
    }
    
    STOP_WORDS_TR = {
        've', 'bir', 'bu', 'şu', 'ile', 'için', 'gibi', 'daha', 'çok',
        'var', 'yok', 'olan', 'olarak', 'veya', 'ama', 'fakat', 'ancak',
        'de', 'da', 'ki', 'mi', 'mı', 'mu', 'mü', 'ise', 'ya', 'hem',
        'ne', 'kadar', 'çünkü', 'iken', 'sonra', 'önce', 'şimdi', 'her',
        'bazı', 'hiç', 'biri', 'kendi', 'aynı', 'başka', 'diğer', 'tüm'
    }
    
    STOP_WORDS = STOP_WORDS_EN | STOP_WORDS_TR
    
    def __init__(self, domain_embeddings=None, enable_intent_detection: bool = True):
        """
        Initialize QueryProcessor.
        
        Args:
            domain_embeddings: Optional domain embeddings engine for enhancement
            enable_intent_detection: Whether to detect query intent
        """
        self.domain_embeddings = domain_embeddings
        self.enable_intent_detection = enable_intent_detection
        logger.info("QueryProcessor initialized")
    
    def process(self, query: str, language: str = "en") -> ProcessedQuery:
        """
        Process a query through the full pipeline.
        
        Args:
            query: Raw user query (fault description)
            language: Query language ("en" or "tr")
            
        Returns:
            ProcessedQuery with all enhancements
        """
        # Step 1: Normalize
        normalized = self.normalize(query)
        
        # Step 2: Extract keywords
        keywords = self.extract_keywords(normalized, limit=10)
        
        # Step 3: Detect intent (optional)
        intent = None
        if self.enable_intent_detection:
            intent = self.detect_intent(normalized)
        
        # Step 4: Enhance with domain knowledge (optional)
        enhanced = normalized
        domain_terms = []
        expansions = []
        
        if self.domain_embeddings:
            try:
                enhancement_result = self.domain_embeddings.enhance_query(query)
                enhanced = enhancement_result.get("enhanced", normalized)
                domain_terms = enhancement_result.get("domain_terms", [])
                expansions = enhancement_result.get("expansions", [])
            except Exception as e:
                logger.warning(f"Domain enhancement failed: {e}")
        
        result = ProcessedQuery(
            original=query,
            normalized=normalized,
            enhanced=enhanced,
            keywords=keywords,
            intent=intent,
            domain_terms=domain_terms,
            expansions=expansions,
            language=language
        )
        
        logger.debug(f"Processed query: {result.to_dict()}")
        return result
    
    def normalize(self, query: str) -> str:
        """
        Normalize query text.
        
        - Lowercase
        - Remove extra whitespace
        - Normalize unicode characters
        - Remove special characters (keep alphanumeric + Turkish)
        """
        if not query:
            return ""
        
        # Strip and lowercase
        normalized = query.strip().lower()
        
        # Normalize multiple whitespace to single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove special characters but keep letters, numbers, spaces
        # Keep Turkish characters: çğıöşüÇĞİÖŞÜ
        normalized = re.sub(r'[^\w\s\-çğıöşüÇĞİÖŞÜ]', ' ', normalized)
        
        # Clean up again
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def extract_keywords(self, text: str, limit: int = 10) -> List[str]:
        """
        Extract meaningful keywords from text.
        
        Args:
            text: Input text (already normalized)
            limit: Maximum number of keywords to return
            
        Returns:
            List of keywords ordered by frequency
        """
        if not text:
            return []
        
        text_lower = text.lower()
        
        # Extract words (min 3 chars, support Turkish)
        words = re.findall(r'\b[a-zA-ZçğıöşüÇĞİÖŞÜ]{3,}\b', text_lower)
        
        # Filter stop words
        keywords = [w for w in words if w not in self.STOP_WORDS]
        
        # Count and return most common
        counter = Counter(keywords)
        return [word for word, _ in counter.most_common(limit)]
    
    def detect_intent(self, query: str) -> Optional[str]:
        """
        Detect the primary intent/category of the query.
        
        Uses relevance_filter's intent detection for consistency.
        
        Returns:
            Intent category or None if no specific intent detected
        """
        try:
            from src.llm.relevance_filter import detect_query_intent
            return detect_query_intent(query)
        except Exception as e:
            logger.warning(f"Intent detection failed: {e}")
            return None
    
    def detect_language(self, query: str) -> str:
        """
        Detect query language (simple heuristic).
        
        Returns:
            "tr" for Turkish, "en" for English (default)
        """
        # Turkish-specific characters
        turkish_chars = set('çğıöşüÇĞİÖŞÜ')
        
        # Check for Turkish characters
        if any(c in turkish_chars for c in query):
            return "tr"
        
        # Check for common Turkish words
        turkish_words = {'ve', 'bir', 'için', 'ile', 'olan', 'gibi', 'çalışmıyor', 'arıza'}
        query_words = set(query.lower().split())
        if query_words & turkish_words:
            return "tr"
        
        return "en"
    
    def expand_query(self, query: str, keywords: List[str]) -> List[str]:
        """
        Generate query expansions using domain knowledge.
        
        Args:
            query: Original query
            keywords: Extracted keywords
            
        Returns:
            List of expansion terms
        """
        expansions = []
        
        # Use domain embeddings if available
        if self.domain_embeddings:
            try:
                for keyword in keywords[:5]:  # Limit to top 5 keywords
                    related = self.domain_embeddings.get_related_terms(keyword)
                    expansions.extend(related)
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_expansions = []
        for exp in expansions:
            if exp.lower() not in seen and exp.lower() not in keywords:
                seen.add(exp.lower())
                unique_expansions.append(exp)
        
        return unique_expansions[:10]  # Limit expansions


# Singleton instance
_query_processor: Optional[QueryProcessor] = None


def get_query_processor(domain_embeddings=None) -> QueryProcessor:
    """
    Get singleton QueryProcessor instance.
    
    Args:
        domain_embeddings: Optional domain embeddings engine
        
    Returns:
        QueryProcessor instance
    """
    global _query_processor
    
    if _query_processor is None:
        _query_processor = QueryProcessor(domain_embeddings=domain_embeddings)
        logger.info("QueryProcessor singleton created")
    
    return _query_processor


def process_query(query: str, language: str = "en", domain_embeddings=None) -> ProcessedQuery:
    """
    Convenience function to process a query.
    
    Args:
        query: Raw user query
        language: Query language
        domain_embeddings: Optional domain embeddings
        
    Returns:
        ProcessedQuery result
    """
    processor = get_query_processor(domain_embeddings)
    return processor.process(query, language)
