"""
RAG Engine - Retrieval-Augmented Generation
Combines vector search with LLM generation
Now with self-learning capabilities from user feedback
Phase 2.2: Hybrid Search (Semantic + BM25) integration
Phase 2.3: Response Caching for improved performance
Phase 3.4: Context Window Optimization
Phase 4.1: Metadata-based Filtering and Boosting
Phase 5.1: Performance Metrics and Monitoring
Phase 6: Self-Learning Feedback Loop
"""
from typing import Dict, List, Optional
from datetime import datetime
import time
import re

from src.documents.embeddings import EmbeddingsGenerator
from src.vectordb.chroma_client import ChromaDBClient
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import get_system_prompt, build_rag_prompt, build_fallback_response
from src.llm.context_optimizer import ContextOptimizer, optimize_context_for_rag
from src.documents.product_extractor import ProductExtractor, IntelligentProductExtractor  # Metadata Enrichment
from src.llm.performance_metrics import get_performance_monitor, QueryTimer, QueryMetrics
from src.database import MongoDBClient
from config.ai_settings import (
    RAG_TOP_K, RAG_SIMILARITY_THRESHOLD, DEFAULT_LANGUAGE,
    USE_HYBRID_SEARCH, HYBRID_SEMANTIC_WEIGHT, HYBRID_BM25_WEIGHT,
    HYBRID_RRF_K, ENABLE_QUERY_EXPANSION,
    USE_CACHE, CACHE_TTL,
    ENABLE_METADATA_BOOST, SERVICE_BULLETIN_BOOST, PROCEDURE_BOOST,
    WARNING_BOOST, IMPORTANCE_BOOST_FACTOR,
    ENABLE_DOMAIN_EMBEDDINGS, ENABLE_FAULT_FILTERING,
    LLM_TIMEOUT_SECONDS, CHROMADB_TIMEOUT_SECONDS
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGEngine:
    """
    RAG Engine for repair suggestions with self-learning capabilities.
    
    Features:
    - Hybrid search (Semantic + BM25)
    - Self-learning from feedback
    - Bulletin score boosting (Phase 7)
    - Error code direct matching
    - Symptom synonym expansion
    """
    
    # Score boosting constants for bulletin prioritization
    SCORE_BOOSTS = {
        'doc_type': {
            'service_bulletin': 2.0,      # Bulletins get 2x boost
            'troubleshooting_guide': 1.5,
            'technical_manual': 1.0,
            'safety_document': 1.2,
            'user_manual': 0.9,
        },
        'source_prefix': {
            'ESDE': 1.5,                   # Official ESDE bulletins get extra boost
        },
        'error_code_match': 2.5,           # Error code in query matches document
        'problem_match_base': 1.5,         # Base boost for problem description match
        'problem_match_per_word': 0.2,     # Additional boost per matching word
    }
    
    # Turkish language indicators
    TURKISH_CHARS = set('çğışöüÇĞİŞÖÜ')
    TURKISH_WORDS = [
        'nedir', 'nasıl', 'neden', 'hangi', 'kaç',
        'hata', 'kodu', 'arıza', 'sorun', 'çözüm', 'onarım',
        'bakım', 'kalibrasyon', 'bağlantı', 'kurulum',
        'çalışmıyor', 'çalışmaz', 'durdu', 'bozuk', 'bozuldu',
        'alet', 'motor', 'tork', 'hız', 'güç', 'kontrol', 'ayar',
        'için', 'ile', 'veya', 'ama', 'fakat', 'mı', 'mi', 'mu', 'mü'
    ]
    
    def __init__(self):
        """Initialize RAG engine"""
        logger.info("Initializing RAG Engine...")
        
        self.embeddings = EmbeddingsGenerator()
        self.vectordb = ChromaDBClient()
        self.llm = OllamaClient()
        self.mongodb = None
        self.feedback_engine = None
        self.hybrid_searcher = None
        self.response_cache = None
        self.self_learning_engine = None  # Phase 6
        self.domain_embeddings = None  # Phase 3.1
        self.query_processor = None  # Phase 2.2
        self.product_extractor = ProductExtractor()  # Metadata Enrichment
        self.context_optimizer = ContextOptimizer(token_budget=8000)  # Phase 3.4
        self.performance_monitor = get_performance_monitor()  # Phase 5.1
        self.intent_detector = None  # Lazy initialized
        
        # Initialize hybrid search if enabled
        if USE_HYBRID_SEARCH:
            self._init_hybrid_search()
        
        # Initialize response cache if enabled (Phase 2.3)
        if USE_CACHE:
            self._init_response_cache()
        
        # Initialize self-learning engine (Phase 6)
        self._init_self_learning()
        
        # Initialize domain embeddings (Phase 3.1) - ONLY if enabled
        if ENABLE_DOMAIN_EMBEDDINGS:
            self._init_domain_embeddings()
        else:
            logger.info("⏭️ Domain embeddings DISABLED (ENABLE_DOMAIN_EMBEDDINGS=false)")
        
        # Initialize query processor (Phase 2.2)
        self._init_query_processor()
        
        logger.info("✅ RAG Engine initialized")
    
    def _init_hybrid_search(self):
        """Lazy initialize hybrid search to avoid circular imports"""
        try:
            from src.llm.hybrid_search import HybridSearcher
            self.hybrid_searcher = HybridSearcher(
                rrf_k=HYBRID_RRF_K,
                semantic_weight=HYBRID_SEMANTIC_WEIGHT,
                bm25_weight=HYBRID_BM25_WEIGHT
            )
            logger.info("✅ Hybrid search enabled (Semantic + BM25)")
        except Exception as e:
            logger.warning(f"Failed to initialize hybrid search: {e}")
            self.hybrid_searcher = None
    
    def _init_response_cache(self):
        """Initialize response cache (Phase 2.3)"""
        try:
            from src.llm.response_cache import get_response_cache
            self.response_cache = get_response_cache(
                max_size=1000,
                default_ttl=CACHE_TTL,
                enable_similarity=True
            )
            logger.info(f"✅ Response cache enabled (TTL: {CACHE_TTL}s)")
        except Exception as e:
            logger.warning(f"Failed to initialize response cache: {e}")
            self.response_cache = None
    
    def _init_self_learning(self):
        """Initialize self-learning engine (Phase 6)"""
        try:
            from src.llm.self_learning import get_self_learning_engine
            self.self_learning_engine = get_self_learning_engine()
            logger.info("✅ Self-learning engine enabled (Phase 6)")
        except Exception as e:
            logger.warning(f"Failed to initialize self-learning engine: {e}")
            self.self_learning_engine = None
    
    def _init_domain_embeddings(self):
        """Initialize domain embeddings engine (Phase 3.1)"""
        try:
            from src.llm.domain_embeddings import get_domain_embeddings_engine
            self.domain_embeddings = get_domain_embeddings_engine()
            logger.info("✅ Domain embeddings enabled (Phase 3.1)")
        except Exception as e:
            logger.warning(f"Failed to initialize domain embeddings: {e}")
            self.domain_embeddings = None
    
    def _init_query_processor(self):
        """Initialize query processor (Phase 2.2)"""
        try:
            from src.llm.query_processor import get_query_processor
            self.query_processor = get_query_processor(self.domain_embeddings)
            logger.info("✅ Query processor enabled (Phase 2.2)")
        except Exception as e:
            logger.warning(f"Failed to initialize query processor: {e}")
            self.query_processor = None
    
    def _detect_language(self, query: str) -> str:
        """
        Auto-detect query language (Turkish vs English).
        
        Detection based on:
        1. Turkish-specific characters (ş, ğ, ı, ö, ü, ç)
        2. Common Turkish words (hata, nedir, nasıl, etc.)
        
        Args:
            query: User query text
            
        Returns:
            "tr" for Turkish, "en" for English (default)
        """
        # Check for Turkish characters first (most reliable)
        if any(c in self.TURKISH_CHARS for c in query):
            logger.debug(f"[LANG] Detected Turkish via special characters")
            return "tr"
        
        # Check for Turkish words
        query_lower = query.lower()
        for word in self.TURKISH_WORDS:
            if word in query_lower:
                logger.debug(f"[LANG] Detected Turkish via keyword: '{word}'")
                return "tr"
        
        # Default to English
        return "en"
    
    def _check_off_topic(self, query: str, part_number: str) -> Optional[str]:
        """
        Check if query is clearly off-topic and should be refused immediately.
        
        Args:
            query: User's query text
            part_number: Product part number
            
        Returns:
            Reason string if off-topic, None if potentially on-topic
        """
        query_lower = query.lower()
        
        # Off-topic keywords (clearly unrelated to industrial tools)
        OFF_TOPIC_KEYWORDS = [
            'capital of', 'president of', 'population of', 'weather in',
            'recipe for', 'how to cook', 'what is the capital',
            'who is the president', 'how many people',
        ]
        
        # Competitor brands
        COMPETITOR_BRANDS = [
            'bosch', 'makita', 'dewalt', 'milwaukee', 'hilti', 'metabo',
            'festool', 'stanley', 'black & decker', 'ryobi', 'ridgid',
            'craftsman', 'kobalt', 'snap-on', 'ingersoll rand', 'chicago pneumatic'
        ]
        
        # Check for off-topic keywords
        for keyword in OFF_TOPIC_KEYWORDS:
            if keyword in query_lower:
                return f"Query appears to be off-topic ('{keyword}')"
        
        # Check for competitor brands (only if Desoutter not mentioned)
        if 'desoutter' not in query_lower:
            for brand in COMPETITOR_BRANDS:
                if brand in query_lower:
                    return f"Query is about competitor product '{brand}', not Desoutter"
        
        # Check for invalid/nonsense product
        if part_number and part_number.upper() in ['INVALID_PRODUCT', 'INVALID', 'NONE', 'NULL']:
            # Also check if query itself looks like nonsense
            words = query_lower.split()
            # Count recognizable words
            nonsense_indicators = ['qwerty', 'xyz123', 'asdf', 'test123', 'random']
            for indicator in nonsense_indicators:
                if indicator in query_lower:
                    return "Query appears to be nonsense or test data"
        
        return None
    
    def _filter_by_product_capabilities(self, docs: List[Dict], capabilities: Dict) -> List[Dict]:
        """
        Filter documents based on product capabilities.
        Excludes wireless/battery content for non-wireless/non-battery tools.
        
        Args:
            docs: List of retrieved documents
            capabilities: Product capabilities dict
            
        Returns:
            Filtered documents list
        """
        if not capabilities or not capabilities.get('product_found'):
            return docs
        
        is_wireless = capabilities.get('wireless', False)
        is_battery = capabilities.get('battery_powered', False)
        
        # Patterns to exclude for non-wireless tools
        wireless_patterns = ['wifi', 'wi-fi', 'wireless', 'access point', 'connect unit', 'pairing']
        
        # Patterns to exclude for non-battery tools
        battery_patterns = ['battery charging', 'battery replacement', 'charger', 'charge level', 'low battery']
        
        filtered = []
        excluded_count = 0
        
        for doc in docs:
            content = doc.get('text', '').lower()
            should_exclude = False
            
            # Check wireless content for non-wireless tools
            if not is_wireless:
                for pattern in wireless_patterns:
                    if content.count(pattern) >= 2:  # Pattern appears multiple times = main topic
                        should_exclude = True
                        break
            
            # Check battery content for non-battery tools
            if not is_battery and not should_exclude:
                for pattern in battery_patterns:
                    if content.count(pattern) >= 2:
                        should_exclude = True
                        break
            
            if should_exclude:
                excluded_count += 1
            else:
                filtered.append(doc)
        
        if excluded_count > 0:
            logger.info(f"[CAPABILITY_FILTER] Excluded {excluded_count} docs (wireless={is_wireless}, battery={is_battery})")
        
        return filtered
    
    def _filter_by_product_strict(self, docs: List[Dict], product_info: Dict) -> List[Dict]:
        """
        STRICT product-specific filtering (Priority 1 - Production Quality).
        Prevents cross-product contamination (e.g., CVIL2 docs for ERS6 queries).
        
        Args:
            docs: List of retrieved documents
            product_info: Product info from MongoDB
            
        Returns:
            Filtered documents list (only matching product or general docs)
        """
        if not product_info:
            return docs
        
        # Extract product identifiers for matching
        target_family = product_info.get('product_family', '').upper()  # e.g., 'EAD', 'ERS'
        target_series = product_info.get('series_name', '').lower()     # e.g., 'ead - transducerized...'
        target_model = product_info.get('model_name', '').upper()       # e.g., 'EAD20-1300'
        target_part = product_info.get('part_number', '')               # e.g., '6151656060'
        target_category = product_info.get('tool_category', '')         # e.g., 'cable_tightening'
        
        # Known product family prefixes that are mutually exclusive
        PRODUCT_FAMILIES = {
            'EAD', 'EPD', 'EFD', 'ERS', 'ECS', 'EPB', 'EPBC', 'EABC', 'EABS',
            'CVI', 'CVIL', 'CVIR', 'CVIC', 'DVT', 'QST', 'PF', 'CONNECT',
            'ELC', 'ELS', 'ELB', 'BLRT', 'XPB', 'EM', 'ERAL', 'EME', 'EMEL'
        }
        
        # WiFi variants map to their base family (C suffix = WiFi)
        FAMILY_ALIASES = {
            'EPBC': 'EPB', 'EPB': 'EPBC',  # EPB <-> EPBC (WiFi)
            'EABC': 'EAB', 'EAB': 'EABC',  # EAB <-> EABC (WiFi)
            'EABS': 'EAB',                  # EABS -> EAB (Standalone)
        }
        
        # Get all matching families (target + aliases)
        matching_families = {target_family}
        if target_family in FAMILY_ALIASES:
            matching_families.add(FAMILY_ALIASES[target_family])
        
        filtered = []
        excluded_count = 0
        
        for doc in docs:
            metadata = doc.get('metadata', {})
            content = doc.get('text', '').upper()
            source = metadata.get('source', '').upper()
            
            # ESDE bulletins are NEVER excluded - they often cover multiple product families
            # Example: ESDE25004 covers ERS2/6/12, ERSA, ERSF, EPB8, EPBA8, EPBC8
            is_esde_bulletin = 'ESDE' in source or 'ESB' in source
            if is_esde_bulletin:
                filtered.append(doc)
                continue
            
            # Extract product family from document
            doc_family = metadata.get('product_family', '')
            doc_categories = metadata.get('product_categories', '')
            
            # Check if document mentions a DIFFERENT product family
            is_other_product = False
            detected_family = None
            
            # Method 1: Check metadata product_family
            if doc_family and doc_family.upper() not in matching_families:
                if doc_family.upper() in PRODUCT_FAMILIES:
                    is_other_product = True
                    detected_family = doc_family.upper()
            
            # Method 2: Check content for other product families
            if not is_other_product and target_family:
                for family in PRODUCT_FAMILIES:
                    if family in matching_families:
                        continue
                    # Check if document is PRIMARILY about another product
                    # (appears in source name or 3+ times in content)
                    if family in source or content.count(family) >= 3:
                        # But also check target family presence
                        target_mentions = content.count(target_family) if target_family else 0
                        other_mentions = content.count(family)
                        
                        if other_mentions > target_mentions + 2:
                            is_other_product = True
                            detected_family = family
                            break
            
            if is_other_product:
                excluded_count += 1
                logger.debug(
                    f"[PRODUCT_FILTER] Excluded: {source[:50]} "
                    f"(detected {detected_family}, target {target_family})"
                )
            else:
                filtered.append(doc)
        
        if excluded_count > 0:
            logger.info(
                f"[PRODUCT_FILTER] Excluded {excluded_count} docs for "
                f"{target_family}/{target_model} (cross-product contamination)"
            )
        
        return filtered

    def _apply_metadata_boost(self, base_score: float, metadata: Dict, query: str = "", content: str = "") -> float:
        """
        Apply metadata-based score boosting (Phase 4.1 + 7.1)
        
        Boosts documents based on:
        - Service bulletins (ESD/ESB) get priority
        - Procedure sections (step-by-step)
        - Warning/caution content
        - Importance score from semantic chunking
        - Error code pattern matching (Phase 7.1)
        - Content keyword matching (Phase 7.1)
        
        Args:
            base_score: Original similarity/relevance score
            metadata: Document metadata dictionary
            query: User's query for pattern matching
            content: Document content for keyword matching
            
        Returns:
            Boosted score
        """
        if not ENABLE_METADATA_BOOST or not metadata:
            return base_score
        
        boost = 1.0
        boost_reasons = []
        
        # 1. Service bulletin boost (ESD/ESB documents)
        source = metadata.get("source", "")
        doc_type = metadata.get("doc_type", "")
        
        is_service_bulletin = (
            "ESD" in source.upper() or 
            "ESB" in source.upper() or
            doc_type == "service_bulletin"
        )
        
        if is_service_bulletin:
            boost *= SERVICE_BULLETIN_BOOST
            boost_reasons.append(f"service_bulletin({SERVICE_BULLETIN_BOOST}x)")
        
        # 2. Procedure section boost
        section_type = metadata.get("section_type", "")
        is_procedure = metadata.get("is_procedure", False)
        
        if section_type == "procedure" or is_procedure:
            boost *= PROCEDURE_BOOST
            boost_reasons.append(f"procedure({PROCEDURE_BOOST}x)")
        
        # 3. Warning/caution boost
        contains_warning = metadata.get("contains_warning", False)
        
        if contains_warning:
            boost *= WARNING_BOOST
            boost_reasons.append(f"warning({WARNING_BOOST}x)")
        
        # 4. Importance score boost
        importance = metadata.get("importance_score")
        if importance is not None:
            try:
                importance_float = float(importance)
                importance_boost = 1 + (importance_float * IMPORTANCE_BOOST_FACTOR)
                boost *= importance_boost
                boost_reasons.append(f"importance({importance_boost:.2f}x)")
            except (ValueError, TypeError):
                pass
        
        # 5. Error code pattern matching boost (Phase 7.1)
        if query:
            error_code_boost = self._calculate_error_code_boost_general(query, source, content)
            if error_code_boost > 1.0:
                boost *= error_code_boost
                boost_reasons.append(f"error_code({error_code_boost:.1f}x)")
        
        # 6. Content keyword matching boost (Phase 7.1)
        if query and content:
            keyword_boost = self._calculate_keyword_boost(query, content)
            if keyword_boost > 1.0:
                boost *= keyword_boost
                boost_reasons.append(f"keyword_match({keyword_boost:.1f}x)")
        
        boosted_score = base_score * boost
        
        if boost > 1.0:
            logger.debug(f"Metadata boost applied: {base_score:.3f} → {boosted_score:.3f} ({', '.join(boost_reasons)})")
        
        return boosted_score
    
    def _calculate_error_code_boost_general(self, query: str, source: str, content: str) -> float:
        """
        General error code pattern matching - not hardcoded!
        
        Detects patterns like: E06, E047, I004, TRD-E06, HW Channel
        and boosts if same pattern appears in document source/content.
        """
        import re
        
        # General error code patterns (works for ANY code)
        error_patterns = [
            r'\b[EI]\d{2,4}\b',           # E06, E047, I004, E018, etc.
            r'\bTrd[-\s]?E\d+\b',          # Trd-E06, TRD E06
            r'\bHW[-\s]?Channel\b',        # HW Channel, HW-Channel
            r'\bSpan[-\s]?[Ff]ailure\b',   # Span Failure
            r'\b\d{4}[-\s]?[A-Z]{2,4}\b',  # 6159-XXX patterns
        ]
        
        query_upper = query.upper()
        source_upper = source.upper() if source else ''
        content_upper = content.upper() if content else ''
        
        for pattern in error_patterns:
            matches = re.findall(pattern, query_upper, re.IGNORECASE)
            for match in matches:
                # Normalize for comparison
                match_normalized = re.sub(r'[-\s]', '', match.upper())
                source_normalized = re.sub(r'[-\s]', '', source_upper)
                content_normalized = re.sub(r'[-\s]', '', content_upper[:2000])  # First 2000 chars
                
                # If error code in query also appears in document
                if match_normalized in source_normalized:
                    return 2.5  # Strong boost for source match
                if match_normalized in content_normalized:
                    return 2.0  # Good boost for content match
        
        return 1.0
    
    def _calculate_keyword_boost(self, query: str, content: str) -> float:
        """
        General keyword matching - boosts when query keywords appear in content.
        
        Two-stage matching:
        1. Phrase matching: Exact phrases from query found in content -> Strong boost
        2. Word matching: Individual keywords found in problem description sections
        """
        import re
        
        if not content:
            return 1.0
        
        content_lower = content.lower()
        query_lower = query.lower()
        
        boost = 1.0
        
        # Stage 1: Phrase matching in the entire content
        # Extract meaningful phrases (2-4 words) from query
        query_words = query_lower.split()
        phrases_to_check = []
        
        # Create bigrams and trigrams
        for i in range(len(query_words)):
            if i + 1 < len(query_words):
                phrases_to_check.append(' '.join(query_words[i:i+2]))
            if i + 2 < len(query_words):
                phrases_to_check.append(' '.join(query_words[i:i+3]))
        
        # Check for phrase matches
        phrase_matches = sum(1 for phrase in phrases_to_check 
                           if len(phrase) > 6 and phrase in content_lower)
        
        if phrase_matches >= 3:
            boost *= 2.5  # Strong phrase match
        elif phrase_matches >= 2:
            boost *= 2.0  # Good phrase match
        elif phrase_matches >= 1:
            boost *= 1.5  # Some phrase match
        
        # Stage 2: Problem section keyword matching (original logic)
        problem_sections = []
        section_patterns = [
            r'description of the issue[:\s]+([^.]{10,100})',
            r'product impact[:\s]+([^.]{10,100})',
            r'cause of issue[:\s]+([^.]{10,100})',
            r'visible symptom[:\s]+([^.]{10,100})',
            r'failure description[:\s]+([^.]{10,100})',
        ]
        
        for pattern in section_patterns:
            match = re.search(pattern, content_lower)
            if match:
                problem_sections.append(match.group(1))
        
        if problem_sections:
            # Get meaningful query words (skip short ones and common words)
            stop_words = {'the', 'is', 'a', 'an', 'and', 'or', 'for', 'to', 'in', 'on', 'with', 'after'}
            meaningful_words = [w for w in query_words if len(w) >= 3 and w not in stop_words]
            
            combined_sections = ' '.join(problem_sections)
            section_matches = sum(1 for w in meaningful_words if w in combined_sections)
            
            if section_matches >= 3:
                boost *= 1.5  # Extra boost for problem section match
            elif section_matches >= 2:
                boost *= 1.3
        
        return min(boost, 4.0)  # Cap at 4x total boost
    
    def _get_feedback_engine(self):
        """Lazy load feedback engine"""
        if self.feedback_engine is None:
            from src.llm.feedback_engine import FeedbackLearningEngine
            self.feedback_engine = FeedbackLearningEngine()
        return self.feedback_engine
    
    def get_product_info(self, part_number: str) -> Optional[Dict]:
        """
        Get product information from MongoDB
        Searches by part_number first, then by model_name
        
        Args:
            part_number: Product part number or model name
            
        Returns:
            Product info dict or None
        """
        try:
            if not self.mongodb:
                self.mongodb = MongoDBClient()
                self.mongodb.connect()
            
            # First try exact part_number match
            products = self.mongodb.get_products(
                filter_dict={"part_number": part_number},
                limit=1
            )
            
            if products:
                return products[0]
            
            # Try model_name match (case-insensitive)
            products = self.mongodb.get_products(
                filter_dict={"model_name": {"$regex": f"^{part_number}$", "$options": "i"}},
                limit=1
            )
            
            if products:
                return products[0]
                
            return None
        except Exception as e:
            logger.error(f"Error getting product info: {e}")
            return None
    
    def _get_product_capabilities(self, product_model: str) -> Dict:
        """
        Get product capabilities for capability-aware responses (Phase 0.2).
        
        Detects:
        - Wireless capability (from MongoDB)
        - Battery powered vs corded (from model code)
        - Standalone vs controller-required (from connection architecture)
        
        Args:
            product_model: Product model name (e.g., "EPBA8-1800-4Q")
            
        Returns:
            Dict with capabilities:
            {
                'wireless': bool,
                'battery_powered': bool,
                'corded': bool,
                'standalone': bool,
                'controller_required': bool,
                'product_found': bool
            }
        """
        capabilities = {
            'wireless': False,
            'battery_powered': False,
            'corded': False,
            'standalone': False,
            'controller_required': False,
            'product_found': False
        }
        
        try:
            # Get product from MongoDB
            product_info = self.get_product_info(product_model)
            
            if product_info:
                capabilities['product_found'] = True
                
                # 1. Wireless capability (from MongoDB)
                wireless_info = product_info.get('wireless', {})
                if isinstance(wireless_info, dict):
                    capabilities['wireless'] = wireless_info.get('capable', False)
                else:
                    capabilities['wireless'] = bool(wireless_info)
                
                # 2. Battery vs Corded (from model code patterns)
                model_upper = product_model.upper()
                
                # Battery-powered tools: EPB, EPBC, EABC, EABS, EAB, BLRT, ELC, XPB, ELS, ELB
                battery_patterns = ['EPB', 'EPBC', 'EABC', 'EABS', 'EAB', 'BLRT', 'ELC', 'XPB', 'ELS', 'ELB']
                capabilities['battery_powered'] = any(model_upper.startswith(p) for p in battery_patterns)
                
                # Corded tools: EAD, EPD, EFD, EIDS, ERS, ECS, MC, EM, ERAL, EME, EMEL
                corded_patterns = ['EAD', 'EPD', 'EFD', 'EIDS', 'ERS', 'ECS', 'MC', 'EM', 'ERAL', 'EME', 'EMEL']
                capabilities['corded'] = any(model_upper.startswith(p) for p in corded_patterns)
                
                # 3. Standalone vs Controller-required (from connection architecture)
                if ENABLE_DOMAIN_EMBEDDINGS and self.domain_embeddings:
                    try:
                        from src.llm.domain_vocabulary import DomainVocabulary
                        connection_info = DomainVocabulary.get_connection_info(product_model)
                        
                        if connection_info:
                            # Standalone: Battery tools without WiFi, or tools that don't need controller
                            standalone_categories = ['STANDALONE_BATTERY']
                            capabilities['standalone'] = connection_info.get('category') in standalone_categories
                            
                            # Controller required: Corded tools, or WiFi tools (need Connect unit)
                            controller_categories = ['CVI3_RANGE', 'CVIC_CVIR_CVIL', 'BATTERY_WIFI']
                            capabilities['controller_required'] = connection_info.get('category') in controller_categories
                    except Exception as e:
                        logger.debug(f"Could not get connection info: {e}")
                
                logger.debug(f"Product capabilities for {product_model}: {capabilities}")
            else:
                logger.warning(f"Product {product_model} not found in MongoDB")
                
        except Exception as e:
            logger.error(f"Error getting product capabilities: {e}")
        
        return capabilities
    
    def _build_product_filter(self, query: str, product_info: Dict = None) -> Optional[Dict]:
        """
        Build ChromaDB filter based on product context.
        NEW: Applies filtering at query time for efficient retrieval.
        
        Args:
            query: User's query text
            product_info: Selected product info from database (if available)
        
        Returns:
            ChromaDB where clause or None for no filtering
        """
        detected_family = None
        
        # Try to extract from product_info first (user selected a specific product)
        if product_info and product_info.get('model_name'):
            # Extract family from product model name using our intelligent extractor
            model_name = product_info.get('model_name', '')
            products = self.product_extractor.extract_from_filename(model_name)
            if products:
                detected_family = products[0].product_family
                logger.info(f"[FILTER] Product family from model: {detected_family}")
        
        # If no product info, try to extract from query itself
        if not detected_family:
            query_context = self.product_extractor.extract_product_from_query(query)
            if query_context.get('has_product_context'):
                detected_family = query_context.get('product_family')
                logger.info(f"[FILTER] Product family from query: {detected_family}")
        
        # If no product context detected, don't filter (return all results)
        if not detected_family:
            logger.info("[FILTER] No product context detected - searching all documents")
            return None
        
        # Build ChromaDB where filter
        # Include: matching family + aliases + GENERAL (generic docs) + UNKNOWN (unclassified)
        
        # Family aliases for cross-product retrieval
        FAMILY_ALIASES = {
            'EPBC': ['EPB'], 'EPB': ['EPBC'],  # EPB/EPBC are interchangeable
            'EABC': ['EAB'], 'EAB': ['EABC'],  # EAB/EABC are interchangeable
            'EABS': ['EAB', 'EABC'],            # EABS maps to EAB family
            'ERS': ['ERSA', 'ERSF'],            # ERS variants
            'ERSA': ['ERS', 'ERSF'],
            'ERSF': ['ERS', 'ERSA'],
        }
        
        # Start with detected family
        families_to_include = [detected_family]
        
        # Add aliases if they exist
        if detected_family in FAMILY_ALIASES:
            families_to_include.extend(FAMILY_ALIASES[detected_family])
            logger.info(f"[FILTER] Including aliases: {FAMILY_ALIASES[detected_family]}")
        
        # Build OR conditions for all families
        family_conditions = [{"product_family": {"$eq": f}} for f in families_to_include]
        family_conditions.extend([
            {"product_family": {"$eq": "GENERAL"}},
            {"product_family": {"$eq": "UNKNOWN"}},
            {"is_generic": {"$eq": True}}
        ])
        
        where_filter = {"$or": family_conditions}
        
        logger.info(f"[FILTER] Applying filter for family: {detected_family}")
        return where_filter
    
    def apply_score_boost(self, base_score: float, metadata: dict, query: str) -> float:
        """
        Apply score boosting based on document metadata and query match.
        
        Boosting factors:
        1. Document type (bulletin > manual)
        2. ESDE prefix (official service documents)
        3. Error code match (E06, I004, etc.)
        4. Problem description match
        
        Args:
            base_score: Original retrieval score
            metadata: Document metadata
            query: User's query
            
        Returns:
            Boosted score
        """
        boost = 1.0
        
        # 1. Document type boost
        doc_type = metadata.get('doc_type', 'unknown')
        boost *= self.SCORE_BOOSTS['doc_type'].get(doc_type, 1.0)
        
        # 2. ESDE prefix boost (official service bulletins)
        source = metadata.get('source', '')
        if source.upper().startswith('ESDE'):
            boost *= self.SCORE_BOOSTS['source_prefix']['ESDE']
        
        # 3. Error code direct match boost
        error_code_boost = self._calculate_error_code_boost(query, metadata)
        boost *= error_code_boost
        
        # 4. Problem description match boost
        problem_boost = self._calculate_problem_match_boost(query, metadata)
        boost *= problem_boost
        
        if boost > 1.0:
            logger.debug(f"[BOOST] {source}: {base_score:.3f} x {boost:.2f} = {base_score * boost:.3f}")
        
        return base_score * boost
    
    def _calculate_error_code_boost(self, query: str, metadata: dict) -> float:
        """
        Boost documents that contain error codes mentioned in query.
        
        Error codes: E06, E047, I004, Trd-E06, HW Channel, etc.
        """
        # Extract error codes from query
        error_patterns = [
            r'\b[EI]\d{2,4}\b',           # E06, E047, I004
            r'\bTrd[-\s]?E\d+\b',          # Trd-E06
            r'\bHW\s*Channel\b',           # HW Channel
            r'\bSpan\s*[Ff]ailure\b',      # Span Failure
        ]
        
        query_upper = query.upper()
        source = metadata.get('source', '').upper()
        chunk_text = str(metadata.get('chunk_text', '')).upper() if metadata.get('chunk_text') else ''
        
        # Also check document text if available
        doc_content = chunk_text or ''
        
        for pattern in error_patterns:
            matches = re.findall(pattern, query_upper, re.IGNORECASE)
            for match in matches:
                match_upper = match.upper().replace(' ', '').replace('-', '')
                # If error code in query also appears in document source or content
                if match_upper in source.replace(' ', '').replace('-', ''):
                    return self.SCORE_BOOSTS['error_code_match']
                if match_upper in doc_content.replace(' ', '').replace('-', ''):
                    return self.SCORE_BOOSTS['error_code_match']
        
        return 1.0
    
    def _calculate_problem_match_boost(self, query: str, metadata: dict) -> float:
        """
        Boost chunks containing explicit problem descriptions matching query.
        
        Patterns:
        - "Failure description: ..."
        - "Description of the issue: ..."
        - "Visible symptom: ..."
        """
        chunk_text = str(metadata.get('chunk_text', '')) if metadata.get('chunk_text') else ''
        if not chunk_text:
            # Try to get from document content if not in chunk_text field
            chunk_text = str(metadata.get('content', '')) if metadata.get('content') else ''
        
        if not chunk_text:
            return 1.0
        
        problem_patterns = [
            r'failure description[:\s]+([^.]+)',
            r'description of the issue[:\s]+([^.]+)',
            r'description of the subject[:\s]+([^.]+)',
            r'visible symptom[:\s]+([^.]+)',
            r'cause of issue[:\s]+([^.]+)',
            r'cause of failure[:\s]+([^.]+)',
        ]
        
        query_words = set(query.lower().split())
        chunk_lower = chunk_text.lower()
        
        for pattern in problem_patterns:
            match = re.search(pattern, chunk_lower)
            if match:
                problem_desc = match.group(1)
                problem_words = set(problem_desc.split())
                
                # Calculate word overlap
                overlap = len(query_words & problem_words)
                if overlap >= 2:
                    boost = self.SCORE_BOOSTS['problem_match_base'] + (overlap * self.SCORE_BOOSTS['problem_match_per_word'])
                    return min(boost, 2.5)  # Cap at 2.5x
        
        return 1.0
    
    def retrieve_context(
        self,
        query: str,
        part_number: Optional[str] = None,
        top_k: int = RAG_TOP_K
    ) -> Dict:
        """
        Retrieve relevant context from vector database
        Uses hybrid search (semantic + BM25) when enabled
        NEW: Applies product filtering at query time
        
        Args:
            query: Search query (fault description)
            part_number: Optional product part number for filtering
            top_k: Number of results to retrieve
            
        Returns:
            Dict with retrieved documents and metadata
        """
        logger.info(f"Retrieving context for query: {query[:50]}...")
        
        # Phase 3.1: Enhance query with domain knowledge (ONLY if enabled)
        enhanced_query = query
        if ENABLE_DOMAIN_EMBEDDINGS and self.domain_embeddings:
            try:
                enhancement = self.domain_embeddings.enhance_query(query)
                enhanced_query = enhancement.get("enhanced", query)
                if enhanced_query != query:
                    logger.info(f"Domain-enhanced query: {enhanced_query[:80]}...")
            except Exception as e:
                logger.warning(f"Domain enhancement failed: {e}")
        
        # Get product info for filtering
        product_info = None
        if part_number:
            product_info = self.get_product_info(part_number)
        
        # NEW: Build product filter for ChromaDB
        product_filter = self._build_product_filter(query, product_info)
        
        # Still get target categories for client-side filtering (fallback)
        target_categories = []
        if part_number:
            raw_cats = self.product_extractor.get_product_categories(part_number)
            if raw_cats:
                target_categories = [c.strip() for c in raw_cats.split(",")]

        # Use hybrid search if available
        if self.hybrid_searcher:
            return self._retrieve_with_hybrid_search(
                enhanced_query, 
                top_k, 
                original_query=query,
                target_categories=target_categories,
                product_filter=product_filter  # NEW: Pass filter
            )
        
        # Fallback to standard semantic search
        return self._retrieve_with_semantic_search(
            enhanced_query, 
            part_number, 
            top_k,
            target_categories=target_categories,
            product_filter=product_filter  # NEW: Pass filter
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for self-learning (Phase 6)"""
        import re
        from collections import Counter
        
        text = text.lower()
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            've', 'bir', 'bu', 'su', 'ile', 'için', 'gibi', 'daha', 'çok',
            'var', 'yok', 'olan', 'olarak', 've', 'veya', 'ama', 'fakat'
        }
        words = re.findall(r'\b[a-zA-ZçğıöşüÇĞİÖŞÜ]{3,}\b', text)
        keywords = [w for w in words if w not in stop_words]
        counter = Counter(keywords)
        return [word for word, _ in counter.most_common(10)]
    
    def _retrieve_with_hybrid_search(
        self, 
        query: str, 
        top_k: int, 
        original_query: str = None,
        target_categories: List[str] = None,
        product_filter: Optional[Dict] = None  # NEW: ChromaDB where clause
    ) -> Dict:
        """Retrieve using hybrid search (semantic + BM25) with metadata filtering and self-learning"""
        # Pass product_filter to hybrid searcher for ChromaDB filtering
        if product_filter:
            logger.info("[HYBRID] Product filter active, passing to ChromaDB query")
        results = self.hybrid_searcher.search(
            query=query,
            top_k=top_k * 2,  # Get more candidates for boosting/reranking
            expand_query=ENABLE_QUERY_EXPANSION,
            use_hybrid=True,
            where_filter=product_filter,  # NEW: Pass product filter to ChromaDB
            min_similarity=RAG_SIMILARITY_THRESHOLD
        )
        
        # Phase 0.1: Apply relevance filtering (ONLY if enabled - can be slow)
        # Filters out documents that don't match query intent (e.g., WiFi query → transducer docs)
        if ENABLE_FAULT_FILTERING:
            try:
                from src.llm.relevance_filter import filter_irrelevant_results
                results = filter_irrelevant_results(query, results)
            except Exception as e:
                logger.warning(f"Relevance filtering failed, using original results: {e}")
                # Continue with original results if filter fails (safety-first)
        else:
            logger.debug("Fault filtering DISABLED (ENABLE_FAULT_FILTERING=false)")

        
        filtered_docs = []
        for result in results:
            meta = result.metadata or {}
            
            # Step 1: Metadata Filtering (Hard Filter or Heavy Demotion)
            # If target categories exist, check for mismatch
            if target_categories and meta.get("product_categories"):
                doc_cats_str = meta.get("product_categories", "")
                doc_cats = [c.strip() for c in doc_cats_str.split(",")] if isinstance(doc_cats_str, str) else doc_cats_str
                
                # Check for intersection
                overlap = set(target_categories) & set(doc_cats)
                
                # If there's a specific product code mismatch (e.g. EPB vs ERS)
                # We want to be strict.
                if not overlap:
                    # Is it a major product code mismatch?
                    # Product codes are typically 3+ chars: EPB, ERS, CVI, CONNECT
                    target_codes = {c for c in target_categories if len(c) >= 3 and c.isupper()}
                    doc_codes = {c for c in doc_cats if len(c) >= 3 and c.isupper()}
                    
                    if target_codes and doc_codes and not (target_codes & doc_codes):
                        logger.debug(f"Filtering out document from {meta.get('source')} due to product mismatch: {doc_codes} vs {target_codes}")
                        continue # Skip this result entirely
            
            base_score = result.similarity if result.similarity > 0 else result.score
            
            # Apply metadata-based boosting (Phase 4.1 + 7.1)
            # Pass query and content for error code + keyword matching
            boosted_score = self._apply_metadata_boost(
                base_score, 
                result.metadata, 
                query=query, 
                content=result.content or ""
            )
            
            filtered_docs.append({
                "text": result.content,
                "metadata": result.metadata,
                "similarity": base_score,
                "boosted_score": boosted_score,
                "search_type": result.source,  # 'semantic', 'bm25', or 'hybrid'
                "bm25_score": result.bm25_score
            })
        
        # Phase 6: Apply self-learning ranking
        if self.self_learning_engine and filtered_docs:
            keywords = self._extract_keywords(query)
            if keywords:
                # Get learned recommendations
                recommendations = self.self_learning_engine.get_recommendations_for_query(keywords)
                boost_sources = set(recommendations.get("boost_sources", []))
                avoid_sources = set(recommendations.get("avoid_sources", []))
                
                # Apply learned boosts
                for doc in filtered_docs:
                    source = doc["metadata"].get("source", "")
                    learned_boost = self.self_learning_engine.ranking_learner.get_source_boost(source)
                    
                    # Extra boost for keyword-recommended sources
                    if source in boost_sources:
                        learned_boost *= 1.25
                    elif source in avoid_sources:
                        learned_boost *= 0.75
                    
                    # Apply learned boost to boosted_score
                    doc["boosted_score"] = doc["boosted_score"] * learned_boost
                    doc["learned_boost"] = learned_boost
                
                if recommendations.get("mappings_found", 0) > 0:
                    logger.info(f"Applied self-learning: {len(boost_sources)} boost, {len(avoid_sources)} avoid sources")
        
        # Re-sort by boosted score and limit to top_k
        filtered_docs.sort(key=lambda x: x.get("boosted_score", x.get("similarity", 0)), reverse=True)
        filtered_docs = filtered_docs[:top_k]
        
        logger.info(f"Hybrid search retrieved {len(filtered_docs)} documents (with metadata + learned boost)")
        return {
            "documents": filtered_docs,
            "query": query,
            "search_type": "hybrid"
        }
    
    def _retrieve_with_semantic_search(
        self, 
        query: str, 
        part_number: Optional[str],
        top_k: int,
        target_categories: List[str] = None,
        product_filter: Optional[Dict] = None  # NEW: ChromaDB where clause
    ) -> Dict:
        """Fallback: retrieve using standard semantic search with product filtering"""
        
        # Generate query embedding
        query_embedding = self.embeddings.generate_embedding(query)
        
        # Use product filter if available, otherwise fall back to client-side filtering
        where = product_filter
        query_n = top_k if product_filter else max(50, top_k * 5)
        
        # Query vector database with filter
        try:
            results = self.vectordb.query(
                query_text=query,
                query_embedding=query_embedding,
                n_results=query_n,
                where=where
            )
        except Exception as e:
            # If filter fails (metadata field doesn't exist), retry without filter
            logger.warning(f"[RETRIEVAL] Filter failed: {e}, retrying without filter")
            results = self.vectordb.query(
                query_text=query,
                query_embedding=query_embedding,
                n_results=max(50, top_k * 5),
                where=None
            )
        
        # Extract results
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        # Filter by distance threshold based on RAG_SIMILARITY_THRESHOLD config
        # L2 distance conversion: similarity_score = max(0, 1 - distance/2)
        # So: distance_threshold = 2 * (1 - similarity_threshold)
        # Example: similarity_threshold=0.7 → distance_threshold=0.6
        #          similarity_threshold=0.85 → distance_threshold=0.3
        similarity_threshold = RAG_SIMILARITY_THRESHOLD
        distance_threshold = 2 * (1 - similarity_threshold)
        
        filtered_docs = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            # Step 1: Metadata Filtering (Hard Filter) 
            if target_categories and meta.get("product_categories"):
                doc_cats_str = meta.get("product_categories", "")
                doc_cats = [c.strip() for c in doc_cats_str.split(",")] if isinstance(doc_cats_str, str) else doc_cats_str
                
                overlap = set(target_categories) & set(doc_cats)
                
                target_codes = {c for c in target_categories if len(c) >= 3 and c.isupper()}
                doc_codes = {c for c in doc_cats if len(c) >= 3 and c.isupper()}
                
                if target_codes and doc_codes and not (target_codes & doc_codes):
                    continue # Strict mismatch skip
                    
            # Skip if distance is too high (not similar enough)
            if dist > distance_threshold:
                similarity_score = max(0, 1 - dist/2)
                logger.debug(f"Skipping doc with distance {dist:.3f} (similarity {similarity_score:.3f}) < threshold {similarity_threshold:.2f}")
                continue

            # Note: We removed the part_number filter because most bulletins 
            # are general and don't have specific product references.
            # The LLM will determine relevance based on the context.

            similarity_score = max(0, 1 - dist/2)
            
            # Apply metadata-based boosting (Phase 4.1 + 7.1)
            # Pass query and content for error code + keyword matching
            boosted_score = self._apply_metadata_boost(
                similarity_score, 
                meta, 
                query=query, 
                content=doc or ""
            )
            
            filtered_docs.append({
                "text": doc,
                "metadata": meta,
                "similarity": similarity_score,
                "boosted_score": boosted_score
            })
        
        # Re-sort by boosted score and limit to top_k
        filtered_docs.sort(key=lambda x: x.get("boosted_score", x.get("similarity", 0)), reverse=True)
        filtered_docs = filtered_docs[:top_k]
        
        logger.info(f"Retrieved {len(filtered_docs)} relevant documents (similarity threshold: {similarity_threshold:.2f}, with metadata boost)")
        
        return {
            "documents": filtered_docs,
            "query": query
        }
    
    def generate_repair_suggestion(
        self,
        part_number: str,
        fault_description: str,
        language: str = DEFAULT_LANGUAGE,
        username: str = "anonymous",
        excluded_sources: List[str] = None,
        is_retry: bool = False,
        retry_of: Optional[str] = None
    ) -> Dict:
        """
        Generate repair suggestion using RAG with learning enhancements
        
        Args:
            part_number: Product part number
            fault_description: Description of the fault
            language: Language code ('en' or 'tr') - auto-detected if empty or 'auto'
            username: User requesting the diagnosis
            excluded_sources: Sources to exclude (for retry)
            is_retry: Whether this is a retry request
            retry_of: Original diagnosis ID if retry
            
        Returns:
            Dict with suggestion and metadata
        """
        start_time = time.time()
        logger.info(f"Generating repair suggestion for {part_number} (retry={is_retry})")
        
        # =====================================================================
        # OFF-TOPIC DETECTION (Priority 0 - Before any processing)
        # =====================================================================
        off_topic_result = self._check_off_topic(fault_description, part_number)
        if off_topic_result:
            from src.llm.context_grounding import build_idk_response
            response_time_ms = int((time.time() - start_time) * 1000)
            
            idk_response = build_idk_response(
                query=fault_description,
                product_model=part_number,
                reason=off_topic_result,
                language=language if language and language != "auto" else "en"
            )
            
            logger.warning(f"[OFF-TOPIC] Refusing query: {off_topic_result}")
            
            return {
                "suggestion": idk_response,
                "confidence": "insufficient_context",
                "confidence_numeric": 0.0,
                "product_model": part_number,
                "part_number": part_number,
                "sources": [],
                "language": language if language and language != "auto" else "en",
                "diagnosis_id": None,
                "response_time_ms": response_time_ms,
                "intent": "general",
                "intent_confidence": 0.0,
                "off_topic_reason": off_topic_result
            }
        
        # =====================================================================
        # AUTO-DETECT LANGUAGE if not specified or set to 'auto'
        # =====================================================================
        if not language or language == "auto":
            language = self._detect_language(fault_description)
            logger.info(f"[LANG] Auto-detected: {language} (query: '{fault_description[:40]}...')")
        elif language not in ("en", "tr"):
            logger.warning(f"[LANG] Unknown language '{language}', defaulting to 'en'")
            language = "en"
        
        # Phase 2.3: Check response cache (skip for retry requests)
        cache_key = None
        if self.response_cache and not is_retry and not excluded_sources:
            cache_key = f"{part_number}:{fault_description}:{language}"
            cached_response = self.response_cache.get(cache_key)
            if cached_response:
                # Add cache hit metadata
                cached_response["from_cache"] = True
                cached_response["response_time_ms"] = int((time.time() - start_time) * 1000)
                logger.info(f"✅ Cache HIT - returning cached response in {cached_response['response_time_ms']}ms")
                return cached_response
        
        # Get learning context from feedback engine
        feedback_engine = self._get_feedback_engine()
        learned_context = feedback_engine.get_learned_context(
            fault_description=fault_description,
            excluded_sources=excluded_sources or []
        )
        
        # Check if we have a high-confidence learned solution
        if learned_context.get("learned_solution") and not is_retry:
            logger.info("Using high-confidence learned solution")
        
        # Get product info
        product_info = self.get_product_info(part_number)
        
        # If product not found, still try RAG with the query
        if not product_info:
            logger.warning(f"Product not found: {part_number}, will use RAG only")
            product_model = part_number
            actual_part_number = part_number
        else:
            product_model = product_info.get("model_name", part_number)
            actual_part_number = product_info.get("part_number", part_number)
        
        # Build enhanced query with product context
        # This ensures retrieval finds product-specific documents
        enhanced_query = f"{product_model} {actual_part_number} {fault_description}"
        logger.info(f"Enhanced retrieval query: {enhanced_query[:80]}...")
        
        # Get product capabilities EARLY for filtering
        capabilities = self._get_product_capabilities(product_model)
        if capabilities.get('product_found'):
            logger.info(f"[CAPABILITIES] wireless={capabilities.get('wireless')}, battery={capabilities.get('battery_powered')}")
        
        # Retrieve relevant context (always try RAG even if product not found)
        context_result = self.retrieve_context(
            query=enhanced_query,
            part_number=actual_part_number
        )
        
        retrieved_docs = context_result["documents"]
        
        # =====================================================================
        # STRICT PRODUCT FILTERING (Priority 1 - Production Quality)
        # Prevents cross-product contamination (CVIL2 docs for ERS6 queries)
        # =====================================================================
        if product_info:
            retrieved_docs = self._filter_by_product_strict(retrieved_docs, product_info)
        
        # =====================================================================
        # FILTER BY PRODUCT CAPABILITIES (Priority 2 - Prevent Hallucination)
        # =====================================================================
        if capabilities.get('product_found'):
            retrieved_docs = self._filter_by_product_capabilities(retrieved_docs, capabilities)
        
        # Apply learning: filter out excluded sources (for retry)
        all_excluded = set(excluded_sources or []) | set(learned_context.get("exclude_sources", []))
        if all_excluded:
            original_count = len(retrieved_docs)
            retrieved_docs = [
                doc for doc in retrieved_docs 
                if doc["metadata"].get("source", "") not in all_excluded
            ]
            logger.info(f"Filtered {original_count - len(retrieved_docs)} excluded sources")
        
        
        # Apply learning: boost sources from positive feedback
        boost_sources = learned_context.get("boost_sources", [])
        if boost_sources:
            # Sort to prioritize boosted sources
            def sort_key(doc):
                source = doc["metadata"].get("source", "")
                # Boosted sources get priority (lower number = higher priority)
                if source in boost_sources:
                    return (0, -doc["similarity"])
                return (1, -doc["similarity"])
            
            retrieved_docs = sorted(retrieved_docs, key=sort_key)
            logger.info(f"Boosted {len(boost_sources)} learned sources")
        
        # NEW: Check context sufficiency (Priority 1 - Response Grounding)
        from config.ai_settings import (
            ENABLE_CONTEXT_GROUNDING, 
            CONTEXT_SUFFICIENCY_THRESHOLD,
            MIN_SIMILARITY_FOR_ANSWER,
            MIN_DOCS_FOR_CONFIDENCE
        )
        
        # Initialize sufficiency variable (will be populated if grounding enabled)
        sufficiency = None
        
        if ENABLE_CONTEXT_GROUNDING and retrieved_docs:
            from src.llm.context_grounding import ContextSufficiencyScorer,build_idk_response
            
            # Calculate average similarity
            similarities = [doc.get("similarity", doc.get("boosted_score", 0.0)) for doc in retrieved_docs]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            
            # Create scorer and check sufficiency
            scorer = ContextSufficiencyScorer(
                sufficiency_threshold=CONTEXT_SUFFICIENCY_THRESHOLD,
                min_similarity=MIN_SIMILARITY_FOR_ANSWER,
                min_docs=MIN_DOCS_FOR_CONFIDENCE
            )
            
            sufficiency = scorer.calculate_sufficiency_score(
                query=fault_description,
                retrieved_docs=retrieved_docs,
                avg_similarity=avg_similarity
            )
            
            # SIMPLE CHECK: Only refuse if NO documents found at all
            # Let the system try to answer if there's any context
            if not sufficiency.is_sufficient and len(retrieved_docs) == 0:
                response_time_ms = int((time.time() - start_time) * 1000)
                
                idk_response = build_idk_response(
                    query=fault_description,
                    product_model=product_model,
                    reason=sufficiency.reason,
                    language=language
                )
                
                logger.warning(f"Insufficient context (score={sufficiency.score:.3f}): {sufficiency.reason}")
                
                # Save to diagnosis history with special confidence
                try:
                    diagnosis_id = feedback_engine.save_diagnosis(
                        part_number=actual_part_number,
                        product_model=product_model,
                        fault_description=fault_description,
                        suggestion=idk_response,
                        confidence="insufficient_context",
                        sources=[],  # No sources to cite
                        username=username,
                        language=language,
                        is_retry=is_retry,
                        retry_of=retry_of,
                        response_time_ms=response_time_ms
                    )
                except Exception as e:
                    logger.error(f"Error saving diagnosis: {e}")
                    diagnosis_id = None
                
                return {
                    "suggestion": idk_response,
                    "confidence": "insufficient_context",
                    "sufficiency_score": sufficiency.score,
                    "sufficiency_factors": sufficiency.factors,
                    "sufficiency_reason": sufficiency.reason,
                    "product_model": product_model,
                    "part_number": actual_part_number,
                    "sources": [],
                    "language": language,
                    "diagnosis_id": diagnosis_id,
                    "response_time_ms": response_time_ms
                }
        
        # Phase 3.4: Optimize context window
        if retrieved_docs:
            # Use context optimizer for better chunk selection and formatting
            optimized_chunks, opt_stats = self.context_optimizer.optimize(
                retrieved_docs=retrieved_docs,
                query=fault_description,
                max_chunks=RAG_TOP_K * 2  # Allow more chunks, optimizer will filter
            )
            
            if optimized_chunks:
                context_str = self.context_optimizer.build_context_string(
                    optimized_chunks,
                    include_metadata=True,
                    group_by_source=False
                )
                
                # Build sources list from optimized chunks
                optimized_sources = [
                    {
                        "source": chunk.source,
                        "similarity": chunk.similarity,
                        "section_type": chunk.section_type,
                        "is_warning": chunk.is_warning,
                        "is_procedure": chunk.is_procedure,
                        "excerpt": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                    }
                    for chunk in optimized_chunks
                ]
                
                logger.info(f"Context optimized: {opt_stats['chunks_in']}→{opt_stats['chunks_out']} chunks, "
                           f"{opt_stats['tokens_used']} tokens, {opt_stats['duplicates_removed']} duplicates removed")
            else:
                # Fallback to simple formatting if optimization returns empty
                context_str = "\n\n".join([
                    f"[Source: {doc['metadata'].get('source', 'Unknown')}]\n{doc['text']}"
                    for doc in retrieved_docs
                ])
                optimized_sources = None
            
            # Phase 0.2: Get product capabilities
            capabilities = self._get_product_capabilities(product_model)
            
            # Build RAG prompt with capabilities
            prompt = build_rag_prompt(
                product_model=product_model,
                part_number=actual_part_number,
                fault_description=fault_description,
                context=context_str,
                language=language,
                capabilities=capabilities
            )
            
            
            # Calculate confidence based on sufficiency score (FIXED: was using chunk count)
            # This is critical for response quality - LLM performs better with accurate confidence
            logger.info(f"DEBUG: sufficiency={sufficiency}, score={sufficiency.score if sufficiency else 'None'}")
            if sufficiency and sufficiency.score >= 0.7:
                confidence = "high"
                logger.info(f"Confidence set to HIGH (score={sufficiency.score:.3f})")
            elif sufficiency and sufficiency.score >= 0.5:
                confidence = "medium"
                logger.info(f"Confidence set to MEDIUM (score={sufficiency.score:.3f})")
            else:
                confidence = "low"
                logger.info(f"Confidence set to LOW (score={sufficiency.score if sufficiency else 'None'})")
        else:
            # No relevant context found - use fallback
            logger.warning("No relevant context found in manuals")
            context_str = build_fallback_response(product_model, language)
            prompt = f"{fault_description}\n\n{context_str}"
            confidence = "low"
            optimized_sources = None
            # Get capabilities for fallback path as well
            capabilities = self._get_product_capabilities(product_model)
        
        # Phase 3.3: Detect Query Intent (Priority 3)
        intent_result = None
        try:
            if not getattr(self, "intent_detector", None):
                from src.llm.intent_detector import IntentDetector
                self.intent_detector = IntentDetector()
            
            intent_result = self.intent_detector.detect_intent(fault_description, product_info)
            logger.info(f"Query intent: {intent_result.intent.value} (confidence: {intent_result.confidence})")
        except Exception as e:
            logger.warning(f"Intent detection failed: {e}")
            
        # Get system prompt (enhanced with intent)
        system_prompt = get_system_prompt(language, intent=intent_result.intent if intent_result else None)
        
        # Build RAG prompt with intent awareness
        prompt = build_rag_prompt(
            product_model=product_model,
            part_number=actual_part_number,
            fault_description=fault_description,
            context=context_str,
            language=language,
            capabilities=capabilities,
            intent=intent_result.intent if intent_result else None
        )
        
        # Generate suggestion from LLM
        logger.info("Generating LLM response...")
        suggestion = self.llm.generate(
            prompt=prompt,
            system=system_prompt
        )
        
        if not suggestion:
            suggestion = "Error: Unable to generate suggestion. Please try again."
            confidence = "low"
        
        # NEW: Validate response (Priority 2 - Response Validation)
        validation_result = None
        from config.ai_settings import (
            ENABLE_RESPONSE_VALIDATION,
            FLAG_UNCERTAINTY_PHRASES,
            VERIFY_NUMERICAL_VALUES,
            MIN_RESPONSE_LENGTH,
            MAX_UNCERTAINTY_COUNT
        )
        
        if ENABLE_RESPONSE_VALIDATION and suggestion:
            from src.llm.response_validator import ResponseValidator
            
            validator = ResponseValidator(
                max_uncertainty_count=MAX_UNCERTAINTY_COUNT,
                min_response_length=MIN_RESPONSE_LENGTH,
                flag_uncertain_responses=FLAG_UNCERTAINTY_PHRASES,
                verify_numbers=VERIFY_NUMERICAL_VALUES
            )
            
            # Get product capabilities for validation
            capabilities = self._get_product_capabilities(product_model)
            
            validation_result = validator.validate_response(
                response=suggestion,
                query=fault_description,
                context=context_str,
                product_info={
                    'model_name': product_model,
                    'wireless': capabilities.get('wireless', False),
                    'battery_powered': capabilities.get('battery_powered', False)
                }
            )
            
            # Adjust confidence if validation suggests
            if validation_result.confidence_adjustment:
                confidence = validation_result.confidence_adjustment
                logger.info(f"Confidence adjusted to '{confidence}' based on validation")
            
            # Log validation results
            if validation_result.issues:
                logger.warning(
                    f"Validation found {len(validation_result.issues)} issue(s), "
                    f"severity={validation_result.severity}"
                )
                for issue in validation_result.issues:
                    logger.debug(f"  - {issue.type}: {issue.description}")
        
        # Prepare response with optimized sources if available
        # Apply source-based deduplication to prevent same document appearing multiple times
        if optimized_sources:
            seen_sources = set()
            sources_list = []
            for src in optimized_sources:
                source_name = src["source"]
                if source_name not in seen_sources:
                    sources_list.append({
                        "source": source_name,
                        "page": src.get("page_number"),
                        "section": src.get("section", ""),
                        "similarity": f"{src['similarity']:.2f}" if isinstance(src['similarity'], float) else src['similarity'],
                        "section_type": src.get("section_type", ""),
                        "is_warning": src.get("is_warning", False),
                        "is_procedure": src.get("is_procedure", False),
                        "excerpt": src["excerpt"]
                    })
                    seen_sources.add(source_name)
        else:
            seen_sources = set()
            sources_list = []
            for doc in (retrieved_docs or []):
                source_name = doc["metadata"].get("source", "Unknown")
                if source_name not in seen_sources:
                    sources_list.append({
                        "source": source_name,
                        "page": doc["metadata"].get("page_number"),
                        "section": doc["metadata"].get("section", ""),
                        "similarity": f"{doc['similarity']:.2f}" if isinstance(doc.get('similarity'), float) else str(doc.get('similarity', 0)),
                        "excerpt": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"]
                    })
                    seen_sources.add(source_name)
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # =================================================================
        # ADVANCED CONFIDENCE SCORING (replaces simple threshold)
        # =================================================================
        try:
            from src.llm.confidence_scorer import ConfidenceScorer
            scorer = ConfidenceScorer()
            
            # Calculate average similarity from sources
            avg_sim = 0.0
            if optimized_sources:
                sims = [s.get('similarity', 0) for s in optimized_sources if isinstance(s.get('similarity'), (int, float))]
                avg_sim = sum(sims) / len(sims) if sims else 0.0
            elif retrieved_docs:
                sims = [d.get('similarity', 0) for d in retrieved_docs if isinstance(d.get('similarity'), (int, float))]
                avg_sim = sum(sims) / len(sims) if sims else 0.0
            
            confidence_result = scorer.calculate_confidence(
                sources=optimized_sources or retrieved_docs or [],
                intent=intent_result.intent.value if intent_result else None,
                intent_confidence=intent_result.confidence if intent_result else 0.0,
                response_text=suggestion,
                sufficiency_score=sufficiency.score if sufficiency else None,
                avg_similarity=avg_sim
            )
            
            confidence = confidence_result.level
            confidence_numeric = confidence_result.score
            confidence_factors = confidence_result.factors
            logger.info(f"Advanced confidence: {confidence} (score={confidence_numeric:.3f}, factors={confidence_factors})")
            
        except Exception as e:
            logger.warning(f"Advanced confidence scoring failed: {e}, using fallback")
            confidence_numeric = 0.3 if confidence == "low" else 0.6 if confidence == "medium" else 0.8
            confidence_factors = {}
        
        # Structure the response
        response = {
            "suggestion": suggestion.strip(),
            "confidence": confidence,
            "confidence_numeric": confidence_numeric,
            "product_model": product_model,
            "part_number": actual_part_number,
            "sources": sources_list,
            "language": language,
            "diagnosis_id": None,
            "response_time_ms": response_time_ms,
            "intent": intent_result.intent.value if intent_result else "general",
            "intent_confidence": intent_result.confidence if intent_result else 0.0
        }
        
        # Add grounding metadata if available (Priority 1)
        if sufficiency is not None:
            response["sufficiency_score"] = sufficiency.score
            response["sufficiency_reason"] = sufficiency.reason
            response["sufficiency_factors"] = sufficiency.factors
            response["sufficiency_recommendation"] = sufficiency.recommendation
            logger.debug(f"Added sufficiency metadata: score={sufficiency.score:.2f}")

        # Add validation metadata if available (Priority 2)
        if validation_result:
            response["validation"] = {
                "is_valid": validation_result.is_valid,
                "severity": validation_result.severity,
                "should_flag": validation_result.should_flag,
                "issues": [
                    {
                        "type": issue.type,
                        "description": issue.description, 
                        "severity": issue.severity,
                        "location": getattr(issue, 'location', ''),
                        "detected_value": getattr(issue, 'detected_value', '')
                    } 
                    for issue in validation_result.issues
                ]
            }

        # Save to diagnosis history
        try:
            # Prepare metadata for storage
            metadata = {
                "sufficiency": sufficiency.__dict__ if sufficiency else None,
                "validation": response.get("validation"),
                "intent": response["intent"],
                "intent_confidence": response["intent_confidence"]
            }
            
            diagnosis_id = feedback_engine.save_diagnosis(
                part_number=actual_part_number,
                product_model=product_model,
                fault_description=fault_description,
                suggestion=suggestion.strip(),
                confidence=confidence,
                sources=[s["source"] for s in sources_list],
                username=username,
                language=language,
                is_retry=is_retry,
                retry_of=retry_of,
                response_time_ms=response_time_ms,
                metadata=metadata
            )
            response["diagnosis_id"] = diagnosis_id
        except Exception as e:
            logger.error(f"Error saving diagnosis history: {e}")
            response["diagnosis_id"] = None
        
        logger.info(f"✅ Generated suggestion with {confidence} confidence in {response_time_ms}ms (Intent: {response['intent']})")
        
        # Phase 2.3: Store in response cache (only for non-retry, successful responses)
        if self.response_cache and cache_key and confidence != "low":
            # Create a copy for caching (without mutable references)
            cache_entry = {
                "suggestion": response["suggestion"],
                "confidence": response["confidence"],
                "product_model": response["product_model"],
                "part_number": response["part_number"],
                "sources": response["sources"],
                "language": response["language"],
                "cached_at": time.time(),
                "intent": response["intent"]
            }
            self.response_cache.set(cache_key, cache_entry)
            logger.info(f"✅ Cached response (key={cache_key[:50]}...)")
        
        return response
    
    def stream_repair_suggestion(
        self,
        part_number: str,
        fault_description: str,
        language: str = DEFAULT_LANGUAGE
    ):
        """
        Stream repair suggestion (for real-time UI updates)
        
        Args:
            part_number: Product part number
            fault_description: Description of the fault
            language: Language code
            
        Yields:
            Response chunks
        """
        # Get product info and context (same as generate_repair_suggestion)
        product_info = self.get_product_info(part_number)
        if not product_info:
            yield {"error": f"Product {part_number} not found"}
            return
        
        product_model = product_info.get("model_name", part_number)
        context_result = self.retrieve_context(fault_description, part_number)
        retrieved_docs = context_result["documents"]
        
        # Phase 0.2: Get product capabilities
        capabilities = self._get_product_capabilities(product_model)
        
        # Build prompt
        if retrieved_docs:
            context_str = "\n\n".join([f"[{doc['metadata'].get('source')}]\n{doc['text']}" for doc in retrieved_docs])
            prompt = build_rag_prompt(product_model, part_number, fault_description, context_str, language, capabilities)
        else:
            context_str = build_fallback_response(product_model, language)
            prompt = f"{fault_description}\n\n{context_str}"
        
        system_prompt = get_system_prompt(language)
        
        # Stream from LLM
        for chunk in self.llm.stream_generate(prompt, system_prompt):
            yield {"chunk": chunk}
