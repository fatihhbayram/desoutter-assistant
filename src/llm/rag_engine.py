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
from src.llm.performance_metrics import get_performance_monitor, QueryTimer, QueryMetrics
from src.database import MongoDBClient
from config.ai_settings import (
    RAG_TOP_K, RAG_SIMILARITY_THRESHOLD, DEFAULT_LANGUAGE,
    USE_HYBRID_SEARCH, HYBRID_SEMANTIC_WEIGHT, HYBRID_BM25_WEIGHT,
    HYBRID_RRF_K, ENABLE_QUERY_EXPANSION,
    USE_CACHE, CACHE_TTL,
    ENABLE_METADATA_BOOST, SERVICE_BULLETIN_BOOST, PROCEDURE_BOOST,
    WARNING_BOOST, IMPORTANCE_BOOST_FACTOR
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGEngine:
    """RAG Engine for repair suggestions with self-learning capabilities"""
    
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
        self.context_optimizer = ContextOptimizer(token_budget=8000)  # Phase 3.4
        self.performance_monitor = get_performance_monitor()  # Phase 5.1
        
        # Initialize hybrid search if enabled
        if USE_HYBRID_SEARCH:
            self._init_hybrid_search()
        
        # Initialize response cache if enabled (Phase 2.3)
        if USE_CACHE:
            self._init_response_cache()
        
        # Initialize self-learning engine (Phase 6)
        self._init_self_learning()
        
        # Initialize domain embeddings (Phase 3.1)
        self._init_domain_embeddings()
        
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
    
    def _apply_metadata_boost(self, base_score: float, metadata: Dict) -> float:
        """
        Apply metadata-based score boosting (Phase 4.1)
        
        Boosts documents based on:
        - Service bulletins (ESD/ESB) get priority
        - Procedure sections (step-by-step)
        - Warning/caution content
        - Importance score from semantic chunking
        
        Args:
            base_score: Original similarity/relevance score
            metadata: Document metadata dictionary
            
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
        
        boosted_score = base_score * boost
        
        if boost > 1.0:
            logger.debug(f"Metadata boost applied: {base_score:.3f} → {boosted_score:.3f} ({', '.join(boost_reasons)})")
        
        return boosted_score
    
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
                if self.domain_embeddings:
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
    
    def retrieve_context(
        self,
        query: str,
        part_number: Optional[str] = None,
        top_k: int = RAG_TOP_K
    ) -> Dict:
        """
        Retrieve relevant context from vector database
        Uses hybrid search (semantic + BM25) when enabled
        
        Args:
            query: Search query (fault description)
            part_number: Optional product part number for filtering
            top_k: Number of results to retrieve
            
        Returns:
            Dict with retrieved documents and metadata
        """
        logger.info(f"Retrieving context for query: {query[:50]}...")
        
        # Phase 3.1: Enhance query with domain knowledge
        enhanced_query = query
        if self.domain_embeddings:
            try:
                enhancement = self.domain_embeddings.enhance_query(query)
                enhanced_query = enhancement.get("enhanced", query)
                if enhanced_query != query:
                    logger.info(f"Domain-enhanced query: {enhanced_query[:80]}...")
            except Exception as e:
                logger.warning(f"Domain enhancement failed: {e}")
        
        # Use hybrid search if available
        if self.hybrid_searcher:
            return self._retrieve_with_hybrid_search(enhanced_query, top_k, original_query=query)
        
        # Fallback to standard semantic search
        return self._retrieve_with_semantic_search(enhanced_query, part_number, top_k)
    
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
    
    def _retrieve_with_hybrid_search(self, query: str, top_k: int, original_query: str = None) -> Dict:
        """Retrieve using hybrid search (semantic + BM25) with metadata boosting and self-learning"""
        results = self.hybrid_searcher.search(
            query=query,
            top_k=top_k * 2,  # Get more candidates for boosting/reranking
            expand_query=ENABLE_QUERY_EXPANSION,
            use_hybrid=True,
            min_similarity=RAG_SIMILARITY_THRESHOLD
        )
        
        # Phase 0.1: Apply relevance filtering (production-safe, can be disabled via config)
        # Filters out documents that don't match query intent (e.g., WiFi query → transducer docs)
        try:
            from src.llm.relevance_filter import filter_irrelevant_results
            results = filter_irrelevant_results(query, results)
        except Exception as e:
            logger.warning(f"Relevance filtering failed, using original results: {e}")
            # Continue with original results if filter fails (safety-first)

        
        filtered_docs = []
        for result in results:
            base_score = result.similarity if result.similarity > 0 else result.score
            
            # Apply metadata-based boosting (Phase 4.1)
            boosted_score = self._apply_metadata_boost(base_score, result.metadata)
            
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
        top_k: int
    ) -> Dict:
        """Fallback: retrieve using standard semantic search"""
        
        # Generate query embedding
        query_embedding = self.embeddings.generate_embedding(query)
        
        # Build filter
        # Note: chroma's where operators are limited. Instead of using a non-supported
        # operator like $contains, we fetch results and apply product filtering client-side.
        where = None
        query_n = top_k
        if part_number:
            # Request more candidates so client-side filtering still returns enough items
            query_n = max(50, top_k * 5)

        # Query vector database
        results = self.vectordb.query(
            query_text=query,
            query_embedding=query_embedding,
            n_results=query_n,
            where=where
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
            # Skip if distance is too high (not similar enough)
            if dist > distance_threshold:
                similarity_score = max(0, 1 - dist/2)
                logger.debug(f"Skipping doc with distance {dist:.3f} (similarity {similarity_score:.3f}) < threshold {similarity_threshold:.2f}")
                continue

            # Note: We removed the part_number filter because most bulletins 
            # are general and don't have specific product references.
            # The LLM will determine relevance based on the context.

            similarity_score = max(0, 1 - dist/2)
            
            # Apply metadata-based boosting (Phase 4.1)
            boosted_score = self._apply_metadata_boost(similarity_score, meta)
            
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
            language: Language code ('en' or 'tr')
            username: User requesting the diagnosis
            excluded_sources: Sources to exclude (for retry)
            is_retry: Whether this is a retry request
            retry_of: Original diagnosis ID if retry
            
        Returns:
            Dict with suggestion and metadata
        """
        start_time = time.time()
        logger.info(f"Generating repair suggestion for {part_number} (retry={is_retry})")
        
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
        
        # Retrieve relevant context (always try RAG even if product not found)
        context_result = self.retrieve_context(
            query=enhanced_query,
            part_number=actual_part_number
        )
        
        retrieved_docs = context_result["documents"]
        
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
            
            # If insufficient context, return "I don't know" response
            if not sufficiency.is_sufficient:
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
            
            confidence = "high" if len(optimized_chunks) >= 3 else "medium" if optimized_chunks else "low"
        else:
            # No relevant context found - use fallback
            logger.warning("No relevant context found in manuals")
            context_str = build_fallback_response(product_model, language)
            prompt = f"{fault_description}\n\n{context_str}"
            confidence = "low"
            optimized_sources = None
        
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
        if optimized_sources:
            sources_list = [
                {
                    "source": src["source"],
                    "page": src.get("page_number"),  # NEW: Page number
                    "section": src.get("section", ""),  # NEW: Section title
                    "similarity": f"{src['similarity']:.2f}" if isinstance(src['similarity'], float) else src['similarity'],
                    "section_type": src.get("section_type", ""),
                    "is_warning": src.get("is_warning", False),
                    "is_procedure": src.get("is_procedure", False),
                    "excerpt": src["excerpt"]
                }
                for src in optimized_sources
            ]
        else:
            sources_list = [
                {
                    "source": doc["metadata"].get("source", "Unknown"),
                    "page": doc["metadata"].get("page_number"),  # NEW: Page number
                    "section": doc["metadata"].get("section", ""),  # NEW: Section title
                    "similarity": f"{doc['similarity']:.2f}" if isinstance(doc.get('similarity'), float) else str(doc.get('similarity', 0)),
                    "excerpt": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"]
                }
                for doc in retrieved_docs
            ] if retrieved_docs else []
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Structure the response
        response = {
            "suggestion": suggestion.strip(),
            "confidence": confidence,
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
