"""
RAG Engine - Retrieval-Augmented Generation
Combines vector search with LLM generation
Now with self-learning capabilities from user feedback
Phase 2.2: Hybrid Search (Semantic + BM25) integration
Phase 2.3: Response Caching for improved performance
Phase 3.4: Context Window Optimization
"""
from typing import Dict, List, Optional
from datetime import datetime
import time

from src.documents.embeddings import EmbeddingsGenerator
from src.vectordb.chroma_client import ChromaDBClient
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import get_system_prompt, build_rag_prompt, build_fallback_response
from src.llm.context_optimizer import ContextOptimizer, optimize_context_for_rag
from src.database import MongoDBClient
from config.ai_settings import (
    RAG_TOP_K, RAG_SIMILARITY_THRESHOLD, DEFAULT_LANGUAGE,
    USE_HYBRID_SEARCH, HYBRID_SEMANTIC_WEIGHT, HYBRID_BM25_WEIGHT,
    HYBRID_RRF_K, ENABLE_QUERY_EXPANSION,
    USE_CACHE, CACHE_TTL
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
        self.context_optimizer = ContextOptimizer(token_budget=8000)  # Phase 3.4
        
        # Initialize hybrid search if enabled
        if USE_HYBRID_SEARCH:
            self._init_hybrid_search()
        
        # Initialize response cache if enabled (Phase 2.3)
        if USE_CACHE:
            self._init_response_cache()
        
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
        
        # Use hybrid search if available
        if self.hybrid_searcher:
            return self._retrieve_with_hybrid_search(query, top_k)
        
        # Fallback to standard semantic search
        return self._retrieve_with_semantic_search(query, part_number, top_k)
    
    def _retrieve_with_hybrid_search(self, query: str, top_k: int) -> Dict:
        """Retrieve using hybrid search (semantic + BM25)"""
        results = self.hybrid_searcher.search(
            query=query,
            top_k=top_k,
            expand_query=ENABLE_QUERY_EXPANSION,
            use_hybrid=True,
            min_similarity=RAG_SIMILARITY_THRESHOLD
        )
        
        filtered_docs = []
        for result in results:
            filtered_docs.append({
                "text": result.content,
                "metadata": result.metadata,
                "similarity": result.similarity if result.similarity > 0 else result.score,
                "search_type": result.source,  # 'semantic', 'bm25', or 'hybrid'
                "bm25_score": result.bm25_score
            })
        
        logger.info(f"Hybrid search retrieved {len(filtered_docs)} documents")
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
            filtered_docs.append({
                "text": doc,
                "metadata": meta,
                "similarity": similarity_score
            })
        
        logger.info(f"Retrieved {len(filtered_docs)} relevant documents (similarity threshold: {similarity_threshold:.2f}, distance threshold: {distance_threshold:.2f})")
        
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
        
        # Retrieve relevant context (always try RAG even if product not found)
        context_result = self.retrieve_context(
            query=fault_description,
            part_number=None  # Don't filter by part_number in RAG
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
            
            # Build RAG prompt
            prompt = build_rag_prompt(
                product_model=product_model,
                part_number=actual_part_number,
                fault_description=fault_description,
                context=context_str,
                language=language
            )
            
            confidence = "high" if len(optimized_chunks) >= 3 else "medium" if optimized_chunks else "low"
        else:
            # No relevant context found - use fallback
            logger.warning("No relevant context found in manuals")
            context_str = build_fallback_response(product_model, language)
            prompt = f"{fault_description}\n\n{context_str}"
            confidence = "low"
            optimized_sources = None
        
        # Get system prompt
        system_prompt = get_system_prompt(language)
        
        # Generate suggestion from LLM
        logger.info("Generating LLM response...")
        suggestion = self.llm.generate(
            prompt=prompt,
            system=system_prompt
        )
        
        if not suggestion:
            suggestion = "Error: Unable to generate suggestion. Please try again."
            confidence = "low"
        
        # Prepare response with optimized sources if available
        if optimized_sources:
            sources_list = [
                {
                    "source": src["source"],
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
                    "similarity": f"{doc['similarity']:.2f}" if isinstance(doc['similarity'], float) else str(doc['similarity']),
                    "excerpt": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"]
                }
                for doc in retrieved_docs
            ] if retrieved_docs else []
        
        response = {
            "suggestion": suggestion.strip(),
            "confidence": confidence,
            "product_model": product_model,
            "part_number": actual_part_number,
            "sources": sources_list,
            "language": language
        }
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Save to diagnosis history
        try:
            diagnosis_id = feedback_engine.save_diagnosis(
                part_number=actual_part_number,
                product_model=product_model,
                fault_description=fault_description,
                suggestion=suggestion.strip(),
                confidence=confidence,
                sources=response["sources"],
                username=username,
                language=language,
                is_retry=is_retry,
                retry_of=retry_of,
                response_time_ms=response_time_ms
            )
            response["diagnosis_id"] = diagnosis_id
            response["response_time_ms"] = response_time_ms
        except Exception as e:
            logger.error(f"Error saving diagnosis history: {e}")
            response["diagnosis_id"] = None
        
        logger.info(f"✅ Generated suggestion with {confidence} confidence in {response_time_ms}ms")
        
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
                "cached_at": time.time()
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
        
        # Build prompt
        if retrieved_docs:
            context_str = "\n\n".join([f"[{doc['metadata'].get('source')}]\n{doc['text']}" for doc in retrieved_docs])
            prompt = build_rag_prompt(product_model, part_number, fault_description, context_str, language)
        else:
            context_str = build_fallback_response(product_model, language)
            prompt = f"{fault_description}\n\n{context_str}"
        
        system_prompt = get_system_prompt(language)
        
        # Stream from LLM
        for chunk in self.llm.stream_generate(prompt, system_prompt):
            yield {"chunk": chunk}
