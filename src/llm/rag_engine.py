"""
RAG Engine - Retrieval-Augmented Generation
Combines vector search with LLM generation
Now with self-learning capabilities from user feedback
"""
from typing import Dict, List, Optional
from datetime import datetime
import time

from src.documents.embeddings import EmbeddingsGenerator
from src.vectordb.chroma_client import ChromaDBClient
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import get_system_prompt, build_rag_prompt, build_fallback_response
from src.database import MongoDBClient
from config.ai_settings import RAG_TOP_K, RAG_SIMILARITY_THRESHOLD, DEFAULT_LANGUAGE
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
        
        logger.info("✅ RAG Engine initialized")
    
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
        
        Args:
            query: Search query (fault description)
            part_number: Optional product part number for filtering
            top_k: Number of results to retrieve
            
        Returns:
            Dict with retrieved documents and metadata
        """
        logger.info(f"Retrieving context for query: {query[:50]}...")
        
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
        
        # Build context string from retrieved documents
        if retrieved_docs:
            context_str = "\n\n".join([
                f"[Source: {doc['metadata'].get('source', 'Unknown')}]\n{doc['text']}"
                for doc in retrieved_docs
            ])
            
            # Build RAG prompt
            prompt = build_rag_prompt(
                product_model=product_model,
                part_number=actual_part_number,
                fault_description=fault_description,
                context=context_str,
                language=language
            )
            
            confidence = "high" if len(retrieved_docs) >= 3 else "medium"
        else:
            # No relevant context found - use fallback
            logger.warning("No relevant context found in manuals")
            context_str = build_fallback_response(product_model, language)
            prompt = f"{fault_description}\n\n{context_str}"
            confidence = "low"
        
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
        
        # Prepare response
        response = {
            "suggestion": suggestion.strip(),
            "confidence": confidence,
            "product_model": product_model,
            "part_number": actual_part_number,
            "sources": [
                {
                    "source": doc["metadata"].get("source", "Unknown"),
                    "similarity": f"{doc['similarity']:.2f}",
                    "excerpt": doc["text"][:200] + "..."
                }
                for doc in retrieved_docs
            ],
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
