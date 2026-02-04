"""
Hybrid Search with Qdrant
=========================
Combines dense (semantic) and sparse (BM25) search with RRF fusion.

Features:
- Dense vector search (all-MiniLM-L6-v2)
- BM25 keyword search (rank-bm25)
- Reciprocal Rank Fusion (RRF)
- Intent-aware filtering
- Product family filtering
- Document type boosting

Usage:
    searcher = HybridSearcher()
    results = searcher.search(
        query="EABC-3000 pset ayarı nasıl yapılır?",
        product_filter="EABC",
        intent_filter=["CONFIGURATION", "PROCEDURE"],
        top_k=10
    )
"""
import os
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Qdrant
from src.vectordb.qdrant_client import (
    QdrantDBClient, SearchResult, QDRANT_AVAILABLE
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RRF_K = 60  # RRF fusion parameter (standard value)


@dataclass
class HybridSearchResult:
    """Hybrid search result with combined score"""
    id: str
    text: str
    metadata: Dict[str, Any]
    dense_score: float = 0.0
    sparse_score: float = 0.0
    combined_score: float = 0.0
    dense_rank: int = 0
    sparse_rank: int = 0


@dataclass
class IntentBoostConfig:
    """Configuration for intent-based document boosting"""
    intent: str
    document_type_boosts: Dict[str, float] = field(default_factory=dict)
    chunk_type_boosts: Dict[str, float] = field(default_factory=dict)


# Intent-specific retrieval strategies
INTENT_BOOST_CONFIGS = {
    "CONFIGURATION": IntentBoostConfig(
        intent="CONFIGURATION",
        document_type_boosts={
            "configuration_guide": 3.0,
            "technical_manual": 2.0,
            "compatibility_matrix": 1.5,
        },
        chunk_type_boosts={
            "semantic_section": 2.0,
            "procedure": 2.5,
        }
    ),
    "COMPATIBILITY": IntentBoostConfig(
        intent="COMPATIBILITY",
        document_type_boosts={
            "compatibility_matrix": 5.0,  # CRITICAL
            "spec_sheet": 2.0,
            "technical_manual": 1.5,
        },
        chunk_type_boosts={
            "table_row": 3.0,
            "version_block": 2.5,
        }
    ),
    "TROUBLESHOOTING": IntentBoostConfig(
        intent="TROUBLESHOOTING",
        document_type_boosts={
            "service_bulletin": 3.0,  # ESDE prioritization
            "error_code_list": 2.5,
            "freshdesk_ticket": 2.0,
        },
        chunk_type_boosts={
            "problem_solution_pair": 3.0,
            "error_code": 2.5,
        }
    ),
    "ERROR_CODE": IntentBoostConfig(
        intent="ERROR_CODE",
        document_type_boosts={
            "error_code_list": 3.0,
            "service_bulletin": 2.5,
            "technical_manual": 1.5,
        },
        chunk_type_boosts={
            "error_code": 3.0,
            "problem_solution_pair": 2.0,
        }
    ),
    "PROCEDURE": IntentBoostConfig(
        intent="PROCEDURE",
        document_type_boosts={
            "procedure_guide": 2.5,
            "technical_manual": 2.0,
            "configuration_guide": 1.5,
        },
        chunk_type_boosts={
            "procedure": 3.0,
            "semantic_section": 2.0,
        }
    ),
    "FIRMWARE": IntentBoostConfig(
        intent="FIRMWARE",
        document_type_boosts={
            "compatibility_matrix": 3.0,  # Version compatibility
            "procedure_guide": 2.5,
            "service_bulletin": 2.0,
        },
        chunk_type_boosts={
            "version_block": 3.0,
            "procedure": 2.5,
        }
    ),
}


class HybridSearcher:
    """
    Hybrid search combining semantic and keyword search.
    
    Uses:
    - Qdrant for dense vector search
    - BM25 for sparse keyword search
    - RRF for score fusion
    """
    
    def __init__(
        self,
        qdrant_client: Optional[QdrantDBClient] = None,
        embedding_model: str = EMBEDDING_MODEL
    ):
        """
        Initialize hybrid searcher.
        
        Args:
            qdrant_client: Optional pre-initialized Qdrant client
            embedding_model: Sentence transformer model name
        """
        # Qdrant client
        if qdrant_client:
            self.qdrant = qdrant_client
        elif QDRANT_AVAILABLE:
            self.qdrant = QdrantDBClient()
        else:
            raise ImportError("qdrant-client not available")
        
        # Embedding model
        self._embedder = None
        self._embedding_model_name = embedding_model
        
        # BM25 index (built lazily)
        self._bm25_index = None
        self._bm25_doc_ids = []
        self._bm25_texts = []
        
        logger.info("HybridSearcher initialized")
    
    @property
    def embedder(self):
        """Lazy load embedding model"""
        if self._embedder is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers not available")
            logger.info(f"Loading embedding model: {self._embedding_model_name}")
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for query"""
        return self.embedder.encode(query, convert_to_numpy=True).tolist()
    
    def build_bm25_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of dicts with 'id' and 'text'
        """
        if not BM25_AVAILABLE:
            logger.warning("rank-bm25 not available, BM25 search disabled")
            return
        
        logger.info(f"Building BM25 index with {len(documents)} documents...")
        
        self._bm25_doc_ids = []
        self._bm25_texts = []
        tokenized_corpus = []
        
        for doc in documents:
            doc_id = doc.get('id', '')
            text = doc.get('text', '')
            
            self._bm25_doc_ids.append(doc_id)
            self._bm25_texts.append(text)
            
            # Simple tokenization
            tokens = text.lower().split()
            tokenized_corpus.append(tokens)
        
        self._bm25_index = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built with {len(tokenized_corpus)} documents")
    
    def search_dense(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        product_filter: Optional[str] = None,
        document_type_filter: Optional[str] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Dense (semantic) search using Qdrant.
        
        Returns:
            List of (id, score, metadata) tuples
        """
        results = self.qdrant.search(
            query_vector=query_embedding,
            top_k=top_k,
            product_filter=product_filter,
            document_type_filter=document_type_filter
        )
        
        return [(r.id, r.score, r.metadata) for r in results]
    
    def search_sparse(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Sparse (BM25) keyword search.
        
        Returns:
            List of (id, score) tuples
        """
        if not self._bm25_index:
            logger.warning("BM25 index not built")
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get scores
        scores = self._bm25_index.get_scores(query_tokens)
        
        # Get top-k
        scored_docs = list(zip(self._bm25_doc_ids, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]
    
    def rrf_fusion(
        self,
        dense_results: List[Tuple[str, float, Dict]],
        sparse_results: List[Tuple[str, float]],
        k: int = RRF_K,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List[HybridSearchResult]:
        """
        Reciprocal Rank Fusion of dense and sparse results.
        
        RRF score = Σ (weight / (k + rank))
        
        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            k: RRF parameter (default 60)
            dense_weight: Weight for dense scores
            sparse_weight: Weight for sparse scores
            
        Returns:
            Fused results sorted by combined score
        """
        # Build ID -> metadata map from dense results
        id_to_metadata = {r[0]: r[2] for r in dense_results}
        id_to_text = {r[0]: r[2].get('text', '') for r in dense_results}
        
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        dense_scores = {}
        sparse_scores = {}
        dense_ranks = {}
        sparse_ranks = {}
        
        # Dense contribution
        for rank, (doc_id, score, _) in enumerate(dense_results, 1):
            rrf_scores[doc_id] += dense_weight / (k + rank)
            dense_scores[doc_id] = score
            dense_ranks[doc_id] = rank
        
        # Sparse contribution
        for rank, (doc_id, score) in enumerate(sparse_results, 1):
            rrf_scores[doc_id] += sparse_weight / (k + rank)
            sparse_scores[doc_id] = score
            sparse_ranks[doc_id] = rank
        
        # Create result objects
        results = []
        for doc_id, combined_score in rrf_scores.items():
            result = HybridSearchResult(
                id=doc_id,
                text=id_to_text.get(doc_id, ''),
                metadata=id_to_metadata.get(doc_id, {}),
                dense_score=dense_scores.get(doc_id, 0.0),
                sparse_score=sparse_scores.get(doc_id, 0.0),
                combined_score=combined_score,
                dense_rank=dense_ranks.get(doc_id, 0),
                sparse_rank=sparse_ranks.get(doc_id, 0)
            )
            results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return results
    
    def apply_intent_boost(
        self,
        results: List[HybridSearchResult],
        intent: str
    ) -> List[HybridSearchResult]:
        """
        Apply intent-specific boosting to results.
        
        Args:
            results: Search results
            intent: Detected query intent
            
        Returns:
            Re-ranked results
        """
        config = INTENT_BOOST_CONFIGS.get(intent.upper())
        if not config:
            return results
        
        for result in results:
            boost = 1.0
            
            # Document type boost
            doc_type = result.metadata.get('document_type', '')
            if doc_type in config.document_type_boosts:
                boost *= config.document_type_boosts[doc_type]
            
            # Chunk type boost
            chunk_type = result.metadata.get('chunk_type', '')
            if chunk_type in config.chunk_type_boosts:
                boost *= config.chunk_type_boosts[chunk_type]
            
            # Apply boost
            result.combined_score *= boost
        
        # Re-sort
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return results
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        product_filter: Optional[str] = None,
        intent_filter: Optional[List[str]] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        apply_intent_boost: bool = True
    ) -> List[HybridSearchResult]:
        """
        Hybrid search with optional filtering and boosting.
        
        Args:
            query: Search query
            top_k: Number of results to return
            product_filter: Filter by product family
            intent_filter: List of intents for filtering/boosting
            dense_weight: Weight for semantic search (0-1)
            sparse_weight: Weight for keyword search (0-1)
            apply_intent_boost: Whether to apply intent-based boosting
            
        Returns:
            List of HybridSearchResult objects
        """
        logger.debug(f"Hybrid search: '{query}' (product={product_filter}, intent={intent_filter})")
        
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Dense search (fetch more for fusion)
        fetch_k = min(top_k * 3, 50)
        dense_results = self.search_dense(
            query_embedding=query_embedding,
            top_k=fetch_k,
            product_filter=product_filter
        )
        
        # Sparse search (if BM25 index available)
        sparse_results = []
        if self._bm25_index and sparse_weight > 0:
            sparse_results = self.search_sparse(query, top_k=fetch_k)
        
        # RRF fusion
        if sparse_results:
            results = self.rrf_fusion(
                dense_results=dense_results,
                sparse_results=sparse_results,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight
            )
        else:
            # Dense only
            results = [
                HybridSearchResult(
                    id=r[0],
                    text=r[2].get('text', ''),
                    metadata=r[2],
                    dense_score=r[1],
                    combined_score=r[1]
                )
                for r in dense_results
            ]
        
        # Apply intent boost
        if apply_intent_boost and intent_filter:
            primary_intent = intent_filter[0] if intent_filter else None
            if primary_intent:
                results = self.apply_intent_boost(results, primary_intent)
        
        return results[:top_k]


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_hybrid_searcher: Optional[HybridSearcher] = None


def get_hybrid_searcher() -> HybridSearcher:
    """Get singleton HybridSearcher instance"""
    global _hybrid_searcher
    
    if _hybrid_searcher is None:
        _hybrid_searcher = HybridSearcher()
    
    return _hybrid_searcher


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("HYBRID SEARCH TEST")
    print("=" * 60)
    
    try:
        searcher = HybridSearcher()
        print("✅ HybridSearcher initialized")
        
        # Test embedding
        test_query = "EABC-3000 pset ayarı nasıl yapılır?"
        embedding = searcher.embed_query(test_query)
        print(f"✅ Query embedding: {len(embedding)} dimensions")
        
        # Test dense search
        results = searcher.search_dense(embedding, top_k=5)
        print(f"✅ Dense search returned {len(results)} results")
        
        # Full hybrid search (if collection has data)
        if results:
            hybrid_results = searcher.search(
                query=test_query,
                top_k=5,
                product_filter="EABC",
                intent_filter=["CONFIGURATION"]
            )
            print(f"✅ Hybrid search returned {len(hybrid_results)} results")
            
            for i, r in enumerate(hybrid_results[:3], 1):
                print(f"\n   {i}. Score: {r.combined_score:.4f}")
                print(f"      Text: {r.text[:100]}...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
