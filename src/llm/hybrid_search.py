"""
=============================================================================
Phase 2.2: Hybrid Search Module
=============================================================================
Combines semantic search (dense) with BM25 keyword search (sparse)
for improved retrieval coverage.

Features:
- Semantic search via Qdrant embeddings
- BM25 keyword search for exact term matching
- Reciprocal Rank Fusion (RRF) for score combination
- Metadata-based filtering
- Query expansion with domain synonyms

Usage:
    from src.llm.hybrid_search import HybridSearcher
    
    searcher = HybridSearcher()
    results = searcher.search("Motor grinding noise", top_k=5)
"""
import re
import math
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass

from src.vectordb.qdrant_client import QdrantDBClient
from src.documents.embeddings import EmbeddingsGenerator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class SearchResult:
    """Unified search result structure"""
    id: str
    content: str
    metadata: Dict
    score: float
    source: str  # 'semantic', 'bm25', or 'hybrid'
    similarity: float = 0.0
    bm25_score: float = 0.0


class BM25Index:
    """
    BM25 (Best Matching 25) keyword search implementation.
    
    BM25 is a probabilistic retrieval model that:
    - Ranks documents by term frequency (TF) with saturation
    - Applies inverse document frequency (IDF) weighting
    - Normalizes by document length
    
    Parameters:
        k1: Term frequency saturation parameter (1.2-2.0 typical)
        b: Document length normalization (0.75 typical)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, str] = {}  # id -> text
        self.doc_lengths: Dict[str, int] = {}  # id -> word count
        self.avg_doc_length: float = 0.0
        self.df: Dict[str, int] = defaultdict(int)  # term -> doc frequency
        self.tf: Dict[str, Dict[str, int]] = {}  # doc_id -> {term: count}
        self.idf: Dict[str, float] = {}
        self.N: int = 0  # Total documents
        
    def add_documents(self, documents: List[Dict]):
        """
        Index documents for BM25 search.
        
        Args:
            documents: List of {id, content, metadata}
        """
        logger.info(f"Indexing {len(documents)} documents for BM25...")
        
        for doc in documents:
            doc_id = doc.get('id', str(hash(doc['content'][:100])))
            content = doc.get('content', '')
            
            if not content:
                continue
                
            # Tokenize
            tokens = self._tokenize(content)
            
            # Store document
            self.documents[doc_id] = content
            self.doc_lengths[doc_id] = len(tokens)
            
            # Compute term frequencies
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            self.tf[doc_id] = dict(tf)
            
            # Update document frequencies
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] += 1
        
        self.N = len(self.documents)
        self.avg_doc_length = sum(self.doc_lengths.values()) / max(1, self.N)
        
        # Compute IDF for all terms
        self._compute_idf()
        
        logger.info(f"✅ BM25 index built: {self.N} documents, {len(self.df)} unique terms")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization with normalization"""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 2]
        return tokens
    
    def _compute_idf(self):
        """Compute IDF scores for all terms"""
        for term, df in self.df.items():
            # IDF with smoothing
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search documents using BM25 scoring.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of SearchResult sorted by BM25 score
        """
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        scores: Dict[str, float] = defaultdict(float)
        
        for doc_id in self.documents:
            score = 0.0
            doc_len = self.doc_lengths[doc_id]
            tf = self.tf.get(doc_id, {})
            
            for token in query_tokens:
                if token not in tf:
                    continue
                    
                # BM25 formula
                term_freq = tf[token]
                idf = self.idf.get(token, 0)
                
                # Term frequency component with saturation
                tf_component = (term_freq * (self.k1 + 1)) / (
                    term_freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                )
                
                score += idf * tf_component
            
            if score > 0:
                scores[doc_id] = score
        
        # Sort by score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in sorted_results:
            results.append(SearchResult(
                id=doc_id,
                content=self.documents[doc_id],
                metadata={},  # Will be populated by HybridSearcher
                score=score,
                source='bm25',
                bm25_score=score
            ))
        
        return results


class QueryExpander:
    """
    Domain-aware query expansion.
    
    Expands user queries with related terms to improve recall.
    Uses centralized domain vocabulary (Phase 1.2 refactor).
    """
    
    def __init__(self):
        # Import centralized vocabulary
        from src.llm.domain_vocabulary import get_domain_vocabulary
        self.vocab = get_domain_vocabulary()
        logger.info("QueryExpander initialized with centralized domain vocabulary")
    
    def expand(self, query: str, max_expansions: int = 5) -> List[str]:
        """
        Expand query with domain synonyms.
        
        Args:
            query: Original user query
            max_expansions: Maximum number of expanded queries
            
        Returns:
            List of query variations including original
        """
        expanded = [query]  # Always include original
        query_lower = query.lower()
        
        # 1. Normalize error codes
        normalized = self.vocab.normalize_error_code(query)
        if normalized != query:
            expanded.append(normalized)
        
        # 2. Expand with domain synonyms
        for term, synonyms in self.vocab.DOMAIN_SYNONYMS.items():
            if term in query_lower:
                for synonym in synonyms[:5]:  # Top 5 synonyms (increased from 2 for better symptom matching)
                    variation = re.sub(
                        rf'\b{re.escape(term)}\b',
                        synonym,
                        query,
                        flags=re.IGNORECASE
                    )
                    if variation != query and variation not in expanded:
                        expanded.append(variation)
        
        # 3. Add product-specific expansions
        product_expansions = self._expand_product_terms(query)
        for exp in product_expansions:
            if exp not in expanded:
                expanded.append(exp)
        
        # Limit and deduplicate
        return list(dict.fromkeys(expanded))[:max_expansions]
    
    def _expand_product_terms(self, query: str) -> List[str]:
        """Expand product-specific abbreviations using centralized vocabulary"""
        expansions = []
        query_lower = query.lower()
        
        # Use centralized PRODUCT_SERIES
        for abbrev, full_name in self.vocab.PRODUCT_SERIES.items():
            if abbrev.lower() in query_lower:
                # Create expansion with full name
                variation = re.sub(
                    rf'\b{re.escape(abbrev)}\b',
                    f"{abbrev} {full_name}",
                    query,
                    flags=re.IGNORECASE
                )
                if variation != query:
                    expansions.append(variation)
        
        return expansions


class MetadataFilter:
    """
    Metadata-based filtering for search results.
    
    Filters and boosts results based on chunk metadata:
    - Document type filtering
    - Importance score boosting
    - Warning/procedure prioritization
    - Fault keyword matching
    """
    
    def __init__(self):
        self.importance_boost = 1.3  # Boost for high-importance chunks
        self.warning_boost = 1.2    # Boost for warning content
        self.procedure_boost = 1.15  # Boost for procedural content
    
    def build_filter(
        self,
        doc_types: Optional[List[str]] = None,
        min_importance: float = 0.0,
        require_warning: bool = False,
        require_procedure: bool = False,
        fault_keywords: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """
        Build ChromaDB where filter from criteria.
        
        Args:
            doc_types: Allowed document types
            min_importance: Minimum importance score
            require_warning: Only warning content
            require_procedure: Only procedural content
            fault_keywords: Required fault keywords
            
        Returns:
            ChromaDB where filter dict or None
        """
        conditions = []
        
        if doc_types:
            conditions.append({
                "doc_type": {"$in": doc_types}
            })
        
        if min_importance > 0:
            conditions.append({
                "importance_score": {"$gte": min_importance}
            })
        
        if require_warning:
            conditions.append({
                "contains_warning": True
            })
        
        if require_procedure:
            conditions.append({
                "is_procedure": True
            })
        
        if not conditions:
            return None
        
        if len(conditions) == 1:
            return conditions[0]
        
        return {"$and": conditions}
    
    def boost_results(
        self,
        results: List[SearchResult],
        query: str
    ) -> List[SearchResult]:
        """
        Apply metadata-based score boosting.
        
        Args:
            results: Search results to boost
            query: Original query for context
            
        Returns:
            Results with boosted scores
        """
        query_lower = query.lower()
        
        for result in results:
            metadata = result.metadata
            boost = 1.0
            
            # Importance boost
            importance = metadata.get('importance_score', 0.5)
            if importance >= 0.8:
                boost *= self.importance_boost
            
            # Warning boost (if query seems error-related)
            if metadata.get('contains_warning', False):
                if any(term in query_lower for term in ['error', 'fault', 'problem', 'e0', 'i0', 'w0']):
                    boost *= self.warning_boost
            
            # Procedure boost (if query seems how-to)
            if metadata.get('is_procedure', False):
                if any(term in query_lower for term in ['how', 'step', 'procedure', 'fix', 'repair']):
                    boost *= self.procedure_boost
            
            # Fault keyword match boost
            fault_keywords = metadata.get('fault_keywords', '')
            if fault_keywords:
                keywords = fault_keywords.split(', ')
                matches = sum(1 for kw in keywords if kw.lower() in query_lower)
                if matches > 0:
                    boost *= (1 + 0.1 * matches)
            
            # Apply boost
            result.score *= boost
        
        # Re-sort by boosted score
        results.sort(key=lambda x: x.score, reverse=True)
        return results


class HybridSearcher:
    """
    Hybrid search combining semantic and keyword (BM25) retrieval.
    
    Uses Reciprocal Rank Fusion (RRF) to combine rankings from:
    - Dense retrieval: Qdrant semantic embeddings
    - Sparse retrieval: BM25 keyword matching
    
    Features:
    - Automatic BM25 index building from Qdrant
    - Query expansion with domain synonyms
    - Metadata-based filtering and boosting
    - Configurable fusion weights
    """
    
    def __init__(
        self,
        rrf_k: int = 60,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        """
        Initialize hybrid searcher.
        
        Args:
            rrf_k: RRF constant (higher = more uniform weighting)
            semantic_weight: Weight for semantic results in fusion
            bm25_weight: Weight for BM25 results in fusion
        """
        logger.info("Initializing HybridSearcher...")
        
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        
        # Initialize components (Qdrant initialized lazily)
        self.embeddings = EmbeddingsGenerator()
        self._qdrant = None
        self.bm25_index = BM25Index()
        self.query_expander = QueryExpander()
        self.metadata_filter = MetadataFilter()
        
        # Build BM25 index from Qdrant if possible
        try:
            self._build_bm25_index()
        except Exception as e:
            logger.warning(f"Could not build BM25 index during init: {e}")
        
        logger.info("✅ HybridSearcher initialized")
    
    @property
    def qdrant(self):
        """Lazy initialization of Qdrant client"""
        if self._qdrant is None:
            self._qdrant = QdrantDBClient()
        return self._qdrant
    
    def _build_bm25_index(self):
        """Build BM25 index from Qdrant documents"""
        try:
            info = self.qdrant.get_collection_info()
            if not info:
                logger.warning("Could not get Qdrant info, BM25 index will be empty")
                return
            
            count = info.get("points_count", 0)
            
            if count == 0:
                logger.warning("Qdrant is empty, BM25 index will be empty")
                return
            
            # Fetch documents from Qdrant in batches
            batch_size = 1000
            all_docs = []
            offset = None
            
            while True:
                results, next_offset = self.qdrant.client.scroll(
                    collection_name=self.qdrant.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True
                )
                for point in results:
                    payload = point.payload or {}
                    all_docs.append({
                        'id': str(point.id),
                        'content': payload.get('text', ''),
                        'metadata': payload
                    })
                if next_offset is None:
                    break
                offset = next_offset
            
            # Build BM25 index
            self.bm25_index.add_documents(all_docs)
            self._doc_metadata = {doc['id']: doc['metadata'] for doc in all_docs}
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self._doc_metadata = {}
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        expand_query: bool = True,
        use_hybrid: bool = True,
        where_filter: Optional[Dict] = None,
        min_similarity: float = 0.3
    ) -> List[SearchResult]:
        """
        Perform hybrid search with symptom synonym expansion and bulletin prioritization.
        
        Args:
            query: Search query
            top_k: Number of results to return
            expand_query: Whether to expand query with synonyms
            use_hybrid: Whether to use hybrid (True) or semantic-only (False)
            where_filter: ChromaDB metadata filter
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of SearchResult sorted by hybrid score
        """
        # Query expansion with domain synonyms
        queries = [query]
        if expand_query:
            queries = self.query_expander.expand(query, max_expansions=3)
            if len(queries) > 1:
                logger.info(f"Expanded query: {queries}")
        
        # Collect results from all query variations
        all_semantic_results = []
        all_bm25_results = []
        
        for q in queries:
            # Semantic search
            semantic_results = self._semantic_search(
                q, 
                top_k=top_k * 2,
                where_filter=where_filter,
                min_similarity=min_similarity
            )
            all_semantic_results.extend(semantic_results)
            
            # BM25 search (if hybrid enabled)
            if use_hybrid:
                bm25_results = self.bm25_index.search(q, top_k=top_k * 2)
                # Enrich with metadata
                for result in bm25_results:
                    result.metadata = self._doc_metadata.get(result.id, {})
                all_bm25_results.extend(bm25_results)
        
        # Combine results
        if use_hybrid and all_bm25_results:
            combined = self._reciprocal_rank_fusion(
                all_semantic_results,
                all_bm25_results,
                top_k=top_k * 2
            )
        else:
            combined = all_semantic_results
        
        # Apply metadata boosting
        combined = self.metadata_filter.boost_results(combined, query)
        
        # NEW: Deduplicate bulletins (keep only highest-scoring chunk per ESDE bulletin)
        combined = self._deduplicate_bulletins(combined)
        
        # Deduplicate and limit
        seen_ids = set()
        unique_results = []
        for result in combined:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)
                if len(unique_results) >= top_k:
                    break
        
        logger.info(f"Hybrid search '{query[:50]}...' -> {len(unique_results)} results")
        return unique_results
    
    def _deduplicate_bulletins(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate chunks from same bulletin document.
        Keep highest scoring chunk per ESDE bulletin.
        
        This prevents bulletins from appearing multiple times with different chunks.
        """
        seen_bulletins = {}  # source -> highest scoring result
        non_bulletin_results = []
        
        for result in results:
            source = result.metadata.get('source', '')
            
            # Check if this is an ESDE bulletin
            if source.upper().startswith('ESDE'):
                # Extract bulletin ID (e.g., "ESDE23007" from "ESDE23007 - Some title.pdf")
                bulletin_id = source.upper().split()[0] if source else ''
                
                if bulletin_id in seen_bulletins:
                    # Keep higher scoring one
                    if result.score > seen_bulletins[bulletin_id].score:
                        seen_bulletins[bulletin_id] = result
                else:
                    seen_bulletins[bulletin_id] = result
            else:
                non_bulletin_results.append(result)
        
        # Combine bulletins + non-bulletins, sort by score
        all_results = list(seen_bulletins.values()) + non_bulletin_results
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        if seen_bulletins:
            logger.debug(f"Deduplicated bulletins: kept {len(seen_bulletins)} unique bulletins")
        
        return all_results
    
    def _semantic_search(
        self,
        query: str,
        top_k: int,
        where_filter: Optional[Dict],
        min_similarity: float
    ) -> List[SearchResult]:
        """Perform semantic search via Qdrant"""
        try:
            query_embedding = self.embeddings.generate_embeddings([query])[0]
            results = self.qdrant.client.search(
                collection_name=self.qdrant.collection_name,
                query_vector=("dense", query_embedding),
                limit=top_k,
                query_filter=where_filter
            )
            search_results = []
            for point in results:
                payload = point.payload or {}
                similarity = point.score
                if similarity >= min_similarity:
                    search_results.append(SearchResult(
                        id=str(point.id),
                        content=payload.get('text', ''),
                        metadata=payload,
                        score=similarity,
                        source='semantic',
                        similarity=similarity
                    ))
            return search_results
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[SearchResult],
        bm25_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        Combine rankings using Reciprocal Rank Fusion.
        
        RRF Score = Σ 1/(k + rank)
        
        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            top_k: Number of results to return
            
        Returns:
            Combined and re-ranked results
        """
        scores: Dict[str, Tuple[float, SearchResult]] = {}
        
        # Score semantic results
        for rank, result in enumerate(semantic_results, 1):
            rrf_score = self.semantic_weight / (self.rrf_k + rank)
            if result.id in scores:
                existing_score, existing_result = scores[result.id]
                scores[result.id] = (
                    existing_score + rrf_score,
                    existing_result
                )
            else:
                scores[result.id] = (rrf_score, result)
        
        # Score BM25 results
        for rank, result in enumerate(bm25_results, 1):
            rrf_score = self.bm25_weight / (self.rrf_k + rank)
            if result.id in scores:
                existing_score, existing_result = scores[result.id]
                # Update scores
                existing_result.bm25_score = result.bm25_score
                existing_result.source = 'hybrid'
                scores[result.id] = (
                    existing_score + rrf_score,
                    existing_result
                )
            else:
                result.source = 'bm25'
                scores[result.id] = (rrf_score, result)
        
        # Sort by combined score
        sorted_items = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        
        results = []
        for doc_id, (score, result) in sorted_items[:top_k]:
            result.score = score
            results.append(result)
        
        return results
    
    def search_with_filter(
        self,
        query: str,
        top_k: int = 5,
        doc_types: Optional[List[str]] = None,
        min_importance: float = 0.0,
        require_warning: bool = False,
        require_procedure: bool = False
    ) -> List[SearchResult]:
        """
        Search with metadata filtering.
        
        Convenience method that builds filters automatically.
        """
        where_filter = self.metadata_filter.build_filter(
            doc_types=doc_types,
            min_importance=min_importance,
            require_warning=require_warning,
            require_procedure=require_procedure
        )
        
        return self.search(
            query=query,
            top_k=top_k,
            where_filter=where_filter
        )


# Convenience function for testing
def test_hybrid_search():
    """Test hybrid search functionality"""
    logger.info("=" * 60)
    logger.info("Testing Hybrid Search")
    logger.info("=" * 60)
    
    searcher = HybridSearcher()
    
    test_queries = [
        "Motor grinding noise E047",
        "CVI3 controller not starting",
        "Battery tool calibration procedure",
        "WiFi connection lost",
    ]
    
    for query in test_queries:
        logger.info(f"\n🔍 Query: {query}")
        
        # Hybrid search
        results = searcher.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. [{result.source}] {result.metadata.get('source', 'unknown')}")
            logger.info(f"     Score: {result.score:.4f} | Similarity: {result.similarity:.4f} | BM25: {result.bm25_score:.4f}")
            logger.info(f"     Content: {result.content[:100]}...")


if __name__ == "__main__":
    test_hybrid_search()
