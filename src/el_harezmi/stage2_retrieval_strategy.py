"""
Stage 2: Retrieval Strategy

Intent-aware retrieval with Qdrant filtering and document type boosting.

Responsibilities:
- Map intents to optimal retrieval strategies
- Apply document type boosting factors
- Construct Qdrant filters based on entities
- Execute hybrid search (dense + sparse vectors)
- Apply RRF fusion for final ranking
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

from .stage1_intent_classifier import IntentType, IntentResult, ExtractedEntities

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Document types in Desoutter knowledge base"""
    TECHNICAL_MANUAL = "technical_manual"
    SERVICE_BULLETIN = "service_bulletin"
    CONFIGURATION_GUIDE = "configuration_guide"
    COMPATIBILITY_MATRIX = "compatibility_matrix"
    SPEC_SHEET = "spec_sheet"
    ERROR_CODE_LIST = "error_code_list"
    PROCEDURE_GUIDE = "procedure_guide"
    FRESHDESK_TICKET = "freshdesk_ticket"
    RELEASE_NOTES = "release_notes"
    QUICK_START_GUIDE = "quick_start_guide"


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with metadata"""
    chunk_id: str
    content: str
    score: float
    document_type: str
    product_model: Optional[str] = None
    product_family: Optional[str] = None
    section_hierarchy: Optional[str] = None
    chunk_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result of retrieval stage"""
    chunks: List[RetrievedChunk]
    strategy_used: str
    filters_applied: Dict[str, Any]
    total_candidates: int
    intent: IntentType
    query: str


@dataclass
class RetrievalStrategy:
    """
    Strategy configuration for a specific intent type
    """
    primary_sources: List[str]
    secondary_sources: List[str]
    boost_factors: Dict[str, float]
    filter_requirements: Dict[str, Any]
    result_limit: int = 20
    score_threshold: float = 0.5
    use_knowledge_graph: bool = False


class RetrievalStrategyManager:
    """
    Stage 2: Intent-Aware Retrieval Strategy Manager
    
    Maps intents to optimal retrieval strategies and executes searches.
    """
    
    # Intent to strategy mapping
    STRATEGIES: Dict[IntentType, RetrievalStrategy] = {
        IntentType.CONFIGURATION: RetrievalStrategy(
            primary_sources=["configuration_guide", "product_manual"],
            secondary_sources=["freshdesk_ticket"],
            boost_factors={
                "configuration_guide": 3.0,
                "product_manual": 2.5,
                "freshdesk_ticket": 2.0,
                "procedure_guide": 1.8,
                "quick_start_guide": 1.5,
                "troubleshooting_guide": 0.5,  # Penalty
            },
            filter_requirements={
                "must": ["product_model"],
                "should": ["contains_procedure", "chunk_type:semantic_section"],
            },
            result_limit=20,
            score_threshold=0.5,
        ),
        
        IntentType.COMPATIBILITY: RetrievalStrategy(
            primary_sources=["compatibility_matrix", "release_notes"],
            secondary_sources=["spec_sheet"],
            boost_factors={
                "compatibility_matrix": 5.0,  # CRITICAL
                "release_notes": 3.0,
                "spec_sheet": 2.0,
                "product_manual": 1.5,
                "generic_manual": 0.3,  # Strong penalty
            },
            filter_requirements={
                "must": ["contains_compatibility_info"],
                "should": ["chunk_type:table_row", "document_type:compatibility_matrix"],
            },
            result_limit=15,
            score_threshold=0.4,
            use_knowledge_graph=True,  # Always verify with hard-coded matrix
        ),
        
        IntentType.TROUBLESHOOT: RetrievalStrategy(
            primary_sources=["service_bulletin", "error_code_list"],
            secondary_sources=["freshdesk_ticket", "troubleshooting_guide"],
            boost_factors={
                "service_bulletin": 3.0,  # ESDE bulletins priority
                "error_code_list": 2.5,
                "freshdesk_ticket": 2.0,
                "troubleshooting_guide": 1.8,
                "product_manual": 1.0,
            },
            filter_requirements={
                "must": ["product_model"],
                "should": ["esde_code", "error_code"],
            },
            result_limit=20,
            score_threshold=0.5,
        ),
        
        IntentType.ERROR_CODE: RetrievalStrategy(
            primary_sources=["error_code_list", "service_bulletin"],
            secondary_sources=["troubleshooting_guide"],
            boost_factors={
                "error_code_list": 4.0,  # Highest priority
                "service_bulletin": 3.5,
                "troubleshooting_guide": 2.0,
                "freshdesk_ticket": 1.5,
            },
            filter_requirements={
                "must": ["error_code"],
                "should": ["chunk_type:error_code"],
            },
            result_limit=15,
            score_threshold=0.4,
        ),
        
        IntentType.PROCEDURE: RetrievalStrategy(
            primary_sources=["procedure_guide", "installation_manual"],
            secondary_sources=["product_manual"],
            boost_factors={
                "procedure_guide": 2.5,
                "installation_manual": 2.0,
                "product_manual": 1.5,
                "quick_start_guide": 1.5,
                "video_tutorial": 1.8,
            },
            filter_requirements={
                "must": ["contains_procedure"],
                "should": ["chunk_type:procedure", "chunk_type:semantic_section"],
            },
            result_limit=20,
            score_threshold=0.5,
        ),
        
        IntentType.SPECIFICATION: RetrievalStrategy(
            primary_sources=["spec_sheet", "product_manual"],
            secondary_sources=["compatibility_matrix"],
            boost_factors={
                "spec_sheet": 3.5,
                "product_manual": 2.0,
                "compatibility_matrix": 1.5,
            },
            filter_requirements={
                "must": ["product_model"],
                "should": ["chunk_type:table_row", "document_type:spec_sheet"],
            },
            result_limit=15,
            score_threshold=0.4,
        ),
        
        IntentType.CALIBRATION: RetrievalStrategy(
            primary_sources=["calibration_guide", "procedure_guide"],
            secondary_sources=["product_manual"],
            boost_factors={
                "calibration_guide": 3.0,
                "procedure_guide": 2.5,
                "product_manual": 1.5,
            },
            filter_requirements={
                "must": ["product_model"],
                "should": ["contains_procedure", "calibration"],
            },
            result_limit=15,
            score_threshold=0.5,
        ),
        
        IntentType.FIRMWARE: RetrievalStrategy(
            primary_sources=["firmware_guide", "release_notes"],
            secondary_sources=["procedure_guide"],
            boost_factors={
                "firmware_guide": 3.0,
                "release_notes": 2.5,
                "procedure_guide": 2.0,
                "product_manual": 1.0,
            },
            filter_requirements={
                "must": ["product_model"],
                "should": ["firmware", "version"],
            },
            result_limit=15,
            score_threshold=0.5,
        ),
        
        IntentType.INSTALLATION: RetrievalStrategy(
            primary_sources=["installation_manual", "quick_start_guide"],
            secondary_sources=["product_manual"],
            boost_factors={
                "installation_manual": 3.0,
                "quick_start_guide": 2.5,
                "product_manual": 1.5,
                "procedure_guide": 1.5,
            },
            filter_requirements={
                "must": ["product_model"],
                "should": ["contains_procedure", "installation", "setup"],
            },
            result_limit=20,
            score_threshold=0.5,
        ),
        
        IntentType.MAINTENANCE: RetrievalStrategy(
            primary_sources=["maintenance_guide", "product_manual"],
            secondary_sources=["procedure_guide"],
            boost_factors={
                "maintenance_guide": 3.0,
                "product_manual": 2.0,
                "procedure_guide": 1.5,
            },
            filter_requirements={
                "must": ["product_model"],
                "should": ["maintenance", "preventive"],
            },
            result_limit=15,
            score_threshold=0.5,
        ),
        
        IntentType.HOW_TO: RetrievalStrategy(
            primary_sources=["product_manual", "procedure_guide"],
            secondary_sources=["freshdesk_ticket"],
            boost_factors={
                "procedure_guide": 2.0,
                "product_manual": 2.0,
                "configuration_guide": 1.8,
                "freshdesk_ticket": 1.5,
            },
            filter_requirements={
                "must": [],
                "should": ["contains_procedure"],
            },
            result_limit=20,
            score_threshold=0.5,
        ),
        
        IntentType.COMPARISON: RetrievalStrategy(
            primary_sources=["spec_sheet", "product_manual"],
            secondary_sources=["compatibility_matrix"],
            boost_factors={
                "spec_sheet": 2.5,
                "product_manual": 2.0,
                "comparison_guide": 3.0,
            },
            filter_requirements={
                "must": [],
                "should": ["specification", "comparison"],
            },
            result_limit=20,
            score_threshold=0.4,
        ),
        
        IntentType.CAPABILITY_QUERY: RetrievalStrategy(
            primary_sources=["spec_sheet", "product_manual"],
            secondary_sources=["compatibility_matrix"],
            boost_factors={
                "spec_sheet": 3.0,
                "product_manual": 2.5,
                "compatibility_matrix": 2.0,
            },
            filter_requirements={
                "must": ["product_model"],
                "should": ["capability", "feature"],
            },
            result_limit=15,
            score_threshold=0.4,
        ),
        
        IntentType.ACCESSORY_QUERY: RetrievalStrategy(
            primary_sources=["accessory_list", "compatibility_matrix"],
            secondary_sources=["product_manual"],
            boost_factors={
                "accessory_list": 3.0,
                "compatibility_matrix": 2.5,
                "product_manual": 1.5,
            },
            filter_requirements={
                "must": ["product_model"],
                "should": ["accessory", "compatible"],
            },
            result_limit=15,
            score_threshold=0.4,
            use_knowledge_graph=True,
        ),
        
        IntentType.GENERAL: RetrievalStrategy(
            primary_sources=["product_manual"],
            secondary_sources=["procedure_guide", "spec_sheet"],
            boost_factors={
                "product_manual": 1.5,
                "procedure_guide": 1.5,
                "spec_sheet": 1.2,
            },
            filter_requirements={
                "must": [],
                "should": [],
            },
            result_limit=20,
            score_threshold=0.5,
        ),
    }
    
    # Default strategy for unmapped intents
    DEFAULT_STRATEGY = RetrievalStrategy(
        primary_sources=["product_manual"],
        secondary_sources=[],
        boost_factors={},
        filter_requirements={"must": [], "should": []},
        result_limit=20,
        score_threshold=0.5,
    )
    
    def __init__(self, qdrant_client=None, embedding_model=None):
        """
        Initialize Retrieval Strategy Manager.
        
        Args:
            qdrant_client: QdrantClient instance for vector search
            embedding_model: Embedding model for query encoding
        """
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
    
    def get_strategy(self, intent: IntentType) -> RetrievalStrategy:
        """Get retrieval strategy for an intent type"""
        return self.STRATEGIES.get(intent, self.DEFAULT_STRATEGY)
    
    def build_qdrant_filter(
        self,
        intent_result: IntentResult,
        strategy: RetrievalStrategy
    ) -> Dict[str, Any]:
        """
        Build Qdrant filter based on intent and entities.
        
        Returns a Qdrant filter dict for the search query.
        """
        must_conditions = []
        should_conditions = []
        
        entities = intent_result.entities
        
        # Product filtering is now done via boosting, not strict filtering
        # because many documents don't have product_model metadata
        if entities.product_model:
            should_conditions.append({
                "key": "product_model",
                "match": {"value": entities.product_model}
            })
        if entities.product_family:
            should_conditions.append({
                "key": "product_family",
                "match": {"value": entities.product_family}
            })
        
        # Add error code filter
        if entities.error_code:
            should_conditions.append({
                "key": "error_code",
                "match": {"value": entities.error_code}
            })
        
        # Add controller filter
        if entities.controller_type:
            should_conditions.append({
                "key": "controller_type",
                "match": {"value": entities.controller_type}
            })
        
        # Add intent relevance filter
        intent_values = [intent_result.primary_intent.value]
        intent_values.extend([s.value for s in intent_result.secondary_intents])
        
        should_conditions.append({
            "key": "intent_relevance",
            "match": {"any": intent_values}
        })
        
        # Add document type boosting via should conditions
        for doc_type, boost in strategy.boost_factors.items():
            if boost >= 2.0:  # Only boost high-priority docs
                should_conditions.append({
                    "key": "document_type",
                    "match": {"value": doc_type}
                })
        
        # Build final filter
        filter_dict = {}
        if must_conditions:
            filter_dict["must"] = must_conditions
        if should_conditions:
            filter_dict["should"] = should_conditions
        
        return filter_dict
    
    def calculate_boosted_score(
        self,
        base_score: float,
        chunk_metadata: Dict[str, Any],
        strategy: RetrievalStrategy
    ) -> float:
        """
        Calculate boosted score based on document type and metadata.
        """
        boosted_score = base_score
        
        # Apply document type boost
        doc_type = chunk_metadata.get("document_type", "")
        if doc_type in strategy.boost_factors:
            boosted_score *= strategy.boost_factors[doc_type]
        
        # Boost ESDE bulletins for troubleshooting
        if chunk_metadata.get("esde_code"):
            boosted_score *= 1.5
        
        # Boost chunks with procedures
        if chunk_metadata.get("contains_procedure"):
            boosted_score *= 1.2
        
        # Boost table rows for compatibility queries
        if chunk_metadata.get("chunk_type") == "table_row":
            boosted_score *= 1.3
        
        return boosted_score
    
    async def retrieve(
        self,
        query: str,
        intent_result: IntentResult,
        top_k: int = 20
    ) -> RetrievalResult:
        """
        Execute retrieval with intent-aware strategy.
        
        Args:
            query: User query string
            intent_result: Result from Stage 1 intent classification
            top_k: Maximum number of results to return
            
        Returns:
            RetrievalResult with ranked chunks
        """
        strategy = self.get_strategy(intent_result.primary_intent)
        
        # Build Qdrant filter
        qdrant_filter = self.build_qdrant_filter(intent_result, strategy)
        
        logger.info(f"Retrieval strategy: {intent_result.primary_intent.value}")
        logger.info(f"Qdrant filter: {qdrant_filter}")
        
        # Execute search (placeholder - actual implementation needs Qdrant client)
        chunks = await self._execute_search(
            query=query,
            filter_dict=qdrant_filter,
            strategy=strategy,
            top_k=top_k
        )
        
        return RetrievalResult(
            chunks=chunks,
            strategy_used=intent_result.primary_intent.value,
            filters_applied=qdrant_filter,
            total_candidates=len(chunks),
            intent=intent_result.primary_intent,
            query=query,
        )
    
    async def _execute_search(
        self,
        query: str,
        filter_dict: Dict[str, Any],
        strategy: RetrievalStrategy,
        top_k: int
    ) -> List[RetrievedChunk]:
        """
        Execute search against Qdrant.
        
        This is a placeholder - actual implementation will use Qdrant client.
        """
        if not self.qdrant_client:
            logger.warning("No Qdrant client configured - returning empty results")
            return []
        
        # Generate query embedding
        if self.embedding_model:
            query_vector = self.embedding_model.encode(query).tolist()
        else:
            logger.warning("No embedding model configured")
            return []
        
        # Execute Qdrant search
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Build proper Qdrant filter
            qdrant_filter = None
            if filter_dict and filter_dict.get("must"):
                conditions = []
                for condition in filter_dict["must"]:
                    if "key" in condition and "match" in condition:
                        conditions.append(
                            FieldCondition(
                                key=condition["key"],
                                match=MatchValue(value=condition["match"]["value"])
                            )
                        )
                if conditions:
                    qdrant_filter = Filter(must=conditions)
            
            # Use older API for Qdrant 1.7.x compatibility with named vector
            results = self.qdrant_client.search(
                collection_name="desoutter_docs_v2",
                query_vector=("dense", query_vector),  # Named vector
                query_filter=qdrant_filter,
                limit=top_k,
                score_threshold=strategy.score_threshold,
            )
            
            # Convert to RetrievedChunk objects
            chunks = []
            for result in results:
                payload = result.payload or {}
                
                # Calculate boosted score
                boosted_score = self.calculate_boosted_score(
                    result.score,
                    payload,
                    strategy
                )
                
                chunk = RetrievedChunk(
                    chunk_id=str(result.id),
                    content=payload.get("content", ""),
                    score=boosted_score,
                    document_type=payload.get("document_type", "unknown"),
                    product_model=payload.get("product_model"),
                    product_family=payload.get("product_family"),
                    section_hierarchy=payload.get("section_hierarchy"),
                    chunk_type=payload.get("chunk_type"),
                    metadata=payload,
                )
                chunks.append(chunk)
            
            # Re-sort by boosted score
            chunks.sort(key=lambda x: x.score, reverse=True)
            
            return chunks[:top_k]
            
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []
    
    def merge_secondary_results(
        self,
        primary_results: List[RetrievedChunk],
        secondary_results: List[RetrievedChunk],
        primary_weight: float = 0.7,
        secondary_weight: float = 0.3
    ) -> List[RetrievedChunk]:
        """
        Merge results from primary and secondary intents using RRF.
        
        Reciprocal Rank Fusion (RRF) formula:
        score = sum(1 / (k + rank)) for each result list
        """
        k = 60  # RRF constant
        
        # Build rank maps
        primary_ranks = {c.chunk_id: i + 1 for i, c in enumerate(primary_results)}
        secondary_ranks = {c.chunk_id: i + 1 for i, c in enumerate(secondary_results)}
        
        # Collect all unique chunks
        all_chunks = {c.chunk_id: c for c in primary_results}
        all_chunks.update({c.chunk_id: c for c in secondary_results})
        
        # Calculate RRF scores
        rrf_scores = {}
        for chunk_id in all_chunks:
            score = 0.0
            if chunk_id in primary_ranks:
                score += primary_weight * (1.0 / (k + primary_ranks[chunk_id]))
            if chunk_id in secondary_ranks:
                score += secondary_weight * (1.0 / (k + secondary_ranks[chunk_id]))
            rrf_scores[chunk_id] = score
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Return merged results
        return [all_chunks[chunk_id] for chunk_id in sorted_ids]
