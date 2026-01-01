"""
Context Grounding Module - SIMPLIFIED
Determines if retrieved context is sufficient to answer user query reliably.
Prevents hallucinations by refusing to answer when context is inadequate.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ContextSufficiencyResult:
    """Result of context sufficiency analysis"""
    score: float  # 0.0-1.0 overall sufficiency score
    is_sufficient: bool  # True if score >= threshold
    reason: str  # Human-readable explanation
    recommendation: str  # What to do: "answer", "answer_with_caution", "refuse"
    factors: Dict[str, float]  # Individual factor scores for debugging
    semantic_overlap: float = 0.0  # Kept for compatibility but not used


class ContextSufficiencyScorer:
    """
    SIMPLIFIED context sufficiency scorer.
    
    Uses basic scoring:
    1. Average similarity (50%) - Are docs semantically relevant?
    2. Top-1 score (30%) - Is best document strong match?
    3. Document count (20%) - Do we have enough context?
    
    NO semantic overlap checks, NO off-topic detection.
    """
    
    def __init__(
        self,
        sufficiency_threshold: float = 0.30,  # Simple threshold
        min_similarity: float = 0.25,
        min_docs: int = 1,
        min_semantic_overlap: float = 0.0  # IGNORED - kept for compatibility
    ):
        """
        Initialize scorer with simple thresholds.
        
        Args:
            sufficiency_threshold: Overall score threshold (0.30 = permissive)
            min_similarity: Minimum similarity for top document
            min_docs: Minimum documents for confident answer
            min_semantic_overlap: IGNORED - kept for API compatibility
        """
        self.sufficiency_threshold = sufficiency_threshold
        self.min_similarity = min_similarity
        self.min_docs = min_docs
        
        logger.info(f"ContextSufficiencyScorer initialized (SIMPLE): threshold={sufficiency_threshold}")
    
    def calculate_sufficiency_score(
        self,
        query: str,
        retrieved_docs: List[Dict],
        avg_similarity: Optional[float] = None
    ) -> ContextSufficiencyResult:
        """
        Calculate SIMPLE sufficiency score based on retrieval quality only.
        
        Args:
            query: User query text (not used in simple mode)
            retrieved_docs: List of retrieved document dicts with 'similarity' and 'text'
            avg_similarity: Optional pre-calculated average
        
        Returns:
            ContextSufficiencyResult with score and recommendation
        """
        # Handle edge cases
        if not retrieved_docs or len(retrieved_docs) == 0:
            return ContextSufficiencyResult(
                score=0.0,
                is_sufficient=False,
                reason="No relevant documents found in knowledge base",
                recommendation="refuse",
                factors={
                    "avg_similarity": 0.0,
                    "top_similarity": 0.0,
                    "doc_count": 0.0
                },
                semantic_overlap=0.0
            )
        
        factors = {}
        
        # Extract similarity scores
        similarities = []
        for doc in retrieved_docs:
            sim = doc.get("similarity", doc.get("boosted_score", 0.0))
            if isinstance(sim, (int, float)):
                similarities.append(float(sim))
            elif isinstance(sim, str):
                try:
                    similarities.append(float(sim))
                except ValueError:
                    similarities.append(0.0)
        
        if not similarities:
            similarities = [0.0]
        
        # Calculate average if not provided
        if avg_similarity is None:
            avg_similarity = sum(similarities) / len(similarities)
        
        # =================================================================
        # Factor 1: Average similarity (50% weight)
        # =================================================================
        factors['avg_similarity'] = min(avg_similarity / 0.5, 1.0)  # Normalize: 0.5 = perfect
        
        # =================================================================
        # Factor 2: Top-1 similarity (30% weight)
        # =================================================================
        top_similarity = max(similarities) if similarities else 0.0
        factors['top_similarity'] = min(top_similarity / 0.6, 1.0)  # Normalize: 0.6 = perfect
        
        # =================================================================
        # Factor 3: Document count (20% weight)
        # =================================================================
        doc_count = len(retrieved_docs)
        factors['doc_count'] = min(doc_count / 3, 1.0)  # Normalize: 3+ docs = perfect
        
        # =================================================================
        # Simple weighted score calculation
        # =================================================================
        weights = {
            'avg_similarity': 0.50,
            'top_similarity': 0.30,
            'doc_count': 0.20
        }
        
        overall_score = sum(factors[k] * weights[k] for k in weights.keys())
        
        # =================================================================
        # Simple sufficiency determination
        # =================================================================
        is_sufficient = overall_score >= self.sufficiency_threshold
        
        if overall_score >= 0.6:
            recommendation = "answer"
            reason = "Good context quality"
        elif overall_score >= self.sufficiency_threshold:
            recommendation = "answer_with_caution"
            reason = "Adequate context - cite sources"
        else:
            recommendation = "refuse"
            is_sufficient = False
            reason = f"Low context quality (score={overall_score:.2f})"
        
        logger.debug(
            f"Context sufficiency: score={overall_score:.3f}, "
            f"sufficient={is_sufficient}, factors={factors}"
        )
        
        return ContextSufficiencyResult(
            score=overall_score,
            is_sufficient=is_sufficient,
            reason=reason,
            recommendation=recommendation,
            factors=factors,
            semantic_overlap=0.0  # Not used
        )


def build_idk_response(
    query: str,
    product_model: str,
    reason: str,
    language: str = "en"
) -> str:
    """
    Build "I don't know" response when context is insufficient.
    
    Args:
        query: User's original query
        product_model: Product model name
        reason: Reason why we can't answer
        language: Response language ('en' or 'tr')
    
    Returns:
        Formatted "I don't know" response
    """
    if language.lower() == "tr":
        return f"""Bu soru hakkında teknik dokümantasyonda yeterli bilgi bulamadım.

**Ürün:** {product_model}
**Neden:** {reason}

**Öneriler:**
1. Sorunuzu farklı kelimelerle tekrar deneyin
2. Desoutter teknik destek ile iletişime geçin

Bu sınırlama, yanlış bilgi vermemek için konulmuştur."""
    else:
        return f"""I don't have enough information in the technical documentation to answer this question accurately.

**Product:** {product_model}
**Reason:** {reason}

**Suggestions:**
1. Try rephrasing your question with different keywords
2. Contact Desoutter technical support for assistance

This limitation ensures I don't provide incorrect information."""
