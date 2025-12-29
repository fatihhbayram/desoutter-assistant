"""
Context Grounding Module
Determines if retrieved context is sufficient to answer user query reliably.
Prevents hallucinations by refusing to answer when context is inadequate.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
import re
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


class ContextSufficiencyScorer:
    """
    Determines if retrieved context is sufficient to answer query.
    
    Uses multi-factor scoring:
    1. Average similarity (40%) - Are docs semantically relevant?
    2. Top-1 score (30%) - Is best document strong match?
    3. Document count (15%) - Do we have enough context?
    4. Query term coverage (15%) - Does context mention query keywords?
    """
    
    def __init__(
        self,
        sufficiency_threshold: float = 0.5,
        min_similarity: float = 0.35,
        min_docs: int = 2
    ):
        """
        Initialize scorer
        
        Args:
            sufficiency_threshold: Overall score threshold (0.5 recommended)
            min_similarity: Minimum similarity for top document
            min_docs: Minimum documents for confident answer
        """
        self.sufficiency_threshold = sufficiency_threshold
        self.min_similarity = min_similarity
        self.min_docs = min_docs
        
        logger.info(f"ContextSufficiencyScorer initialized: threshold={sufficiency_threshold}")
    
    def calculate_sufficiency_score(
        self,
        query: str,
        retrieved_docs: List[Dict],
        avg_similarity: Optional[float] = None
    ) -> ContextSufficiencyResult:
        """
        Calculate multi-factor sufficiency score
        
        Args:
            query: User query text
            retrieved_docs: List of retrieved document dicts with 'similarity' and 'text'
            avg_similarity: Optional pre-calculated average (for efficiency)
        
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
                    "doc_count": 0.0,
                    "term_coverage": 0.0
                }
            )
        
        # Extract similarities
        similarities = []
        for doc in retrieved_docs:
            # Handle both 'similarity' and 'boosted_score' keys
            sim = doc.get('similarity', doc.get('boosted_score', 0.0))
            if isinstance(sim, str):
                try:
                    sim = float(sim)
                except ValueError:
                    sim = 0.0
            similarities.append(sim)
        
        # Calculate individual factors
        factors = {}
        
        # Factor 1: Average similarity score (40% weight)
        if avg_similarity is None:
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        factors['avg_similarity'] = min(avg_similarity / 0.7, 1.0)  # Normalize: 0.7 = perfect
        
        # Factor 2: Top-1 similarity (30% weight)
        top_similarity = max(similarities) if similarities else 0.0
        factors['top_similarity'] = min(top_similarity / 0.8, 1.0)  # Normalize: 0.8 = perfect
        
        # Factor 3: Document count (15% weight)
        doc_count = len(retrieved_docs)
        factors['doc_count'] = min(doc_count / 3, 1.0)  # Normalize: 3+ docs = perfect
        
        # Factor 4: Query term coverage (15% weight)
        term_coverage = self._calculate_term_coverage(query, retrieved_docs)
        factors['term_coverage'] = term_coverage
        
        # Weighted score calculation
        weights = {
            'avg_similarity': 0.40,
            'top_similarity': 0.30,
            'doc_count': 0.15,
            'term_coverage': 0.15
        }
        
        overall_score = sum(factors[k] * weights[k] for k in weights.keys())
        
        # Determine sufficiency and recommendation
        is_sufficient = overall_score >= self.sufficiency_threshold
        
        if overall_score >= 0.7:
            recommendation = "answer"
            reason = "Strong context match with high confidence"
        elif overall_score >= self.sufficiency_threshold:
            recommendation = "answer_with_caution"
            reason = "Adequate context but cite sources extensively"
        elif overall_score >= 0.3:
            recommendation = "answer_with_caution"
            reason = "Borderline context quality - answer carefully with low confidence"
        else:
            recommendation = "refuse"
            
            # Detailed reason for refusal
            reasons = []
            if top_similarity < self.min_similarity:
                reasons.append(f"Best match similarity ({top_similarity:.2f}) below threshold ({self.min_similarity})")
            if doc_count < self.min_docs:
                reasons.append(f"Only {doc_count} document(s) found (need {self.min_docs}+)")
            if term_coverage < 0.3:
                reasons.append(f"Low query term coverage ({term_coverage:.1%}) in context")
            
            reason = "; ".join(reasons) if reasons else "Insufficient context quality"
        
        logger.info(
            f"Context sufficiency: score={overall_score:.3f}, "
            f"recommendation={recommendation}, "
            f"factors={factors}"
        )
        
        return ContextSufficiencyResult(
            score=overall_score,
            is_sufficient=is_sufficient,
            reason=reason,
            recommendation=recommendation,
            factors=factors
        )
    
    def _calculate_term_coverage(self, query: str, retrieved_docs: List[Dict]) -> float:
        """
        Calculate what percentage of query terms appear in retrieved context
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
        
        Returns:
            Coverage score 0.0-1.0
        """
        # Extract meaningful query terms (filter stop words)
        query_terms = self._extract_meaningful_terms(query)
        
        if not query_terms:
            return 0.0
        
        # Combine all context text
        context_text = " ".join(doc.get('text', '') for doc in retrieved_docs).lower()
        
        # Count how many query terms appear in context
        matched_terms = sum(1 for term in query_terms if term in context_text)
        
        coverage = matched_terms / len(query_terms)
        
        logger.debug(f"Term coverage: {matched_terms}/{len(query_terms)} = {coverage:.1%}")
        
        return coverage
    
    def _extract_meaningful_terms(self, text: str) -> List[str]:
        """
        Extract meaningful terms from text (filter stop words)
        
        Args:
            text: Input text
        
        Returns:
            List of lowercase meaningful terms
        """
        # Common stop words to filter
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'must',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
            'what', 'which', 'who', 'when', 'where', 'why', 'how',
            'to', 'from', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'as', 'of',
            've', 'bir', 'bu', 'su', 'ile', 'için', 'gibi', 'daha', 'çok',
            'var', 'yok', 'olan', 'olarak', 've', 'veya', 'ama', 'fakat'
        }
        
        # Tokenize and filter
        text_lower = text.lower()
        
        # Extract words (alphanumeric + some special chars for product codes)
        words = re.findall(r'\b[a-zA-Z0-9çğıöşüÇĞİÖŞÜ]+\b', text_lower)
        
        # Filter stop words and too-short terms
        meaningful_terms = [
            word for word in words
            if word not in stop_words and len(word) >= 3
        ]
        
        return meaningful_terms


def build_idk_response(
    query: str,
    product_model: str,
    reason: str,
    language: str = "en"
) -> str:
    """
    Build "I don't know" response when context is insufficient
    
    Args:
        query: User's original query
        product_model: Product model name
        reason: Reason why we can't answer
        language: Response language
    
    Returns:
        Formatted "I don't know" response
    """
    if language.lower() == "tr":
        return f"""Mevcut dokümantasyonda bu soruyu güvenilir şekilde yanıtlamak için yeterli bilgiye sahip değilim.

**Sorgu:** {query}
**Ürün:** {product_model}

**Neden yanıtlayamıyorum:**
{reason}

**Ne yapabilirsiniz:**
1. Desoutter Teknik Destek ile iletişime geçin: support@desoutter.com
2. Ürün kılavuzunu doğrudan kontrol edin (elinizde varsa)
3. Yetkili servis merkezini ziyaret edin
4. Sorunuzu daha spesifik detaylarla yeniden ifade edin

**Not:** Sadece resmi dokümantasyona dayanarak doğru olduğundan emin olduğum yanıtlar veriyorum. Ekipmanınıza zarar verebilecek veya güvenlik sorunlarına yol açabilecek yanlış bilgi vermektense "bilmiyorum" demek daha iyidir.
"""
    else:
        return f"""I don't have sufficient information in the available documentation to answer this question reliably.

**Query:** {query}
**Product:** {product_model}

**Why I can't answer:**
{reason}

**What you can do:**
1. Contact Desoutter Technical Support: support@desoutter.com
2. Check the product manual directly (if you have it)
3. Visit an authorized service center
4. Rephrase your question with more specific details

**Note:** I only provide answers when I'm confident they're accurate based on official documentation. It's better to say "I don't know" than to provide incorrect information that could damage equipment or cause safety issues.
"""
