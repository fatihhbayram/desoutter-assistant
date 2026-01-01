"""
Confidence Scoring Module
=========================
Calculates confidence scores for RAG responses based on multiple factors.
Provides both numeric scores and categorical levels (high/medium/low).

Version: 1.0
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceResult:
    """Result of confidence calculation"""
    level: str  # "high", "medium", "low"
    score: float  # 0.0-1.0 numeric score
    factors: Dict[str, float]  # Individual factor contributions
    explanation: str  # Human-readable explanation


class ConfidenceScorer:
    """
    Calculate confidence scores based on multiple factors:
    1. Source quality (number and similarity of retrieved docs)
    2. Intent match (how well we understand the query)
    3. Response quality (length, structure)
    4. Context sufficiency (from grounding check)
    
    Thresholds:
    - HIGH: >= 0.7
    - MEDIUM: >= 0.5
    - LOW: < 0.5
    """
    
    # Thresholds for confidence levels
    HIGH_THRESHOLD = 0.7
    MEDIUM_THRESHOLD = 0.5
    
    def __init__(self):
        logger.info("ConfidenceScorer initialized")
    
    def calculate_confidence(
        self,
        sources: List[Dict],
        intent: Optional[str] = None,
        intent_confidence: float = 0.0,
        response_text: str = "",
        sufficiency_score: Optional[float] = None,
        avg_similarity: float = 0.0
    ) -> ConfidenceResult:
        """
        Calculate overall confidence score.
        
        Args:
            sources: List of source documents used
            intent: Detected intent type
            intent_confidence: Confidence of intent detection (0-1)
            response_text: Generated response text
            sufficiency_score: Context sufficiency score (0-1) if available
            avg_similarity: Average similarity of retrieved docs
            
        Returns:
            ConfidenceResult with level, score, and breakdown
        """
        factors = {}
        
        # =================================================================
        # Factor 1: Source Quality (35% weight)
        # =================================================================
        source_score = self._calculate_source_score(sources, avg_similarity)
        factors['sources'] = source_score
        
        # =================================================================
        # Factor 2: Intent Understanding (25% weight)
        # =================================================================
        intent_score = self._calculate_intent_score(intent, intent_confidence)
        factors['intent'] = intent_score
        
        # =================================================================
        # Factor 3: Response Quality (25% weight)
        # =================================================================
        response_score = self._calculate_response_score(response_text)
        factors['response'] = response_score
        
        # =================================================================
        # Factor 4: Context Sufficiency (15% weight)
        # =================================================================
        if sufficiency_score is not None:
            context_score = min(sufficiency_score / 0.7, 1.0)  # Normalize: 0.7 = perfect
        else:
            # Estimate from sources if not provided
            context_score = source_score * 0.8
        factors['context'] = context_score
        
        # =================================================================
        # Weighted Total
        # =================================================================
        weights = {
            'sources': 0.35,
            'intent': 0.25,
            'response': 0.25,
            'context': 0.15
        }
        
        total_score = sum(factors[k] * weights[k] for k in weights.keys())
        
        # Determine level
        if total_score >= self.HIGH_THRESHOLD:
            level = "high"
        elif total_score >= self.MEDIUM_THRESHOLD:
            level = "medium"
        else:
            level = "low"
        
        # Build explanation
        explanation = self._build_explanation(factors, total_score, level)
        
        logger.info(
            f"[CONFIDENCE] {level} ({total_score:.3f}) - "
            f"sources={factors['sources']:.2f}, intent={factors['intent']:.2f}, "
            f"response={factors['response']:.2f}, context={factors['context']:.2f}"
        )
        
        return ConfidenceResult(
            level=level,
            score=round(total_score, 3),
            factors={k: round(v, 3) for k, v in factors.items()},
            explanation=explanation
        )
    
    def _calculate_source_score(
        self, 
        sources: List[Dict], 
        avg_similarity: float = 0.0
    ) -> float:
        """
        Score based on source quantity and quality.
        
        Scoring:
        - 3+ docs with good similarity = 1.0
        - 2 docs = 0.7
        - 1 doc = 0.4
        - 0 docs = 0.0
        
        Adjusted by average similarity.
        """
        doc_count = len(sources) if sources else 0
        
        # Base score from document count
        if doc_count >= 3:
            base = 1.0
        elif doc_count == 2:
            base = 0.7
        elif doc_count == 1:
            base = 0.4
        else:
            return 0.0
        
        # Adjust by similarity quality
        if avg_similarity > 0:
            # Normalize: 0.5 similarity = 1.0 factor, 0.3 = 0.6 factor
            similarity_factor = min(avg_similarity / 0.5, 1.0)
        else:
            # Estimate from sources if avg not provided
            if sources:
                sims = []
                for src in sources:
                    sim = src.get('similarity', 0)
                    if isinstance(sim, str):
                        try:
                            sim = float(sim)
                        except:
                            sim = 0.3
                    sims.append(sim)
                avg_similarity = sum(sims) / len(sims) if sims else 0.3
                similarity_factor = min(avg_similarity / 0.5, 1.0)
            else:
                similarity_factor = 0.5
        
        return base * similarity_factor
    
    def _calculate_intent_score(
        self, 
        intent: Optional[str], 
        intent_confidence: float
    ) -> float:
        """
        Score based on intent detection quality.
        
        High-confidence intents that match well-defined categories score higher.
        """
        if not intent:
            return 0.3  # Default for unknown intent
        
        # Intent type weights (some intents are easier to answer accurately)
        intent_weights = {
            'error_code': 0.9,      # Very specific, easy to verify
            'specifications': 0.85,  # Factual, from docs
            'calibration': 0.8,      # Procedural, documented
            'installation': 0.75,    # Procedural
            'maintenance': 0.75,     # Procedural
            'connection': 0.7,       # Can be complex
            'troubleshooting': 0.65, # Can require inference
            'general': 0.5           # Vague queries
        }
        
        intent_weight = intent_weights.get(intent, 0.5)
        
        # Combine with detection confidence
        # High confidence + good intent = high score
        combined = (intent_weight * 0.6) + (intent_confidence * 0.4)
        
        return min(combined, 1.0)
    
    def _calculate_response_score(self, response_text: str) -> float:
        """
        Score based on response quality indicators.
        
        Factors:
        - Length (not too short, not too long)
        - Structure (has steps, bullet points)
        - Specificity (mentions product names, numbers)
        """
        if not response_text:
            return 0.0
        
        text = response_text.strip()
        length = len(text)
        
        # Length score (optimal: 200-800 chars)
        if length >= 200:
            length_score = min(1.0, length / 400)  # Max at 400+ chars
        elif length >= 100:
            length_score = 0.6
        elif length >= 50:
            length_score = 0.4
        else:
            length_score = 0.2
        
        # Structure score (has numbered steps, bullets)
        structure_score = 0.0
        if any(marker in text for marker in ['1.', '2.', '•', '-', 'Step']):
            structure_score = 0.3
        if any(marker in text for marker in ['Warning', 'Caution', 'Note', '⚠', 'Uyarı', 'Dikkat']):
            structure_score += 0.2
        
        # Specificity score (mentions specific values)
        specificity_score = 0.0
        # Check for numbers with units
        if re.search(r'\d+\s*(Nm|rpm|mm|kg|V|A|W|bar|psi)', text, re.IGNORECASE):
            specificity_score = 0.3
        # Check for error codes
        if re.search(r'E\d{2,4}|error\s+\d+|fault\s+\d+', text, re.IGNORECASE):
            specificity_score = max(specificity_score, 0.4)
        
        # Combine factors
        total = (length_score * 0.5) + (structure_score * 0.25) + (specificity_score * 0.25)
        
        return min(total, 1.0)
    
    def _build_explanation(
        self, 
        factors: Dict[str, float], 
        total: float, 
        level: str
    ) -> str:
        """Build human-readable explanation of confidence."""
        parts = []
        
        if factors['sources'] >= 0.7:
            parts.append("good source coverage")
        elif factors['sources'] >= 0.4:
            parts.append("limited sources")
        else:
            parts.append("few sources found")
        
        if factors['intent'] >= 0.7:
            parts.append("clear intent")
        elif factors['intent'] < 0.5:
            parts.append("unclear query intent")
        
        if factors['response'] >= 0.7:
            parts.append("detailed response")
        elif factors['response'] < 0.4:
            parts.append("brief response")
        
        explanation = f"{level.upper()} confidence ({total:.0%}): {', '.join(parts)}"
        return explanation


# =============================================================================
# Singleton instance
# =============================================================================
_confidence_scorer = None

def get_confidence_scorer() -> ConfidenceScorer:
    """Get singleton confidence scorer instance"""
    global _confidence_scorer
    if _confidence_scorer is None:
        _confidence_scorer = ConfidenceScorer()
    return _confidence_scorer


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    scorer = ConfidenceScorer()
    
    # Test case 1: Good response
    result1 = scorer.calculate_confidence(
        sources=[{"similarity": 0.6}, {"similarity": 0.5}, {"similarity": 0.4}],
        intent="error_code",
        intent_confidence=0.9,
        response_text="The E804 error code indicates a communication failure. Steps: 1. Check cable connections 2. Verify power supply 3. Reset the controller. Warning: Ensure power is off before inspection.",
        sufficiency_score=0.7,
        avg_similarity=0.5
    )
    print(f"Test 1 (Good): {result1.level} ({result1.score}) - {result1.explanation}")
    
    # Test case 2: Mediocre response
    result2 = scorer.calculate_confidence(
        sources=[{"similarity": 0.4}, {"similarity": 0.3}],
        intent="troubleshooting",
        intent_confidence=0.6,
        response_text="Check the motor connections and ensure power supply is working.",
        sufficiency_score=0.4,
        avg_similarity=0.35
    )
    print(f"Test 2 (Medium): {result2.level} ({result2.score}) - {result2.explanation}")
    
    # Test case 3: Poor response
    result3 = scorer.calculate_confidence(
        sources=[{"similarity": 0.2}],
        intent="general",
        intent_confidence=0.5,
        response_text="I'm not sure.",
        sufficiency_score=0.2,
        avg_similarity=0.2
    )
    print(f"Test 3 (Low): {result3.level} ({result3.score}) - {result3.explanation}")
