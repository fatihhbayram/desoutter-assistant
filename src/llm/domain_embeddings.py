"""
Phase 3.1: Domain Embeddings
=============================
Domain-specific embedding enhancement for Desoutter repair domain.

Components:
1. DomainVocabulary - Desoutter-specific technical terminology
2. DomainEmbeddingAdapter - Enhances generic embeddings with domain knowledge
3. ContrastiveLearningTrainer - Trains on feedback data
4. DomainQueryEnhancer - Domain-aware query preprocessing

This module works with the existing sentence-transformers model and
enhances its performance for the repair domain without requiring
full model fine-tuning.

Created: 22 December 2025
"""
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
import math
import numpy as np
from collections import defaultdict
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# DESOUTTER DOMAIN VOCABULARY (Phase 1.2 Refactor)
# =============================================================================
# Import centralized vocabulary instead of defining locally
from src.llm.domain_vocabulary import DomainVocabulary, get_domain_vocabulary

logger.info("✅ Using centralized DomainVocabulary (Phase 1.2 refactor)")

# DOMAIN EMBEDDING ADAPTER
# =============================================================================

@dataclass
class DomainTermWeight:
    """Weight adjustment for domain terms"""
    term: str
    weight: float  # 1.0 = neutral, >1 = boost, <1 = reduce
    category: str
    confidence: float = 1.0


class DomainEmbeddingAdapter:
    """
    Adapts generic embeddings for domain-specific use.
    
    Strategies:
    1. Term importance weighting
    2. Domain term boosting in similarity
    3. Learned adjustments from feedback
    """
    
    def __init__(self, mongodb=None):
        self.vocabulary = DomainVocabulary()
        self.term_weights: Dict[str, DomainTermWeight] = {}
        self.mongodb = mongodb
        
        # Initialize default weights
        self._init_default_weights()
        
        # Load learned weights if available
        if mongodb:
            self._load_learned_weights()
        
        logger.info(f"DomainEmbeddingAdapter initialized with {len(self.term_weights)} term weights")
    
    def _init_default_weights(self):
        """Initialize default weights for domain terms"""
        # Product series get high boost (very specific)
        for code in DomainVocabulary.PRODUCT_SERIES.keys():
            self.term_weights[code.lower()] = DomainTermWeight(
                term=code.lower(),
                weight=2.0,
                category="product_series"
            )
        
        # Error codes get high boost
        for code in DomainVocabulary.ERROR_CODES.keys():
            self.term_weights[code.lower()] = DomainTermWeight(
                term=code.lower(),
                weight=2.0,
                category="error_code"
            )
        
        # Technical terms get medium boost
        for component in DomainVocabulary.COMPONENTS.keys():
            self.term_weights[component.lower()] = DomainTermWeight(
                term=component.lower(),
                weight=1.5,
                category="component"
            )
        
        # Symptoms get medium-high boost (diagnostic relevance)
        for symptom_key in DomainVocabulary.SYMPTOMS.keys():
            self.term_weights[symptom_key.lower()] = DomainTermWeight(
                term=symptom_key.lower(),
                weight=1.7,
                category="symptom"
            )
    
    def _load_learned_weights(self):
        """Load weights learned from feedback"""
        if not self.mongodb:
            return
        
        try:
            # Load from domain_term_weights collection
            cursor = self.mongodb.db.domain_term_weights.find({})
            for doc in cursor:
                term = doc.get("term", "").lower()
                if term:
                    self.term_weights[term] = DomainTermWeight(
                        term=term,
                        weight=doc.get("weight", 1.0),
                        category=doc.get("category", "learned"),
                        confidence=doc.get("confidence", 0.5)
                    )
            logger.debug(f"Loaded {len(self.term_weights)} term weights from DB")
        except Exception as e:
            logger.warning(f"Could not load learned weights: {e}")
    
    def get_term_weight(self, term: str) -> float:
        """Get weight for a term"""
        term_lower = term.lower()
        if term_lower in self.term_weights:
            return self.term_weights[term_lower].weight
        return 1.0
    
    def weight_query(self, query: str) -> Tuple[str, Dict[str, float]]:
        """
        Analyze query and return weighted terms.
        
        Returns:
            Tuple of (enhanced_query, term_weights_dict)
        """
        words = query.lower().split()
        weights = {}
        enhanced_parts = []
        
        for word in words:
            weight = self.get_term_weight(word)
            weights[word] = weight
            
            # Repeat high-weight terms to boost their importance
            if weight >= 2.0:
                enhanced_parts.extend([word] * 2)
            elif weight >= 1.5:
                enhanced_parts.append(word)
                enhanced_parts.append(word)
            else:
                enhanced_parts.append(word)
        
        enhanced_query = " ".join(enhanced_parts)
        return enhanced_query, weights
    
    def compute_domain_similarity(
        self,
        query_embedding: np.ndarray,
        doc_embedding: np.ndarray,
        query_terms: List[str],
        doc_terms: List[str]
    ) -> float:
        """
        Compute domain-aware similarity score.
        
        Combines:
        - Base cosine similarity
        - Domain term overlap bonus
        - Error code matching bonus
        """
        # Base cosine similarity
        base_sim = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        
        # Domain term overlap
        query_domain = set(t.lower() for t in query_terms if self.get_term_weight(t) > 1.0)
        doc_domain = set(t.lower() for t in doc_terms if self.get_term_weight(t) > 1.0)
        
        if query_domain and doc_domain:
            overlap = len(query_domain & doc_domain)
            overlap_bonus = overlap * 0.05  # 5% bonus per matching domain term
        else:
            overlap_bonus = 0
        
        # Error code exact match bonus
        error_codes = set(DomainVocabulary.ERROR_CODES.keys())
        query_errors = set(t.upper() for t in query_terms if t.upper() in error_codes)
        doc_errors = set(t.upper() for t in doc_terms if t.upper() in error_codes)
        
        if query_errors and doc_errors:
            error_match = len(query_errors & doc_errors)
            error_bonus = error_match * 0.1  # 10% bonus per matching error code
        else:
            error_bonus = 0
        
        # Product series match bonus
        series_codes = set(code.lower() for code in DomainVocabulary.PRODUCT_SERIES.keys())
        query_series = set(t.lower() for t in query_terms if t.lower() in series_codes)
        doc_series = set(t.lower() for t in doc_terms if t.lower() in series_codes)
        
        if query_series and doc_series:
            series_match = len(query_series & doc_series)
            series_bonus = series_match * 0.15  # 15% bonus per matching series
        else:
            series_bonus = 0
        
        # Combine scores (cap at 1.0)
        final_sim = min(1.0, base_sim + overlap_bonus + error_bonus + series_bonus)
        
        return float(final_sim)
    
    def update_weight_from_feedback(
        self,
        term: str,
        is_positive: bool,
        context: str = ""
    ):
        """Update term weight based on feedback"""
        term_lower = term.lower()
        
        if term_lower not in self.term_weights:
            self.term_weights[term_lower] = DomainTermWeight(
                term=term_lower,
                weight=1.0,
                category="learned"
            )
        
        current = self.term_weights[term_lower]
        
        # Adjust weight based on feedback
        adjustment = 0.1 if is_positive else -0.05
        new_weight = max(0.5, min(3.0, current.weight + adjustment))
        
        current.weight = new_weight
        current.confidence = min(1.0, current.confidence + 0.1)
        
        # Save to MongoDB if available
        if self.mongodb:
            try:
                self.mongodb.db.domain_term_weights.update_one(
                    {"term": term_lower},
                    {"$set": {
                        "term": term_lower,
                        "weight": new_weight,
                        "category": current.category,
                        "confidence": current.confidence,
                        "updated_at": datetime.now().isoformat()
                    }},
                    upsert=True
                )
            except Exception as e:
                logger.warning(f"Could not save term weight: {e}")


# =============================================================================
# DOMAIN QUERY ENHANCER
# =============================================================================

class DomainQueryEnhancer:
    """
    Enhances queries with domain knowledge.
    
    Features:
    - Synonym expansion
    - Error code expansion
    - Component relationship inference
    """
    
    def __init__(self):
        self.vocabulary = DomainVocabulary()
        self._synonym_cache: Dict[str, List[str]] = {}
    
    def enhance_query(self, query: str, expand_synonyms: bool = True) -> str:
        """
        Enhance query with domain knowledge.
        
        Args:
            query: Original query
            expand_synonyms: Whether to add synonyms
            
        Returns:
            Enhanced query string
        """
        words = query.split()
        enhanced_parts = list(words)  # Start with original words
        
        for word in words:
            word_lower = word.lower()
            
            # Add synonyms
            if expand_synonyms:
                synonyms = self.vocabulary.get_synonyms(word_lower)
                for syn in synonyms[:2]:  # Limit to 2 synonyms per term
                    if syn.lower() != word_lower and syn not in enhanced_parts:
                        enhanced_parts.append(syn)
            
            # Add related error codes for symptoms
            if word_lower in self.vocabulary.SYMPTOMS:
                related_errors = self.vocabulary.get_related_errors(word_lower)
                for error in related_errors[:2]:  # Limit to 2 errors
                    if error not in enhanced_parts:
                        enhanced_parts.append(error)
        
        return " ".join(enhanced_parts)
    
    def extract_domain_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract domain entities from text.
        
        Returns:
            Dict with entity types and found entities
        """
        text_lower = text.lower()
        entities = {
            "product_series": [],
            "error_codes": [],
            "components": [],
            "symptoms": [],
            "actions": [],
        }
        
        # Find product series
        for code in DomainVocabulary.PRODUCT_SERIES.keys():
            if code.lower() in text_lower:
                entities["product_series"].append(code)
        
        # Find error codes
        for code in DomainVocabulary.ERROR_CODES.keys():
            if code.lower() in text_lower:
                entities["error_codes"].append(code)
        
        # Find components
        for component, variants in DomainVocabulary.COMPONENTS.items():
            for variant in variants:
                if variant.lower() in text_lower:
                    if component not in entities["components"]:
                        entities["components"].append(component)
                    break
        
        # Find symptoms
        for symptom, variants in DomainVocabulary.SYMPTOMS.items():
            for variant in variants:
                if variant.lower() in text_lower:
                    if symptom not in entities["symptoms"]:
                        entities["symptoms"].append(symptom)
                    break
        
        # Find actions/procedures
        for action, variants in DomainVocabulary.PROCEDURES.items():
            for variant in variants:
                if variant.lower() in text_lower:
                    if action not in entities["actions"]:
                        entities["actions"].append(action)
                    break
        
        return entities
    
    def get_context_keywords(self, entities: Dict[str, List[str]]) -> List[str]:
        """
        Get additional context keywords based on extracted entities.
        
        This helps expand search to related content.
        """
        context = []
        
        # If we have product series, add related tool type
        for series in entities.get("product_series", []):
            series_upper = series.upper()
            if series_upper.startswith("EB") or series_upper.startswith("EP"):
                context.extend(["battery", "cordless", "wireless"])
            elif series_upper.startswith("EC"):
                context.extend(["cable", "corded", "power"])
            elif series_upper.startswith("EF"):
                context.extend(["fixtured", "spindle", "automation"])
            elif series_upper.startswith("CVI"):
                context.extend(["controller", "interface", "programming"])
            # EPBC specific - crowfoot tools
            if "EPBC" in series_upper or "EBC" in series_upper:
                context.extend(["crowfoot", "reindex", "reindexing", "position"])
        
        # If we have symptoms, add related keywords
        for symptom in entities.get("symptoms", []):
            if symptom in ["not_connected", "communication", "vision_error"]:
                context.extend(["cvi3", "controller", "socket tray", "communication", "firmware"])
        
        # If we have error codes, add related content
        for error in entities.get("error_codes", []):
            error_upper = error.upper()
            if error_upper in DomainVocabulary.ERROR_CODES:
                desc = DomainVocabulary.ERROR_CODES[error_upper]
                context.extend(desc.lower().split())
        
        # If we have components, add related symptoms
        component_symptom_map = {
            "motor": ["noise", "overheating", "power_loss"],
            "gearbox": ["noise", "vibration"],
            "battery": ["power_loss", "overheating"],
            "encoder": ["display_error", "intermittent"],
            "transducer": ["calibration"],
        }
        
        for component in entities.get("components", []):
            if component in component_symptom_map:
                context.extend(component_symptom_map[component])
        
        return list(set(context))


# =============================================================================
# CONTRASTIVE LEARNING DATA MANAGER
# =============================================================================

@dataclass
class ContrastivePair:
    """A training pair for contrastive learning"""
    anchor: str  # Query text
    positive: str  # Relevant document text
    negative: Optional[str] = None  # Irrelevant document text
    weight: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "anchor": self.anchor,
            "positive": self.positive,
            "negative": self.negative,
            "weight": self.weight,
            "created_at": self.created_at
        }


class ContrastiveLearningManager:
    """
    Manages contrastive learning data collection and training.
    
    Collects triplets from feedback:
    - Anchor: User query
    - Positive: Documents marked as relevant
    - Negative: Documents marked as irrelevant
    """
    
    def __init__(self, mongodb=None):
        self.mongodb = mongodb
        self.pairs: List[ContrastivePair] = []
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure MongoDB collection exists"""
        if not self.mongodb:
            return
        
        try:
            db = self.mongodb.db
            if "contrastive_pairs" not in db.list_collection_names():
                db.create_collection("contrastive_pairs")
                db.contrastive_pairs.create_index("created_at")
                logger.info("Created contrastive_pairs collection")
        except Exception as e:
            logger.warning(f"Could not ensure contrastive collection: {e}")
    
    def add_pair_from_feedback(
        self,
        query: str,
        relevant_docs: List[str],
        irrelevant_docs: List[str],
        weight: float = 1.0
    ) -> int:
        """
        Add training pairs from feedback.
        
        Creates pairs:
        - (query, relevant_doc) for each relevant doc
        - (query, relevant_doc, irrelevant_doc) for triplets
        
        Returns number of pairs added.
        """
        pairs_added = 0
        
        for relevant in relevant_docs:
            # Simple positive pair
            pair = ContrastivePair(
                anchor=query,
                positive=relevant,
                negative=irrelevant_docs[0] if irrelevant_docs else None,
                weight=weight
            )
            
            self.pairs.append(pair)
            pairs_added += 1
            
            # Save to MongoDB
            if self.mongodb:
                try:
                    self.mongodb.db.contrastive_pairs.insert_one(pair.to_dict())
                except Exception as e:
                    logger.warning(f"Could not save pair: {e}")
        
        logger.info(f"Added {pairs_added} contrastive pairs")
        return pairs_added
    
    def get_training_pairs(self, limit: int = 1000) -> List[ContrastivePair]:
        """Get pairs for training"""
        if self.mongodb:
            try:
                cursor = self.mongodb.db.contrastive_pairs.find({}).sort(
                    "created_at", -1
                ).limit(limit)
                
                return [
                    ContrastivePair(
                        anchor=doc["anchor"],
                        positive=doc["positive"],
                        negative=doc.get("negative"),
                        weight=doc.get("weight", 1.0),
                        created_at=doc.get("created_at", "")
                    )
                    for doc in cursor
                ]
            except Exception as e:
                logger.warning(f"Could not load pairs: {e}")
        
        return self.pairs[-limit:]
    
    def get_stats(self) -> Dict:
        """Get training data statistics"""
        if self.mongodb:
            try:
                total = self.mongodb.db.contrastive_pairs.count_documents({})
                with_negatives = self.mongodb.db.contrastive_pairs.count_documents({
                    "negative": {"$ne": None}
                })
                return {
                    "total_pairs": total,
                    "pairs_with_negatives": with_negatives,
                    "pairs_positive_only": total - with_negatives,
                    "ready_for_training": total >= 50
                }
            except Exception as e:
                logger.warning(f"Could not get stats: {e}")
        
        return {
            "total_pairs": len(self.pairs),
            "pairs_with_negatives": sum(1 for p in self.pairs if p.negative),
            "pairs_positive_only": sum(1 for p in self.pairs if not p.negative),
            "ready_for_training": len(self.pairs) >= 50
        }


# =============================================================================
# MAIN DOMAIN EMBEDDINGS ENGINE
# =============================================================================

class DomainEmbeddingsEngine:
    """
    Main orchestrator for domain-enhanced embeddings.
    
    Integrates:
    - Domain vocabulary
    - Embedding adaptation
    - Query enhancement
    - Contrastive learning
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Initialize MongoDB connection
        self.mongodb = None
        try:
            from src.database.mongo_client import MongoDBClient
            self.mongodb = MongoDBClient()
            self.mongodb.connect()
        except Exception as e:
            logger.warning(f"MongoDB not available for domain embeddings: {e}")
        
        # Initialize components
        self.vocabulary = DomainVocabulary()
        self.adapter = DomainEmbeddingAdapter(self.mongodb)
        self.enhancer = DomainQueryEnhancer()
        self.learning_manager = ContrastiveLearningManager(self.mongodb)
        
        self._initialized = True
        logger.info("✅ DomainEmbeddingsEngine initialized (Phase 3.1)")
    
    def enhance_query(self, query: str) -> Dict:
        """
        Enhance a query with domain knowledge.
        
        Returns:
            {
                "original": str,
                "enhanced": str,
                "entities": dict,
                "context_keywords": list,
                "term_weights": dict
            }
        """
        # Extract entities
        entities = self.enhancer.extract_domain_entities(query)
        
        # Get context keywords
        context_keywords = self.enhancer.get_context_keywords(entities)
        
        # Enhance query with synonyms
        enhanced_text = self.enhancer.enhance_query(query)
        
        # Add context keywords
        if context_keywords:
            enhanced_text = f"{enhanced_text} {' '.join(context_keywords[:5])}"
        
        # Get term weights
        _, term_weights = self.adapter.weight_query(query)
        
        return {
            "original": query,
            "enhanced": enhanced_text,
            "entities": entities,
            "context_keywords": context_keywords,
            "term_weights": term_weights
        }
    
    def compute_enhanced_similarity(
        self,
        query: str,
        query_embedding: List[float],
        doc_text: str,
        doc_embedding: List[float]
    ) -> float:
        """
        Compute domain-enhanced similarity.
        
        Args:
            query: Original query text
            query_embedding: Query embedding vector
            doc_text: Document text
            doc_embedding: Document embedding vector
            
        Returns:
            Enhanced similarity score (0-1)
        """
        # Extract terms
        query_terms = query.lower().split()
        doc_terms = doc_text.lower().split()
        
        # Compute domain-aware similarity
        similarity = self.adapter.compute_domain_similarity(
            query_embedding=np.array(query_embedding),
            doc_embedding=np.array(doc_embedding),
            query_terms=query_terms,
            doc_terms=doc_terms
        )
        
        return similarity
    
    def learn_from_feedback(
        self,
        query: str,
        relevant_docs: List[Dict],
        irrelevant_docs: List[Dict]
    ):
        """
        Learn from user feedback.
        
        Updates:
        - Term weights based on relevant/irrelevant documents
        - Contrastive learning pairs
        """
        # Extract terms from query
        query_terms = query.lower().split()
        
        # Update term weights
        for doc in relevant_docs:
            doc_text = doc.get("text", "")
            for term in query_terms:
                if term in doc_text.lower():
                    self.adapter.update_weight_from_feedback(term, is_positive=True)
        
        for doc in irrelevant_docs:
            doc_text = doc.get("text", "")
            for term in query_terms:
                if term in doc_text.lower():
                    self.adapter.update_weight_from_feedback(term, is_positive=False)
        
        # Add contrastive pairs
        relevant_texts = [doc.get("text", "")[:500] for doc in relevant_docs]
        irrelevant_texts = [doc.get("text", "")[:500] for doc in irrelevant_docs]
        
        if relevant_texts:
            self.learning_manager.add_pair_from_feedback(
                query=query,
                relevant_docs=relevant_texts,
                irrelevant_docs=irrelevant_texts
            )
    
    def get_stats(self) -> Dict:
        """Get domain embeddings statistics"""
        learning_stats = self.learning_manager.get_stats()
        
        return {
            "vocabulary_terms": len(self.vocabulary.get_all_terms()),
            "term_weights": len(self.adapter.term_weights),
            "product_series": len(DomainVocabulary.PRODUCT_SERIES),
            "error_codes": len(DomainVocabulary.ERROR_CODES),
            "components": len(DomainVocabulary.COMPONENTS),
            "symptoms": len(DomainVocabulary.SYMPTOMS),
            "contrastive_learning": learning_stats
        }
    
    def get_vocabulary_info(self) -> Dict:
        """Get vocabulary information"""
        return {
            "tool_types": list(DomainVocabulary.TOOL_TYPES.keys()),
            "product_series": list(DomainVocabulary.PRODUCT_SERIES.keys()),
            "error_codes": list(DomainVocabulary.ERROR_CODES.keys()),
            "components": list(DomainVocabulary.COMPONENTS.keys()),
            "symptoms": list(DomainVocabulary.SYMPTOMS.keys()),
            "procedures": list(DomainVocabulary.PROCEDURES.keys()),
        }


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

def get_domain_embeddings_engine() -> DomainEmbeddingsEngine:
    """Get singleton instance of DomainEmbeddingsEngine"""
    return DomainEmbeddingsEngine()
