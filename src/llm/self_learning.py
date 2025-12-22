"""
Phase 6: Self-Learning Feedback Loop
=====================================
Implements continuous learning from user feedback to improve RAG system.

Components:
1. FeedbackSignalProcessor - Processes and propagates feedback signals
2. SourceRankingLearner - Learns source rankings from feedback
3. EmbeddingRetrainer - Schedules periodic embedding re-training
4. SelfLearningEngine - Main orchestrator

Created: 22 December 2025
"""
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import json
import math
import threading
import time

from src.database.mongo_client import MongoDBClient
from src.database.feedback_models import (
    DiagnosisHistory,
    DiagnosisFeedback,
    FeedbackType,
    NegativeFeedbackReason,
    SourceRelevance
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SourceScore:
    """Represents a learned score for a document source"""
    source: str
    base_score: float = 1.0
    positive_signals: int = 0
    negative_signals: int = 0
    relevance_score: float = 0.5
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    keywords: List[str] = field(default_factory=list)
    
    @property
    def total_signals(self) -> int:
        return self.positive_signals + self.negative_signals
    
    @property
    def confidence(self) -> float:
        """Calculate confidence based on signal count"""
        if self.total_signals == 0:
            return 0.0
        return min(1.0, math.log(self.total_signals + 1) / math.log(20))
    
    def calculate_score(self) -> float:
        """
        Calculate final score using Wilson score interval
        More statistically robust than simple ratio
        """
        if self.total_signals == 0:
            return self.base_score
        
        n = self.total_signals
        p = self.positive_signals / n
        z = 1.96  # 95% confidence
        
        # Wilson score lower bound
        wilson = (p + z*z/(2*n) - z * math.sqrt((p*(1-p) + z*z/(4*n))/n)) / (1 + z*z/n)
        
        # Blend with base score using confidence
        return self.base_score * (1 - self.confidence) + wilson * self.confidence * 2
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "base_score": self.base_score,
            "positive_signals": self.positive_signals,
            "negative_signals": self.negative_signals,
            "relevance_score": self.relevance_score,
            "calculated_score": self.calculate_score(),
            "confidence": self.confidence,
            "last_updated": self.last_updated,
            "keywords": self.keywords
        }


@dataclass
class KeywordMapping:
    """Maps keywords to preferred sources and solutions"""
    keywords: Tuple[str, ...]  # Frozenset of keywords as tuple for hashing
    preferred_sources: List[str] = field(default_factory=list)
    avoided_sources: List[str] = field(default_factory=list)
    solution_patterns: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5
    
    def to_dict(self) -> Dict:
        return {
            "keywords": list(self.keywords),
            "keyword_hash": hashlib.md5("_".join(sorted(self.keywords)).encode()).hexdigest()[:12],
            "preferred_sources": self.preferred_sources,
            "avoided_sources": self.avoided_sources,
            "solution_patterns": self.solution_patterns,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "last_updated": self.last_updated
        }


# =============================================================================
# FEEDBACK SIGNAL PROCESSOR
# =============================================================================

class FeedbackSignalProcessor:
    """
    Processes user feedback and converts to learning signals.
    
    Signal Types:
    - Explicit: User clicked positive/negative
    - Implicit: Retry indicates dissatisfaction
    - Source-level: Per-source relevance feedback
    """
    
    def __init__(self, mongodb: MongoDBClient):
        self.mongodb = mongodb
        self._ensure_collections()
    
    def _ensure_collections(self):
        """Ensure learning collections exist"""
        try:
            db = self.mongodb.db
            collections = db.list_collection_names()
            
            if "source_learning_scores" not in collections:
                db.create_collection("source_learning_scores")
                db.source_learning_scores.create_index("source", unique=True)
                db.source_learning_scores.create_index("calculated_score")
                logger.info("Created source_learning_scores collection")
            
            if "keyword_mappings" not in collections:
                db.create_collection("keyword_mappings")
                db.keyword_mappings.create_index("keyword_hash", unique=True)
                db.keyword_mappings.create_index("keywords")
                db.keyword_mappings.create_index("success_rate")
                logger.info("Created keyword_mappings collection")
            
            if "learning_events" not in collections:
                db.create_collection("learning_events")
                db.learning_events.create_index("event_type")
                db.learning_events.create_index("created_at")
                # TTL index: auto-delete after 90 days
                db.learning_events.create_index(
                    "created_at",
                    expireAfterSeconds=90*24*60*60,
                    name="learning_events_ttl"
                )
                logger.info("Created learning_events collection with TTL")
                
        except Exception as e:
            logger.error(f"Error ensuring learning collections: {e}")
    
    def process_feedback_signal(
        self,
        feedback_type: str,  # "positive" or "negative"
        sources: List[str],
        keywords: List[str],
        source_relevance: Optional[List[Dict]] = None,
        solution: Optional[str] = None,
        is_retry: bool = False
    ) -> Dict:
        """
        Process a feedback signal and update learning models.
        
        Args:
            feedback_type: "positive" or "negative"
            sources: List of document sources used
            keywords: Extracted fault keywords
            source_relevance: Per-source relevance [{source, relevant}, ...]
            solution: The solution that was given
            is_retry: Whether this was a retry (implicit negative)
            
        Returns:
            Processing result with updated scores
        """
        result = {
            "sources_updated": 0,
            "mappings_updated": 0,
            "events_logged": 0
        }
        
        try:
            # 1. Process source-level signals
            if source_relevance:
                # Use explicit per-source feedback
                for sr in source_relevance:
                    source = sr.get("source", "")
                    relevant = sr.get("relevant", True)
                    if source:
                        self._update_source_score(
                            source=source,
                            is_positive=relevant,
                            keywords=keywords
                        )
                        result["sources_updated"] += 1
            else:
                # Use overall feedback for all sources
                is_positive = feedback_type == "positive"
                for source in sources:
                    if source:
                        self._update_source_score(
                            source=source,
                            is_positive=is_positive,
                            keywords=keywords
                        )
                        result["sources_updated"] += 1
            
            # 2. Update keyword mappings
            if keywords:
                self._update_keyword_mapping(
                    keywords=keywords,
                    sources=sources,
                    is_positive=feedback_type == "positive",
                    solution=solution
                )
                result["mappings_updated"] += 1
            
            # 3. Log learning event
            self._log_learning_event(
                event_type="feedback_processed",
                data={
                    "feedback_type": feedback_type,
                    "sources_count": len(sources),
                    "keywords": keywords[:5],
                    "is_retry": is_retry,
                    "has_source_relevance": source_relevance is not None
                }
            )
            result["events_logged"] += 1
            
            logger.info(f"Processed feedback signal: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing feedback signal: {e}")
            return result
    
    def _update_source_score(
        self,
        source: str,
        is_positive: bool,
        keywords: List[str]
    ):
        """Update source learning score"""
        db = self.mongodb.db
        
        # Get or create source score
        existing = db.source_learning_scores.find_one({"source": source})
        
        if existing:
            # Update existing
            update = {
                "$inc": {
                    "positive_signals" if is_positive else "negative_signals": 1
                },
                "$set": {
                    "last_updated": datetime.now().isoformat()
                },
                "$addToSet": {
                    "keywords": {"$each": keywords[:5]}
                }
            }
            db.source_learning_scores.update_one({"source": source}, update)
            
            # Recalculate score
            updated = db.source_learning_scores.find_one({"source": source})
            score = SourceScore(
                source=source,
                positive_signals=updated.get("positive_signals", 0),
                negative_signals=updated.get("negative_signals", 0)
            )
            db.source_learning_scores.update_one(
                {"source": source},
                {"$set": {
                    "calculated_score": score.calculate_score(),
                    "confidence": score.confidence
                }}
            )
        else:
            # Create new
            score = SourceScore(
                source=source,
                positive_signals=1 if is_positive else 0,
                negative_signals=0 if is_positive else 1,
                keywords=keywords[:5]
            )
            db.source_learning_scores.insert_one(score.to_dict())
    
    def _update_keyword_mapping(
        self,
        keywords: List[str],
        sources: List[str],
        is_positive: bool,
        solution: Optional[str] = None
    ):
        """Update keyword-to-source mapping"""
        if not keywords:
            return
        
        db = self.mongodb.db
        
        # Create stable hash from top keywords
        top_keywords = tuple(sorted(keywords[:5]))
        keyword_hash = hashlib.md5("_".join(top_keywords).encode()).hexdigest()[:12]
        
        existing = db.keyword_mappings.find_one({"keyword_hash": keyword_hash})
        
        if existing:
            # Update existing mapping
            if is_positive:
                update = {
                    "$inc": {"success_count": 1},
                    "$addToSet": {"preferred_sources": {"$each": sources[:3]}},
                    "$set": {"last_updated": datetime.now().isoformat()}
                }
                if solution:
                    update["$addToSet"]["solution_patterns"] = solution[:200]
            else:
                update = {
                    "$inc": {"failure_count": 1},
                    "$addToSet": {"avoided_sources": {"$each": sources[:3]}},
                    "$set": {"last_updated": datetime.now().isoformat()}
                }
            
            db.keyword_mappings.update_one({"keyword_hash": keyword_hash}, update)
            
            # Recalculate success rate
            updated = db.keyword_mappings.find_one({"keyword_hash": keyword_hash})
            success = updated.get("success_count", 0)
            failure = updated.get("failure_count", 0)
            total = success + failure
            rate = success / total if total > 0 else 0.5
            
            db.keyword_mappings.update_one(
                {"keyword_hash": keyword_hash},
                {"$set": {"success_rate": rate}}
            )
        else:
            # Create new mapping
            mapping = KeywordMapping(
                keywords=top_keywords,
                preferred_sources=sources[:3] if is_positive else [],
                avoided_sources=sources[:3] if not is_positive else [],
                solution_patterns=[solution[:200]] if solution and is_positive else [],
                success_count=1 if is_positive else 0,
                failure_count=0 if is_positive else 1
            )
            db.keyword_mappings.insert_one(mapping.to_dict())
    
    def _log_learning_event(self, event_type: str, data: Dict):
        """Log a learning event for analytics"""
        self.mongodb.db.learning_events.insert_one({
            "event_type": event_type,
            "data": data,
            "created_at": datetime.now().isoformat()
        })


# =============================================================================
# SOURCE RANKING LEARNER
# =============================================================================

class SourceRankingLearner:
    """
    Learns to rank document sources based on accumulated feedback.
    
    Provides boost/demote factors for RAG retrieval.
    """
    
    def __init__(self, mongodb: MongoDBClient):
        self.mongodb = mongodb
        self._score_cache: Dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_time = 0
    
    def get_source_boost(self, source: str) -> float:
        """
        Get boost factor for a source.
        
        Returns:
            Float multiplier (1.0 = no change, >1 = boost, <1 = demote)
        """
        self._refresh_cache_if_needed()
        return self._score_cache.get(source, 1.0)
    
    def get_source_boosts_batch(self, sources: List[str]) -> Dict[str, float]:
        """Get boost factors for multiple sources"""
        self._refresh_cache_if_needed()
        return {s: self._score_cache.get(s, 1.0) for s in sources}
    
    def get_recommended_sources_for_keywords(
        self,
        keywords: List[str],
        limit: int = 5
    ) -> Tuple[List[str], List[str]]:
        """
        Get recommended and avoided sources for given keywords.
        
        Returns:
            Tuple of (recommended_sources, avoided_sources)
        """
        if not keywords:
            return [], []
        
        db = self.mongodb.db
        
        # Find matching keyword mappings
        mappings = list(db.keyword_mappings.find({
            "keywords": {"$in": keywords[:5]},
            "success_rate": {"$gte": 0.6}
        }).sort("success_rate", -1).limit(3))
        
        recommended = set()
        avoided = set()
        
        for m in mappings:
            for s in m.get("preferred_sources", []):
                recommended.add(s)
            for s in m.get("avoided_sources", []):
                avoided.add(s)
        
        return list(recommended)[:limit], list(avoided)[:limit]
    
    def apply_learned_ranking(
        self,
        results: List[Dict],
        keywords: List[str]
    ) -> List[Dict]:
        """
        Apply learned ranking to search results.
        
        Args:
            results: List of search results with 'source' and 'score'
            keywords: Query keywords
            
        Returns:
            Re-ranked results with adjusted scores
        """
        if not results:
            return results
        
        # Get recommendations for keywords
        recommended, avoided = self.get_recommended_sources_for_keywords(keywords)
        recommended_set = set(recommended)
        avoided_set = set(avoided)
        
        # Apply adjustments
        adjusted = []
        for r in results:
            source = r.get("source", r.get("metadata", {}).get("source", ""))
            score = r.get("score", r.get("similarity", 0.5))
            
            # Get learned boost
            boost = self.get_source_boost(source)
            
            # Apply keyword-based boost/demote
            if source in recommended_set:
                boost *= 1.3  # 30% boost for recommended
            elif source in avoided_set:
                boost *= 0.7  # 30% demote for avoided
            
            adjusted_score = score * boost
            
            adjusted.append({
                **r,
                "original_score": score,
                "score": adjusted_score,
                "learned_boost": boost
            })
        
        # Re-sort by adjusted score
        adjusted.sort(key=lambda x: x["score"], reverse=True)
        
        return adjusted
    
    def _refresh_cache_if_needed(self):
        """Refresh score cache if TTL expired"""
        now = time.time()
        if now - self._cache_time > self._cache_ttl:
            self._load_score_cache()
            self._cache_time = now
    
    def _load_score_cache(self):
        """Load all source scores into cache"""
        try:
            db = self.mongodb.db
            scores = db.source_learning_scores.find({}, {"source": 1, "calculated_score": 1})
            
            self._score_cache = {}
            for s in scores:
                source = s.get("source", "")
                score = s.get("calculated_score", 1.0)
                if source:
                    self._score_cache[source] = score
            
            logger.debug(f"Loaded {len(self._score_cache)} source scores into cache")
        except Exception as e:
            logger.error(f"Error loading score cache: {e}")


# =============================================================================
# EMBEDDING RETRAINER (Placeholder for future domain embedding training)
# =============================================================================

class EmbeddingRetrainer:
    """
    Manages periodic re-training of embeddings based on feedback.
    
    Note: Full embedding fine-tuning requires significant compute resources.
    This provides infrastructure for:
    - Collecting training data from feedback
    - Scheduling retraining jobs
    - Tracking retraining history
    """
    
    def __init__(self, mongodb: MongoDBClient):
        self.mongodb = mongodb
        self._ensure_collections()
    
    def _ensure_collections(self):
        """Ensure retraining collections exist"""
        try:
            db = self.mongodb.db
            if "retraining_data" not in db.list_collection_names():
                db.create_collection("retraining_data")
                db.retraining_data.create_index("created_at")
                db.retraining_data.create_index("used_for_training")
                logger.info("Created retraining_data collection")
                
            if "retraining_history" not in db.list_collection_names():
                db.create_collection("retraining_history")
                logger.info("Created retraining_history collection")
        except Exception as e:
            logger.error(f"Error ensuring retraining collections: {e}")
    
    def collect_training_sample(
        self,
        query: str,
        positive_docs: List[str],
        negative_docs: List[str]
    ):
        """
        Collect a training sample from feedback.
        
        Creates contrastive learning pairs:
        - (query, positive_doc) -> similar
        - (query, negative_doc) -> dissimilar
        """
        try:
            sample = {
                "query": query,
                "positive_docs": positive_docs,
                "negative_docs": negative_docs,
                "created_at": datetime.now().isoformat(),
                "used_for_training": False
            }
            self.mongodb.db.retraining_data.insert_one(sample)
            logger.debug(f"Collected training sample: {len(positive_docs)} positive, {len(negative_docs)} negative")
        except Exception as e:
            logger.error(f"Error collecting training sample: {e}")
    
    def get_training_data_stats(self) -> Dict:
        """Get statistics about collected training data"""
        db = self.mongodb.db
        
        total = db.retraining_data.count_documents({})
        unused = db.retraining_data.count_documents({"used_for_training": False})
        
        # Recent samples
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        recent = db.retraining_data.count_documents({
            "created_at": {"$gte": week_ago}
        })
        
        return {
            "total_samples": total,
            "unused_samples": unused,
            "samples_last_week": recent,
            "ready_for_training": unused >= 100  # Minimum samples for training
        }
    
    def get_retraining_history(self, limit: int = 10) -> List[Dict]:
        """Get recent retraining history"""
        return list(self.mongodb.db.retraining_history.find({}).sort(
            "completed_at", -1
        ).limit(limit))
    
    def schedule_retraining(self) -> Dict:
        """
        Schedule a retraining job.
        
        Note: Actual training would be done by a separate process.
        This just marks the data and creates a job record.
        """
        stats = self.get_training_data_stats()
        
        if not stats["ready_for_training"]:
            return {
                "status": "not_ready",
                "message": f"Need at least 100 samples, have {stats['unused_samples']}"
            }
        
        job = {
            "job_id": hashlib.md5(
                datetime.now().isoformat().encode()
            ).hexdigest()[:12],
            "status": "scheduled",
            "samples_count": stats["unused_samples"],
            "created_at": datetime.now().isoformat(),
            "completed_at": None
        }
        
        self.mongodb.db.retraining_history.insert_one(job)
        
        logger.info(f"Scheduled retraining job {job['job_id']} with {stats['unused_samples']} samples")
        
        return {
            "status": "scheduled",
            "job_id": job["job_id"],
            "samples": stats["unused_samples"]
        }


# =============================================================================
# SELF-LEARNING ENGINE - Main Orchestrator
# =============================================================================

class SelfLearningEngine:
    """
    Main orchestrator for the self-learning feedback loop.
    
    Integrates:
    - Feedback signal processing
    - Source ranking learning
    - Embedding retraining scheduling
    
    Usage:
        engine = SelfLearningEngine()
        
        # Process feedback
        engine.learn_from_feedback(
            feedback_type="positive",
            sources=["manual_1.pdf", "bulletin_2.pdf"],
            keywords=["error", "E50", "torque"]
        )
        
        # Apply learned ranking
        results = engine.apply_learned_ranking(search_results, keywords)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.mongodb = MongoDBClient()
        self.mongodb.connect()
        
        self.signal_processor = FeedbackSignalProcessor(self.mongodb)
        self.ranking_learner = SourceRankingLearner(self.mongodb)
        self.embedding_retrainer = EmbeddingRetrainer(self.mongodb)
        
        self._initialized = True
        logger.info("✅ SelfLearningEngine initialized (Phase 6)")
    
    def learn_from_feedback(
        self,
        feedback_type: str,
        sources: List[str],
        keywords: List[str],
        source_relevance: Optional[List[Dict]] = None,
        solution: Optional[str] = None,
        is_retry: bool = False,
        query: Optional[str] = None
    ) -> Dict:
        """
        Process feedback and update learning models.
        
        This is the main entry point for feedback learning.
        """
        # 1. Process signals
        result = self.signal_processor.process_feedback_signal(
            feedback_type=feedback_type,
            sources=sources,
            keywords=keywords,
            source_relevance=source_relevance,
            solution=solution,
            is_retry=is_retry
        )
        
        # 2. Collect training data for embedding retraining
        if source_relevance:
            positive_docs = [sr["source"] for sr in source_relevance if sr.get("relevant")]
            negative_docs = [sr["source"] for sr in source_relevance if not sr.get("relevant")]
            
            if query and (positive_docs or negative_docs):
                self.embedding_retrainer.collect_training_sample(
                    query=query,
                    positive_docs=positive_docs,
                    negative_docs=negative_docs
                )
                result["training_sample_collected"] = True
        
        return result
    
    def apply_learned_ranking(
        self,
        results: List[Dict],
        keywords: List[str]
    ) -> List[Dict]:
        """
        Apply learned ranking to search results.
        
        This should be called after initial retrieval to re-rank
        results based on accumulated feedback.
        """
        return self.ranking_learner.apply_learned_ranking(results, keywords)
    
    def get_recommendations_for_query(
        self,
        keywords: List[str]
    ) -> Dict:
        """
        Get source recommendations based on learned patterns.
        
        Returns:
            {
                "boost_sources": [...],
                "avoid_sources": [...],
                "confidence": float
            }
        """
        recommended, avoided = self.ranking_learner.get_recommended_sources_for_keywords(
            keywords
        )
        
        # Calculate overall confidence
        db = self.mongodb.db
        relevant_mappings = db.keyword_mappings.count_documents({
            "keywords": {"$in": keywords[:5]}
        })
        
        confidence = min(1.0, relevant_mappings / 5)  # Max confidence at 5+ mappings
        
        return {
            "boost_sources": recommended,
            "avoid_sources": avoided,
            "mappings_found": relevant_mappings,
            "confidence": round(confidence, 2)
        }
    
    def get_learning_stats(self) -> Dict:
        """Get comprehensive learning statistics"""
        db = self.mongodb.db
        
        # Source scores
        source_count = db.source_learning_scores.count_documents({})
        high_confidence = db.source_learning_scores.count_documents({
            "confidence": {"$gte": 0.7}
        })
        
        # Keyword mappings
        mapping_count = db.keyword_mappings.count_documents({})
        successful_mappings = db.keyword_mappings.count_documents({
            "success_rate": {"$gte": 0.7}
        })
        
        # Learning events (last 7 days)
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        recent_events = db.learning_events.count_documents({
            "created_at": {"$gte": week_ago}
        })
        
        # Training data
        training_stats = self.embedding_retrainer.get_training_data_stats()
        
        return {
            "source_learning": {
                "total_sources": source_count,
                "high_confidence_sources": high_confidence,
                "confidence_rate": round(high_confidence / source_count * 100, 1) if source_count > 0 else 0
            },
            "keyword_mappings": {
                "total_mappings": mapping_count,
                "successful_mappings": successful_mappings,
                "success_rate": round(successful_mappings / mapping_count * 100, 1) if mapping_count > 0 else 0
            },
            "activity": {
                "events_last_week": recent_events,
                "avg_events_per_day": round(recent_events / 7, 1)
            },
            "embedding_training": training_stats,
            "system_status": "active" if recent_events > 0 else "idle"
        }
    
    def get_top_learned_sources(self, limit: int = 10) -> List[Dict]:
        """Get top performing sources based on learning"""
        db = self.mongodb.db
        
        sources = list(db.source_learning_scores.find({}).sort(
            "calculated_score", -1
        ).limit(limit))
        
        return [{
            "source": s.get("source"),
            "score": round(s.get("calculated_score", 1.0), 3),
            "positive": s.get("positive_signals", 0),
            "negative": s.get("negative_signals", 0),
            "confidence": round(s.get("confidence", 0), 2)
        } for s in sources]
    
    # =========================================================================
    # USER-FACING API (Phase 2.1 Consolidation)
    # =========================================================================
    # These methods provide the same interface as feedback_engine.py
    # but use the superior Phase 6 learning algorithms
    
    def save_diagnosis(
        self,
        part_number: str,
        product_model: str,
        fault_description: str,
        suggestion: str,
        confidence: str,
        sources: List[dict],
        username: str,
        language: str = "en",
        is_retry: bool = False,
        retry_of: Optional[str] = None,
        response_time_ms: Optional[int] = None
    ) -> str:
        """
        Save diagnosis to history.
        
        Args:
            part_number: Product part number
            product_model: Product model name
            fault_description: User's fault description
            suggestion: Generated repair suggestion
            confidence: Confidence level (low/medium/high)
            sources: List of source documents used
            username: User who requested diagnosis
            language: Language code (en/tr)
            is_retry: Whether this is a retry
            retry_of: Original diagnosis_id if retry
            response_time_ms: Response time in milliseconds
            
        Returns:
            diagnosis_id
        """
        history = DiagnosisHistory(
            part_number=part_number,
            product_model=product_model,
            fault_description=fault_description,
            suggestion=suggestion,
            confidence=confidence,
            sources=sources,
            username=username,
            language=language,
            is_retry=is_retry,
            retry_of=retry_of,
            response_time_ms=response_time_ms
        )
        
        if is_retry and retry_of:
            # Get retry count from original
            original = self.mongodb.db.diagnosis_history.find_one({"diagnosis_id": retry_of})
            if original:
                history.retry_count = original.get("retry_count", 0) + 1
        
        self.mongodb.db.diagnosis_history.insert_one(history.to_dict())
        logger.info(f"Saved diagnosis {history.diagnosis_id} for user {username}")
        
        return history.diagnosis_id
    
    def get_user_history(
        self,
        username: str,
        limit: int = 20,
        skip: int = 0
    ) -> List[dict]:
        """
        Get diagnosis history for a user.
        
        Args:
            username: Username to get history for
            limit: Maximum number of records
            skip: Number of records to skip (pagination)
            
        Returns:
            List of diagnosis records
        """
        cursor = self.mongodb.db.diagnosis_history.find(
            {"username": username}
        ).sort("created_at", -1).skip(skip).limit(limit)
        
        return list(cursor)
    
    def submit_feedback(
        self,
        diagnosis_id: str,
        feedback_type: FeedbackType,
        username: str,
        negative_reason: Optional[NegativeFeedbackReason] = None,
        user_comment: Optional[str] = None,
        correct_solution: Optional[str] = None,
        source_relevance: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Submit feedback for a diagnosis.
        
        Args:
            diagnosis_id: ID of the diagnosis
            feedback_type: FeedbackType.POSITIVE or FeedbackType.NEGATIVE
            username: User submitting feedback
            negative_reason: Reason for negative feedback
            user_comment: Optional user comment
            correct_solution: User's correct solution (if known)
            source_relevance: List of {"source": str, "relevant": bool}
            
        Returns:
            {
                "feedback_id": str,
                "can_retry": bool,
                "message": str
            }
        """
        # Get original diagnosis
        diagnosis = self.mongodb.db.diagnosis_history.find_one({"diagnosis_id": diagnosis_id})
        if not diagnosis:
            return {"error": "Diagnosis not found"}
        
        # Convert source_relevance dicts to SourceRelevance models
        source_relevance_models = None
        if source_relevance:
            source_relevance_models = [
                SourceRelevance(source=sr["source"], relevant=sr["relevant"])
                for sr in source_relevance
            ]
        
        # Create feedback record
        feedback = DiagnosisFeedback(
            diagnosis_id=diagnosis_id,
            part_number=diagnosis["part_number"],
            fault_description=diagnosis["fault_description"],
            suggestion=diagnosis["suggestion"],
            sources_used=[s.get("source", "") for s in diagnosis.get("sources", [])],
            feedback_type=feedback_type,
            negative_reason=negative_reason,
            user_comment=user_comment,
            correct_solution=correct_solution,
            source_relevance=source_relevance_models,
            username=username
        )
        
        self.mongodb.db.diagnosis_feedback.insert_one(feedback.to_dict())
        
        # Update diagnosis history
        self.mongodb.db.diagnosis_history.update_one(
            {"diagnosis_id": diagnosis_id},
            {"$set": {
                "feedback_type": feedback_type.value,
                "feedback_given": True
            }}
        )
        
        # Extract keywords for learning
        keywords = self._extract_keywords(diagnosis["fault_description"])
        
        # Process feedback with Phase 6 learning
        self.learn_from_feedback(
            feedback_type=feedback_type.value,
            sources=feedback.sources_used,
            keywords=keywords,
            source_relevance=source_relevance,
            solution=diagnosis["suggestion"],
            is_retry=False,
            query=diagnosis["fault_description"]
        )
        
        logger.info(f"Feedback {feedback_type.value} submitted for diagnosis {diagnosis_id}")
        
        return {
            "feedback_id": feedback.feedback_id,
            "can_retry": feedback_type == FeedbackType.NEGATIVE,
            "message": "Feedback saved successfully"
        }
    
    def get_feedback_stats(self) -> Dict:
        """
        Get feedback statistics.
        
        Returns:
            {
                "total_feedback": int,
                "positive_count": int,
                "negative_count": int,
                "positive_rate": float,
                "total_diagnoses": int,
                "feedback_rate": float
            }
        """
        db = self.mongodb.db
        
        total_feedback = db.diagnosis_feedback.count_documents({})
        positive = db.diagnosis_feedback.count_documents({"feedback_type": "positive"})
        negative = db.diagnosis_feedback.count_documents({"feedback_type": "negative"})
        
        total_diagnoses = db.diagnosis_history.count_documents({})
        
        return {
            "total_feedback": total_feedback,
            "positive_count": positive,
            "negative_count": negative,
            "positive_rate": round(positive / total_feedback * 100, 1) if total_feedback > 0 else 0,
            "total_diagnoses": total_diagnoses,
            "feedback_rate": round(total_feedback / total_diagnoses * 100, 1) if total_diagnoses > 0 else 0
        }
    
    def get_dashboard_stats(self) -> Dict:
        """
        Get comprehensive dashboard statistics.
        
        Returns:
            Comprehensive stats for admin dashboard including:
            - Feedback stats
            - Top products
            - Top faults
            - Learning stats
            - Recent activity
        """
        db = self.mongodb.db
        
        # Basic feedback stats
        feedback_stats = self.get_feedback_stats()
        
        # Top products (by diagnosis count)
        top_products_pipeline = [
            {"$group": {
                "_id": "$part_number",
                "count": {"$sum": 1},
                "product_model": {"$first": "$product_model"}
            }},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        top_products = list(db.diagnosis_history.aggregate(top_products_pipeline))
        
        # Top fault keywords
        # Extract from recent diagnoses
        recent_diagnoses = list(db.diagnosis_history.find({}).sort("created_at", -1).limit(100))
        fault_keywords = defaultdict(int)
        for diag in recent_diagnoses:
            keywords = self._extract_keywords(diag.get("fault_description", ""))
            for kw in keywords[:3]:  # Top 3 keywords per diagnosis
                fault_keywords[kw] += 1
        
        top_faults = [
            {"keyword": kw, "count": count}
            for kw, count in sorted(fault_keywords.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Learning stats
        learning_stats = self.get_learning_stats()
        
        # Recent activity (last 7 days)
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        recent_diagnoses_count = db.diagnosis_history.count_documents({
            "created_at": {"$gte": week_ago}
        })
        recent_feedback_count = db.diagnosis_feedback.count_documents({
            "created_at": {"$gte": week_ago}
        })
        
        # Top learned sources
        top_sources = self.get_top_learned_sources(limit=10)
        
        return {
            "feedback": feedback_stats,
            "top_products": [
                {
                    "part_number": p["_id"],
                    "product_model": p.get("product_model", "Unknown"),
                    "diagnosis_count": p["count"]
                }
                for p in top_products
            ],
            "top_faults": top_faults,
            "learning": learning_stats,
            "recent_activity": {
                "diagnoses_last_week": recent_diagnoses_count,
                "feedback_last_week": recent_feedback_count,
                "avg_diagnoses_per_day": round(recent_diagnoses_count / 7, 1),
                "avg_feedback_per_day": round(recent_feedback_count / 7, 1)
            },
            "top_sources": top_sources
        }
    
    def get_alternative_sources(
        self,
        original_sources: List[str],
        fault_description: str
    ) -> List[str]:
        """
        Get alternative sources for retry (excluding original sources).
        
        Args:
            original_sources: Sources used in original diagnosis
            fault_description: Fault description to find alternatives for
            
        Returns:
            List of alternative source names
        """
        keywords = self._extract_keywords(fault_description)
        
        # Get recommendations from learned mappings
        recommended, avoided = self.ranking_learner.get_recommended_sources_for_keywords(
            keywords,
            limit=10
        )
        
        # Filter out original sources and avoided sources
        original_set = set(original_sources)
        avoided_set = set(avoided)
        
        alternatives = [
            s for s in recommended
            if s not in original_set and s not in avoided_set
        ]
        
        # If not enough alternatives, get high-scoring sources
        if len(alternatives) < 3:
            top_sources = self.get_top_learned_sources(limit=20)
            for source_info in top_sources:
                source = source_info["source"]
                if source not in original_set and source not in avoided_set and source not in alternatives:
                    alternatives.append(source)
                if len(alternatives) >= 5:
                    break
        
        return alternatives[:5]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from text.
        
        Same logic as feedback_engine.py for compatibility.
        """
        import re
        from collections import Counter
        
        # Lowercase and clean
        text = text.lower()
        
        # Remove common stop words (TR + EN)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'then', 'once',
            've', 'bir', 'bu', 'su', 'ile', 'için', 'gibi', 'daha', 'çok',
            'var', 'yok', 'olan', 'olarak', 've', 'veya', 'ama', 'fakat',
            'de', 'da', 'den', 'dan', 'ne', 'ki', 'mi', 'mu', 'mı'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-ZçğıöşüÇĞİÖŞÜ]{3,}\b', text)
        
        # Filter and get unique keywords
        keywords = [w for w in words if w not in stop_words]
        
        # Return most common (up to 10)
        counter = Counter(keywords)
        return [word for word, _ in counter.most_common(10)]
    
    # =========================================================================
    # END USER-FACING API
    # =========================================================================
    
    def reset_learning(self, confirm: bool = False) -> Dict:
        """
        Reset all learned data.
        
        WARNING: This deletes all accumulated learning!
        """
        if not confirm:
            return {"status": "cancelled", "message": "Set confirm=True to reset"}
        
        db = self.mongodb.db
        
        deleted = {
            "source_scores": db.source_learning_scores.delete_many({}).deleted_count,
            "keyword_mappings": db.keyword_mappings.delete_many({}).deleted_count,
            "learning_events": db.learning_events.delete_many({}).deleted_count,
            "training_data": db.retraining_data.delete_many({}).deleted_count
        }
        
        # Invalidate cache
        self.ranking_learner._score_cache = {}
        self.ranking_learner._cache_time = 0
        
        logger.warning(f"Learning data reset: {deleted}")
        
        return {
            "status": "reset_complete",
            "deleted": deleted
        }


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

def get_self_learning_engine() -> SelfLearningEngine:
    """Get singleton instance of SelfLearningEngine"""
    return SelfLearningEngine()
