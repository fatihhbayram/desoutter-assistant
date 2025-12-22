"""
RAG Performance Metrics - Phase 5
Tracks and monitors RAG system performance

Metrics tracked:
- Query latency (retrieval, LLM, total)
- Cache hit/miss rates
- Retrieval quality scores
- User feedback accuracy
- System health indicators
"""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import threading
import json

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query"""
    query_id: str
    timestamp: datetime
    query_text: str
    
    # Timing metrics (milliseconds)
    retrieval_time_ms: float = 0.0
    llm_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Retrieval metrics
    documents_retrieved: int = 0
    avg_similarity_score: float = 0.0
    cache_hit: bool = False
    
    # Quality indicators
    confidence: str = "unknown"
    user_feedback: Optional[bool] = None  # True=positive, False=negative
    
    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "timestamp": self.timestamp.isoformat(),
            "query_text": self.query_text[:100],  # Truncate for storage
            "retrieval_time_ms": self.retrieval_time_ms,
            "llm_time_ms": self.llm_time_ms,
            "total_time_ms": self.total_time_ms,
            "documents_retrieved": self.documents_retrieved,
            "avg_similarity_score": self.avg_similarity_score,
            "cache_hit": self.cache_hit,
            "confidence": self.confidence,
            "user_feedback": self.user_feedback
        }


@dataclass
class PerformanceStats:
    """Aggregated performance statistics"""
    period_start: datetime
    period_end: datetime
    
    # Query counts
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Latency stats (ms)
    avg_retrieval_time: float = 0.0
    avg_llm_time: float = 0.0
    avg_total_time: float = 0.0
    p95_total_time: float = 0.0
    p99_total_time: float = 0.0
    
    # Quality stats
    avg_similarity: float = 0.0
    avg_docs_retrieved: float = 0.0
    
    # Confidence distribution
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0
    
    # Feedback stats
    positive_feedback_count: int = 0
    negative_feedback_count: int = 0
    feedback_accuracy: float = 0.0  # positive / (positive + negative)
    
    def to_dict(self) -> Dict:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat()
            },
            "queries": {
                "total": self.total_queries,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / max(self.total_queries, 1)
            },
            "latency_ms": {
                "avg_retrieval": round(self.avg_retrieval_time, 2),
                "avg_llm": round(self.avg_llm_time, 2),
                "avg_total": round(self.avg_total_time, 2),
                "p95_total": round(self.p95_total_time, 2),
                "p99_total": round(self.p99_total_time, 2)
            },
            "retrieval_quality": {
                "avg_similarity": round(self.avg_similarity, 4),
                "avg_docs_retrieved": round(self.avg_docs_retrieved, 2)
            },
            "confidence_distribution": {
                "high": self.high_confidence_count,
                "medium": self.medium_confidence_count,
                "low": self.low_confidence_count
            },
            "feedback": {
                "positive": self.positive_feedback_count,
                "negative": self.negative_feedback_count,
                "accuracy": round(self.feedback_accuracy, 4)
            }
        }


class PerformanceMonitor:
    """
    Centralized performance monitoring for RAG system
    
    Features:
    - Real-time query metrics collection
    - Aggregated statistics calculation
    - Historical data retention (configurable)
    - Thread-safe operations
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, retention_hours: int = 24, max_queries: int = 10000):
        """
        Initialize performance monitor
        
        Args:
            retention_hours: How long to keep query metrics
            max_queries: Maximum queries to keep in memory
        """
        if self._initialized:
            return
            
        self.retention_hours = retention_hours
        self.max_queries = max_queries
        self.queries: List[QueryMetrics] = []
        self._query_lock = threading.RLock()
        self._query_counter = 0
        self._initialized = True
        
        logger.info(f"Performance Monitor initialized (retention: {retention_hours}h, max: {max_queries})")
    
    def generate_query_id(self) -> str:
        """Generate unique query ID"""
        with self._query_lock:
            self._query_counter += 1
            return f"q_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._query_counter}"
    
    def record_query(self, metrics: QueryMetrics):
        """Record query metrics"""
        with self._query_lock:
            self.queries.append(metrics)
            
            # Trim old queries
            self._cleanup_old_queries()
            
            # Log significant issues
            if metrics.total_time_ms > 30000:  # > 30 seconds
                logger.warning(f"Slow query detected: {metrics.total_time_ms}ms - {metrics.query_text[:50]}")
    
    def _cleanup_old_queries(self):
        """Remove queries older than retention period"""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        
        # Remove old queries
        self.queries = [q for q in self.queries if q.timestamp > cutoff]
        
        # Trim to max size if needed
        if len(self.queries) > self.max_queries:
            self.queries = self.queries[-self.max_queries:]
    
    def record_feedback(self, query_id: str, is_positive: bool):
        """Record user feedback for a query"""
        with self._query_lock:
            for query in reversed(self.queries):
                if query.query_id == query_id:
                    query.user_feedback = is_positive
                    break
    
    def get_stats(self, hours: int = 1) -> PerformanceStats:
        """
        Calculate aggregated statistics for a time period
        
        Args:
            hours: Number of hours to include
            
        Returns:
            PerformanceStats object with aggregated metrics
        """
        with self._query_lock:
            cutoff = datetime.now() - timedelta(hours=hours)
            recent = [q for q in self.queries if q.timestamp > cutoff]
            
            if not recent:
                return PerformanceStats(
                    period_start=cutoff,
                    period_end=datetime.now()
                )
            
            # Calculate stats
            total = len(recent)
            cache_hits = sum(1 for q in recent if q.cache_hit)
            
            retrieval_times = [q.retrieval_time_ms for q in recent]
            llm_times = [q.llm_time_ms for q in recent]
            total_times = [q.total_time_ms for q in recent]
            
            # Confidence counts
            high = sum(1 for q in recent if q.confidence == "high")
            medium = sum(1 for q in recent if q.confidence == "medium")
            low = sum(1 for q in recent if q.confidence == "low")
            
            # Feedback stats
            with_feedback = [q for q in recent if q.user_feedback is not None]
            positive = sum(1 for q in with_feedback if q.user_feedback)
            negative = len(with_feedback) - positive
            
            # Calculate percentiles
            sorted_times = sorted(total_times)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)
            
            return PerformanceStats(
                period_start=cutoff,
                period_end=datetime.now(),
                total_queries=total,
                cache_hits=cache_hits,
                cache_misses=total - cache_hits,
                avg_retrieval_time=sum(retrieval_times) / total if total else 0,
                avg_llm_time=sum(llm_times) / total if total else 0,
                avg_total_time=sum(total_times) / total if total else 0,
                p95_total_time=sorted_times[p95_idx] if sorted_times else 0,
                p99_total_time=sorted_times[p99_idx] if sorted_times else 0,
                avg_similarity=sum(q.avg_similarity_score for q in recent) / total if total else 0,
                avg_docs_retrieved=sum(q.documents_retrieved for q in recent) / total if total else 0,
                high_confidence_count=high,
                medium_confidence_count=medium,
                low_confidence_count=low,
                positive_feedback_count=positive,
                negative_feedback_count=negative,
                feedback_accuracy=positive / (positive + negative) if (positive + negative) > 0 else 0
            )
    
    def get_recent_queries(self, limit: int = 20) -> List[Dict]:
        """Get recent queries for debugging"""
        with self._query_lock:
            recent = self.queries[-limit:]
            return [q.to_dict() for q in reversed(recent)]
    
    def get_slow_queries(self, threshold_ms: float = 10000, limit: int = 10) -> List[Dict]:
        """Get slow queries exceeding threshold"""
        with self._query_lock:
            slow = [q for q in self.queries if q.total_time_ms > threshold_ms]
            slow.sort(key=lambda x: x.total_time_ms, reverse=True)
            return [q.to_dict() for q in slow[:limit]]
    
    def get_health_status(self) -> Dict:
        """Get overall system health status"""
        stats_1h = self.get_stats(hours=1)
        stats_24h = self.get_stats(hours=24)
        
        # Determine health status
        health = "healthy"
        issues = []
        
        # Check latency
        if stats_1h.avg_total_time > 30000:
            health = "degraded"
            issues.append(f"High average latency: {stats_1h.avg_total_time:.0f}ms")
        
        # Check cache hit rate
        if stats_1h.total_queries > 10:
            hit_rate = stats_1h.cache_hits / stats_1h.total_queries
            if hit_rate < 0.1:
                issues.append(f"Low cache hit rate: {hit_rate:.1%}")
        
        # Check feedback accuracy
        if stats_24h.positive_feedback_count + stats_24h.negative_feedback_count > 10:
            if stats_24h.feedback_accuracy < 0.5:
                health = "warning"
                issues.append(f"Low feedback accuracy: {stats_24h.feedback_accuracy:.1%}")
        
        return {
            "status": health,
            "issues": issues,
            "last_hour": stats_1h.to_dict(),
            "last_24h": stats_24h.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
    
    def reset(self):
        """Reset all metrics (for testing)"""
        with self._query_lock:
            self.queries.clear()
            self._query_counter = 0
            logger.info("Performance metrics reset")


# Global instance getter
_monitor_instance: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PerformanceMonitor()
    return _monitor_instance


class QueryTimer:
    """Context manager for timing query phases"""
    
    def __init__(self, query_id: str, query_text: str):
        self.query_id = query_id
        self.query_text = query_text
        self.start_time = None
        self.retrieval_start = None
        self.llm_start = None
        
        self.retrieval_time = 0.0
        self.llm_time = 0.0
        self.total_time = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.total_time = (time.time() - self.start_time) * 1000
    
    def start_retrieval(self):
        """Mark start of retrieval phase"""
        self.retrieval_start = time.time()
    
    def end_retrieval(self):
        """Mark end of retrieval phase"""
        if self.retrieval_start:
            self.retrieval_time = (time.time() - self.retrieval_start) * 1000
    
    def start_llm(self):
        """Mark start of LLM phase"""
        self.llm_start = time.time()
    
    def end_llm(self):
        """Mark end of LLM phase"""
        if self.llm_start:
            self.llm_time = (time.time() - self.llm_start) * 1000
    
    def create_metrics(
        self,
        documents_retrieved: int = 0,
        avg_similarity: float = 0.0,
        cache_hit: bool = False,
        confidence: str = "unknown"
    ) -> QueryMetrics:
        """Create QueryMetrics object from timer data"""
        return QueryMetrics(
            query_id=self.query_id,
            timestamp=datetime.now(),
            query_text=self.query_text,
            retrieval_time_ms=self.retrieval_time,
            llm_time_ms=self.llm_time,
            total_time_ms=self.total_time,
            documents_retrieved=documents_retrieved,
            avg_similarity_score=avg_similarity,
            cache_hit=cache_hit,
            confidence=confidence
        )
