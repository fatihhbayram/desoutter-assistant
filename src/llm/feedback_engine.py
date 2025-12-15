"""
Feedback Learning Engine - Self-Improving RAG System
Processes user feedback to improve future recommendations
"""
from typing import Dict, List, Optional
from datetime import datetime
import re
from collections import Counter

from src.database.mongo_client import MongoDBClient
from src.database.feedback_models import (
    DiagnosisFeedback, 
    FeedbackType, 
    NegativeFeedbackReason,
    LearnedMapping,
    DiagnosisHistory
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FeedbackLearningEngine:
    """
    Engine for processing feedback and improving RAG responses
    
    Learning strategies:
    1. Positive feedback → Boost similar document rankings
    2. Negative feedback → Demote sources, try alternatives
    3. Pattern extraction → Learn fault-solution mappings
    """
    
    def __init__(self):
        self.mongodb = None
        self._ensure_connection()
        self._ensure_collections()
    
    def _ensure_connection(self):
        """Ensure MongoDB connection"""
        if not self.mongodb:
            self.mongodb = MongoDBClient()
            self.mongodb.connect()
    
    def _ensure_collections(self):
        """Ensure required collections exist with indexes"""
        try:
            db = self.mongodb.db
            
            # Create collections if not exist
            collections = db.list_collection_names()
            
            if "diagnosis_feedback" not in collections:
                db.create_collection("diagnosis_feedback")
                db.diagnosis_feedback.create_index("diagnosis_id")
                db.diagnosis_feedback.create_index("part_number")
                db.diagnosis_feedback.create_index("feedback_type")
                db.diagnosis_feedback.create_index("processed")
                logger.info("Created diagnosis_feedback collection")
            
            if "learned_mappings" not in collections:
                db.create_collection("learned_mappings")
                db.learned_mappings.create_index("fault_keywords")
                db.learned_mappings.create_index("confidence_score")
                logger.info("Created learned_mappings collection")
            
            if "diagnosis_history" not in collections:
                db.create_collection("diagnosis_history")
                db.diagnosis_history.create_index("username")
                db.diagnosis_history.create_index("part_number")
                db.diagnosis_history.create_index("created_at")
                db.diagnosis_history.create_index("feedback_given")
                logger.info("Created diagnosis_history collection")
                
        except Exception as e:
            logger.error(f"Error ensuring collections: {e}")
    
    # =========================================================================
    # DIAGNOSIS HISTORY
    # =========================================================================
    
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
        Save diagnosis to history
        
        Returns:
            diagnosis_id
        """
        self._ensure_connection()
        
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
        """Get diagnosis history for a user"""
        self._ensure_connection()
        
        cursor = self.mongodb.db.diagnosis_history.find(
            {"username": username}
        ).sort("created_at", -1).skip(skip).limit(limit)
        
        return list(cursor)
    
    # =========================================================================
    # FEEDBACK PROCESSING
    # =========================================================================
    
    def submit_feedback(
        self,
        diagnosis_id: str,
        feedback_type: FeedbackType,
        username: str,
        negative_reason: Optional[NegativeFeedbackReason] = None,
        user_comment: Optional[str] = None,
        correct_solution: Optional[str] = None
    ) -> Dict:
        """
        Submit feedback for a diagnosis
        
        Returns:
            feedback_id and whether retry is available
        """
        self._ensure_connection()
        
        # Get original diagnosis
        diagnosis = self.mongodb.db.diagnosis_history.find_one({"diagnosis_id": diagnosis_id})
        if not diagnosis:
            return {"error": "Diagnosis not found"}
        
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
        
        # Process feedback for learning
        self._process_feedback_for_learning(feedback)
        
        logger.info(f"Feedback {feedback_type.value} submitted for diagnosis {diagnosis_id}")
        
        return {
            "feedback_id": feedback.feedback_id,
            "can_retry": feedback_type == FeedbackType.NEGATIVE,
            "message": "Feedback saved successfully"
        }
    
    def _process_feedback_for_learning(self, feedback: DiagnosisFeedback):
        """
        Process feedback to update learned mappings
        """
        try:
            # Extract keywords from fault description
            keywords = self._extract_keywords(feedback.fault_description)
            
            if feedback.feedback_type == FeedbackType.POSITIVE:
                # Positive: reinforce or create mapping
                self._reinforce_mapping(
                    keywords=keywords,
                    sources=feedback.sources_used,
                    solution_summary=feedback.suggestion[:500]  # First 500 chars as summary
                )
            else:
                # Negative: demote mapping
                self._demote_mapping(
                    keywords=keywords,
                    sources=feedback.sources_used
                )
            
            # Mark feedback as processed
            self.mongodb.db.diagnosis_feedback.update_one(
                {"feedback_id": feedback.feedback_id},
                {"$set": {
                    "processed": True,
                    "processed_at": datetime.now().isoformat()
                }}
            )
            
        except Exception as e:
            logger.error(f"Error processing feedback for learning: {e}")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
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
    
    def _reinforce_mapping(
        self,
        keywords: List[str],
        sources: List[str],
        solution_summary: str
    ):
        """Reinforce or create a positive mapping"""
        if not keywords:
            return
        
        # Look for existing mapping with similar keywords
        existing = self.mongodb.db.learned_mappings.find_one({
            "fault_keywords": {"$in": keywords[:3]}  # Match on top 3 keywords
        })
        
        if existing:
            # Update existing mapping
            mapping = LearnedMapping(**existing)
            mapping.positive_count += 1
            mapping.calculate_confidence()
            mapping.updated_at = datetime.now().isoformat()
            
            # Add new sources if not already present
            for source in sources:
                if source and source not in mapping.recommended_sources:
                    mapping.recommended_sources.append(source)
            
            self.mongodb.db.learned_mappings.update_one(
                {"mapping_id": mapping.mapping_id},
                {"$set": mapping.to_dict()}
            )
            logger.info(f"Reinforced mapping {mapping.mapping_id}, confidence: {mapping.confidence_score:.2f}")
        else:
            # Create new mapping
            mapping = LearnedMapping(
                fault_keywords=keywords,
                recommended_sources=[s for s in sources if s],
                solution_summary=solution_summary
            )
            mapping.calculate_confidence()
            
            self.mongodb.db.learned_mappings.insert_one(mapping.to_dict())
            logger.info(f"Created new mapping {mapping.mapping_id}")
    
    def _demote_mapping(self, keywords: List[str], sources: List[str]):
        """Demote mapping based on negative feedback"""
        if not keywords:
            return
        
        existing = self.mongodb.db.learned_mappings.find_one({
            "fault_keywords": {"$in": keywords[:3]}
        })
        
        if existing:
            mapping = LearnedMapping(**existing)
            mapping.negative_count += 1
            mapping.calculate_confidence()
            mapping.updated_at = datetime.now().isoformat()
            
            self.mongodb.db.learned_mappings.update_one(
                {"mapping_id": mapping.mapping_id},
                {"$set": mapping.to_dict()}
            )
            logger.info(f"Demoted mapping {mapping.mapping_id}, confidence: {mapping.confidence_score:.2f}")
    
    # =========================================================================
    # LEARNING-ENHANCED RETRIEVAL
    # =========================================================================
    
    def get_learned_context(
        self,
        fault_description: str,
        excluded_sources: List[str] = None
    ) -> Dict:
        """
        Get learned context to boost RAG retrieval
        
        Returns:
            - boost_sources: Sources to prioritize
            - exclude_sources: Sources to avoid
            - learned_solution: Pre-learned solution if high confidence
        """
        self._ensure_connection()
        
        keywords = self._extract_keywords(fault_description)
        if not keywords:
            return {"boost_sources": [], "exclude_sources": [], "learned_solution": None}
        
        # Find matching mappings
        mappings = list(self.mongodb.db.learned_mappings.find({
            "fault_keywords": {"$in": keywords[:5]},
            "confidence_score": {"$gte": 0.6}  # Only use confident mappings
        }).sort("confidence_score", -1).limit(3))
        
        boost_sources = []
        exclude_sources = excluded_sources or []
        learned_solution = None
        
        for mapping in mappings:
            # High confidence mapping - can use as learned solution
            if mapping.get("confidence_score", 0) >= 0.8:
                learned_solution = mapping.get("solution_summary")
            
            # Add sources to boost
            for source in mapping.get("recommended_sources", []):
                if source not in boost_sources and source not in exclude_sources:
                    boost_sources.append(source)
        
        # Also get sources from negative feedback to exclude
        negative_feedbacks = list(self.mongodb.db.diagnosis_feedback.find({
            "feedback_type": "negative",
            "part_number": {"$exists": True}
        }).sort("created_at", -1).limit(50))
        
        for fb in negative_feedbacks:
            fb_keywords = self._extract_keywords(fb.get("fault_description", ""))
            # If keywords overlap significantly, exclude those sources
            overlap = len(set(keywords) & set(fb_keywords))
            if overlap >= 2:
                for source in fb.get("sources_used", []):
                    if source and source not in exclude_sources:
                        exclude_sources.append(source)
        
        return {
            "boost_sources": boost_sources[:5],  # Top 5 to boost
            "exclude_sources": exclude_sources[:10],  # Max 10 to exclude
            "learned_solution": learned_solution
        }
    
    def get_alternative_sources(
        self,
        original_sources: List[str],
        fault_description: str
    ) -> List[str]:
        """
        Get alternative sources for retry (excluding original sources)
        """
        # Sources to avoid
        excluded = set(original_sources)
        
        # Get sources from other successful diagnoses with similar keywords
        keywords = self._extract_keywords(fault_description)
        
        alternatives = []
        
        # Search in positive feedback
        positive_feedbacks = list(self.mongodb.db.diagnosis_feedback.find({
            "feedback_type": "positive"
        }).sort("created_at", -1).limit(100))
        
        for fb in positive_feedbacks:
            fb_keywords = self._extract_keywords(fb.get("fault_description", ""))
            overlap = len(set(keywords) & set(fb_keywords))
            
            if overlap >= 2:  # Similar problem
                for source in fb.get("sources_used", []):
                    if source and source not in excluded and source not in alternatives:
                        alternatives.append(source)
        
        return alternatives[:10]
    
    # =========================================================================
    # ANALYTICS & DASHBOARD
    # =========================================================================
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics"""
        self._ensure_connection()
        
        total = self.mongodb.db.diagnosis_feedback.count_documents({})
        positive = self.mongodb.db.diagnosis_feedback.count_documents({"feedback_type": "positive"})
        negative = self.mongodb.db.diagnosis_feedback.count_documents({"feedback_type": "negative"})
        
        # Negative reasons breakdown
        reasons = {}
        for reason in NegativeFeedbackReason:
            count = self.mongodb.db.diagnosis_feedback.count_documents({
                "feedback_type": "negative",
                "negative_reason": reason.value
            })
            if count > 0:
                reasons[reason.value] = count
        
        # Learned mappings
        mappings_count = self.mongodb.db.learned_mappings.count_documents({})
        high_confidence = self.mongodb.db.learned_mappings.count_documents({"confidence_score": {"$gte": 0.8}})
        
        return {
            "total_feedback": total,
            "positive_feedback": positive,
            "negative_feedback": negative,
            "satisfaction_rate": round(positive / total * 100, 1) if total > 0 else 0,
            "negative_reasons": reasons,
            "learned_mappings": mappings_count,
            "high_confidence_mappings": high_confidence
        }

    def get_dashboard_stats(self) -> Dict:
        """
        Get comprehensive dashboard statistics
        Returns stats for charts, top products, fault trends, etc.
        """
        self._ensure_connection()
        from datetime import datetime, timedelta
        
        # =====================================================================
        # TIME RANGES
        # =====================================================================
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        # =====================================================================
        # DIAGNOSIS COUNTS
        # =====================================================================
        total_diagnoses = self.mongodb.db.diagnosis_history.count_documents({})
        today_diagnoses = self.mongodb.db.diagnosis_history.count_documents({
            "created_at": {"$gte": today_start.isoformat()}
        })
        week_diagnoses = self.mongodb.db.diagnosis_history.count_documents({
            "created_at": {"$gte": week_ago.isoformat()}
        })
        month_diagnoses = self.mongodb.db.diagnosis_history.count_documents({
            "created_at": {"$gte": month_ago.isoformat()}
        })
        
        # =====================================================================
        # TOP PRODUCTS (Most diagnosed)
        # =====================================================================
        top_products_pipeline = [
            {"$group": {
                "_id": "$part_number",
                "count": {"$sum": 1},
                "model": {"$first": "$product_model"}
            }},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        top_products = list(self.mongodb.db.diagnosis_history.aggregate(top_products_pipeline))
        
        # =====================================================================
        # TOP FAULT KEYWORDS
        # =====================================================================
        fault_keywords = {}
        diagnoses = self.mongodb.db.diagnosis_history.find({}, {"fault_description": 1}).limit(500)
        for diag in diagnoses:
            keywords = self._extract_keywords(diag.get("fault_description", ""))
            for kw in keywords:
                fault_keywords[kw] = fault_keywords.get(kw, 0) + 1
        
        # Sort and get top 15
        top_faults = sorted(fault_keywords.items(), key=lambda x: x[1], reverse=True)[:15]
        
        # =====================================================================
        # DAILY TREND (Last 7 days)
        # =====================================================================
        daily_trend = []
        for i in range(7):
            day = now - timedelta(days=6-i)
            day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            
            count = self.mongodb.db.diagnosis_history.count_documents({
                "created_at": {
                    "$gte": day_start.isoformat(),
                    "$lt": day_end.isoformat()
                }
            })
            daily_trend.append({
                "date": day_start.strftime("%Y-%m-%d"),
                "day": day_start.strftime("%a"),
                "count": count
            })
        
        # =====================================================================
        # FEEDBACK BREAKDOWN
        # =====================================================================
        feedback_stats = self.get_feedback_stats()
        
        # =====================================================================
        # RESPONSE TIME STATS
        # =====================================================================
        response_times = list(self.mongodb.db.diagnosis_history.find(
            {"response_time_ms": {"$exists": True, "$ne": None}},
            {"response_time_ms": 1}
        ).limit(100))
        
        avg_response_time = 0
        if response_times:
            times = [r["response_time_ms"] for r in response_times if r.get("response_time_ms")]
            if times:
                avg_response_time = sum(times) / len(times)
        
        # =====================================================================
        # CONFIDENCE BREAKDOWN
        # =====================================================================
        confidence_counts = {
            "high": self.mongodb.db.diagnosis_history.count_documents({"confidence": "high"}),
            "medium": self.mongodb.db.diagnosis_history.count_documents({"confidence": "medium"}),
            "low": self.mongodb.db.diagnosis_history.count_documents({"confidence": "low"})
        }
        
        # =====================================================================
        # ACTIVE USERS (Last 7 days)
        # =====================================================================
        active_users_pipeline = [
            {"$match": {"created_at": {"$gte": week_ago.isoformat()}}},
            {"$group": {"_id": "$username"}},
            {"$count": "count"}
        ]
        active_users_result = list(self.mongodb.db.diagnosis_history.aggregate(active_users_pipeline))
        active_users = active_users_result[0]["count"] if active_users_result else 0
        
        # =====================================================================
        # RETRY STATS
        # =====================================================================
        retry_count = self.mongodb.db.diagnosis_history.count_documents({"is_retry": True})
        retry_rate = round(retry_count / total_diagnoses * 100, 1) if total_diagnoses > 0 else 0
        
        return {
            "overview": {
                "total_diagnoses": total_diagnoses,
                "today_diagnoses": today_diagnoses,
                "week_diagnoses": week_diagnoses,
                "month_diagnoses": month_diagnoses,
                "active_users_week": active_users,
                "avg_response_time_ms": round(avg_response_time),
                "retry_rate": retry_rate
            },
            "top_products": [
                {"part_number": p["_id"], "model": p.get("model", ""), "count": p["count"]}
                for p in top_products
            ],
            "top_faults": [
                {"keyword": kw, "count": count}
                for kw, count in top_faults
            ],
            "daily_trend": daily_trend,
            "confidence_breakdown": confidence_counts,
            "feedback": feedback_stats
        }
