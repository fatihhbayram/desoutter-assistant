"""
Feedback and Learning Models for Self-Improving RAG
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class FeedbackType(str, Enum):
    """Feedback types"""
    POSITIVE = "positive"  # 👍 Helpful
    NEGATIVE = "negative"  # 👎 Not helpful


class NegativeFeedbackReason(str, Enum):
    """Reasons for negative feedback"""
    WRONG_PRODUCT = "wrong_product"           # Yanlış ürün/parça önerildi
    WRONG_FAULT_TYPE = "wrong_fault_type"     # Arıza tipi farklı
    INCOMPLETE_INFO = "incomplete_info"        # Eksik bilgi var
    INCORRECT_STEPS = "incorrect_steps"        # Adımlar yanlış
    OTHER = "other"                            # Diğer


class SourceRelevance(BaseModel):
    """Relevance feedback for a single source document"""
    source: str = Field(..., description="Document name/path")
    relevant: bool = Field(..., description="Whether the source was relevant")


class DiagnosisFeedback(BaseModel):
    """Feedback model for diagnosis results"""
    
    feedback_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))
    
    # Diagnosis info
    diagnosis_id: str = Field(..., description="Original diagnosis ID")
    part_number: str = Field(..., description="Product part number")
    fault_description: str = Field(..., description="Original fault description")
    suggestion: str = Field(..., description="AI suggestion that was given")
    sources_used: List[str] = Field(default=[], description="Document sources used")
    
    # Feedback
    feedback_type: FeedbackType = Field(..., description="Positive or negative")
    negative_reason: Optional[NegativeFeedbackReason] = Field(default=None)
    user_comment: Optional[str] = Field(default=None, description="Additional user comment")
    correct_solution: Optional[str] = Field(default=None, description="User provided correct solution")
    
    # Source relevance feedback (per-document relevance ratings)
    source_relevance: Optional[List[SourceRelevance]] = Field(
        default=None, 
        description="Per-source relevance feedback from user"
    )
    
    # User info
    username: str = Field(..., description="User who gave feedback")
    
    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Learning status
    processed: bool = Field(default=False, description="Whether feedback has been processed for learning")
    processed_at: Optional[str] = Field(default=None)
    
    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


class LearnedMapping(BaseModel):
    """
    Learned fault-solution mappings from feedback
    These are high-confidence patterns extracted from positive feedback
    """
    
    mapping_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))
    
    # Pattern info
    fault_keywords: List[str] = Field(..., description="Keywords from fault description")
    product_category: Optional[str] = Field(default=None, description="Product category if specific")
    product_series: Optional[str] = Field(default=None, description="Product series if specific")
    
    # Solution info
    recommended_sources: List[str] = Field(..., description="Document sources that worked")
    solution_summary: str = Field(..., description="Summarized solution")
    
    # Confidence
    positive_count: int = Field(default=1, description="Number of positive feedbacks")
    negative_count: int = Field(default=0, description="Number of negative feedbacks")
    confidence_score: float = Field(default=0.5, description="Confidence score 0-1")
    
    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    def calculate_confidence(self) -> float:
        """Calculate confidence score based on feedback ratio"""
        total = self.positive_count + self.negative_count
        if total == 0:
            return 0.5
        
        # Base confidence from ratio
        ratio = self.positive_count / total
        
        # Boost confidence with more samples (max boost at 10+ samples)
        sample_boost = min(total / 10, 1.0) * 0.2
        
        self.confidence_score = min(ratio + sample_boost, 1.0)
        return self.confidence_score
    
    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


class DiagnosisHistory(BaseModel):
    """
    History of all diagnoses for analytics and learning
    """
    
    diagnosis_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))
    
    # Request info
    part_number: str
    product_model: Optional[str] = None
    fault_description: str
    language: str = "en"
    
    # Response info
    suggestion: str
    confidence: str
    sources: List[str] = Field(default=[])  # List of source document names
    
    # User info
    username: str
    
    # Feedback (updated later)
    feedback_type: Optional[FeedbackType] = None
    feedback_given: bool = False
    
    # Retry info
    is_retry: bool = Field(default=False)
    retry_of: Optional[str] = Field(default=None, description="Original diagnosis ID if this is a retry")
    retry_count: int = Field(default=0)
    
    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    response_time_ms: Optional[int] = None
    metadata: Optional[dict] = Field(default=None, description="Additional context like sufficiency, intent")
    
    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True)
