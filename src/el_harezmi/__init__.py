"""
El-Harezmi 5-Stage RAG Pipeline

A structured, intent-driven retrieval and generation system for
Desoutter industrial tool technical support.

Stages:
    1. Intent Classification - Multi-label intent detection + entity extraction
    2. Retrieval Strategy - Intent-aware Qdrant filtering and boosting
    3. Info Extraction - Structured JSON extraction from chunks
    4. KG Validation - Compatibility matrix validation
    5. Response Formatter - Turkish response templates

Usage:
    from src.el_harezmi import ElHarezmiPipeline
    
    pipeline = ElHarezmiPipeline()
    response = await pipeline.process("EABC-3000 tork ayarı nasıl yapılır?")
"""

from .stage1_intent_classifier import IntentClassifier, IntentResult, IntentType, ExtractedEntities
from .stage2_retrieval_strategy import RetrievalStrategy, RetrievalResult
from .stage3_info_extraction import InfoExtractor, ExtractionResult
from .stage4_kg_validation import KGValidator, ValidationResult
from .stage5_response_formatter import ResponseFormatter, FormattedResponse, Language
from .pipeline import ElHarezmiPipeline

__all__ = [
    # Pipeline
    "ElHarezmiPipeline",
    
    # Stage 1
    "IntentClassifier",
    "IntentResult",
    "IntentType",
    "ExtractedEntities",
    
    # Stage 2
    "RetrievalStrategy",
    "RetrievalResult",
    
    # Stage 3
    "InfoExtractor",
    "ExtractionResult",
    
    # Stage 4
    "KGValidator",
    "ValidationResult",
    
    # Stage 5
    "ResponseFormatter",
    "FormattedResponse",
    "Language",
]

__version__ = "1.0.0"
