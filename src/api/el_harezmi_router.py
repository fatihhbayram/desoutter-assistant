"""
El-Harezmi API Router

FastAPI router for the new 5-stage RAG pipeline.
Provides /api/v2/chat and /api/v2/diagnose endpoints.
"""

import asyncio
import logging
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Header, Query
from pydantic import BaseModel, Field

from config.feature_flags import (
    get_feature_flags, get_pipeline_version, 
    PipelineVersion, is_el_harezmi_enabled
)

logger = logging.getLogger(__name__)

# Create router with v2 prefix
router = APIRouter(prefix="/api/v2", tags=["El-Harezmi"])


# -----------------------------------------------------------------------------
# REQUEST/RESPONSE MODELS
# -----------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., description="User message/query")
    product_model: Optional[str] = Field(None, description="Product model (e.g., EABC-3000)")
    language: str = Field("tr", description="Response language: 'tr' or 'en'")
    force_pipeline: Optional[str] = Field(None, description="Force pipeline: 'legacy' or 'el_harezmi'")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")


class SourceInfo(BaseModel):
    """Source document information"""
    document: str
    score: float
    chunk_type: Optional[str] = None
    product_family: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="Generated response")
    intent: str = Field(..., description="Detected intent type")
    confidence: float = Field(..., description="Response confidence score")
    product_model: Optional[str] = Field(None, description="Extracted product model")
    sources: List[SourceInfo] = Field(default_factory=list, description="Source documents used")
    pipeline_version: str = Field(..., description="Pipeline version used")
    language: str = Field(..., description="Response language")
    validation_status: Optional[str] = Field(None, description="KG validation status")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    warnings: List[str] = Field(default_factory=list, description="Any warnings")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    pipeline_version: str
    el_harezmi_enabled: bool
    rollout_percentage: float
    qdrant_connected: bool
    timestamp: str


# -----------------------------------------------------------------------------
# PIPELINE INITIALIZATION
# -----------------------------------------------------------------------------

# Lazy-loaded El-Harezmi pipeline
_el_harezmi_pipeline = None
_qdrant_client = None
_embedding_model = None


def get_el_harezmi_pipeline():
    """Get or create El-Harezmi pipeline singleton"""
    global _el_harezmi_pipeline, _qdrant_client, _embedding_model
    
    if _el_harezmi_pipeline is None:
        logger.info("Initializing El-Harezmi pipeline...")
        
        try:
            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer
            from src.el_harezmi import ElHarezmiPipeline
            
            # Initialize Qdrant client
            _qdrant_client = QdrantClient(
                host="qdrant",
                port=6333,
                timeout=30
            )
            
            # Initialize embedding model
            _embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Initialize pipeline
            _el_harezmi_pipeline = ElHarezmiPipeline(
                qdrant_client=_qdrant_client,
                embedding_model=_embedding_model
            )
            
            logger.info("El-Harezmi pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize El-Harezmi pipeline: {e}")
            raise
    
    return _el_harezmi_pipeline


def is_qdrant_connected() -> bool:
    """Check if Qdrant is connected"""
    global _qdrant_client
    
    if _qdrant_client is None:
        return False
    
    try:
        _qdrant_client.get_collections()
        return True
    except:
        return False


# -----------------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for El-Harezmi pipeline.
    """
    flags = get_feature_flags()
    
    return HealthResponse(
        status="healthy",
        pipeline_version="el_harezmi_v1",
        el_harezmi_enabled=is_el_harezmi_enabled(),
        rollout_percentage=flags.el_harezmi_rollout,
        qdrant_connected=is_qdrant_connected(),
        timestamp=datetime.now().isoformat()
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    authorization: str = Header(default=None)
):
    """
    Chat endpoint using El-Harezmi 5-stage pipeline.
    
    Stages:
    1. Intent Classification - Detect query intent and extract entities
    2. Retrieval Strategy - Intent-aware Qdrant search
    3. Info Extraction - Structured data extraction (with LLM)
    4. KG Validation - Compatibility matrix validation
    5. Response Formatting - Turkish/English response generation
    """
    start_time = datetime.now()
    
    # Get username from token
    user_id = "anonymous"
    if authorization and authorization.startswith("Bearer "):
        try:
            import jwt
            from src.api.main import JWT_SECRET, JWT_ALG
            token = authorization.split(" ")[1]
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
            user_id = payload.get("sub", "anonymous")
        except:
            pass
    
    # Determine pipeline version
    pipeline_version = get_pipeline_version(user_id, request.force_pipeline)
    
    flags = get_feature_flags()
    if flags.log_pipeline_selection:
        logger.info(f"Pipeline selection: {pipeline_version.value} for user {user_id}")
    
    # Use El-Harezmi pipeline
    if pipeline_version == PipelineVersion.EL_HAREZMI:
        try:
            return await _process_with_el_harezmi(request, user_id, start_time)
        except Exception as e:
            logger.error(f"El-Harezmi error: {e}")
            
            # Fallback to legacy if configured
            if flags.fallback_to_legacy_on_error:
                logger.warning("Falling back to legacy pipeline")
                return await _process_with_legacy(request, user_id, start_time)
            else:
                raise HTTPException(status_code=500, detail=str(e))
    else:
        # Use legacy pipeline
        return await _process_with_legacy(request, user_id, start_time)


async def _process_with_el_harezmi(
    request: ChatRequest,
    user_id: str,
    start_time: datetime
) -> ChatResponse:
    """Process request with El-Harezmi pipeline"""
    
    from src.el_harezmi import Language
    
    pipeline = get_el_harezmi_pipeline()
    
    # Map language
    language = Language.TURKISH if request.language.lower() in ["tr", "turkish"] else Language.ENGLISH
    
    # Process query
    result = await pipeline.process(
        query=request.message,
        language=language
    )
    
    # Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    # Build source info
    sources = []
    if result.retrieval_result and result.retrieval_result.chunks:
        for chunk in result.retrieval_result.chunks[:5]:  # Top 5 sources
            sources.append(SourceInfo(
                document=chunk.metadata.get("source", "unknown") if chunk.metadata else "unknown",
                score=chunk.score,
                chunk_type=chunk.chunk_type,
                product_family=chunk.product_family
            ))
    
    # Build warnings
    warnings = []
    if result.validation_result and result.validation_result.issues:
        for issue in result.validation_result.issues:
            warnings.append(f"{issue.severity}: {issue.message}")
    
    return ChatResponse(
        response=result.response.content,
        intent=result.intent_result.primary_intent.value,
        confidence=result.response.confidence,
        product_model=result.response.product_model,
        sources=sources,
        pipeline_version="el_harezmi",
        language=request.language,
        validation_status=result.validation_result.status.value if result.validation_result else None,
        processing_time_ms=processing_time,
        warnings=warnings
    )


async def _process_with_legacy(
    request: ChatRequest,
    user_id: str,
    start_time: datetime
) -> ChatResponse:
    """Process request with legacy RAG pipeline"""
    
    from src.llm.rag_engine import RAGEngine
    
    # Get or create RAG engine
    rag = RAGEngine()
    
    # Run in thread pool to avoid blocking
    result = await asyncio.to_thread(
        rag.generate_repair_suggestion,
        part_number=request.product_model or "",
        fault_description=request.message,
        language=request.language,
        username=user_id
    )
    
    # Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    # Build source info from legacy result
    sources = []
    if result.get("sources"):
        for src in result["sources"][:5]:
            sources.append(SourceInfo(
                document=src.get("source", "unknown"),
                score=src.get("score", 0.0),
                chunk_type=None,
                product_family=None
            ))
    
    return ChatResponse(
        response=result.get("suggestion", ""),
        intent="general",  # Legacy doesn't have intent
        confidence=result.get("confidence", 0.5),
        product_model=request.product_model,
        sources=sources,
        pipeline_version="legacy",
        language=request.language,
        validation_status=None,
        processing_time_ms=processing_time,
        warnings=[]
    )


@router.post("/diagnose", response_model=ChatResponse)
async def diagnose(
    request: ChatRequest,
    authorization: str = Header(default=None)
):
    """
    Alias for /chat endpoint (backward compatibility).
    """
    return await chat(request, authorization)


@router.get("/pipeline/status")
async def pipeline_status():
    """
    Get current pipeline configuration status.
    """
    flags = get_feature_flags()
    
    return {
        "el_harezmi_rollout": flags.el_harezmi_rollout,
        "force_pipeline": flags.force_pipeline,
        "enable_kg_validation": flags.enable_kg_validation,
        "enable_llm_extraction": flags.enable_llm_extraction,
        "ab_test_enabled": flags.ab_test_enabled,
        "fallback_to_legacy_on_error": flags.fallback_to_legacy_on_error,
        "qdrant_connected": is_qdrant_connected()
    }


@router.post("/pipeline/rollout")
async def set_rollout(
    percentage: float = Query(..., ge=0.0, le=1.0, description="Rollout percentage (0.0 - 1.0)"),
    authorization: str = Header(default=None)
):
    """
    Set El-Harezmi rollout percentage (admin only).
    
    - 0.0 = All traffic to legacy
    - 0.5 = 50/50 split
    - 1.0 = All traffic to El-Harezmi
    """
    # TODO: Add admin authentication check
    
    from config.feature_flags import set_rollout_percentage
    set_rollout_percentage(percentage)
    
    flags = get_feature_flags()
    logger.info(f"Rollout percentage set to {percentage} (was {flags.el_harezmi_rollout})")
    
    return {
        "status": "success",
        "rollout_percentage": flags.el_harezmi_rollout
    }
