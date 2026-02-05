"""
El-Harezmi Pipeline Orchestrator

Coordinates the 5-stage RAG pipeline for Desoutter technical support.

Pipeline Stages:
1. Intent Classification - Multi-label intent detection + entity extraction
2. Retrieval Strategy - Intent-aware Qdrant filtering and boosting
3. Info Extraction - Structured JSON extraction from chunks
4. KG Validation - Compatibility matrix validation
5. Response Formatter - Turkish response templates
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from .stage1_intent_classifier import IntentClassifier, IntentResult, IntentType
from .stage2_retrieval_strategy import RetrievalStrategyManager, RetrievalResult
from .stage3_info_extraction import InfoExtractor, ExtractionResult
from .stage4_kg_validation import KGValidator, ValidationResult, ValidationStatus
from .stage5_response_formatter import ResponseFormatter, FormattedResponse

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Metrics for pipeline execution"""
    total_time_ms: float = 0.0
    stage1_time_ms: float = 0.0
    stage2_time_ms: float = 0.0
    stage3_time_ms: float = 0.0
    stage4_time_ms: float = 0.0
    stage5_time_ms: float = 0.0
    chunks_retrieved: int = 0
    validation_status: str = ""
    final_confidence: float = 0.0


@dataclass
class PipelineResult:
    """Complete result from El-Harezmi pipeline"""
    response: FormattedResponse
    intent_result: IntentResult
    retrieval_result: Optional[RetrievalResult]
    extraction_result: Optional[ExtractionResult]
    validation_result: Optional[ValidationResult]
    metrics: PipelineMetrics
    success: bool = True
    error: Optional[str] = None


class ElHarezmiPipeline:
    """
    El-Harezmi 5-Stage RAG Pipeline Orchestrator
    
    Coordinates intent classification, retrieval, extraction, validation,
    and response formatting for Desoutter technical support queries.
    """
    
    # Intents that don't require retrieval
    NO_RETRIEVAL_INTENTS = [
        IntentType.GREETING,
        IntentType.OFF_TOPIC,
    ]
    
    def __init__(
        self,
        qdrant_client=None,
        embedding_model=None,
        llm_client=None,
        confidence_threshold: float = 0.75,
        auto_init_llm: bool = True
    ):
        """
        Initialize El-Harezmi Pipeline.
        
        Args:
            qdrant_client: Qdrant client for vector search
            embedding_model: Embedding model for query encoding
            llm_client: LLM client for extraction
            confidence_threshold: Minimum confidence for intent classification
            auto_init_llm: Auto-initialize LLM client if not provided
        """
        # Auto-initialize LLM client if needed
        if llm_client is None and auto_init_llm:
            try:
                from .async_llm_client import get_async_llm_client
                llm_client = get_async_llm_client()
                logger.info("Auto-initialized async LLM client")
            except Exception as e:
                logger.warning(f"Failed to auto-initialize LLM: {e}")
        
        # Initialize stages
        self.stage1_classifier = IntentClassifier(confidence_threshold=confidence_threshold)
        self.stage2_retriever = RetrievalStrategyManager(
            qdrant_client=qdrant_client,
            embedding_model=embedding_model
        )
        self.stage3_extractor = InfoExtractor(llm_client=llm_client)
        self.stage4_validator = KGValidator()
        self.stage5_formatter = ResponseFormatter()
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        
        logger.info("El-Harezmi Pipeline initialized")
    
    async def process(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        language = None  # Language enum from stage5
    ) -> PipelineResult:
        """
        Process a user query through the 5-stage pipeline.
        
        Args:
            query: User query string
            context: Optional context (conversation history, user info, etc.)
            language: Response language (Language.TURKISH or Language.ENGLISH)
            
        Returns:
            PipelineResult with response and all intermediate results
        """
        # Set formatter language if provided
        if language:
            self.stage5_formatter.set_language(language)
        
        metrics = PipelineMetrics()
        start_time = time.time()
        
        logger.info(f"Processing query: {query[:50]}...")
        
        try:
            # ============================================
            # Stage 1: Intent Classification
            # ============================================
            stage1_start = time.time()
            intent_result = self.stage1_classifier.classify(query)
            metrics.stage1_time_ms = (time.time() - stage1_start) * 1000
            
            logger.info(f"Stage 1 complete: {intent_result.primary_intent.value} "
                       f"(confidence={intent_result.confidence:.2f})")
            
            # Handle special intents (no retrieval needed)
            if intent_result.primary_intent in self.NO_RETRIEVAL_INTENTS:
                response = self._handle_special_intent(intent_result)
                metrics.total_time_ms = (time.time() - start_time) * 1000
                
                return PipelineResult(
                    response=response,
                    intent_result=intent_result,
                    retrieval_result=None,
                    extraction_result=None,
                    validation_result=None,
                    metrics=metrics,
                )
            
            # ============================================
            # Stage 2: Retrieval Strategy
            # ============================================
            stage2_start = time.time()
            retrieval_result = await self.stage2_retriever.retrieve(
                query=query,
                intent_result=intent_result,
                top_k=20
            )
            metrics.stage2_time_ms = (time.time() - stage2_start) * 1000
            metrics.chunks_retrieved = len(retrieval_result.chunks)
            
            logger.info(f"Stage 2 complete: {len(retrieval_result.chunks)} chunks retrieved")
            
            # Handle no results
            if not retrieval_result.chunks:
                response = self.stage5_formatter.format_no_result(intent_result)
                metrics.total_time_ms = (time.time() - start_time) * 1000
                
                return PipelineResult(
                    response=response,
                    intent_result=intent_result,
                    retrieval_result=retrieval_result,
                    extraction_result=None,
                    validation_result=None,
                    metrics=metrics,
                )
            
            # ============================================
            # Stage 3: Information Extraction
            # ============================================
            stage3_start = time.time()
            extraction_result = await self.stage3_extractor.extract(
                intent_result=intent_result,
                retrieval_result=retrieval_result
            )
            metrics.stage3_time_ms = (time.time() - stage3_start) * 1000
            
            logger.info(f"Stage 3 complete: confidence={extraction_result.confidence:.2f}")
            
            # ============================================
            # Stage 4: Knowledge Graph Validation
            # ============================================
            stage4_start = time.time()
            validation_result = self.stage4_validator.validate(
                extraction_result=extraction_result,
                intent_result=intent_result
            )
            metrics.stage4_time_ms = (time.time() - stage4_start) * 1000
            metrics.validation_status = validation_result.status.value
            
            logger.info(f"Stage 4 complete: {validation_result.status.value}")
            
            # ============================================
            # Stage 5: Response Formatting
            # ============================================
            stage5_start = time.time()
            response = self.stage5_formatter.format(
                intent_result=intent_result,
                extraction_result=extraction_result,
                validation_result=validation_result
            )
            metrics.stage5_time_ms = (time.time() - stage5_start) * 1000
            metrics.final_confidence = response.confidence
            
            logger.info(f"Stage 5 complete: formatted response ready")
            
            # Calculate total time
            metrics.total_time_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Pipeline complete in {metrics.total_time_ms:.0f}ms")
            
            return PipelineResult(
                response=response,
                intent_result=intent_result,
                retrieval_result=retrieval_result,
                extraction_result=extraction_result,
                validation_result=validation_result,
                metrics=metrics,
            )
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            metrics.total_time_ms = (time.time() - start_time) * 1000
            
            # Return error response
            error_response = FormattedResponse(
                content=f"Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.\n\nHata: {str(e)}",
                intent=IntentType.GENERAL,
                product_model=None,
                confidence=0.0,
                sources=[],
                warnings=["İşlem sırasında hata oluştu"],
                validation_status=ValidationStatus.UNKNOWN,
            )
            
            return PipelineResult(
                response=error_response,
                intent_result=IntentResult(
                    primary_intent=IntentType.GENERAL,
                    raw_query=query
                ),
                retrieval_result=None,
                extraction_result=None,
                validation_result=None,
                metrics=metrics,
                success=False,
                error=str(e)
            )
    
    def _handle_special_intent(self, intent_result: IntentResult) -> FormattedResponse:
        """Handle intents that don't require retrieval"""
        
        if intent_result.primary_intent == IntentType.GREETING:
            return self.stage5_formatter.format_greeting(intent_result)
        
        if intent_result.primary_intent == IntentType.OFF_TOPIC:
            return self.stage5_formatter.format_off_topic(intent_result)
        
        # Fallback
        return self.stage5_formatter.format_no_result(intent_result)
    
    def process_sync(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Synchronous wrapper for process method.
        
        Uses asyncio to run the async pipeline.
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.process(query, context))
    
    def check_compatibility(
        self,
        product_model: str,
        controller: str,
        controller_version: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        Quick compatibility check without full pipeline.
        
        Args:
            product_model: Product model (e.g., "EABC-3000")
            controller: Controller model (e.g., "CVI3")
            controller_version: Optional controller version
            
        Returns:
            (is_compatible, message)
        """
        return self.stage4_validator.check_tool_controller_compatibility(
            product_model=product_model,
            controller=controller,
            controller_version=controller_version
        )
    
    def classify_intent(self, query: str) -> IntentResult:
        """
        Quick intent classification without full pipeline.
        
        Useful for routing or logging purposes.
        """
        return self.stage1_classifier.classify(query)
    
    def get_retrieval_strategy(self, intent: IntentType) -> Dict[str, Any]:
        """Get retrieval strategy for an intent type"""
        strategy = self.stage2_retriever.get_strategy(intent)
        return {
            "primary_sources": strategy.primary_sources,
            "secondary_sources": strategy.secondary_sources,
            "boost_factors": strategy.boost_factors,
            "result_limit": strategy.result_limit,
            "score_threshold": strategy.score_threshold,
            "use_knowledge_graph": strategy.use_knowledge_graph,
        }
    
    def get_product_info(self, product_model: str) -> Optional[Dict[str, Any]]:
        """Get known information about a product from knowledge graph"""
        return self.stage4_validator.get_product_info(product_model)
    
    def get_controller_info(self, controller: str) -> Optional[Dict[str, Any]]:
        """Get known information about a controller"""
        return self.stage4_validator.get_controller_info(controller)


# Factory function for easy initialization
def create_pipeline(
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    llm_model_name: str = "qwen2.5:7b-instruct",
    ollama_host: str = "localhost",
    ollama_port: int = 11434
) -> ElHarezmiPipeline:
    """
    Factory function to create El-Harezmi pipeline with default configuration.
    
    Args:
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        embedding_model_name: Sentence transformer model name
        llm_model_name: Ollama model name for LLM
        ollama_host: Ollama server host
        ollama_port: Ollama server port
        
    Returns:
        Configured ElHarezmiPipeline instance
    """
    # Import dependencies
    try:
        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
    except Exception as e:
        logger.warning(f"Failed to connect to Qdrant: {e}")
        qdrant_client = None
    
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(embedding_model_name)
        logger.info(f"Loaded embedding model: {embedding_model_name}")
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}")
        embedding_model = None
    
    # Create simple LLM client wrapper
    class OllamaClient:
        def __init__(self, host: str, port: int, model: str):
            self.base_url = f"http://{host}:{port}"
            self.model = model
        
        async def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2000) -> str:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "temperature": temperature,
                        "stream": False
                    }
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("response", "")
                    else:
                        raise Exception(f"Ollama error: {resp.status}")
    
    try:
        llm_client = OllamaClient(ollama_host, ollama_port, llm_model_name)
        logger.info(f"Configured Ollama client: {llm_model_name}")
    except Exception as e:
        logger.warning(f"Failed to configure Ollama client: {e}")
        llm_client = None
    
    return ElHarezmiPipeline(
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
        llm_client=llm_client
    )
