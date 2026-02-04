"""
Chunker Factory
===============
Factory pattern for selecting appropriate chunker based on document type.

Usage:
    from src.documents.document_classifier import DocumentClassifier, DocumentType
    from src.documents.chunkers import ChunkerFactory
    
    classifier = DocumentClassifier()
    result = classifier.classify(text, filename)
    
    chunker = ChunkerFactory.get_chunker(result.document_type)
    chunks = chunker.chunk(text)
"""
import logging
from typing import Dict, Type, Optional, Any

from .base_chunker import BaseChunker, Chunk
from .semantic_chunker import SemanticChunker
from .table_aware_chunker import TableAwareChunker
from .entity_chunker import EntityChunker
from .problem_solution_chunker import ProblemSolutionChunker
from .step_preserving_chunker import StepPreservingChunker
from .hybrid_chunker import HybridChunker

# Import DocumentType enum
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from src.documents.document_classifier import DocumentType
except ImportError:
    # Fallback enum if import fails
    from enum import Enum
    class DocumentType(str, Enum):
        TECHNICAL_MANUAL = "technical_manual"
        SERVICE_BULLETIN = "service_bulletin"
        CONFIGURATION_GUIDE = "configuration_guide"
        COMPATIBILITY_MATRIX = "compatibility_matrix"
        SPEC_SHEET = "spec_sheet"
        ERROR_CODE_LIST = "error_code_list"
        PROCEDURE_GUIDE = "procedure_guide"
        FRESHDESK_TICKET = "freshdesk_ticket"

logger = logging.getLogger(__name__)


# =============================================================================
# CHUNKER CONFIGURATION
# =============================================================================

CHUNKER_CONFIG: Dict[str, Dict[str, Any]] = {
    DocumentType.TECHNICAL_MANUAL.value: {
        "chunker_class": SemanticChunker,
        "max_chunk_size": 1000,
        "min_chunk_size": 100,
        "split_by": "section_headers",
    },
    
    DocumentType.SERVICE_BULLETIN.value: {
        "chunker_class": ProblemSolutionChunker,
        "max_chunk_size": 1500,
        "min_chunk_size": 200,
        "preserve_esde_blocks": True,
    },
    
    DocumentType.CONFIGURATION_GUIDE.value: {
        "chunker_class": SemanticChunker,
        "max_chunk_size": 1000,
        "min_chunk_size": 100,
        "preserve_procedures": True,
    },
    
    DocumentType.COMPATIBILITY_MATRIX.value: {
        "chunker_class": TableAwareChunker,
        "max_chunk_size": 1500,
        "min_chunk_size": 50,
        "preserve_headers": True,
        "chunk_by_row": True,
    },
    
    DocumentType.SPEC_SHEET.value: {
        "chunker_class": SemanticChunker,
        "max_chunk_size": 500,
        "min_chunk_size": 50,
        "split_by": "section_headers",
    },
    
    DocumentType.ERROR_CODE_LIST.value: {
        "chunker_class": EntityChunker,
        "max_chunk_size": 800,
        "min_chunk_size": 50,
        "entity_type": "error_code",
    },
    
    DocumentType.PROCEDURE_GUIDE.value: {
        "chunker_class": StepPreservingChunker,
        "max_chunk_size": 1500,
        "min_chunk_size": 100,
        "preserve_steps": True,
    },
    
    DocumentType.FRESHDESK_TICKET.value: {
        "chunker_class": ProblemSolutionChunker,
        "max_chunk_size": 1000,
        "min_chunk_size": 100,
        "preserve_qa_format": True,
    },
}

# Default configuration for unknown document types
DEFAULT_CONFIG = {
    "chunker_class": HybridChunker,
    "max_chunk_size": 1000,
    "min_chunk_size": 100,
}


class ChunkerFactory:
    """
    Factory for creating document-type specific chunkers.
    
    Selects the appropriate chunking strategy based on document type
    to preserve important structural elements during retrieval.
    """
    
    _instances: Dict[str, BaseChunker] = {}
    
    @classmethod
    def get_chunker(
        cls,
        document_type: DocumentType,
        use_cache: bool = True,
        **override_config
    ) -> BaseChunker:
        """
        Get chunker for document type.
        
        Args:
            document_type: Type of document to chunk
            use_cache: Whether to reuse cached instances
            **override_config: Override default configuration
            
        Returns:
            Appropriate BaseChunker instance
        """
        doc_type_str = document_type.value if hasattr(document_type, 'value') else str(document_type)
        
        # Check cache
        cache_key = f"{doc_type_str}_{hash(frozenset(override_config.items()))}"
        if use_cache and cache_key in cls._instances:
            return cls._instances[cache_key]
        
        # Get configuration
        config = CHUNKER_CONFIG.get(doc_type_str, DEFAULT_CONFIG).copy()
        config.update(override_config)
        
        # Extract chunker class
        chunker_class = config.pop('chunker_class', HybridChunker)
        
        # Create instance
        try:
            chunker = chunker_class(**config)
            logger.debug(f"Created {chunker_class.__name__} for {doc_type_str}")
        except Exception as e:
            logger.warning(f"Failed to create {chunker_class.__name__}: {e}, using HybridChunker")
            chunker = HybridChunker()
        
        # Cache
        if use_cache:
            cls._instances[cache_key] = chunker
        
        return chunker
    
    @classmethod
    def get_chunker_for_file(
        cls,
        text: str,
        filename: str,
        classifier=None
    ) -> BaseChunker:
        """
        Get chunker by classifying the file first.
        
        Args:
            text: Document text
            filename: Document filename
            classifier: Optional DocumentClassifier instance
            
        Returns:
            Appropriate BaseChunker instance
        """
        if classifier is None:
            from src.documents.document_classifier import DocumentClassifier
            classifier = DocumentClassifier()
        
        # Classify document
        result = classifier.classify(text, filename)
        logger.info(f"Classified '{filename}' as {result.document_type.value} "
                   f"(confidence: {result.confidence:.2f})")
        
        return cls.get_chunker(result.document_type)
    
    @classmethod
    def clear_cache(cls):
        """Clear cached chunker instances"""
        cls._instances.clear()
    
    @classmethod
    def list_chunkers(cls) -> Dict[str, str]:
        """List available chunkers for each document type"""
        return {
            doc_type: config['chunker_class'].__name__
            for doc_type, config in CHUNKER_CONFIG.items()
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def chunk_document(
    text: str,
    filename: str,
    document_type: Optional[DocumentType] = None,
    classifier=None
) -> list:
    """
    Convenience function to chunk a document.
    
    Args:
        text: Document text
        filename: Document filename
        document_type: Optional explicit document type
        classifier: Optional DocumentClassifier instance
        
    Returns:
        List of Chunk objects
    """
    if document_type:
        chunker = ChunkerFactory.get_chunker(document_type)
    else:
        chunker = ChunkerFactory.get_chunker_for_file(text, filename, classifier)
    
    return chunker.chunk(text, metadata={'source': filename})


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("CHUNKER FACTORY TEST")
    print("=" * 60)
    
    # List available chunkers
    print("\nAvailable chunkers:")
    for doc_type, chunker_name in ChunkerFactory.list_chunkers().items():
        print(f"  {doc_type}: {chunker_name}")
    
    # Test getting chunkers
    print("\nTesting chunker creation:")
    for doc_type in DocumentType:
        chunker = ChunkerFactory.get_chunker(doc_type)
        print(f"  {doc_type.value}: {chunker.__class__.__name__}")
