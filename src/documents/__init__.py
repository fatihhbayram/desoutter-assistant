"""Documents processing module"""
from .document_processor import DocumentProcessor, PDFProcessor, SUPPORTED_EXTENSIONS
# from .chunker import TextChunker  # REMOVED: Replaced by SemanticChunker in Phase 1
from .embeddings import EmbeddingsGenerator

__all__ = ['DocumentProcessor', 'PDFProcessor', 'EmbeddingsGenerator', 'SUPPORTED_EXTENSIONS']
