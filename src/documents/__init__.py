"""Documents processing module"""
from .document_processor import DocumentProcessor, PDFProcessor, SUPPORTED_EXTENSIONS
# from .chunker import TextChunker  # REMOVED: Replaced by SemanticChunker in Phase 1

# Lazy import to avoid torch dependency when only using chunkers
def get_embeddings_generator():
    """Lazy import of EmbeddingsGenerator to avoid torch dependency"""
    from .embeddings import EmbeddingsGenerator
    return EmbeddingsGenerator

__all__ = ['DocumentProcessor', 'PDFProcessor', 'get_embeddings_generator', 'SUPPORTED_EXTENSIONS']
