"""
Base Chunker Interface
======================
Abstract base class for all chunking strategies.
"""
import re
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a document chunk with metadata"""
    text: str
    chunk_index: int
    chunk_type: str  # semantic_section, table_row, error_code, procedure, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional fields
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    section_hierarchy: Optional[str] = None  # "Chapter 4 > Section 4.3"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'text': self.text,
            'chunk_index': self.chunk_index,
            'chunk_type': self.chunk_type,
            'section_hierarchy': self.section_hierarchy,
            'metadata': self.metadata
        }
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        return len(self.text)


class BaseChunker(ABC):
    """
    Abstract base class for document chunkers.
    
    Each chunker implements a specific strategy for splitting documents
    while preserving important structural elements.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        overlap: int = 50
    ):
        """
        Initialize chunker.
        
        Args:
            max_chunk_size: Maximum tokens per chunk
            min_chunk_size: Minimum tokens per chunk (merge smaller)
            overlap: Token overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        self.chunk_type = "generic"
    
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Document text to chunk
            metadata: Optional metadata to include in chunks
            
        Returns:
            List of Chunk objects
        """
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Approximate: 1 token ≈ 4 characters for English
        # Turkish might be slightly different
        return len(text) // 4
    
    def split_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """Split text into chunks of approximately max_tokens"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for word in words:
            word_tokens = len(word) // 4 + 1
            if current_tokens + word_tokens > max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep overlap
                overlap_words = current_chunk[-self.overlap // 4:] if self.overlap else []
                current_chunk = overlap_words
                current_tokens = sum(len(w) // 4 + 1 for w in current_chunk)
            
            current_chunk.append(word)
            current_tokens += word_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge chunks smaller than min_chunk_size"""
        if not chunks:
            return chunks
        
        merged = []
        buffer = None
        
        for chunk in chunks:
            if buffer is None:
                buffer = chunk
            elif self.estimate_tokens(buffer.text) < self.min_chunk_size:
                # Merge with current chunk
                buffer.text = buffer.text + "\n\n" + chunk.text
                # Keep first chunk's metadata, update end position
                if chunk.end_char:
                    buffer.end_char = chunk.end_char
            else:
                merged.append(buffer)
                buffer = chunk
        
        if buffer:
            merged.append(buffer)
        
        # Re-index
        for i, chunk in enumerate(merged):
            chunk.chunk_index = i
        
        return merged
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()
        return text
    
    def extract_section_hierarchy(self, text: str) -> Optional[str]:
        """Extract section hierarchy from text (e.g., Chapter > Section)"""
        # Look for chapter/section patterns
        patterns = [
            r'(Chapter\s+\d+[^>]*)',
            r'(Section\s+\d+\.\d+[^>]*)',
            r'(Bölüm\s+\d+[^>]*)',
            r'(\d+\.\d+\.?\d*\s+[A-Z][^>]*)'
        ]
        
        hierarchy_parts = []
        for pattern in patterns:
            match = re.search(pattern, text[:500], re.IGNORECASE)
            if match:
                hierarchy_parts.append(match.group(1).strip()[:50])
        
        return ' > '.join(hierarchy_parts) if hierarchy_parts else None
