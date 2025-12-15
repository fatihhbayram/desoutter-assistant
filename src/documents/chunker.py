"""
Text Chunking Module
Split documents into semantic chunks for embedding
"""
from typing import List, Dict
import re
from config.ai_settings import CHUNK_SIZE, CHUNK_OVERLAP
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TextChunker:
    """Split text into semantic chunks"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize chunker
        
        Args:
            chunk_size: Target size of each chunk (in tokens/words)
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        Split text into chunks by sentences
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Keep last few sentences for overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_size = sum(len(s.split()) for s in overlap_sentences)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with NLTK/spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap"""
        overlap_size = 0
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            sentence_size = len(sentence.split())
            if overlap_size + sentence_size > self.chunk_overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_size += sentence_size
        
        return overlap_sentences
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Chunk a document and create chunks with metadata
        
        Args:
            document: Document dict from PDFProcessor
            
        Returns:
            List of chunk dicts with metadata
        """
        text = document.get("text", "")
        if not text:
            return []
        
        # Split into chunks
        text_chunks = self.chunk_by_sentences(text)
        
        # Create chunk objects with metadata
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                "text": chunk_text,
                "chunk_id": f"{document['filename']}_chunk_{i}",
                "chunk_index": i,
                "source": document["filename"],
                "metadata": {
                    **document.get("metadata", {}),
                    "chunk_index": i,
                    "total_chunks": len(text_chunks)
                }
            }
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {document['filename']}")
        return chunks
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of document dicts
            
        Returns:
            List of all chunks
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
