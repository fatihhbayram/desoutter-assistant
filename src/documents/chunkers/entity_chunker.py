"""
Entity Chunker
==============
Groups entities (error codes, parameters) with their descriptions.

Best for:
- Error code lists
- Parameter definitions
- Glossaries

Preserves:
- Entity code + description pairs
- Related entities grouped together
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from .base_chunker import BaseChunker, Chunk

logger = logging.getLogger(__name__)


# Error code patterns
ERROR_CODE_PATTERNS = [
    # Standard error codes: E01, E012, E0123
    r'(E\d{2,4})\s*[:\-–]\s*(.+?)(?=\n(?:E\d{2,4}|$))',
    # Product-prefixed: EABC-001, SPD-E06, CVI-E12
    r'([A-Z]{2,5}[-\s]?E?\d{2,4})\s*[:\-–]\s*(.+?)(?=\n(?:[A-Z]{2,5}[-\s]?E?\d{2,4}|$))',
    # With description header: "Error E01: Description"
    r'(?:Error|Hata|Fault|Arıza)\s+([A-Z]?\d{2,4})\s*[:\-–]\s*(.+?)(?=\n(?:Error|Hata|Fault|Arıza|\Z))',
]


class EntityChunker(BaseChunker):
    """
    Chunks documents by grouping entities with descriptions.
    
    Strategy:
    1. Detect entity patterns (error codes, parameters)
    2. Extract entity + description pairs
    3. Group related entities
    4. Ensure code + description stay together
    """
    
    def __init__(
        self,
        max_chunk_size: int = 800,
        min_chunk_size: int = 50,
        overlap: int = 0,
        entity_type: str = "error_code",
        entities_per_chunk: int = 3,
        **kwargs
    ):
        super().__init__(max_chunk_size, min_chunk_size, overlap)
        self.chunk_type = entity_type
        self.entity_type = entity_type
        self.entities_per_chunk = entities_per_chunk
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into entity-based chunks"""
        text = self.clean_text(text)
        metadata = metadata or {}
        
        # Extract entities
        entities = self._extract_entities(text)
        
        if not entities:
            # No entities found, fall back to semantic chunking
            logger.debug("No entities found, using semantic chunking")
            from .semantic_chunker import SemanticChunker
            return SemanticChunker(self.max_chunk_size, self.min_chunk_size).chunk(text, metadata)
        
        # Group entities into chunks
        chunks = self._group_entities(entities, metadata)
        
        logger.debug(f"EntityChunker created {len(chunks)} chunks from {len(entities)} entities")
        return chunks
    
    def _extract_entities(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract entity-description pairs.
        
        Returns:
            List of (entity_code, description, full_text) tuples
        """
        entities = []
        
        # Try each pattern
        for pattern in ERROR_CODE_PATTERNS:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    code = match[0].strip()
                    description = match[1].strip()
                    full_text = f"{code}: {description}"
                    
                    # Avoid duplicates
                    if not any(e[0] == code for e in entities):
                        entities.append((code, description, full_text))
        
        # If no regex matches, try line-by-line detection
        if not entities:
            entities = self._extract_by_lines(text)
        
        return entities
    
    def _extract_by_lines(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract entities by analyzing lines"""
        entities = []
        lines = text.split('\n')
        
        current_code = None
        current_desc = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with error code
            code_match = re.match(r'^([A-Z]{0,5}[-\s]?E?\d{2,4})\s*[:\-–]?\s*(.*)$', line)
            
            if code_match:
                # Save previous entity
                if current_code and current_desc:
                    desc = ' '.join(current_desc)
                    entities.append((current_code, desc, f"{current_code}: {desc}"))
                
                # Start new entity
                current_code = code_match.group(1)
                rest = code_match.group(2).strip()
                current_desc = [rest] if rest else []
            elif current_code:
                # Continue description
                current_desc.append(line)
        
        # Save last entity
        if current_code and current_desc:
            desc = ' '.join(current_desc)
            entities.append((current_code, desc, f"{current_code}: {desc}"))
        
        return entities
    
    def _group_entities(
        self,
        entities: List[Tuple[str, str, str]],
        metadata: Dict
    ) -> List[Chunk]:
        """Group entities into chunks"""
        chunks = []
        
        for i in range(0, len(entities), self.entities_per_chunk):
            group = entities[i:i + self.entities_per_chunk]
            
            # Combine entity texts
            chunk_text = '\n\n'.join(e[2] for e in group)
            
            # Extract codes for metadata
            codes = [e[0] for e in group]
            
            chunk = Chunk(
                text=chunk_text,
                chunk_index=len(chunks),
                chunk_type=self.chunk_type,
                metadata={
                    **metadata,
                    'entity_type': self.entity_type,
                    'entity_codes': codes,
                    'entity_count': len(group)
                }
            )
            chunks.append(chunk)
        
        return chunks
