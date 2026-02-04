"""
Hybrid Chunker
==============
Fallback chunker that combines multiple strategies.

Best for:
- Mixed content documents
- Documents with varied structure
- Fallback when type detection fails

Strategy:
- Detects and preserves tables
- Detects and preserves procedures
- Detects error codes
- Uses semantic chunking for prose
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from .base_chunker import BaseChunker, Chunk

logger = logging.getLogger(__name__)


class HybridChunker(BaseChunker):
    """
    Combines multiple chunking strategies based on content detection.
    
    Flow:
    1. Split document into segments by type
    2. Apply appropriate chunker to each segment
    3. Combine results with proper ordering
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 50,
        overlap: int = 50,
        **kwargs
    ):
        super().__init__(max_chunk_size, min_chunk_size, overlap)
        self.chunk_type = "hybrid"
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text using hybrid approach"""
        text = self.clean_text(text)
        metadata = metadata or {}
        
        # Segment document by content type
        segments = self._segment_by_type(text)
        
        # Process each segment with appropriate chunker
        all_chunks = []
        
        for segment in segments:
            segment_chunks = self._chunk_segment(segment, metadata)
            
            # Update chunk indices
            for chunk in segment_chunks:
                chunk.chunk_index = len(all_chunks)
                all_chunks.append(chunk)
        
        # Post-process: merge small chunks
        if all_chunks:
            all_chunks = self.merge_small_chunks(all_chunks)
        
        logger.debug(f"HybridChunker created {len(all_chunks)} chunks from {len(segments)} segments")
        return all_chunks
    
    def _segment_by_type(self, text: str) -> List[Dict[str, Any]]:
        """Split document into typed segments"""
        segments = []
        
        # Find all special segments (tables, procedures, error lists)
        special_regions = []
        
        # Find tables
        table_pattern = r'(\|[^\n]+\|(?:\n\|[^\n]+\|)+)'
        for match in re.finditer(table_pattern, text):
            special_regions.append({
                'start': match.start(),
                'end': match.end(),
                'type': 'table',
                'text': match.group(0)
            })
        
        # Find procedures (numbered steps)
        procedure_pattern = r'((?:^\s*\d+\.\s+.+\n?)+)'
        for match in re.finditer(procedure_pattern, text, re.MULTILINE):
            if len(match.group(0).strip()) > 50:  # Minimum length
                special_regions.append({
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'procedure',
                    'text': match.group(0)
                })
        
        # Find error code blocks
        error_pattern = r'((?:[A-Z]{0,5}[-\s]?E?\d{2,4}\s*[:\-][^\n]+\n?){3,})'
        for match in re.finditer(error_pattern, text):
            special_regions.append({
                'start': match.start(),
                'end': match.end(),
                'type': 'error_codes',
                'text': match.group(0)
            })
        
        # Sort by start position
        special_regions.sort(key=lambda x: x['start'])
        
        # Remove overlapping regions (keep larger)
        filtered_regions = []
        for region in special_regions:
            overlaps = False
            for existing in filtered_regions:
                if (region['start'] < existing['end'] and 
                    region['end'] > existing['start']):
                    # Overlap - keep larger
                    if len(region['text']) > len(existing['text']):
                        filtered_regions.remove(existing)
                    else:
                        overlaps = True
                    break
            if not overlaps:
                filtered_regions.append(region)
        
        # Build segments including prose sections
        pos = 0
        for region in filtered_regions:
            # Add prose before this region
            if region['start'] > pos:
                prose_text = text[pos:region['start']].strip()
                if prose_text:
                    segments.append({
                        'type': 'prose',
                        'text': prose_text
                    })
            
            # Add special region
            segments.append(region)
            pos = region['end']
        
        # Add remaining prose
        if pos < len(text):
            remaining = text[pos:].strip()
            if remaining:
                segments.append({
                    'type': 'prose',
                    'text': remaining
                })
        
        # If no segments created, treat entire text as prose
        if not segments:
            segments.append({
                'type': 'prose',
                'text': text
            })
        
        return segments
    
    def _chunk_segment(
        self,
        segment: Dict[str, Any],
        metadata: Dict
    ) -> List[Chunk]:
        """Chunk a segment with the appropriate chunker"""
        segment_type = segment['type']
        text = segment['text']
        
        if segment_type == 'table':
            from .table_aware_chunker import TableAwareChunker
            chunker = TableAwareChunker(self.max_chunk_size, self.min_chunk_size)
        elif segment_type == 'procedure':
            from .step_preserving_chunker import StepPreservingChunker
            chunker = StepPreservingChunker(self.max_chunk_size, self.min_chunk_size)
        elif segment_type == 'error_codes':
            from .entity_chunker import EntityChunker
            chunker = EntityChunker(self.max_chunk_size, self.min_chunk_size)
        else:  # prose
            from .semantic_chunker import SemanticChunker
            chunker = SemanticChunker(self.max_chunk_size, self.min_chunk_size)
        
        chunks = chunker.chunk(text, metadata)
        
        # Tag chunks with their segment type
        for chunk in chunks:
            chunk.metadata['segment_type'] = segment_type
        
        return chunks
