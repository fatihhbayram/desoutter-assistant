"""
Step Preserving Chunker
=======================
Keeps procedure steps together without splitting.

Best for:
- Installation guides
- Procedure guides
- Firmware update instructions
- Calibration procedures

Preserves:
- Numbered steps
- Prerequisites
- Warnings
- Step sequences
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from .base_chunker import BaseChunker, Chunk

logger = logging.getLogger(__name__)


# Step detection patterns
STEP_PATTERNS = [
    r'^\s*(\d+)\.\s+',  # "1. Do this"
    r'^\s*Step\s+(\d+)\s*[:\-]?\s*',  # "Step 1:"
    r'^\s*Adım\s+(\d+)\s*[:\-]?\s*',  # "Adım 1:" (Turkish)
    r'^\s*\((\d+)\)\s+',  # "(1) Do this"
    r'^\s*(\d+)\)\s+',  # "1) Do this"
]

# Warning patterns
WARNING_PATTERNS = [
    r'(?:WARNING|UYARI|CAUTION|DİKKAT|NOTE|NOT|IMPORTANT|ÖNEMLİ)\s*[:\-]',
]

# Prerequisite patterns
PREREQUISITE_PATTERNS = [
    r'(?:Prerequisites?|Ön\s*Koşullar?|Requirements?|Gereksinimler|Before\s+(?:you\s+)?begin)\s*[:\-]',
]


class StepPreservingChunker(BaseChunker):
    """
    Chunks documents while preserving step sequences.
    
    Strategy:
    1. Detect numbered step patterns
    2. Group related steps together
    3. Keep prerequisites with first steps
    4. Preserve warnings with their steps
    5. Never split in middle of a step
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1500,  # Larger to keep steps together
        min_chunk_size: int = 100,
        overlap: int = 0,
        steps_per_chunk: int = 5,  # Max steps per chunk
        **kwargs
    ):
        super().__init__(max_chunk_size, min_chunk_size, overlap)
        self.chunk_type = "procedure"
        self.steps_per_chunk = steps_per_chunk
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into step-based chunks"""
        text = self.clean_text(text)
        metadata = metadata or {}
        
        # Extract prerequisites if present
        prerequisites, remaining_text = self._extract_prerequisites(text)
        
        # Extract procedure steps
        steps = self._extract_steps(remaining_text)
        
        if not steps:
            # No steps found, fall back to semantic chunking
            logger.debug("No steps found, using semantic chunking")
            from .semantic_chunker import SemanticChunker
            return SemanticChunker(self.max_chunk_size, self.min_chunk_size).chunk(text, metadata)
        
        # Group steps into chunks
        chunks = self._group_steps(steps, prerequisites, metadata)
        
        logger.debug(f"StepPreservingChunker created {len(chunks)} chunks from {len(steps)} steps")
        return chunks
    
    def _extract_prerequisites(self, text: str) -> Tuple[Optional[str], str]:
        """Extract prerequisites section from text"""
        for pattern in PREREQUISITE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start = match.start()
                # Find end of prerequisites (next major section or steps)
                end = len(text)
                
                for step_pattern in STEP_PATTERNS:
                    step_match = re.search(step_pattern, text[match.end():], re.MULTILINE)
                    if step_match:
                        end = match.end() + step_match.start()
                        break
                
                prerequisites = text[start:end].strip()
                remaining = text[end:].strip()
                
                return prerequisites, remaining
        
        return None, text
    
    def _extract_steps(self, text: str) -> List[Dict[str, Any]]:
        """Extract numbered steps from text"""
        steps = []
        
        # Find all step starts
        step_positions = []
        for pattern in STEP_PATTERNS:
            for match in re.finditer(pattern, text, re.MULTILINE):
                step_num = int(match.group(1))
                step_positions.append({
                    'start': match.start(),
                    'number': step_num,
                    'header_end': match.end()
                })
        
        # Sort by position
        step_positions.sort(key=lambda x: x['start'])
        
        # Remove duplicates at same position
        unique_positions = []
        seen = set()
        for pos in step_positions:
            if pos['start'] not in seen:
                unique_positions.append(pos)
                seen.add(pos['start'])
        
        # Extract step content
        for i, pos in enumerate(unique_positions):
            # Determine end of step
            if i + 1 < len(unique_positions):
                end = unique_positions[i + 1]['start']
            else:
                end = len(text)
            
            step_text = text[pos['start']:end].strip()
            
            # Check for warnings in step
            has_warning = any(
                re.search(pattern, step_text, re.IGNORECASE)
                for pattern in WARNING_PATTERNS
            )
            
            steps.append({
                'number': pos['number'],
                'text': step_text,
                'has_warning': has_warning
            })
        
        return steps
    
    def _group_steps(
        self,
        steps: List[Dict[str, Any]],
        prerequisites: Optional[str],
        metadata: Dict
    ) -> List[Chunk]:
        """Group steps into chunks"""
        chunks = []
        current_steps = []
        current_text_parts = []
        
        # Add prerequisites to first chunk
        if prerequisites:
            current_text_parts.append(prerequisites)
        
        for step in steps:
            step_tokens = self.estimate_tokens(step['text'])
            current_tokens = sum(self.estimate_tokens(t) for t in current_text_parts)
            
            # Check if adding step exceeds limits
            should_create_chunk = (
                (current_tokens + step_tokens > self.max_chunk_size and current_steps) or
                len(current_steps) >= self.steps_per_chunk
            )
            
            if should_create_chunk:
                # Create chunk from current steps
                chunk = self._create_step_chunk(
                    current_steps,
                    current_text_parts,
                    len(chunks),
                    metadata,
                    include_prerequisites=(len(chunks) == 0 and prerequisites)
                )
                chunks.append(chunk)
                
                # Reset
                current_steps = []
                current_text_parts = []
            
            # Add step
            current_steps.append(step)
            current_text_parts.append(step['text'])
        
        # Create final chunk
        if current_steps:
            chunk = self._create_step_chunk(
                current_steps,
                current_text_parts,
                len(chunks),
                metadata,
                include_prerequisites=(len(chunks) == 0 and prerequisites)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_step_chunk(
        self,
        steps: List[Dict[str, Any]],
        text_parts: List[str],
        chunk_index: int,
        metadata: Dict,
        include_prerequisites: bool = False
    ) -> Chunk:
        """Create a chunk from steps"""
        chunk_text = '\n\n'.join(text_parts)
        
        step_numbers = [s['number'] for s in steps]
        has_warnings = any(s['has_warning'] for s in steps)
        
        return Chunk(
            text=chunk_text,
            chunk_index=chunk_index,
            chunk_type=self.chunk_type,
            metadata={
                **metadata,
                'step_numbers': step_numbers,
                'step_range': f"{min(step_numbers)}-{max(step_numbers)}" if step_numbers else None,
                'has_warnings': has_warnings,
                'has_prerequisites': include_prerequisites,
                'contains_procedure': True
            }
        )
