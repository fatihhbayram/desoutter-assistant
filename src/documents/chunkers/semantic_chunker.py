"""
Semantic Chunker
================
Chunks documents by section headers while preserving semantic coherence.

Best for:
- Configuration guides
- Technical manuals
- Spec sheets

Preserves:
- Section hierarchy
- Paragraph boundaries
- Code blocks
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from .base_chunker import BaseChunker, Chunk

logger = logging.getLogger(__name__)


# Section header patterns (ordered by specificity)
SECTION_PATTERNS = [
    # Numbered sections: 1.2.3 Title
    (r'^(\d+\.)+\d*\s+[A-ZÇĞİÖŞÜ]', 'numbered_section'),
    # Chapter/Section keywords
    (r'^(Chapter|Section|Bölüm|Kısım)\s+\d+', 'chapter_section'),
    # ALL CAPS headers
    (r'^[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ\s]{5,50}$', 'caps_header'),
    # Title Case headers (at least 3 words)
    (r'^[A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜa-zçğıöşü]+){2,}$', 'title_header'),
    # Markdown-style headers
    (r'^#{1,6}\s+', 'markdown_header'),
]


class SemanticChunker(BaseChunker):
    """
    Chunks documents by semantic sections.
    
    Strategy:
    1. Detect section headers
    2. Split at section boundaries
    3. Merge small sections
    4. Preserve code blocks and tables
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        overlap: int = 50,
        split_by: str = "section_headers",
        preserve_procedures: bool = True,
        **kwargs
    ):
        super().__init__(max_chunk_size, min_chunk_size, overlap)
        self.chunk_type = "semantic_section"
        self.split_by = split_by
        self.preserve_procedures = preserve_procedures
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into semantic chunks"""
        text = self.clean_text(text)
        metadata = metadata or {}
        
        # Find section boundaries
        sections = self._split_by_sections(text)
        
        # Convert sections to chunks
        chunks = []
        for i, (section_text, section_type, hierarchy) in enumerate(sections):
            # Check if section is too large
            if self.estimate_tokens(section_text) > self.max_chunk_size:
                # Split further by paragraphs
                sub_chunks = self._split_large_section(section_text, hierarchy)
                for j, sub_text in enumerate(sub_chunks):
                    chunk = Chunk(
                        text=sub_text,
                        chunk_index=len(chunks),
                        chunk_type=self.chunk_type,
                        section_hierarchy=hierarchy,
                        metadata={
                            **metadata,
                            'section_type': section_type,
                            'is_subsection': True
                        }
                    )
                    chunks.append(chunk)
            else:
                chunk = Chunk(
                    text=section_text,
                    chunk_index=len(chunks),
                    chunk_type=self.chunk_type,
                    section_hierarchy=hierarchy,
                    metadata={
                        **metadata,
                        'section_type': section_type
                    }
                )
                chunks.append(chunk)
        
        # Merge small chunks
        chunks = self.merge_small_chunks(chunks)
        
        logger.debug(f"SemanticChunker created {len(chunks)} chunks")
        return chunks
    
    def _split_by_sections(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Split text by section headers.
        
        Returns:
            List of (section_text, section_type, hierarchy) tuples
        """
        lines = text.split('\n')
        sections = []
        current_section = []
        current_type = 'body'
        current_hierarchy = []
        
        for line in lines:
            # Check if line is a section header
            header_match = self._is_section_header(line)
            
            if header_match:
                # Save current section
                if current_section:
                    section_text = '\n'.join(current_section).strip()
                    if section_text:
                        hierarchy_str = ' > '.join(current_hierarchy) if current_hierarchy else None
                        sections.append((section_text, current_type, hierarchy_str))
                
                # Start new section
                current_section = [line]
                current_type = header_match[1]
                
                # Update hierarchy
                header_level = self._get_header_level(line)
                header_text = self._clean_header(line)
                
                # Trim hierarchy to current level
                current_hierarchy = current_hierarchy[:header_level]
                if len(current_hierarchy) < header_level + 1:
                    current_hierarchy.append(header_text)
                else:
                    current_hierarchy[header_level] = header_text
            else:
                current_section.append(line)
        
        # Save last section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                hierarchy_str = ' > '.join(current_hierarchy) if current_hierarchy else None
                sections.append((section_text, current_type, hierarchy_str))
        
        return sections
    
    def _is_section_header(self, line: str) -> Optional[Tuple[str, str]]:
        """Check if line is a section header"""
        line = line.strip()
        if not line or len(line) > 100:
            return None
        
        for pattern, header_type in SECTION_PATTERNS:
            if re.match(pattern, line, re.MULTILINE):
                return (line, header_type)
        
        return None
    
    def _get_header_level(self, line: str) -> int:
        """Determine header hierarchy level (0 = top)"""
        line = line.strip()
        
        # Markdown style
        if line.startswith('#'):
            return line.count('#') - 1
        
        # Numbered sections
        match = re.match(r'^(\d+\.)+', line)
        if match:
            return match.group().count('.') - 1
        
        # Chapter/Section
        if re.match(r'^Chapter\s+\d+', line, re.IGNORECASE):
            return 0
        if re.match(r'^Section\s+\d+', line, re.IGNORECASE):
            return 1
        
        # Default
        return 0
    
    def _clean_header(self, line: str) -> str:
        """Clean header text for hierarchy"""
        line = line.strip()
        # Remove markdown #
        line = re.sub(r'^#+\s*', '', line)
        # Truncate
        return line[:50]
    
    def _split_large_section(self, text: str, hierarchy: Optional[str]) -> List[str]:
        """Split large section by paragraphs"""
        paragraphs = re.split(r'\n\n+', text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.estimate_tokens(para)
            
            # Check for procedure blocks (preserve together)
            if self.preserve_procedures and self._is_procedure_block(para):
                # Save current chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Add procedure as its own chunk
                chunks.append(para)
                continue
            
            if current_tokens + para_tokens > self.max_chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(para)
            current_tokens += para_tokens
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _is_procedure_block(self, text: str) -> bool:
        """Check if text is a procedure/steps block"""
        # Look for numbered steps
        step_count = len(re.findall(r'^\s*\d+[\.\)]\s+', text, re.MULTILINE))
        return step_count >= 3
