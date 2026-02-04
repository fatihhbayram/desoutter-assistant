"""
Table-Aware Chunker
===================
Preserves table structure during chunking for compatibility matrices and changelogs.

Best for:
- Compatibility matrices
- CHANGELOG files
- Version tables
- Spec tables

Preserves:
- Table headers with each row
- Row integrity (no mid-row splits)
- Table context
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from .base_chunker import BaseChunker, Chunk

logger = logging.getLogger(__name__)


class TableAwareChunker(BaseChunker):
    """
    Chunks documents while preserving table structure.
    
    Strategy:
    1. Detect tables in text
    2. Extract table headers
    3. Create chunks with header + row(s)
    4. Preserve non-table content separately
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 50,
        overlap: int = 0,  # No overlap for tables
        preserve_headers: bool = True,
        chunk_by_row: bool = True,
        rows_per_chunk: int = 5,
        **kwargs
    ):
        super().__init__(max_chunk_size, min_chunk_size, overlap)
        self.chunk_type = "table_row"
        self.preserve_headers = preserve_headers
        self.chunk_by_row = chunk_by_row
        self.rows_per_chunk = rows_per_chunk
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text while preserving tables"""
        text = self.clean_text(text)
        metadata = metadata or {}
        
        chunks = []
        
        # Split into table and non-table segments
        segments = self._split_into_segments(text)
        
        for segment_type, segment_text in segments:
            if segment_type == 'table':
                # Process table
                table_chunks = self._chunk_table(segment_text, metadata)
                chunks.extend(table_chunks)
            else:
                # Process non-table text
                if segment_text.strip():
                    text_chunks = self._chunk_text(segment_text, metadata)
                    chunks.extend(text_chunks)
        
        # Re-index chunks
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
        
        logger.debug(f"TableAwareChunker created {len(chunks)} chunks")
        return chunks
    
    def _split_into_segments(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text into table and non-table segments.
        
        Returns:
            List of (segment_type, segment_text) tuples
        """
        segments = []
        lines = text.split('\n')
        
        current_segment = []
        current_type = 'text'
        
        for line in lines:
            line_type = self._detect_line_type(line)
            
            if line_type == 'table' and current_type != 'table':
                # Switch to table
                if current_segment:
                    segments.append((current_type, '\n'.join(current_segment)))
                current_segment = [line]
                current_type = 'table'
            elif line_type != 'table' and current_type == 'table':
                # Switch from table
                if current_segment:
                    segments.append((current_type, '\n'.join(current_segment)))
                current_segment = [line]
                current_type = 'text'
            else:
                current_segment.append(line)
        
        if current_segment:
            segments.append((current_type, '\n'.join(current_segment)))
        
        return segments
    
    def _detect_line_type(self, line: str) -> str:
        """Detect if line is part of a table"""
        line = line.strip()
        
        # Pipe-delimited table
        if '|' in line and line.count('|') >= 2:
            return 'table'
        
        # Tab-delimited (common in Word exports)
        if '\t' in line and line.count('\t') >= 2:
            return 'table'
        
        # Version patterns (common in changelogs)
        if re.match(r'^[CAFV]?\d+\.\d+\.\d+', line):
            return 'table'
        
        # Row with multiple columns separated by whitespace
        parts = re.split(r'\s{2,}', line)
        if len(parts) >= 3 and all(len(p) < 50 for p in parts):
            return 'table'
        
        return 'text'
    
    def _chunk_table(self, table_text: str, metadata: Dict) -> List[Chunk]:
        """
        Chunk a table while preserving headers.
        
        Each chunk contains:
        - Table header(s)
        - N rows of data
        - Context (table title if available)
        """
        lines = table_text.strip().split('\n')
        
        if not lines:
            return []
        
        # Detect header rows (usually first 1-2 rows)
        header_rows = self._detect_header_rows(lines)
        header_text = '\n'.join(header_rows)
        
        # Get data rows
        data_start = len(header_rows)
        data_rows = lines[data_start:]
        
        # Filter out separator rows (------|------|------)
        data_rows = [r for r in data_rows if not re.match(r'^[\s\-\|]+$', r)]
        
        if not data_rows:
            # Table has only headers
            return [Chunk(
                text=header_text,
                chunk_index=0,
                chunk_type='table_header',
                metadata={**metadata, 'is_table': True}
            )]
        
        chunks = []
        
        if self.chunk_by_row:
            # Create chunk for each row (or group of rows)
            for i in range(0, len(data_rows), self.rows_per_chunk):
                row_group = data_rows[i:i + self.rows_per_chunk]
                
                # Combine header + rows
                if self.preserve_headers:
                    chunk_text = header_text + '\n' + '\n'.join(row_group)
                else:
                    chunk_text = '\n'.join(row_group)
                
                chunk = Chunk(
                    text=chunk_text,
                    chunk_index=len(chunks),
                    chunk_type=self.chunk_type,
                    metadata={
                        **metadata,
                        'is_table': True,
                        'row_start': i,
                        'row_end': min(i + self.rows_per_chunk, len(data_rows)),
                        'total_rows': len(data_rows)
                    }
                )
                chunks.append(chunk)
        else:
            # Keep entire table as one chunk (if not too large)
            full_table = header_text + '\n' + '\n'.join(data_rows)
            
            if self.estimate_tokens(full_table) <= self.max_chunk_size:
                chunks.append(Chunk(
                    text=full_table,
                    chunk_index=0,
                    chunk_type='table_complete',
                    metadata={
                        **metadata,
                        'is_table': True,
                        'total_rows': len(data_rows)
                    }
                ))
            else:
                # Fall back to row-by-row chunking
                return self._chunk_table_by_rows(header_text, data_rows, metadata)
        
        return chunks
    
    def _detect_header_rows(self, lines: List[str]) -> List[str]:
        """Detect table header rows"""
        if not lines:
            return []
        
        headers = []
        
        for i, line in enumerate(lines[:3]):  # Check first 3 rows
            # Header indicators:
            # - Contains column names (Product, Version, etc.)
            # - Followed by separator row
            # - ALL CAPS or Title Case
            
            is_header = False
            
            # Check for header keywords
            header_keywords = [
                'product', 'version', 'release', 'tool', 'controller',
                'firmware', 'software', 'date', 'features', 'model',
                'ürün', 'sürüm', 'tarih', 'özellik'
            ]
            line_lower = line.lower()
            if any(kw in line_lower for kw in header_keywords):
                is_header = True
            
            # Check if next line is separator
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if re.match(r'^[\s\-\|\+:]+$', next_line):
                    is_header = True
            
            if is_header:
                headers.append(line)
                # Also include separator if present
                if i + 1 < len(lines) and re.match(r'^[\s\-\|\+:]+$', lines[i + 1]):
                    headers.append(lines[i + 1])
            elif not headers:
                # First row is likely header even if not detected
                headers.append(line)
                break
            else:
                break
        
        return headers if headers else [lines[0]] if lines else []
    
    def _chunk_table_by_rows(
        self,
        header_text: str,
        data_rows: List[str],
        metadata: Dict
    ) -> List[Chunk]:
        """Chunk table row by row with headers"""
        chunks = []
        
        for i, row in enumerate(data_rows):
            chunk_text = f"{header_text}\n{row}"
            
            chunk = Chunk(
                text=chunk_text,
                chunk_index=i,
                chunk_type=self.chunk_type,
                metadata={
                    **metadata,
                    'is_table': True,
                    'row_index': i,
                    'total_rows': len(data_rows)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_text(self, text: str, metadata: Dict) -> List[Chunk]:
        """Chunk non-table text"""
        from .semantic_chunker import SemanticChunker
        
        semantic_chunker = SemanticChunker(
            max_chunk_size=self.max_chunk_size,
            min_chunk_size=self.min_chunk_size
        )
        
        return semantic_chunker.chunk(text, metadata)
