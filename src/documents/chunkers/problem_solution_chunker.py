"""
Problem-Solution Chunker
========================
Preserves problem-solution pairs from ESDE bulletins.

Best for:
- Service bulletins (ESDE)
- Troubleshooting guides
- FAQ documents

Preserves:
- Problem description + solution together
- Affected models list
- ESDE code relationships
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from .base_chunker import BaseChunker, Chunk

logger = logging.getLogger(__name__)


# ESDE bulletin patterns
ESDE_PATTERNS = {
    'esde_code': r'ESDE[-\s]?\d{4,5}',
    'problem_start': r'(?:Problem|Sorun|Issue|Symptom|Belirtiler)\s*[:\-]',
    'solution_start': r'(?:Solution|Çözüm|Resolution|Fix|Düzeltme)\s*[:\-]',
    'affected_models': r'(?:Affected\s*(?:Models?|Products?)|Etkilenen\s*(?:Model|Ürün)ler?)\s*[:\-]',
    'root_cause': r'(?:Root\s*Cause|Kök\s*Neden|Cause)\s*[:\-]',
}

# Section headers for bulletins
BULLETIN_SECTIONS = [
    'Problem', 'Sorun', 'Issue', 'Symptom',
    'Solution', 'Çözüm', 'Resolution', 'Fix',
    'Affected', 'Etkilenen', 'Root Cause', 'Kök Neden',
    'Background', 'Arka Plan', 'Description', 'Açıklama'
]


class ProblemSolutionChunker(BaseChunker):
    """
    Chunks documents by preserving problem-solution pairs.
    
    Strategy:
    1. Detect ESDE code and bulletin structure
    2. Extract problem + solution as unit
    3. Include affected models as metadata
    4. Never split problem from solution
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1500,  # Larger to keep pairs together
        min_chunk_size: int = 100,
        overlap: int = 0,
        **kwargs
    ):
        super().__init__(max_chunk_size, min_chunk_size, overlap)
        self.chunk_type = "problem_solution_pair"
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into problem-solution chunks"""
        text = self.clean_text(text)
        metadata = metadata or {}
        
        # Extract ESDE code if present
        esde_match = re.search(ESDE_PATTERNS['esde_code'], text)
        esde_code = esde_match.group(0) if esde_match else None
        
        # Try to extract structured problem-solution pairs
        pairs = self._extract_problem_solution_pairs(text)
        
        if pairs:
            chunks = self._create_pair_chunks(pairs, metadata, esde_code)
        else:
            # Fall back to section-based chunking
            chunks = self._chunk_by_sections(text, metadata, esde_code)
        
        logger.debug(f"ProblemSolutionChunker created {len(chunks)} chunks")
        return chunks
    
    def _extract_problem_solution_pairs(self, text: str) -> List[Dict[str, str]]:
        """Extract problem-solution pairs from text"""
        pairs = []
        
        # Find problem sections
        problem_pattern = re.compile(
            ESDE_PATTERNS['problem_start'],
            re.IGNORECASE
        )
        solution_pattern = re.compile(
            ESDE_PATTERNS['solution_start'],
            re.IGNORECASE
        )
        
        # Split text into lines for analysis
        lines = text.split('\n')
        
        current_section = None
        current_content = []
        sections = {}
        
        for line in lines:
            # Check for section headers
            for section in BULLETIN_SECTIONS:
                if re.match(rf'^{section}\s*[:\-]', line, re.IGNORECASE):
                    # Save previous section
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content)
                    
                    current_section = section.lower()
                    # Get content after header
                    rest = re.sub(rf'^{section}\s*[:\-]\s*', '', line, flags=re.IGNORECASE)
                    current_content = [rest] if rest.strip() else []
                    break
            else:
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        # Combine problem and solution
        if sections:
            pair = {}
            
            # Find problem
            for key in ['problem', 'sorun', 'issue', 'symptom']:
                if key in sections:
                    pair['problem'] = sections[key]
                    break
            
            # Find solution
            for key in ['solution', 'çözüm', 'resolution', 'fix']:
                if key in sections:
                    pair['solution'] = sections[key]
                    break
            
            # Find affected models
            for key in ['affected', 'etkilenen']:
                for section_key in sections:
                    if key in section_key:
                        pair['affected_models'] = sections[section_key]
                        break
            
            # Find root cause
            for key in ['root cause', 'kök neden', 'cause']:
                for section_key in sections:
                    if key in section_key:
                        pair['root_cause'] = sections[section_key]
                        break
            
            if 'problem' in pair or 'solution' in pair:
                pairs.append(pair)
        
        return pairs
    
    def _create_pair_chunks(
        self,
        pairs: List[Dict[str, str]],
        metadata: Dict,
        esde_code: Optional[str]
    ) -> List[Chunk]:
        """Create chunks from problem-solution pairs"""
        chunks = []
        
        for idx, pair in enumerate(pairs):
            # Build chunk text
            text_parts = []
            
            if esde_code:
                text_parts.append(f"ESDE: {esde_code}")
            
            if 'problem' in pair:
                text_parts.append(f"Problem:\n{pair['problem'].strip()}")
            
            if 'root_cause' in pair:
                text_parts.append(f"Root Cause:\n{pair['root_cause'].strip()}")
            
            if 'solution' in pair:
                text_parts.append(f"Solution:\n{pair['solution'].strip()}")
            
            chunk_text = '\n\n'.join(text_parts)
            
            # Extract affected models
            affected_models = []
            if 'affected_models' in pair:
                # Parse model names
                models_text = pair['affected_models']
                model_matches = re.findall(
                    r'[A-Z]{2,5}[-\s]?\d{3,4}[A-Z]*',
                    models_text,
                    re.IGNORECASE
                )
                affected_models = list(set(model_matches))
            
            chunk = Chunk(
                text=chunk_text,
                chunk_index=idx,
                chunk_type=self.chunk_type,
                metadata={
                    **metadata,
                    'esde_code': esde_code,
                    'has_problem': 'problem' in pair,
                    'has_solution': 'solution' in pair,
                    'affected_models': affected_models
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_sections(
        self,
        text: str,
        metadata: Dict,
        esde_code: Optional[str]
    ) -> List[Chunk]:
        """Fallback: chunk by sections while trying to keep related content"""
        from .semantic_chunker import SemanticChunker
        
        # Use semantic chunker as fallback
        semantic = SemanticChunker(self.max_chunk_size, self.min_chunk_size)
        base_chunks = semantic.chunk(text, metadata)
        
        # Update chunk type and add ESDE code
        chunks = []
        for chunk in base_chunks:
            chunk.chunk_type = self.chunk_type
            chunk.metadata['esde_code'] = esde_code
            chunks.append(chunk)
        
        return chunks
    
    def _extract_affected_models(self, text: str) -> List[str]:
        """Extract affected model numbers from text"""
        # Common Desoutter model patterns
        patterns = [
            r'[A-Z]{2,5}[-\s]?\d{3,4}[A-Z]*',  # EABC-3000, SPD1200
            r'E[A-Z]{2,3}[-\s]?\d{3,4}',  # EABC, EFD, ERSF series
        ]
        
        models = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            models.extend(matches)
        
        return list(set(models))
