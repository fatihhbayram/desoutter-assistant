"""
Semantic Chunking Module
Splits documents at semantic boundaries while preserving structure and context
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentType(Enum):
    """Document type classification"""
    TECHNICAL_MANUAL = "technical_manual"
    SERVICE_BULLETIN = "service_bulletin"
    TROUBLESHOOTING_GUIDE = "troubleshooting_guide"
    PARTS_CATALOG = "parts_catalog"
    SAFETY_DOCUMENT = "safety_document"
    UNKNOWN = "unknown"


class SectionType(Enum):
    """Type of content section"""
    HEADING = "heading"
    PROCEDURE = "procedure"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    WARNING = "warning"
    CODE_BLOCK = "code_block"
    EXAMPLE = "example"


@dataclass
class ChunkMetadata:
    """Rich metadata for a chunk"""
    source: str                          # Filename
    section: str                         # Section title (e.g., "3.2 Troubleshooting")
    heading_level: int                   # 0-3 (higher = more important)
    section_type: str                    # Type of section
    doc_type: str                        # Document classification
    product_categories: List[str]        # e.g., ["Battery Tools", "Torque Wrenches"]
    fault_keywords: List[str]            # Extracted fault-related terms
    importance_score: float              # 0.0-1.0 based on heading level + structure
    is_procedure: bool                   # Contains step-by-step instructions
    contains_warning: bool               # Contains CAUTION/WARNING
    contains_numbers: bool               # Contains numeric specifications
    position: float                      # 0.0-1.0 relative position in document
    page_number: Optional[int] = None    # Page number if available
    word_count: int = 0                  # Number of words in chunk
    content_hash: str = ""               # SHA-256 hash for deduplication


class DocumentTypeDetector:
    """Detect document type and apply type-specific processing"""
    
    def detect_type(self, text: str) -> DocumentType:
        """Detect document type from content - prioritizes more specific patterns"""
        
        text_lower = text.lower()
        
        # Priority order: More specific patterns first to avoid false matches
        
        # 1. Safety documents (very distinct)
        if any(keyword in text_lower for keyword in ["caution", "warning", "danger", "safety", "hazard"]):
            return DocumentType.SAFETY_DOCUMENT
        
        # 2. Service bulletins (specific to desoutter updates)
        if any(keyword in text_lower for keyword in ["service bulletin", "technical bulletin", "bulletin", "service notice"]):
            return DocumentType.SERVICE_BULLETIN
        
        # 3. Parts catalogs (tabular/list heavy)
        if any(keyword in text_lower for keyword in ["part number", "catalog", "parts list", "assembly", "component list"]):
            return DocumentType.PARTS_CATALOG
        
        # 4. Troubleshooting (problem-solution structure)
        if any(keyword in text_lower for keyword in ["troubleshooting", "troubleshoot", "problem:", "symptom", "solution:"]):
            return DocumentType.TROUBLESHOOTING_GUIDE
        
        # 5. Technical manuals (default for instruction content)
        if any(keyword in text_lower for keyword in ["manual", "chapter", "section", "procedure", "instruction", "operation", "maintenance"]):
            return DocumentType.TECHNICAL_MANUAL
        
        # Default fallback
        return DocumentType.UNKNOWN
    
    def get_chunk_strategy(self, doc_type: DocumentType) -> str:
        """Get chunking strategy for document type"""
        
        strategies = {
            DocumentType.TECHNICAL_MANUAL: "preserve_procedures",
            DocumentType.SERVICE_BULLETIN: "preserve_structure",
            DocumentType.TROUBLESHOOTING_GUIDE: "problem_solution_pairs",
            DocumentType.PARTS_CATALOG: "table_aware",
            DocumentType.SAFETY_DOCUMENT: "warning_aware",
            DocumentType.UNKNOWN: "recursive"
        }
        
        return strategies.get(doc_type, "recursive")


class FaultKeywordExtractor:
    """Extract fault-related keywords from text"""
    
    # Repair domain vocabulary
    FAULT_KEYWORDS = {
        "motor": ["motor", "drive", "engine"],
        "noise": ["noise", "sound", "grinding", "squeaking", "clicking", "humming"],
        "mechanical": ["grinding", "grinding", "jamming", "stuck", "resistance"],
        "electrical": ["short", "voltage", "current", "resistance", "continuity"],
        "calibration": ["calibration", "calibrate", "accurate", "precision"],
        "leakage": ["leak", "seepage", "drip", "moisture"],
        "corrosion": ["corrosion", "oxidation", "rust", "tarnish"],
        "wear": ["wear", "worn", "degraded", "damaged"],
        "connection": ["connection", "contact", "loose", "disconnected", "cable"],
        "torque": ["torque", "tightness", "tension"]
    }
    
    def extract(self, text: str) -> List[str]:
        """Extract fault keywords from text"""
        
        text_lower = text.lower()
        found_keywords = set()
        
        for category, keywords in self.FAULT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.add(category)
        
        return list(found_keywords)


class SemanticChunker:
    """
    Recursive character chunking that preserves semantic meaning
    Uses document structure to split intelligently
    """
    
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
        max_recursion_depth: int = 3
    ):
        """
        Initialize semantic chunker
        
        Args:
            chunk_size: Target chunk size in tokens (approximate)
            chunk_overlap: Overlap between chunks in tokens
            max_recursion_depth: Max recursion levels (section → paragraph → sentence → char)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_recursion_depth = max_recursion_depth
        
        self.doc_type_detector = DocumentTypeDetector()
        self.keyword_extractor = FaultKeywordExtractor()
        
        logger.info(f"✅ SemanticChunker initialized: size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_document(
        self,
        text: str,
        source_filename: str,
        doc_type: Optional[DocumentType] = None,
        product_metadata: Optional[Dict] = None  # NEW: Full product metadata dict
    ) -> List[Dict]:
        """
        Chunk document with semantic structure preservation
        
        Args:
            text: Document text
            source_filename: Source document filename
            doc_type: Optional document type (auto-detected if None)
            product_metadata: Optional product metadata from IntelligentProductExtractor
        
        Returns:
            List of chunks with rich metadata
        """
        
        # Auto-detect document type
        if doc_type is None:
            doc_type = self.doc_type_detector.detect_type(text)
        
        logger.info(f"Chunking '{source_filename}' as {doc_type.value}")
        
        # Check if this is a service bulletin - apply special handling
        is_bulletin = (
            doc_type == DocumentType.SERVICE_BULLETIN or
            source_filename.upper().startswith('ESDE') or
            'bulletin' in source_filename.lower()
        )
        
        # Extract affected products from bulletin content
        affected_products = None
        if is_bulletin:
            affected_products = self._extract_affected_products(text)
            if affected_products:
                logger.info(f"Bulletin {source_filename}: affected products = {affected_products}")
        
        # For short bulletins, keep as single chunk to preserve context
        word_count = len(text.split())
        if is_bulletin and word_count < 800:
            logger.info(f"Short bulletin ({word_count} words) - keeping as single chunk")
            return [self._create_single_bulletin_chunk(
                text, source_filename, doc_type, product_metadata, affected_products
            )]
        
        # Split by paragraphs first (preserve structure)
        paragraphs = self._split_by_paragraphs(text)
        
        chunks = []
        current_section = "Introduction"
        section_heading_level = 0
        position_ratio = 0.0
        current_page = None  # Track current page number
        
        for para_idx, paragraph in enumerate(paragraphs):
            # Extract page number from markers at start of paragraph
            # Format: "--- Page X ---\n[content]" or just "--- Page X ---"
            # Use search and check start position to be more robust against leading whitespace/BOMs
            page_match = re.search(r'---\s*Page\s+(\d+)\s*---', paragraph)
            if page_match and page_match.start() < 10:
                current_page = int(page_match.group(1))
                logger.debug(f"Parsed page number: {current_page} from marker at pos {page_match.start()}")
                # Remove the page marker and any surrounding whitespace/newlines
                # We remove from the start of the match to the end of the line it's on
                marker_line_end = paragraph.find('\n', page_match.end())
                if marker_line_end != -1:
                    paragraph = paragraph[marker_line_end:].strip()
                else:
                    paragraph = paragraph[page_match.end():].strip()
                
                # If paragraph is now empty, it was just a page marker
                if not paragraph:
                    continue
            
            # Detect section heading
            if self._is_heading(paragraph):
                current_section = paragraph.strip()
                section_heading_level = self._get_heading_level(paragraph)
            
            # Update position in document
            position_ratio = para_idx / len(paragraphs) if paragraphs else 0.0
            
            # Chunk paragraph if too large
            para_chunks = self._chunk_paragraph(
                paragraph,
                section=current_section,
                heading_level=section_heading_level,
                position=position_ratio,
                source=source_filename,
                doc_type=doc_type,
                page_number=current_page,  # Pass page number to chunks
                product_metadata=product_metadata,  # Pass full product metadata
                affected_products=affected_products  # NEW: Pass affected products for bulletins
            )
            
            chunks.extend(para_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {source_filename}")
        
        return chunks
    
    def _extract_affected_products(self, text: str) -> Optional[str]:
        """
        Extract affected products from bulletin content.
        
        Looks for patterns like:
        - "Products impacted: EABS, EPBC, EABC"
        - "Products potentially impacted: ExBx tools"
        - "Applicable products: ERS, ERSA"
        """
        # Product alias expansion
        PRODUCT_ALIASES = {
            'ExB': ['EPB', 'EAB'],
            'ExBC': ['EPBC', 'EABC'],
            'ExBx': ['EABC', 'EPBC', 'EABS'],
        }
        
        patterns = [
            r'Products?\s+(?:impacted|affected|applicable|potentially\s+impacted)[:\s]+([^\n]+)',
            r'Applicable\s+(?:to|for)[:\s]+([^\n]+)',
            r'This\s+bulletin\s+(?:applies|concerns)[:\s]+([^\n]+)',
        ]
        
        text_lower = text.lower()
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                products_text = match.group(1).strip()
                # Clean up the text
                products_text = re.sub(r'[^\w\s,/-]', '', products_text)
                
                # Extract individual product codes
                products = []
                
                # Check for known product codes
                product_patterns = [
                    r'\bE[A-Z]{1,4}\d*\b',  # EABS, EPBC, EAB, EPB, ERS, etc.
                    r'\bExB[xCc]?\b',       # ExB, ExBx, ExBC
                    r'\bBLRT[Cc]?\b',       # BLRT, BLRTC
                ]
                
                for prod_pattern in product_patterns:
                    found = re.findall(prod_pattern, products_text, re.IGNORECASE)
                    products.extend([p.upper() for p in found])
                
                # Expand aliases
                expanded = []
                for prod in products:
                    if prod in PRODUCT_ALIASES:
                        expanded.extend(PRODUCT_ALIASES[prod])
                    expanded.append(prod)
                
                # Remove duplicates and return
                unique_products = list(dict.fromkeys(expanded))
                if unique_products:
                    return ','.join(unique_products)
        
        return None
    
    def _create_single_bulletin_chunk(
        self,
        text: str,
        source: str,
        doc_type: DocumentType,
        product_metadata: Optional[Dict],
        affected_products: Optional[str]
    ) -> Dict:
        """Create a single chunk for short bulletins to preserve full context."""
        word_count = len(text.split())
        content_hash = self._calculate_content_hash(text)
        fault_keywords = self.keyword_extractor.extract(text)
        
        # Build metadata
        metadata = {
            'source': source,
            'section': 'Full Bulletin',
            'heading_level': 0,
            'section_type': 'procedure',
            'doc_type': doc_type.value,
            'product_categories': product_metadata.get('product_family', 'UNKNOWN') if product_metadata else 'UNKNOWN',
            'fault_keywords': ', '.join(list(fault_keywords)[:5]) if fault_keywords else '',
            'importance_score': 1.0,  # Bulletins are high importance
            'is_procedure': True,
            'contains_warning': bool(re.search(r'\b(warning|caution|danger|important)\b', text, re.I)),
            'contains_numbers': bool(re.search(r'\d', text)),
            'position': 0.0,
            'word_count': word_count,
            'content_hash': content_hash,
            'is_complete_bulletin': True,  # Flag this as a complete bulletin
        }
        
        # Add product metadata fields
        if product_metadata:
            metadata['product_family'] = str(product_metadata.get('product_family', 'UNKNOWN'))
            metadata['product_models'] = str(product_metadata.get('product_models', ''))
            metadata['product_category'] = str(product_metadata.get('product_category', 'UNKNOWN'))
            metadata['is_generic'] = product_metadata.get('is_generic', False)
        else:
            metadata['product_family'] = 'UNKNOWN'
            metadata['product_models'] = ''
            metadata['product_category'] = 'UNKNOWN'
            metadata['is_generic'] = False
        
        # Add affected products if extracted
        if affected_products:
            metadata['affected_products'] = affected_products
        
        return {
            'text': text,
            'metadata': metadata
        }
    
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs (blank lines)"""
        
        # Split by multiple newlines
        paragraphs = re.split(r'\n\s*\n', text.strip())
        
        # Filter empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _is_heading(self, text: str) -> bool:
        """Check if text is a heading"""
        
        text_stripped = text.strip()
        if not text_stripped or len(text_stripped) < 2:
            return False
        
        # Markdown-style headings
        if re.match(r'^#+\s', text_stripped):
            return True
        
        # Special markers (>>, •, -, etc.)
        if re.match(r'^(>>|•|►|→)\s*[A-Z]', text_stripped):
            return True
        
        # Numbered section headings (e.g., "3.2 Troubleshooting")
        if re.match(r'^\d+(\.\d+)*\s+[A-Z]', text_stripped):
            return True
        
        # ALL CAPS headings (relaxed length limit for PDFs)
        # Allow up to 150 chars and allow some digits (for product names)
        if text_stripped.isupper() and len(text_stripped) < 150:
            # Must have at least 2 words
            words = text_stripped.split()
            if len(words) >= 2:
                return True
        
        # Short lines that look like headings (common in PDFs)
        # Must be < 60 chars, start with capital, no ending punctuation
        if len(text_stripped) < 60 and text_stripped[0].isupper():
            # No ending punctuation (except :)
            if not text_stripped[-1] in '.!?,;' or text_stripped.endswith(':'):
                # Contains heading keywords
                heading_keywords = [
                    'introduction', 'overview', 'benefits', 'features',
                    'installation', 'configuration', 'operation', 'maintenance',
                    'troubleshooting', 'specifications', 'safety', 'warning',
                    'caution', 'note', 'important', 'description', 'procedure',
                    'assembly', 'disassembly', 'repair', 'replacement'
                ]
                text_lower = text_stripped.lower()
                if any(keyword in text_lower for keyword in heading_keywords):
                    return True
        
        return False
    
    def _get_heading_level(self, text: str) -> int:
        """Extract heading level (0-3)"""
        
        # Markdown-style
        markdown_match = re.match(r'^(#+)', text)
        if markdown_match:
            return min(len(markdown_match.group(1)) - 1, 3)
        
        # Numbered sections
        if re.match(r'^\d\s+', text):  # "1 Title" = level 0
            return 0
        elif re.match(r'^\d\.\d\s+', text):  # "1.1 Title" = level 1
            return 1
        elif re.match(r'^\d\.\d\.\d\s+', text):  # "1.1.1 Title" = level 2
            return 2
        
        # Default for all-caps
        return 3
    
    def _get_chunk_size_for_type(self, doc_type: DocumentType) -> Tuple[int, int]:
        """
        Get adaptive chunk size and overlap based on document type.
        Target smaller chunks for problem/solution content (precision).
        Target larger chunks for general manuals (context).
        """
        if doc_type == DocumentType.TROUBLESHOOTING_GUIDE:
            return 200, 50 # Smaller chunks for precise problem/solution matching
        elif doc_type == DocumentType.SERVICE_BULLETIN:
            return 300, 75 # Medium chunks for technical updates
        elif doc_type == DocumentType.SAFETY_DOCUMENT:
            return 250, 50 # Smaller chunks for safety warnings
        else:
            return self.chunk_size, self.chunk_overlap # Default (e.g. 400, 100)

    def _chunk_paragraph(
        self,
        paragraph: str,
        section: str,
        heading_level: int,
        position: float,
        source: str,
        doc_type: DocumentType,
        page_number: Optional[int] = None,  # NEW: Page number from PDF
        product_metadata: Optional[Dict] = None,  # NEW: Full product metadata
        affected_products: Optional[str] = None  # NEW: Affected products for bulletins
    ) -> List[Dict]:
        """Chunk a paragraph into semantic pieces"""
        
        # Adaptive chunk sizing (Priority 5)
        target_chunk_size, target_overlap = self._get_chunk_size_for_type(doc_type)

        # If paragraph is small enough, return as single chunk
        word_count = len(paragraph.split())
        
        if word_count < target_chunk_size * 0.7:  # Less than 70% of target size
            return [self._create_chunk(
                text=paragraph,
                section=section,
                heading_level=heading_level,
                position=position,
                source=source,
                doc_type=doc_type,
                word_count=word_count,
                page_number=page_number,  # Pass page number
                product_metadata=product_metadata,
                affected_products=affected_products
            )]
        
        # Split into sentences
        sentences = self._split_by_sentences(paragraph)
        
        chunks = []
        current_chunk = ""
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # Check if adding sentence would exceed chunk size
            if current_word_count + sentence_words > target_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(self._create_chunk(
                    text=current_chunk.strip(),
                    section=section,
                    heading_level=heading_level,
                    position=position,
                    source=source,
                    doc_type=doc_type,
                    word_count=current_word_count,
                    page_number=page_number,  # Pass page number
                    product_metadata=product_metadata,
                    affected_products=affected_products
                ))
                
                # Start new chunk (with overlap)
                current_chunk = ""
                current_word_count = 0
            
            current_chunk += sentence + " "
            current_word_count += sentence_words
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                text=current_chunk.strip(),
                section=section,
                heading_level=heading_level,
                position=position,
                source=source,
                doc_type=doc_type,
                word_count=current_word_count,
                page_number=page_number,  # Pass page number
                product_metadata=product_metadata,
                affected_products=affected_products
            ))
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving structure"""
        
        # Handle abbreviations and edge cases
        text = re.sub(r'(\w\.)(\s+)([A-Z])', r'\1\n\2\3', text)
        
        # Split by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunk(
        self,
        text: str,
        section: str,
        heading_level: int,
        position: float,
        source: str,
        doc_type: DocumentType,
        word_count: int,
        page_number: Optional[int] = None,  # NEW: Page number from PDF
        product_metadata: Optional[Dict] = None,  # NEW: Full product metadata
        affected_products: Optional[str] = None  # NEW: Affected products from bulletin
    ) -> Dict:
        """Create a chunk with metadata"""
        
        # Calculate importance score (heading level + position)
        # Higher heading level = more important
        # Earlier in document = slightly more important
        importance = (1.0 - (heading_level / 4)) * 0.7 + (1.0 - position) * 0.3
        
        # Extract fault keywords
        fault_keywords = self.keyword_extractor.extract(text)
        
        # Detect section type
        section_type = self._detect_section_type(text)
        
        # Check for warnings
        contains_warning = bool(re.search(r'\b(CAUTION|WARNING|DANGER|NOTE)\b', text, re.IGNORECASE))
        
        metadata = ChunkMetadata(
            source=source,
            section=section,
            heading_level=heading_level,
            section_type=section_type.value,
            doc_type=doc_type.value,
            # Product metadata - stored as ChromaDB-compatible fields
            product_categories=[product_metadata.get("product_category", "UNKNOWN")] if product_metadata else [],
            fault_keywords=fault_keywords,
            importance_score=importance,
            is_procedure=self._is_procedure(text),
            contains_warning=contains_warning,
            contains_numbers=bool(re.search(r'\d+', text)),
            position=position,
            page_number=page_number,  # Store page number
            word_count=word_count,
            content_hash=self._calculate_content_hash(text)  # NEW: Content hash for deduplication
        )
        
        # Build return dict with additional product fields for ChromaDB filtering
        chunk_metadata = metadata.__dict__.copy()
        
        # Add product-specific fields (strings for ChromaDB compatibility)
        if product_metadata:
            chunk_metadata["product_family"] = product_metadata.get("product_family", "UNKNOWN")
            chunk_metadata["product_models"] = product_metadata.get("product_models", "")
            chunk_metadata["product_category"] = product_metadata.get("product_category", "UNKNOWN")
            chunk_metadata["is_generic"] = product_metadata.get("is_generic", False)
        else:
            chunk_metadata["product_family"] = "UNKNOWN"
            chunk_metadata["product_models"] = ""
            chunk_metadata["product_category"] = "UNKNOWN"
            chunk_metadata["is_generic"] = True
        
        # Add affected products for bulletins
        if affected_products:
            chunk_metadata["affected_products"] = affected_products
        
        return {
            "text": text,
            "metadata": chunk_metadata
        }

    def _calculate_content_hash(self, text: str) -> str:
        """Calculate SHA-256 hash of normalized content"""
        import hashlib
        
        # Normalize text: lowercase, remove excess whitespace
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def _detect_section_type(self, text: str) -> SectionType:
        """Detect the type of section"""
        
        text_lower = text.lower()
        
        if "warning" in text_lower or "caution" in text_lower:
            return SectionType.WARNING
        
        if any(keyword in text_lower for keyword in ["step", "procedure", "follow", "then"]):
            return SectionType.PROCEDURE
        
        if re.match(r'^[\d•\-]\s+', text):
            return SectionType.LIST
        
        if "|" in text and "\n" in text:  # Likely a table
            return SectionType.TABLE
        
        return SectionType.PARAGRAPH
    
    def _is_procedure(self, text: str) -> bool:
        """Check if text contains step-by-step procedures"""
        
        procedure_indicators = [
            r'^\d+\s+',  # Numbered steps
            r'\bstep\b',
            r'\bprocedure\b',
            r'\bfollow\b',
            r'\bthen\b',
            r'\bfirst\b.*\bthen\b'
        ]
        
        text_lower = text.lower()
        
        return sum(1 for pattern in procedure_indicators if re.search(pattern, text_lower)) >= 2
