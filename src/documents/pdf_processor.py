"""
PDF Document Processing Module
Extracts text from repair manuals and bulletins
"""
import re
from pathlib import Path
from typing import List, Dict, Optional
import PyPDF2
import pdfplumber
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PDFProcessor:
    """Process PDF documents and extract text"""
    
    def __init__(self):
        """Initialize PDF processor"""
        pass
    
    def extract_text_pypdf2(self, pdf_path: Path) -> str:
        """
        Extract text using PyPDF2 (faster, basic)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        try:
            text_content = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
                        
            return "\n\n".join(text_content)
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_text_pdfplumber(self, pdf_path: Path, extract_tables: bool = True) -> Dict:
        """
        Extract text and tables using pdfplumber (better quality)
        
        Args:
            pdf_path: Path to PDF file
            extract_tables: Whether to extract tables
            
        Returns:
            Dictionary with text and tables
        """
        try:
            text_content = []
            tables_content = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text()
                    if text and text.strip():
                        text_content.append(f"--- Page {page_num + 1} ---\n{text}")
                    
                    # Extract tables if requested
                    if extract_tables:
                        tables = page.extract_tables()
                        for table_idx, table in enumerate(tables):
                            if table:
                                table_text = self._format_table(table)
                                tables_content.append(
                                    f"--- Page {page_num + 1}, Table {table_idx + 1} ---\n{table_text}"
                                )
            
            return {
                "text": "\n\n".join(text_content),
                "tables": "\n\n".join(tables_content) if tables_content else "",
                "combined": "\n\n".join(text_content + tables_content)
            }
        except Exception as e:
            logger.error(f"Error extracting from {pdf_path}: {e}")
            return {"text": "", "tables": "", "combined": ""}
    
    def _format_table(self, table: List[List[str]]) -> str:
        """Format extracted table as text"""
        if not table:
            return ""
        
        # Convert table to text representation
        formatted_rows = []
        for row in table:
            # Clean None values
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            formatted_rows.append(" | ".join(cleaned_row))
        
        return "\n".join(formatted_rows)
    
    def process_document(
        self,
        pdf_path: Path,
        method: str = "pdfplumber",
        extract_tables: bool = True
    ) -> Dict:
        """
        Process a PDF document
        
        Args:
            pdf_path: Path to PDF file
            method: Extraction method ('pypdf2' or 'pdfplumber')
            extract_tables: Whether to extract tables
            
        Returns:
            Dictionary with document metadata and content
        """
        logger.info(f"Processing: {pdf_path.name}")
        
        if not pdf_path.exists():
            logger.error(f"File not found: {pdf_path}")
            return None
        
        # Extract text
        if method == "pdfplumber":
            extracted = self.extract_text_pdfplumber(pdf_path, extract_tables)
            text = extracted["combined"]
        else:
            text = self.extract_text_pypdf2(pdf_path)
        
        if not text or len(text.strip()) < 100:
            logger.warning(f"No meaningful content extracted from {pdf_path.name}")
            return None
        
        # Clean text
        text = self.clean_text(text)
        
        # Extract metadata
        metadata = self.extract_metadata(pdf_path, text)
        
        return {
            "filename": pdf_path.name,
            "filepath": str(pdf_path),
            "text": text,
            "metadata": metadata,
            "word_count": len(text.split()),
            "char_count": len(text)
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (basic)
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        
        # Remove common PDF artifacts
        text = re.sub(r'\x00', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def extract_metadata(self, pdf_path: Path, text: str) -> Dict:
        """
        Extract metadata from PDF and text
        
        Args:
            pdf_path: Path to PDF file
            text: Extracted text
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "source": pdf_path.name,
            "type": self._infer_document_type(pdf_path),
        }
        
        # Try to extract product references from filename and text
        part_numbers = self._extract_part_numbers(pdf_path.name + " " + text[:1000])
        if part_numbers:
            metadata["related_products"] = part_numbers
        
        # Try to extract language
        metadata["language"] = self._detect_language(text[:500])
        
        return metadata
    
    def _infer_document_type(self, pdf_path: Path) -> str:
        """Infer document type from path"""
        path_str = str(pdf_path).lower()
        
        if "manual" in path_str:
            return "manual"
        elif "bulletin" in path_str:
            return "bulletin"
        elif "guide" in path_str:
            return "guide"
        else:
            return "unknown"
    
    def _extract_part_numbers(self, text: str) -> List[str]:
        """Extract Desoutter part numbers from text"""
        patterns = [
            r'\b(6151[0-9]{6})\b',
            r'\b(8920[0-9]{6})\b',
        ]
        
        found = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            found.update(matches)
        
        return sorted(list(found))
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        # Turkish specific characters
        turkish_chars = ['ş', 'Ş', 'ğ', 'Ğ', 'ı', 'İ', 'ö', 'Ö', 'ü', 'Ü', 'ç', 'Ç']
        
        # Count Turkish characters
        turkish_count = sum(1 for char in text if char in turkish_chars)
        
        if turkish_count > 5:
            return "tr"
        else:
            return "en"
    
    def process_directory(
        self,
        directory: Path,
        method: str = "pdfplumber",
        extract_tables: bool = True
    ) -> List[Dict]:
        """
        Process all PDFs in a directory
        
        Args:
            directory: Directory containing PDF files
            method: Extraction method
            extract_tables: Whether to extract tables
            
        Returns:
            List of processed documents
        """
        logger.info(f"Processing PDFs in: {directory}")
        
        pdf_files = list(directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        documents = []
        for pdf_file in pdf_files:
            doc = self.process_document(pdf_file, method, extract_tables)
            if doc:
                documents.append(doc)
                logger.info(f"✓ Processed: {pdf_file.name} ({doc['word_count']} words)")
            else:
                logger.warning(f"✗ Failed: {pdf_file.name}")
        
        logger.info(f"Successfully processed {len(documents)}/{len(pdf_files)} documents")
        return documents
