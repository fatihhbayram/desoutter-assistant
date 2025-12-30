#!/usr/bin/env python3
"""
Debug: Check if section metadata exists in chunks
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.documents.document_processor import DocumentProcessor
from src.documents.semantic_chunker import SemanticChunker

# Process a sample document
processor = DocumentProcessor()
chunker = SemanticChunker()

# Get a sample PDF
sample_pdf = Path("/app/documents/bulletins/CVI3_ENG_2020-06.pdf")

if sample_pdf.exists():
    print(f"Processing: {sample_pdf.name}")
    
    # Extract text
    result = processor.extract_text_pdfplumber(sample_pdf)
    text = result.get("combined", "")
    
    # Chunk it
    chunks = chunker.chunk_document(
        text=text,
        source_filename=sample_pdf.name
    )
    
    print(f"\nCreated {len(chunks)} chunks")
    print(f"\nFirst chunk metadata:")
    if chunks:
        meta = chunks[0].get("metadata", {})
        print(f"  source: {meta.get('source')}")
        print(f"  section: {meta.get('section')}")
        print(f"  page_number: {meta.get('page_number')}")
        print(f"  doc_type: {meta.get('doc_type')}")
        print(f"\n  All keys: {list(meta.keys())}")
else:
    print(f"File not found: {sample_pdf}")
