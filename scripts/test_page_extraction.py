#!/usr/bin/env python3
"""
Test page number extraction from PDFs
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.documents.document_processor import DocumentProcessor
from src.documents.semantic_chunker import SemanticChunker

# Process a sample PDF
processor = DocumentProcessor()
chunker = SemanticChunker()

sample_pdf = Path("/app/documents/bulletins/CVI3_ENG_2020-06.pdf")

if sample_pdf.exists():
    print(f"Testing page number extraction: {sample_pdf.name}")
    
    # Extract text (includes page markers)
    result = processor.extract_text_pdfplumber(sample_pdf)
    text = result.get("combined", "")
    
    # Check for page markers
    import re
    page_markers = re.findall(r'--- Page (\d+) ---', text)
    print(f"\nPage markers found: {len(page_markers)}")
    if page_markers:
        print(f"Pages: {', '.join(page_markers[:5])}...")
    
    # Chunk it
    chunks = chunker.chunk_document(
        text=text,
        source_filename=sample_pdf.name
    )
    
    print(f"\nCreated {len(chunks)} chunks")
    
    # Check page numbers in chunks
    chunks_with_pages = sum(1 for c in chunks if c.get('metadata', {}).get('page_number'))
    print(f"Chunks with page numbers: {chunks_with_pages}/{len(chunks)} ({chunks_with_pages/len(chunks)*100:.0f}%)")
    
    # Show sample chunks
    print(f"\nSample chunks:")
    for idx, chunk in enumerate(chunks[:3], 1):
        meta = chunk.get('metadata', {})
        print(f"\n  Chunk {idx}:")
        print(f"    Page: {meta.get('page_number', 'N/A')}")
        print(f"    Section: {meta.get('section', 'N/A')[:40]}")
        print(f"    Text: {chunk.get('text', '')[:60]}...")
        
else:
    print(f"File not found: {sample_pdf}")
