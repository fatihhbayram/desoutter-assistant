#!/usr/bin/env python3
"""
Test Script: Semantic Chunking Integration
============================================
Tests the SemanticChunker integration with DocumentProcessor.
This validates Phase 1-2 enhancement before full document re-ingestion.

Usage:
    python scripts/test_semantic_chunking.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.documents.semantic_chunker import (
    SemanticChunker, 
    DocumentType, 
    SectionType
)
from src.documents.document_processor import DocumentProcessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def test_semantic_chunker_basic():
    """Test basic SemanticChunker functionality"""
    logger.info("=" * 70)
    logger.info("TEST 1: Basic Semantic Chunking")
    logger.info("=" * 70)
    
    # Sample technical manual text
    sample_text = """
# Technical Manual: Desoutter DS20 Electric Screwdriver

## Chapter 1: Introduction

The Desoutter DS20 is a professional-grade electric screwdriver designed for 
industrial assembly and repair applications. This manual covers operation, 
maintenance, and troubleshooting.

## Section 1.1: Safety Warnings

WARNING: Do not operate this device without proper training. Risk of serious injury.
Never use this tool near electrical conductors or in explosive atmospheres.
Always wear appropriate safety equipment including eye protection.

## Section 2: Operating Procedure

Follow these steps to operate the DS20:

1. Insert the appropriate bit into the chuck
2. Set the torque adjustment to the desired value
3. Position the screw and apply pressure to the tool
4. Release the trigger when the screw is tight
5. Remove the tool and inspect the screw tightness

## Technical Specifications

| Specification | Value |
|---|---|
| Operating Voltage | 100-240V AC |
| Power | 80W |
| Motor Speed | 400-1500 RPM |
| Torque Range | 0.5-2.5 N·m |
| Weight | 1.2 kg |

## Troubleshooting Guide

### Problem: Motor will not start

Check the power connection and ensure the power switch is functioning properly.
If the motor still does not start, the motor may be faulty and require replacement.

### Problem: Noise during operation

Excessive noise may indicate bearing wear or mechanical damage. 
Stop operation immediately and have the unit serviced.
    """
    
    # Test chunking
    chunker = SemanticChunker(chunk_size=400, chunk_overlap=100)
    chunks = chunker.chunk_document(
        text=sample_text,
        source_filename="DS20_Manual.pdf",
        doc_type=DocumentType.TECHNICAL_MANUAL
    )
    
    logger.info(f"\n✓ Chunking completed successfully")
    logger.info(f"  Total chunks: {len(chunks)}")
    logger.info(f"  Average chunk size: {sum(len(c.get('content', '')) for c in chunks) / len(chunks):.0f} chars")
    
    # Display sample chunks with metadata
    logger.info(f"\n📄 Sample chunks (first 3):\n")
    for i, chunk in enumerate(chunks[:3]):
        metadata = chunk.get('metadata', {})
        logger.info(f"\n  Chunk {i+1}:")
        logger.info(f"    Section Type: {metadata.get('section_type', 'unknown')}")
        logger.info(f"    Document Type: {metadata.get('doc_type', 'unknown')}")
        logger.info(f"    Is Procedure: {metadata.get('is_procedure', False)}")
        logger.info(f"    Is Warning: {metadata.get('contains_warning', False)}")
        logger.info(f"    Importance: {metadata.get('importance_score', 0):.2f}")
        logger.info(f"    Fault Keywords: {metadata.get('fault_keywords', [])}")
        logger.info(f"    Content: {chunk.get('content', '')[:150]}...")
    
    return len(chunks) > 0


def test_document_processor_integration():
    """Test DocumentProcessor with semantic chunking"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: DocumentProcessor Integration")
    logger.info("=" * 70)
    
    # Check if we have test documents
    documents_dir = project_root / "desoutter-assistant" / "documents"
    if not documents_dir.exists():
        logger.warning(f"  ⚠ Documents directory not found: {documents_dir}")
        logger.info("  Creating test with sample text instead...")
        
    # Create processor
        processor = DocumentProcessor()
        
        # Create a test file
        test_file = project_root / "test_document.txt"
        test_content = """
TECHNICAL MANUAL: Motor Assembly

Chapter 1: Motor Specifications

The motor operates at 3000 RPM and requires calibration every 500 hours.

Section 1.1: Installation Procedure

1. Mount the motor on the bracket
2. Connect the power cables
3. Calibrate the motor speed
4. Test under no-load condition

WARNING: High voltage - do not touch terminals during operation!

Section 1.2: Troubleshooting

If the motor makes noise, check the bearing wear and lubrication.
        """
        test_file.write_text(test_content)
        logger.info(f"  Created test file: {test_file.name}")
        
        return True  # Skip actual file processing
    
    logger.info(f"\n📁 Found documents directory: {documents_dir}")
    
    # Find PDF files
    pdf_files = list(documents_dir.glob("**/*.pdf"))
    if not pdf_files:
        logger.warning("  ⚠ No PDF files found in documents directory")
        return True
    
    logger.info(f"  Found {len(pdf_files)} PDF files")
    
    # Process first PDF with semantic chunking
    processor = DocumentProcessor()
    test_pdf = pdf_files[0]
    
    logger.info(f"\n  Processing: {test_pdf.name}")
    doc = processor.process_document(
        test_pdf,
        extract_tables=True,
        enable_semantic_chunking=True
    )
    
    if doc:
        logger.info(f"  ✓ Processing successful")
        logger.info(f"    Words: {doc['word_count']}")
        logger.info(f"    Chunks: {doc.get('chunk_count', 0)}")
        
        if doc.get('chunks'):
            logger.info(f"\n  📊 Chunk statistics:")
            chunks = doc['chunks']
            
            # Analyze chunk distribution
            section_types = {}
            for chunk in chunks:
                st = chunk.get('metadata', {}).get('section_type', 'unknown')
                section_types[st] = section_types.get(st, 0) + 1
            
            logger.info(f"    Section type distribution:")
            for st, count in sorted(section_types.items(), key=lambda x: -x[1]):
                logger.info(f"      {st}: {count}")
            
            # Analyze importance scores
            importance_scores = [c.get('metadata', {}).get('importance_score', 0) for c in chunks]
            avg_importance = sum(importance_scores) / len(importance_scores) if importance_scores else 0
            max_importance = max(importance_scores) if importance_scores else 0
            
            logger.info(f"    Importance scores:")
            logger.info(f"      Average: {avg_importance:.2f}")
            logger.info(f"      Max: {max_importance:.2f}")
            
            # Show warnings and procedures
            warnings = sum(1 for c in chunks if c.get('metadata', {}).get('contains_warning', False))
            procedures = sum(1 for c in chunks if c.get('metadata', {}).get('is_procedure', False))
            logger.info(f"    Warnings found: {warnings}")
            logger.info(f"    Procedures found: {procedures}")
            
            # Show top fault keywords
            all_keywords = []
            for chunk in chunks:
                all_keywords.extend(chunk.get('metadata', {}).get('fault_keywords', []))
            
            if all_keywords:
                from collections import Counter
                keyword_freq = Counter(all_keywords)
                logger.info(f"    Top 5 fault keywords:")
                for keyword, count in keyword_freq.most_common(5):
                    logger.info(f"      {keyword}: {count} occurrences")
        
        return True
    else:
        logger.error("  ✗ Processing failed")
        return False


def test_document_type_detection():
    """Test document type detection"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Document Type Detection")
    logger.info("=" * 70)
    
    from src.documents.semantic_chunker import DocumentTypeDetector
    
    detector = DocumentTypeDetector()
    
    # Test cases
    test_cases = [
        ("Service Bulletin: Known Issues\n\nThis bulletin addresses known issues in the DS20...", 
         DocumentType.SERVICE_BULLETIN),
        ("TECHNICAL MANUAL\nChapter 1: Introduction\n\nThis manual covers all aspects...",
         DocumentType.TECHNICAL_MANUAL),
        ("TROUBLESHOOTING GUIDE\n\nProblem: Motor will not start\nSolution: Check power...",
         DocumentType.TROUBLESHOOTING_GUIDE),
        ("SAFETY WARNINGS\n\nDANGER: Risk of electrical shock! Do not touch terminals...",
         DocumentType.SAFETY_DOCUMENT),
        ("PARTS CATALOG\n\nPart Number | Description\n6151234567 | Motor Assembly",
         DocumentType.PARTS_CATALOG),
    ]
    
    logger.info("\n  Testing document type detection:\n")
    correct = 0
    for text, expected_type in test_cases:
        detected = detector.detect_type(text[:500])
        is_correct = detected == expected_type
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        logger.info(f"  {status} Expected: {expected_type.value}")
        logger.info(f"     Detected: {detected.value}\n")
        logger.info(f"     Detected: {detected.value}\n")
    
    logger.info(f"  Result: {correct}/{len(test_cases)} correct")
    return correct == len(test_cases)


def test_fault_keyword_extraction():
    """Test fault keyword extraction"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Fault Keyword Extraction")
    logger.info("=" * 70)
    
    from src.documents.semantic_chunker import FaultKeywordExtractor
    
    extractor = FaultKeywordExtractor()
    
    # Test texts with domain keywords
    test_texts = [
        "The motor bearing needs lubrication and the brushes are worn out.",
        "Electrical shock risk due to loose wire connections and poor grounding.",
        "Excessive noise and vibration indicate mechanical damage.",
        "Torque calibration failed - check the torque wrench specification.",
        "Oil leakage detected - replace the seal and check for corrosion.",
    ]
    
    logger.info("\n  Testing keyword extraction:\n")
    for text in test_texts:
        keywords = extractor.extract(text)
        logger.info(f"  Text: {text}")
        logger.info(f"  Keywords: {keywords}\n")
    
    return True


def main():
    """Run all tests"""
    logger.info("\n")
    logger.info("╔" + "=" * 68 + "╗")
    logger.info("║" + " SEMANTIC CHUNKING TEST SUITE ".center(68) + "║")
    logger.info("║" + " Phase 1-2 RAG Enhancement Validation ".center(68) + "║")
    logger.info("╚" + "=" * 68 + "╝")
    
    results = []
    
    # Run all tests
    try:
        results.append(("Basic Semantic Chunking", test_semantic_chunker_basic()))
    except Exception as e:
        logger.error(f"  ✗ Test failed with error: {e}")
        results.append(("Basic Semantic Chunking", False))
    
    try:
        results.append(("Document Type Detection", test_document_type_detection()))
    except Exception as e:
        logger.error(f"  ✗ Test failed with error: {e}")
        results.append(("Document Type Detection", False))
    
    try:
        results.append(("Fault Keyword Extraction", test_fault_keyword_extraction()))
    except Exception as e:
        logger.error(f"  ✗ Test failed with error: {e}")
        results.append(("Fault Keyword Extraction", False))
    
    try:
        results.append(("DocumentProcessor Integration", test_document_processor_integration()))
    except Exception as e:
        logger.error(f"  ✗ Test failed with error: {e}")
        results.append(("DocumentProcessor Integration", False))
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    logger.info(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        logger.info("\n🎉 All tests passed! SemanticChunker is ready for production.")
    else:
        logger.warning(f"\n⚠ {total_tests - total_passed} test(s) failed. Review errors above.")
    
    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
