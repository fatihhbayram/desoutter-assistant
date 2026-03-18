#!/usr/bin/env python3
"""
Test product filtering in retrieval.
Verifies that queries only return relevant product documents.

Usage:
    python scripts/test_product_filtering.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.documents.product_extractor import IntelligentProductExtractor
from src.vectordb.chroma_client import ChromaDBClient
from src.documents.embeddings import EmbeddingsGenerator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# Test cases with expected product families
TEST_CASES = [
    {
        "name": "ERS6 Reverse Switch",
        "query": "ERS6 reverse switch not working",
        "expected_families": ["ERS", "GENERAL", "UNKNOWN"],
        "unexpected_families": ["ELRT", "EABS", "EPB", "CVI3"]
    },
    {
        "name": "EABS Battery Issue",
        "query": "EABS battery not charging",
        "expected_families": ["EABS", "GENERAL", "UNKNOWN"],
        "unexpected_families": ["ERS", "CVI3", "ELRT"]
    },
    {
        "name": "CVI3 Controller Error",
        "query": "CVI3 error code E-105",
        "expected_families": ["CVI3", "GENERAL", "UNKNOWN"],
        "unexpected_families": ["ERS", "EABS", "EPB"]
    },
    {
        "name": "EPB Battery Tool WiFi",
        "query": "EPB WiFi not connecting to access point",
        "expected_families": ["EPB", "GENERAL", "UNKNOWN"],
        "unexpected_families": ["ERS", "ELRT"]
    },
    {
        "name": "Generic Torque Query (No Product)",
        "query": "how to calibrate torque settings",
        "expected_families": None,  # Should return mixed results
        "unexpected_families": []
    }
]


def test_product_extractor():
    """Test the intelligent product extractor"""
    print("\n" + "=" * 60)
    print("🔍 TESTING PRODUCT EXTRACTOR")
    print("=" * 60)
    
    extractor = IntelligentProductExtractor()
    
    passed = 0
    failed = 0
    
    # Test filename extraction
    test_filenames = [
        ("ERS6 - Product Instructions.pdf", "ERS"),
        ("EABS12-1100-4S Maintenance.docx", "EABS"),
        ("CVI3 VISION User Manual.pdf", "CVI3"),
        ("General Safety Guidelines.pdf", "GENERAL"),
        ("EPB8 Troubleshooting Guide.pdf", "EPB"),
        ("ELRT025 Calibration.pdf", "ELRT"),
    ]
    
    print("\n📁 Filename extraction tests:")
    for filename, expected_family in test_filenames:
        metadata = extractor.get_product_metadata(filename)
        detected = metadata.get('product_family', 'NONE')
        
        if detected == expected_family:
            print(f"   ✅ {filename} → {detected}")
            passed += 1
        else:
            print(f"   ❌ {filename} → {detected} (expected {expected_family})")
            failed += 1
    
    # Test query extraction
    print("\n🔍 Query extraction tests:")
    for test in TEST_CASES:
        context = extractor.extract_product_from_query(test['query'])
        detected = context.get('product_family', None)
        has_context = context.get('has_product_context', False)
        
        print(f"   Query: \"{test['query'][:40]}...\"")
        print(f"   → Family: {detected}, Has context: {has_context}")
    
    print(f"\n📊 Results: {passed}/{passed+failed} tests passed")
    return passed, failed


def test_vector_db_metadata():
    """Check that ChromaDB chunks have product metadata"""
    print("\n" + "=" * 60)
    print("🗄️  CHECKING VECTORDB METADATA")
    print("=" * 60)
    
    try:
        chroma = ChromaDBClient()
        count = chroma.get_count()
        
        if count == 0:
            print("⚠️  No documents in vector database!")
            print("   Run reingest_documents.py first.")
            return 0, 1
        
        print(f"📊 Total documents: {count}")
        
        # Sample some documents
        results = chroma.collection.get(
            limit=10,
            include=["metadatas"]
        )
        
        metadatas = results.get('metadatas', [])
        
        # Check for required fields
        required_fields = ['product_family', 'is_generic']
        missing_fields = []
        
        for meta in metadatas:
            for field in required_fields:
                if field not in meta:
                    missing_fields.append(field)
        
        if missing_fields:
            print(f"⚠️  Some documents missing fields: {set(missing_fields)}")
            print("   Re-ingestion may be needed.")
            return 0, 1
        
        # Show sample
        print("\n📋 Sample documents:")
        for i, meta in enumerate(metadatas[:5]):
            family = meta.get('product_family', 'N/A')
            generic = meta.get('is_generic', 'N/A')
            source = meta.get('source', 'N/A')[:30]
            print(f"   [{i+1}] Family: {family}, Generic: {generic}, Source: {source}...")
        
        # Count families
        family_counts = {}
        for meta in metadatas:
            family = meta.get('product_family', 'UNKNOWN')
            family_counts[family] = family_counts.get(family, 0) + 1
        
        print("\n📊 Family distribution (sample):")
        for family, count in sorted(family_counts.items()):
            print(f"   {family}: {count}")
        
        print("\n✅ Metadata structure looks good!")
        return 1, 0
        
    except Exception as e:
        print(f"❌ Error checking vector DB: {e}")
        return 0, 1


def test_retrieval_filtering():
    """Test retrieval with product filtering"""
    print("\n" + "=" * 60)
    print("🔎 TESTING RETRIEVAL FILTERING")
    print("=" * 60)
    
    try:
        from src.llm.rag_engine import RAGEngine
        rag = RAGEngine()
    except Exception as e:
        print(f"⚠️  Could not initialize RAG engine: {e}")
        print("   Some dependencies may not be available.")
        return 0, 1
    
    passed = 0
    failed = 0
    
    for test in TEST_CASES:
        print(f"\n📋 Test: {test['name']}")
        print(f"   Query: {test['query']}")
        
        try:
            # Get retrieval results
            results = rag.retrieve_context(
                query=test['query'],
                top_k=5
            )
            
            # Check returned document families
            returned_families = set()
            docs = results.get('documents', [])
            
            for doc in docs:
                meta = doc.get('metadata', {})
                family = meta.get('product_family', 'UNKNOWN')
                returned_families.add(family)
            
            print(f"   Returned families: {returned_families}")
            
            # Validate
            test_passed = True
            
            if test['expected_families']:
                # Check that at least one expected family is present
                if not returned_families.intersection(set(test['expected_families'])):
                    if docs:  # Only fail if we got results but wrong families
                        print(f"   ❌ FAIL: No expected families found")
                        test_passed = False
            
            # Check that no unexpected families are present
            unexpected_found = returned_families.intersection(set(test['unexpected_families']))
            if unexpected_found:
                print(f"   ❌ FAIL: Unexpected families found: {unexpected_found}")
                test_passed = False
            
            if test_passed:
                print(f"   ✅ PASS")
                passed += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            failed += 1
    
    print(f"\n📊 Results: {passed}/{passed+failed} tests passed")
    return passed, failed


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🧪 PRODUCT FILTERING TEST SUITE")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    
    # Test 1: Product extractor
    p, f = test_product_extractor()
    total_passed += p
    total_failed += f
    
    # Test 2: Vector DB metadata
    p, f = test_vector_db_metadata()
    total_passed += p
    total_failed += f
    
    # Test 3: Retrieval filtering
    p, f = test_retrieval_filtering()
    total_passed += p
    total_failed += f
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 OVERALL RESULTS")
    print("=" * 60)
    
    total = total_passed + total_failed
    if total > 0:
        rate = (total_passed / total) * 100
        status = "✅ PASS" if rate >= 80 else "⚠️ NEEDS WORK" if rate >= 50 else "❌ FAIL"
        print(f"\n{status}: {total_passed}/{total} tests passed ({rate:.1f}%)")
    else:
        print("\n⚠️ No tests were run!")
    
    print("=" * 60)
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
