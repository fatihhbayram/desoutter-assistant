"""
Test script for general RAG improvements.
Tests score boosting, bulletin preservation, and deduplication.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger

logger = setup_logger("test_rag_improvements")


# Test cases for general bulletin retrieval
TEST_CASES = [
    {
        "name": "Error Code Query (I004)",
        "query": "EPB8 I004 Span Failure",
        "product": "EPB8-1800-4Q",
        "should_have_bulletin": True,
    },
    {
        "name": "Error Code Query (E06)",
        "query": "ERS error E06 unbalance NOK",
        "product": "ERS6",
        "should_have_bulletin": True,
    },
    {
        "name": "Symptom Query (WiFi)",
        "query": "EABS WiFi weak signal performance",
        "product": "EABS8-1500",
        "should_have_bulletin": True,
    },
    {
        "name": "Symptom Query (No Boot)",
        "query": "EABS tool not starting no LEDs",
        "product": "EABS8-1500",
        "should_have_bulletin": True,
    },
    {
        "name": "General Manual Query",
        "query": "How to calibrate torque sensor",
        "product": "CVI3",
        "should_have_bulletin": False,  # Manual content expected
    },
]


def test_score_boosting():
    """Test that score boosting is properly applied."""
    print("\n" + "=" * 60)
    print("📋 TEST: Score Boosting")
    print("=" * 60)
    
    try:
        from src.llm.rag_engine import RAGEngine
        rag = RAGEngine()
    except Exception as e:
        print(f"⚠️  RAG Engine initialization failed: {e}")
        return False
    
    # Test that SCORE_BOOSTS is defined
    if hasattr(rag, 'SCORE_BOOSTS'):
        print(f"✅ SCORE_BOOSTS defined")
        print(f"   service_bulletin boost: {rag.SCORE_BOOSTS['doc_type'].get('service_bulletin', 'N/A')}x")
        print(f"   ESDE prefix boost: {rag.SCORE_BOOSTS['source_prefix'].get('ESDE', 'N/A')}x")
    else:
        print("❌ SCORE_BOOSTS not defined")
        return False
    
    # Test apply_score_boost method
    if hasattr(rag, 'apply_score_boost'):
        # Test bulletin boost
        bulletin_score = rag.apply_score_boost(
            0.5,
            {'doc_type': 'service_bulletin', 'source': 'ESDE23007.pdf'},
            'test query'
        )
        manual_score = rag.apply_score_boost(
            0.5,
            {'doc_type': 'technical_manual', 'source': 'User Manual.pdf'},
            'test query'
        )
        
        print(f"\n   Bulletin (base=0.5): boosted to {bulletin_score:.3f}")
        print(f"   Manual (base=0.5): boosted to {manual_score:.3f}")
        
        if bulletin_score > manual_score:
            print("   ✅ Bulletins are boosted higher than manuals")
            return True
        else:
            print("   ❌ Bulletin boost not working correctly")
            return False
    else:
        print("❌ apply_score_boost method not found")
        return False


def test_bulletin_preservation():
    """Test that short bulletins are preserved as single chunks."""
    print("\n" + "=" * 60)
    print("📋 TEST: Bulletin Preservation")
    print("=" * 60)
    
    try:
        from src.documents.semantic_chunker import SemanticChunker, DocumentType
        chunker = SemanticChunker()
    except Exception as e:
        print(f"⚠️  SemanticChunker initialization failed: {e}")
        return False
    
    # Create a short bulletin-like text (under 800 words)
    short_bulletin = """
    Service Bulletin ESDE99999
    
    Products impacted: EABS, EPBC, EABC
    
    Description of the issue:
    Tools may experience communication failure after firmware update.
    
    Cause of issue:
    Configuration mismatch in WiFi module.
    
    Corrective actions:
    1. Reset WiFi settings to factory default
    2. Re-pair with controller
    3. Update firmware to version 7.5.0
    
    Contact support if issue persists.
    """
    
    chunks = chunker.chunk_document(
        short_bulletin,
        "ESDE99999_Test_Bulletin.pdf",
        DocumentType.SERVICE_BULLETIN
    )
    
    if len(chunks) == 1:
        print(f"✅ Short bulletin preserved as single chunk")
        if chunks[0]['metadata'].get('is_complete_bulletin'):
            print(f"   is_complete_bulletin: True")
        affected = chunks[0]['metadata'].get('affected_products', '')
        if affected:
            print(f"   affected_products: {affected}")
        return True
    else:
        print(f"❌ Short bulletin was split into {len(chunks)} chunks")
        return False


def test_bulletin_deduplication():
    """Test that bulletins are deduplicated in results."""
    print("\n" + "=" * 60)
    print("📋 TEST: Bulletin Deduplication")
    print("=" * 60)
    
    try:
        from src.llm.hybrid_search import HybridSearcher, SearchResult
    except Exception as e:
        print(f"⚠️  HybridSearcher import failed: {e}")
        return False
    
    # Check if _deduplicate_bulletins method exists
    if hasattr(HybridSearcher, '_deduplicate_bulletins'):
        print("✅ _deduplicate_bulletins method exists")
        
        # Create mock results with duplicate bulletins
        mock_results = [
            SearchResult(id="1", content="chunk1", metadata={'source': 'ESDE23007 - Title.pdf'}, score=0.9, source='hybrid'),
            SearchResult(id="2", content="chunk2", metadata={'source': 'ESDE23007 - Title.pdf'}, score=0.8, source='hybrid'),
            SearchResult(id="3", content="chunk3", metadata={'source': 'Manual.pdf'}, score=0.85, source='hybrid'),
            SearchResult(id="4", content="chunk4", metadata={'source': 'ESDE23007 - Title.pdf'}, score=0.7, source='hybrid'),
        ]
        
        # Test deduplication
        searcher = HybridSearcher.__new__(HybridSearcher)
        deduplicated = searcher._deduplicate_bulletins(mock_results)
        
        bulletin_count = sum(1 for r in deduplicated if 'ESDE' in r.metadata.get('source', ''))
        
        if bulletin_count == 1:
            print(f"✅ Duplicate bulletin chunks removed (4 → {len(deduplicated)} results)")
            return True
        else:
            print(f"❌ Deduplication not working (still {bulletin_count} bulletin chunks)")
            return False
    else:
        print("❌ _deduplicate_bulletins method not found")
        return False


def test_retrieval_with_error_code():
    """Test retrieval with error code query."""
    print("\n" + "=" * 60)
    print("📋 TEST: Retrieval with Error Code")
    print("=" * 60)
    
    try:
        from src.llm.rag_engine import RAGEngine
        rag = RAGEngine()
    except Exception as e:
        print(f"⚠️  RAG Engine initialization failed: {e}")
        return False
    
    # Test error code query
    result = rag.retrieve_context(
        "EPB8 I004 Span Failure error",
        part_number="EPB8-1800-4Q",
        top_k=5
    )
    
    documents = result.get('documents', [])
    
    if documents:
        print(f"✅ Retrieved {len(documents)} documents")
        
        # Check if any ESDE bulletins in results
        esde_count = 0
        for doc in documents:
            source = doc.get('metadata', {}).get('source', '')
            if source.upper().startswith('ESDE'):
                esde_count += 1
                print(f"   📄 {source}")
        
        if esde_count > 0:
            print(f"   ✅ Found {esde_count} service bulletin(s)")
            return True
        else:
            print(f"   ⚠️  No ESDE bulletins found in top 5")
            return False
    else:
        print("❌ No documents retrieved")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("🧪 RAG IMPROVEMENTS TEST SUITE")
    print("=" * 70)
    
    results = []
    
    # Test 1: Score Boosting
    results.append(("Score Boosting", test_score_boosting()))
    
    # Test 2: Bulletin Preservation
    results.append(("Bulletin Preservation", test_bulletin_preservation()))
    
    # Test 3: Bulletin Deduplication
    results.append(("Bulletin Deduplication", test_bulletin_deduplication()))
    
    # Test 4: Retrieval with Error Code
    results.append(("Error Code Retrieval", test_retrieval_with_error_code()))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n🎯 OVERALL: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed - review needed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
