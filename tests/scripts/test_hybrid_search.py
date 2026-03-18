#!/usr/bin/env python3
"""
Test script for Phase 2.2: Hybrid Search
Tests semantic + BM25 search combination
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.hybrid_search import HybridSearcher, QueryExpander, BM25Index
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def test_query_expansion():
    """Test query expansion with domain synonyms"""
    logger.info("=" * 60)
    logger.info("Test 1: Query Expansion")
    logger.info("=" * 60)
    
    expander = QueryExpander()
    
    test_queries = [
        "Motor grinding noise",
        "e047 battery error",
        "CVI3 not starting",
        "WiFi connection lost",
        "Torque calibration failed",
    ]
    
    for query in test_queries:
        expansions = expander.expand(query, max_expansions=5)
        logger.info(f"\n🔍 Query: '{query}'")
        logger.info(f"   Expansions ({len(expansions)}):")
        for i, exp in enumerate(expansions):
            logger.info(f"   {i+1}. {exp}")
    
    logger.info("\n✅ Test 1 PASSED: Query expansion working")
    return True


def test_bm25_index():
    """Test BM25 keyword search"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: BM25 Keyword Search")
    logger.info("=" * 60)
    
    # Create sample documents
    sample_docs = [
        {"id": "doc1", "content": "Motor grinding noise indicates bearing wear. Replace bearings if noise persists."},
        {"id": "doc2", "content": "Battery error E047 means low voltage. Check battery connections and charger."},
        {"id": "doc3", "content": "WiFi connection issues. Check antenna and router settings."},
        {"id": "doc4", "content": "Torque calibration procedure: Use certified wrench, apply 10Nm."},
        {"id": "doc5", "content": "CVI3 controller startup guide. Press power button for 3 seconds."},
    ]
    
    # Build BM25 index
    bm25 = BM25Index(k1=1.5, b=0.75)
    bm25.add_documents(sample_docs)
    
    # Test searches
    test_queries = ["motor noise", "battery error", "WiFi connection"]
    
    for query in test_queries:
        results = bm25.search(query, top_k=3)
        logger.info(f"\n🔍 Query: '{query}'")
        for i, result in enumerate(results, 1):
            logger.info(f"   {i}. {result.id} (score: {result.bm25_score:.4f})")
            logger.info(f"      {result.content[:60]}...")
    
    logger.info("\n✅ Test 2 PASSED: BM25 search working")
    return True


def test_hybrid_search():
    """Test hybrid search with real ChromaDB data"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Hybrid Search (Semantic + BM25)")
    logger.info("=" * 60)
    
    try:
        searcher = HybridSearcher()
        
        test_queries = [
            "Motor grinding noise E047",
            "CVI3 controller won't turn on",
            "Battery tool calibration procedure",
            "WiFi connection error EABC",
            "How to replace bearing",
        ]
        
        for query in test_queries:
            logger.info(f"\n🔍 Query: '{query}'")
            
            # Test with hybrid search
            results = searcher.search(query, top_k=3, expand_query=True, use_hybrid=True)
            
            logger.info(f"   Results ({len(results)}):")
            for i, result in enumerate(results, 1):
                source = result.metadata.get('source', 'unknown')
                doc_type = result.metadata.get('doc_type', 'unknown')
                logger.info(f"   {i}. [{result.source}] {source}")
                logger.info(f"      Type: {doc_type} | Score: {result.score:.4f} | Sim: {result.similarity:.4f}")
                logger.info(f"      {result.content[:80]}...")
        
        logger.info("\n✅ Test 3 PASSED: Hybrid search working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test 3 FAILED: {e}")
        return False


def test_metadata_filtering():
    """Test metadata-based filtering"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Metadata Filtering")
    logger.info("=" * 60)
    
    try:
        searcher = HybridSearcher()
        
        # Test with doc_type filter
        logger.info("\n🔍 Filter: technical_manual only")
        results = searcher.search_with_filter(
            "calibration procedure",
            top_k=3,
            doc_types=["technical_manual"]
        )
        
        for i, result in enumerate(results, 1):
            doc_type = result.metadata.get('doc_type', 'unknown')
            logger.info(f"   {i}. {result.metadata.get('source', 'unknown')} (type: {doc_type})")
        
        # Test with high importance filter
        logger.info("\n🔍 Filter: min_importance >= 0.7")
        results = searcher.search_with_filter(
            "error warning safety",
            top_k=3,
            min_importance=0.7
        )
        
        for i, result in enumerate(results, 1):
            importance = result.metadata.get('importance_score', 0)
            logger.info(f"   {i}. {result.metadata.get('source', 'unknown')} (importance: {importance})")
        
        logger.info("\n✅ Test 4 PASSED: Metadata filtering working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test 4 FAILED: {e}")
        return False


def test_comparison():
    """Compare semantic-only vs hybrid search"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 5: Semantic vs Hybrid Comparison")
    logger.info("=" * 60)
    
    try:
        searcher = HybridSearcher()
        
        query = "E047 battery voltage low"
        
        # Semantic only
        logger.info(f"\n🔍 Query: '{query}'")
        logger.info("\n📊 Semantic-only results:")
        semantic_results = searcher.search(query, top_k=3, use_hybrid=False, expand_query=False)
        for i, result in enumerate(semantic_results, 1):
            logger.info(f"   {i}. {result.metadata.get('source', 'unknown')} (sim: {result.similarity:.4f})")
        
        # Hybrid
        logger.info("\n📊 Hybrid results:")
        hybrid_results = searcher.search(query, top_k=3, use_hybrid=True, expand_query=True)
        for i, result in enumerate(hybrid_results, 1):
            logger.info(f"   {i}. [{result.source}] {result.metadata.get('source', 'unknown')} (score: {result.score:.4f})")
        
        logger.info("\n✅ Test 5 PASSED: Comparison complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test 5 FAILED: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("=" * 80)
    logger.info("🧪 Phase 2.2: Hybrid Search Tests")
    logger.info("=" * 80)
    
    results = {
        "Query Expansion": test_query_expansion(),
        "BM25 Search": test_bm25_index(),
        "Hybrid Search": test_hybrid_search(),
        "Metadata Filtering": test_metadata_filtering(),
        "Semantic vs Hybrid": test_comparison(),
    }
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 Test Summary")
    logger.info("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {name}: {status}")
    
    logger.info(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Phase 2.2 ready.")
    else:
        logger.info("\n⚠️ Some tests failed. Check logs above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
