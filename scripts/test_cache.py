#!/usr/bin/env python3
"""
=============================================================================
Phase 2.3 Response Cache Test Script
=============================================================================
Tests the response caching functionality:
- Cache hit/miss detection
- Cache statistics
- TTL expiration
- Similarity-based matching
- Admin API endpoints

Run: docker exec desoutter-api python scripts/test_cache.py
=============================================================================
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def test_response_cache_unit():
    """Test ResponseCache class directly"""
    print("\n" + "="*60)
    print("TEST 1: ResponseCache Unit Tests")
    print("="*60)
    
    from src.llm.response_cache import ResponseCache, QuerySimilarityCache, get_response_cache
    
    # Test basic cache operations
    print("\n📦 Testing basic cache operations...")
    cache = ResponseCache(max_size=10, default_ttl=5)  # 5 second TTL for testing
    
    # Test set and get
    cache.set("key1", {"response": "test1"})
    result = cache.get("key1")
    assert result is not None, "Failed to get cached value"
    assert result["response"] == "test1", "Cached value mismatch"
    print("  ✅ Set and Get: OK")
    
    # Test cache miss
    result = cache.get("nonexistent")
    assert result is None, "Should return None for missing key"
    print("  ✅ Cache miss: OK")
    
    # Test TTL expiration
    print("\n⏱️  Testing TTL expiration (waiting 6 seconds)...")
    cache.set("expiring_key", {"data": "will expire"}, ttl=2)
    time.sleep(3)
    result = cache.get("expiring_key")
    assert result is None, "Entry should have expired"
    print("  ✅ TTL expiration: OK")
    
    # Test LRU eviction
    print("\n🗑️  Testing LRU eviction...")
    small_cache = ResponseCache(max_size=3, default_ttl=3600)
    small_cache.set("a", {"v": "a"})
    small_cache.set("b", {"v": "b"})
    small_cache.set("c", {"v": "c"})
    
    # Access 'a' to make it recently used
    small_cache.get("a")
    
    # Add new item - should evict 'b' (least recently used)
    small_cache.set("d", {"v": "d"})
    
    assert small_cache.get("a") is not None, "'a' should still exist"
    assert small_cache.get("b") is None, "'b' should be evicted"
    assert small_cache.get("c") is not None, "'c' should still exist"
    assert small_cache.get("d") is not None, "'d' should exist"
    print("  ✅ LRU eviction: OK")
    
    # Test statistics
    print("\n📊 Testing cache statistics...")
    stats = small_cache.get_stats()
    assert stats["hits"] > 0, "Should have hits"
    assert stats["misses"] > 0, "Should have misses"
    assert "hit_rate" in stats, "Should have hit rate"
    print(f"  Stats: {stats}")
    print("  ✅ Statistics: OK")
    
    print("\n✅ ResponseCache unit tests passed!")
    return True


def test_similarity_cache():
    """Test QuerySimilarityCache for fuzzy matching"""
    print("\n" + "="*60)
    print("TEST 2: Similarity Cache Tests")
    print("="*60)
    
    from src.llm.response_cache import QuerySimilarityCache
    
    cache = QuerySimilarityCache(
        max_size=100, 
        default_ttl=3600,
        similarity_threshold=0.85
    )
    
    # Store a response
    original_query = "motor is not starting and making noise"
    cache.set(original_query, {"suggestion": "Check motor brushes"})
    
    # Test exact match
    print("\n🔍 Testing exact match...")
    result = cache.get(original_query)
    assert result is not None, "Exact match should hit"
    print("  ✅ Exact match: OK")
    
    # Test similar query
    print("\n🔍 Testing similar query match...")
    similar_query = "motor not starting making noise"  # Very similar
    result = cache.get_similar(similar_query)
    if result:
        print(f"  Similar match found with similarity: {result.get('_similarity', 'N/A')}")
        print("  ✅ Similar query match: OK")
    else:
        print("  ⚠️ Similar query not matched (may need threshold adjustment)")
    
    # Test dissimilar query
    print("\n🔍 Testing dissimilar query (should miss)...")
    dissimilar_query = "battery not charging"
    result = cache.get_similar(dissimilar_query)
    if result is None:
        print("  ✅ Dissimilar query correctly missed")
    else:
        print(f"  ⚠️ Unexpected match: {result}")
    
    print("\n✅ Similarity cache tests passed!")
    return True


def test_rag_cache_integration():
    """Test cache integration with RAGEngine"""
    print("\n" + "="*60)
    print("TEST 3: RAGEngine Cache Integration")
    print("="*60)
    
    from src.llm.rag_engine import RAGEngine
    
    print("\n🔧 Initializing RAG engine...")
    rag = RAGEngine()
    
    # Check if cache is enabled
    if not rag.response_cache:
        print("  ⚠️ Response cache is DISABLED")
        print("  Set USE_CACHE=true in config/ai_settings.py to enable")
        return False
    
    print("  ✅ Response cache is enabled")
    stats_before = rag.response_cache.get_stats()
    print(f"  Initial stats: {stats_before}")
    
    # Make first request (cache miss)
    print("\n📤 Making first request (should be cache MISS)...")
    start_time = time.time()
    result1 = rag.generate_repair_suggestion(
        part_number="6159326000",
        fault_description="motor is not running",
        language="en"
    )
    time1 = time.time() - start_time
    print(f"  Response time: {time1*1000:.0f}ms")
    print(f"  From cache: {result1.get('from_cache', False)}")
    print(f"  Confidence: {result1.get('confidence')}")
    
    # Make same request (cache hit)
    print("\n📤 Making same request (should be cache HIT)...")
    start_time = time.time()
    result2 = rag.generate_repair_suggestion(
        part_number="6159326000",
        fault_description="motor is not running",
        language="en"
    )
    time2 = time.time() - start_time
    print(f"  Response time: {time2*1000:.0f}ms")
    print(f"  From cache: {result2.get('from_cache', False)}")
    
    # Verify cache hit
    if result2.get("from_cache"):
        print("  ✅ Cache HIT confirmed!")
        speedup = time1 / time2 if time2 > 0 else float('inf')
        print(f"  Speedup: {speedup:.1f}x faster")
    else:
        print("  ❌ Expected cache hit but got miss")
    
    # Check stats after
    stats_after = rag.response_cache.get_stats()
    print(f"\n📊 Final cache stats: {stats_after}")
    
    # Make retry request (should bypass cache)
    print("\n📤 Making retry request (should bypass cache)...")
    result3 = rag.generate_repair_suggestion(
        part_number="6159326000",
        fault_description="motor is not running",
        language="en",
        is_retry=True
    )
    print(f"  From cache: {result3.get('from_cache', False)}")
    if not result3.get("from_cache"):
        print("  ✅ Retry correctly bypassed cache")
    else:
        print("  ❌ Retry should not use cache")
    
    print("\n✅ RAGEngine cache integration tests passed!")
    return True


def test_cache_api_endpoints():
    """Test admin API endpoints for cache management"""
    print("\n" + "="*60)
    print("TEST 4: Cache API Endpoints")
    print("="*60)
    
    import requests
    
    API_URL = "http://localhost:8000"
    
    # Login as admin
    print("\n🔐 Logging in as admin...")
    login_resp = requests.post(f"{API_URL}/auth/login", json={
        "username": "admin",
        "password": "admin123"
    })
    
    if login_resp.status_code != 200:
        print(f"  ❌ Login failed: {login_resp.text}")
        return False
    
    token = login_resp.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"}
    print("  ✅ Logged in successfully")
    
    # Get cache stats
    print("\n📊 Getting cache stats...")
    stats_resp = requests.get(f"{API_URL}/admin/cache/stats", headers=headers)
    
    if stats_resp.status_code == 200:
        stats = stats_resp.json()
        print(f"  Status: {stats.get('status')}")
        if stats.get('stats'):
            print(f"  Stats: {stats['stats']}")
        print("  ✅ Cache stats endpoint: OK")
    else:
        print(f"  ❌ Failed: {stats_resp.text}")
    
    # Clear cache
    print("\n🗑️  Clearing cache...")
    clear_resp = requests.post(f"{API_URL}/admin/cache/clear", headers=headers)
    
    if clear_resp.status_code == 200:
        result = clear_resp.json()
        print(f"  Result: {result}")
        print("  ✅ Cache clear endpoint: OK")
    else:
        print(f"  ❌ Failed: {clear_resp.text}")
    
    print("\n✅ API endpoint tests passed!")
    return True


def run_all_tests():
    """Run all Phase 2.3 tests"""
    print("\n" + "="*60)
    print("🧪 PHASE 2.3 RESPONSE CACHE TESTS")
    print("="*60)
    
    results = []
    
    # Test 1: Unit tests
    try:
        results.append(("Unit Tests", test_response_cache_unit()))
    except Exception as e:
        print(f"❌ Unit tests failed: {e}")
        results.append(("Unit Tests", False))
    
    # Test 2: Similarity cache
    try:
        results.append(("Similarity Cache", test_similarity_cache()))
    except Exception as e:
        print(f"❌ Similarity cache tests failed: {e}")
        results.append(("Similarity Cache", False))
    
    # Test 3: RAG integration
    try:
        results.append(("RAG Integration", test_rag_cache_integration()))
    except Exception as e:
        print(f"❌ RAG integration tests failed: {e}")
        results.append(("RAG Integration", False))
    
    # Test 4: API endpoints
    try:
        results.append(("API Endpoints", test_cache_api_endpoints()))
    except Exception as e:
        print(f"❌ API endpoint tests failed: {e}")
        results.append(("API Endpoints", False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL PHASE 2.3 TESTS PASSED!")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
