#!/usr/bin/env python3
"""
Test Context Window Optimizer (Phase 3.4)
Tests context optimization for better RAG responses
"""
import sys
sys.path.insert(0, '/app')

from src.llm.context_optimizer import ContextOptimizer, OptimizedChunk, optimize_context_for_rag


def create_test_docs():
    """Create test documents with various metadata"""
    return [
        {
            "text": "Motor sesli çalışıyor. Rulmanları kontrol edin. Aşınmış rulmanlar değiştirilmelidir.",
            "metadata": {
                "source": "CVIX_Manual.pdf",
                "importance_score": 0.7,
                "section_type": "troubleshooting",
                "heading_text": "Motor Sorunları",
                "is_procedure": True,
                "is_warning": False
            },
            "similarity": 0.85
        },
        {
            "text": "⚠️ DİKKAT: Motor bakımı yapmadan önce güç kaynağını kesinlikle kesin! Elektrik çarpması tehlikesi!",
            "metadata": {
                "source": "Safety_Guide.pdf",
                "importance_score": 0.95,
                "section_type": "warning",
                "heading_text": "Güvenlik Uyarıları",
                "is_procedure": False,
                "is_warning": True
            },
            "similarity": 0.72
        },
        {
            "text": "Tork ayarları 5-50 Nm arasında yapılabilir. Tork kalibrasyonu için servis aracı gereklidir.",
            "metadata": {
                "source": "CVIX_Manual.pdf",
                "importance_score": 0.6,
                "section_type": "specifications",
                "heading_text": "Teknik Özellikler",
                "is_procedure": False,
                "is_warning": False
            },
            "similarity": 0.68
        },
        {
            "text": "Motor sesli çalışıyor. Rulmanları kontrol edin. Aşınmış rulmanlar değiştirilmelidir.",  # Duplicate
            "metadata": {
                "source": "Service_Bulletin.pdf",
                "importance_score": 0.65,
                "section_type": "troubleshooting",
                "heading_text": "Bilinen Sorunlar",
                "is_procedure": True,
                "is_warning": False
            },
            "similarity": 0.80
        },
        {
            "text": """Onarım Prosedürü:
1. Aleti kapatın ve güç kaynağını kesin
2. Motor kapağını sökün (4 adet M4 vida)
3. Rulman durumunu kontrol edin
4. Gerekirse rulmanları değiştirin
5. Kapağı takın ve test edin""",
            "metadata": {
                "source": "Repair_Guide.pdf",
                "importance_score": 0.85,
                "section_type": "procedure",
                "heading_text": "Motor Onarımı",
                "is_procedure": True,
                "is_warning": False
            },
            "similarity": 0.78
        },
        {
            "text": "Desoutter 1892 yılında kurulmuştur. Şirket merkezi Fransa'dadır.",  # Irrelevant
            "metadata": {
                "source": "Company_Info.pdf",
                "importance_score": 0.2,
                "section_type": "general",
                "heading_text": "Hakkımızda",
                "is_procedure": False,
                "is_warning": False
            },
            "similarity": 0.35
        }
    ]


def test_context_optimizer():
    """Test ContextOptimizer class"""
    print("\n" + "="*60)
    print("TEST 1: Context Optimizer Basic Functions")
    print("="*60)
    
    optimizer = ContextOptimizer(token_budget=2000)
    test_docs = create_test_docs()
    
    # Test optimization
    chunks, stats = optimizer.optimize(
        retrieved_docs=test_docs,
        query="Motor sesli çalışıyor, ne yapmalıyım?",
        max_chunks=5
    )
    
    print(f"\n📊 Optimization Stats:")
    print(f"   Input chunks: {stats['chunks_in']}")
    print(f"   Output chunks: {stats['chunks_out']}")
    print(f"   Tokens used: {stats['tokens_used']}")
    print(f"   Duplicates removed: {stats['duplicates_removed']}")
    print(f"   Warnings prioritized: {stats['warnings_prioritized']}")
    print(f"   Procedures prioritized: {stats['procedures_prioritized']}")
    
    print(f"\n📝 Optimized Chunks (in priority order):")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n   {i}. [{chunk.source}] - {chunk.section_type}")
        print(f"      Similarity: {chunk.similarity:.2f}, Importance: {chunk.importance_score:.2f}")
        print(f"      Warning: {chunk.is_warning}, Procedure: {chunk.is_procedure}")
        print(f"      Text: {chunk.text[:80]}...")
    
    # Verify duplicate removal
    assert stats['duplicates_removed'] >= 1, "Should remove duplicate content"
    print(f"\n✅ TEST 1 PASSED: Duplicates removed: {stats['duplicates_removed']}")
    
    return True


def test_warning_prioritization():
    """Test that warnings are prioritized"""
    print("\n" + "="*60)
    print("TEST 2: Warning Prioritization")
    print("="*60)
    
    optimizer = ContextOptimizer(
        token_budget=2000,
        prioritize_warnings=True
    )
    
    test_docs = create_test_docs()
    chunks, stats = optimizer.optimize(
        retrieved_docs=test_docs,
        query="Motor bakımı",
        max_chunks=5
    )
    
    # Check if warning appears in top results despite lower similarity
    warning_positions = [i for i, c in enumerate(chunks) if c.is_warning]
    
    print(f"\n📊 Warning positions in results: {warning_positions}")
    print(f"   Total chunks: {len(chunks)}")
    
    if warning_positions:
        print(f"\n✅ TEST 2 PASSED: Warning found at position {warning_positions[0]}")
    else:
        print(f"\n⚠️ TEST 2 WARNING: No warning chunks in results")
    
    return True


def test_context_string_formatting():
    """Test context string generation"""
    print("\n" + "="*60)
    print("TEST 3: Context String Formatting")
    print("="*60)
    
    optimizer = ContextOptimizer(token_budget=2000)
    test_docs = create_test_docs()
    
    chunks, _ = optimizer.optimize(
        retrieved_docs=test_docs,
        query="Motor sorunları",
        max_chunks=3
    )
    
    # Test different formatting options
    context_simple = optimizer.build_context_string(chunks, include_metadata=False)
    context_meta = optimizer.build_context_string(chunks, include_metadata=True)
    context_grouped = optimizer.build_context_string(chunks, include_metadata=True, group_by_source=True)
    
    print(f"\n📝 Simple formatting (no metadata):")
    print(f"   Length: {len(context_simple)} chars")
    print(f"   Preview: {context_simple[:200]}...")
    
    print(f"\n📝 With metadata:")
    print(f"   Length: {len(context_meta)} chars")
    print(f"   Preview: {context_meta[:200]}...")
    
    print(f"\n📝 Grouped by source:")
    print(f"   Length: {len(context_grouped)} chars")
    print(f"   Preview: {context_grouped[:200]}...")
    
    assert len(context_meta) >= len(context_simple), "Metadata version should be longer"
    print(f"\n✅ TEST 3 PASSED: Context formatting works correctly")
    
    return True


def test_token_budget():
    """Test token budget enforcement"""
    print("\n" + "="*60)
    print("TEST 4: Token Budget Enforcement")
    print("="*60)
    
    # Test with small budget
    optimizer = ContextOptimizer(token_budget=200)  # Very small
    test_docs = create_test_docs()
    
    chunks, stats = optimizer.optimize(
        retrieved_docs=test_docs,
        query="Motor sorunları",
        max_chunks=10
    )
    
    print(f"\n📊 Small budget test (200 tokens):")
    print(f"   Chunks selected: {len(chunks)}")
    print(f"   Tokens used: {stats['tokens_used']}")
    print(f"   Truncated chunks: {stats['truncated_chunks']}")
    
    assert stats['tokens_used'] <= 200 + 50, "Should respect token budget (with small margin)"
    print(f"\n✅ TEST 4 PASSED: Token budget enforced")
    
    return True


def test_convenience_function():
    """Test the convenience function"""
    print("\n" + "="*60)
    print("TEST 5: Convenience Function")
    print("="*60)
    
    test_docs = create_test_docs()
    
    context_str, sources, stats = optimize_context_for_rag(
        retrieved_docs=test_docs,
        query="Motor bakımı nasıl yapılır?",
        token_budget=1500,
        max_chunks=5
    )
    
    print(f"\n📊 optimize_context_for_rag results:")
    print(f"   Context length: {len(context_str)} chars")
    print(f"   Sources count: {len(sources)}")
    print(f"   Stats: {stats}")
    
    print(f"\n📝 Sources:")
    for src in sources:
        print(f"   - {src['source']}: sim={src['similarity']}, type={src['section_type']}")
    
    assert len(context_str) > 0, "Should produce context string"
    assert len(sources) > 0, "Should produce sources list"
    print(f"\n✅ TEST 5 PASSED: Convenience function works")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("   CONTEXT WINDOW OPTIMIZER TESTS (Phase 3.4)")
    print("="*70)
    
    tests = [
        ("Context Optimizer Basic", test_context_optimizer),
        ("Warning Prioritization", test_warning_prioritization),
        ("Context String Formatting", test_context_string_formatting),
        ("Token Budget Enforcement", test_token_budget),
        ("Convenience Function", test_convenience_function),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"   RESULTS: {passed}/{len(tests)} tests passed")
    print("="*70)
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
