#!/usr/bin/env python3
"""
Test Context Grounding & "I Don't Know" Logic
Tests the multi-factor sufficiency scoring and IDK response generation
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.llm.context_grounding import ContextSufficiencyScorer, build_idk_response


def test_insufficient_context():
    """Test 1: Insufficient context (low similarity) → refuse to answer"""
    print("\n" + "="*80)
    print("TEST 1: Insufficient Context (Low Similarity)")
    print("="*80)
    
    scorer = ContextSufficiencyScorer(
        sufficiency_threshold=0.5,
        min_similarity=0.35,
        min_docs=2
    )
    
    # Simulate poor retrieval results (below threshold)
    retrieved_docs = [
        {"text": "Some random text", "similarity": 0.20},
        {"text": "Unrelated content", "similarity": 0.15}
    ]
    
    result = scorer.calculate_sufficiency_score(
        query="tool not starting",
        retrieved_docs=retrieved_docs,
        avg_similarity=0.175
    )
    
    print(f"✓ Sufficiency Score: {result.score:.3f}")
    print(f"✓ Is Sufficient: {result.is_sufficient}")
    print(f"✓ Recommendation: {result.recommendation}")
    print(f"✓ Reason: {result.reason}")
    print(f"✓ Factors: {result.factors}")
    
    assert not result.is_sufficient, "Should be insufficient"
    assert result.recommendation == "refuse", f"Should refuse, got {result.recommendation}"
    print("\n✅ TEST 1 PASSED: System correctly refuses low-quality context")


def test_borderline_context():
    """Test 2: Borderline context (0.3-0.5) → answer with caution"""
    print("\n" + "="*80)
    print("TEST 2: Borderline Context (Moderate Quality)")
    print("="*80)
    
    scorer = ContextSufficiencyScorer(sufficiency_threshold=0.5)
    
    # Simulate moderate retrieval results
    retrieved_docs = [
        {"text": "Motor troubleshooting section", "similarity": 0.45},
        {"text": "Power supply check", "similarity": 0.40},
        {"text": "General maintenance", "similarity": 0.35}
    ]
    
    result = scorer.calculate_sufficiency_score(
        query="motor not working",
        retrieved_docs=retrieved_docs,
        avg_similarity=0.40
    )
    
    print(f"✓ Sufficiency Score: {result.score:.3f}")
    print(f"✓ Is Sufficient: {result.is_sufficient}")
    print(f"✓ Recommendation: {result.recommendation}")
    print(f"✓ Reason: {result.reason}")
    
    # Borderline should be cautious
    assert result.recommendation == "answer_with_caution", f"Should be cautious, got {result.recommendation}"
    print("\n✅ TEST 2 PASSED: System correctly flags borderline context")


def test_sufficient_context():
    """Test 3: Sufficient context (>0.5) → answer normally"""
    print("\n" + "="*80)
    print("TEST 3: Sufficient Context (High Quality)")
    print("="*80)
    
    scorer = ContextSufficiencyScorer(sufficiency_threshold=0.5)
    
    # Simulate good retrieval results
    retrieved_docs = [
        {"text": "Motor troubleshooting: Check power connections, verify motor cable integrity", "similarity": 0.75},
        {"text": "Motor fault diagnosis: Test motor resistance, check for shorts", "similarity": 0.70},
        {"text": "Motor replacement procedure: Remove housing, disconnect wiring", "similarity": 0.65},
        {"text": "Motor specifications: 24V DC, 150W nominal power", "similarity": 0.55}
    ]
    
    result = scorer.calculate_sufficiency_score(
        query="motor not working properly",
        retrieved_docs=retrieved_docs,
        avg_similarity=0.66
    )
    
    print(f"✓ Sufficiency Score: {result.score:.3f}")
    print(f"✓ Is Sufficient: {result.is_sufficient}")
    print(f"✓ Recommendation: {result.recommendation}")
    print(f"✓ Reason: {result.reason}")
    
    assert result.is_sufficient, "Should be sufficient"
    assert result.recommendation in ["answer", "answer_with_caution"], f"Should answer, got {result.recommendation}"
    print("\n✅ TEST 3 PASSED: System correctly accepts high-quality context")


def test_no_documents():
    """Test 4: No documents retrieved → refuse"""
    print("\n" + "="*80)
    print("TEST 4: No Documents Retrieved")
    print("="*80)
    
    scorer = ContextSufficiencyScorer(sufficiency_threshold=0.5)
    
    result = scorer.calculate_sufficiency_score(
        query="completely unknown query",
        retrieved_docs=[],
        avg_similarity=0.0
    )
    
    print(f"✓ Sufficiency Score: {result.score:.3f}")
    print(f"✓ Recommendation: {result.recommendation}")
    print(f"✓ Reason: {result.reason}")
    
    assert result.score == 0.0, "Score should be 0.0"
    assert not result.is_sufficient, "Should be insufficient"
    assert result.recommendation == "refuse", "Should refuse"
    print("\n✅ TEST 4 PASSED: System correctly handles no documents")


def test_term_coverage():
    """Test 5: Term coverage calculation"""
    print("\n" + "="*80)
    print("TEST 5: Query Term Coverage Calculation")
    print("="*80)
    
    scorer = ContextSufficiencyScorer(sufficiency_threshold=0.5)
    
    # Query with specific terms
    query = "WiFi connection problem with CVI3 controller"
    
    # Good term coverage
    good_docs = [
        {"text": "WiFi connectivity issues with CVI3: Check network settings, verify WiFi module", "similarity": 0.60},
        {"text": "CVI3 controller WiFi troubleshooting: Reset connection, check signal strength", "similarity": 0.55}
    ]
    
    # Poor term coverage
    poor_docs = [
        {"text": "General maintenance procedures for tools", "similarity": 0.40},
        {"text": "Battery replacement steps", "similarity": 0.35}
    ]
    
    good_result = scorer.calculate_sufficiency_score(query, good_docs, 0.575)
    poor_result = scorer.calculate_sufficiency_score(query, poor_docs, 0.375)
    
    print(f"✓ Good Coverage - Term Factor: {good_result.factors['term_coverage']:.2f}")
    print(f"✓ Poor Coverage - Term Factor: {poor_result.factors['term_coverage']:.2f}")
    
    assert good_result.factors['term_coverage'] > poor_result.factors['term_coverage'], \
        "Good docs should have higher term coverage"
    print("\n✅ TEST 5 PASSED: Term coverage calculation working correctly")


def test_idk_response_en():
    """Test 6: "I don't know" response generation (English)"""
    print("\n" + "="*80)
    print("TEST 6: IDK Response Generation (English)")
    print("="*80)
    
    response = build_idk_response(
        query="What color is the CVI3 unit?",
        product_model="CVI3",
        reason="Information about product appearance not available in technical documentation",
        language="en"
    )
    
    print("Generated Response:")
    print("-" * 80)
    print(response)
    print("-" * 80)
    
    assert "don't have sufficient information" in response.lower(), "Should mention insufficient info"
    assert "CVI3" in response, "Should mention product"
    assert "support@desoutter.com" in response, "Should provide support contact"
    print("\n✅ TEST 6 PASSED: English IDK response generated correctly")


def test_idk_response_tr():
    """Test 7: "I don't know" response generation (Turkish)"""
    print("\n" + "="*80)
    print("TEST 7: IDK Response Generation (Turkish)")
    print("="*80)
    
    response = build_idk_response(
        query="CVI3 ünitesinin rengi nedir?",
        product_model="CVI3",
        reason="Teknik dokümantasyonda ürün görünümü hakkında bilgi mevcut değil",
        language="tr"
    )
    
    print("Generated Response (Turkish):")
    print("-" * 80)
    print(response)
    print("-" * 80)
    
    assert "yeterli bilgiye sahip değilim" in response.lower(), "Should mention insufficient info in Turkish"
    assert "CVI3" in response, "Should mention product"
    assert "support@desoutter.com" in response, "Should provide support contact"
    print("\n✅ TEST 7 PASSED: Turkish IDK response generated correctly")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("CONTEXT GROUNDING & \"I DON'T KNOW\" LOGIC TEST SUITE")
    print("="*80)
    print("Testing multi-factor sufficiency scoring and response grounding")
    
    tests = [
        test_insufficient_context,
        test_borderline_context,
        test_sufficient_context,
        test_no_documents,
        test_term_coverage,
        test_idk_response_en,
        test_idk_response_tr
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n❌ TEST FAILED: {test_func.__name__}")
            print(f"   Error: {e}")
        except Exception as e:
            failed += 1
            print(f"\n❌ TEST ERROR: {test_func.__name__}")
            print(f"   Error: {type(e).__name__}: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total: {passed + failed} tests")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nContext grounding system is working correctly:")
        print("  • Multi-factor sufficiency scoring operational")
        print("  • Handles insufficient/borderline/sufficient contexts appropriately")
        print("  • Term coverage calculation working")
        print("  • IDK responses generated for EN/TR languages")
        print("\nReady for integration testing with RAG engine.")
        return 0
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
