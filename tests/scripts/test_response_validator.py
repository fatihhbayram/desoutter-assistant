#!/usr/bin/env python3
"""
Test Response Validation System
Tests hallucination detection through uncertainty phrases, number verification, etc.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.llm.response_validator import ResponseValidator, ValidationIssue


def test_uncertainty_detection():
    """Test 1: Detect uncertainty phrases"""
    print("\n" + "="*80)
    print("TEST 1: Uncertainty Phrase Detection")
    print("="*80)
    
    validator = ResponseValidator(
        max_uncertainty_count=2,
        flag_uncertain_responses=True
    )
    
    # Response with uncertainty phrases
    response = """The tool might be experiencing power issues. It could be a battery problem,
    or possibly a motor fault. I think you should probably check the connections first."""
    
    result = validator.validate_response(
        response=response,
        query="tool not working",
        context="Power supply troubleshooting section",
        product_info=None
    )
    
    print(f"✓ Issues Found: {len(result.issues)}")
    print(f"✓ Severity: {result.severity}")
    print(f"✓ Should Flag: {result.should_flag}")
    
    for issue in result.issues:
        print(f"  - {issue.type}: {issue.description}")
    
    assert len(result.issues) > 0, "Should detect uncertainty phrases"
    assert any(i.type == "uncertainty" for i in result.issues), "Should have uncertainty issues"
    print("\n✅ TEST 1 PASSED: Uncertainty phrases detected correctly")


def test_number_verification():
    """Test 2: Verify numerical values exist in context"""
    print("\n" + "="*80)
    print("TEST 2: Numerical Value Verification")
    print("="*80)
    
    validator = ResponseValidator(verify_numbers=True)
    
    # Response with numbers
    response = "The torque range is 5.0-15.0 Nm and the speed is 500 rpm"
    
    # Context WITHOUT those numbers (hallucination)
    context = "This is a general troubleshooting guide. Check motor connections."
    
    result = validator.validate_response(
        response=response,
        query="what are the specifications",
        context=context,
        product_info=None
    )
    
    print(f"✓ Issues Found: {len(result.issues)}")
    print(f"✓ Severity: {result.severity}")
    
    for issue in result.issues:
        print(f"  - {issue.type}: '{issue.detected_value}' - {issue.description}")
    
    # Should detect hallucinated numbers
    assert any(i.type == "number_mismatch" for i in result.issues), "Should detect hallucinated numbers"
    print("\n✅ TEST 2 PASSED: Hallucinated numbers detected")


def test_number_verification_valid():
    """Test 3: Valid numbers in context should pass"""
    print("\n" + "="*80)
    print("TEST 3: Valid Numerical Values (Should Pass)")
    print("="*80)
    
    validator = ResponseValidator(verify_numbers=True)
    
    # Response with numbers
    response = "The torque range is 5.0-15.0 Nm and the speed is 500 rpm"
    
    # Context WITH those numbers (valid)
    context = """Specifications:
    - Torque range: 5.0-15.0 Nm
    - Speed: 500 rpm
    - Weight: 1.2 kg"""
    
    result = validator.validate_response(
        response=response,
        query="what are the specifications",
        context=context,
        product_info=None
    )
    
    print(f"✓ Issues Found: {len(result.issues)}")
    print(f"✓ Severity: {result.severity}")
    
    # Should NOT detect number mismatches
    number_issues = [i for i in result.issues if i.type == "number_mismatch"]
    assert len(number_issues) == 0, "Should not flag valid numbers"
    print("\n✅ TEST 3 PASSED: Valid numbers not flagged")


def test_short_response():
    """Test 4: Detect too-short responses"""
    print("\n" + "="*80)
    print("TEST 4: Short Response Detection")
    print("="*80)
    
    validator = ResponseValidator(min_response_length=30)
    
    # Very short response
    response = "Check battery."
    
    result = validator.validate_response(
        response=response,
        query="tool not working",
        context="Troubleshooting guide",
        product_info=None
    )
    
    print(f"✓ Response Length: {len(response)} chars")
    print(f"✓ Issues Found: {len(result.issues)}")
    print(f"✓ Severity: {result.severity}")
    
    assert any(i.type == "short_response" for i in result.issues), "Should detect short response"
    assert result.severity == "high", "Short response should be high severity"
    print("\n✅ TEST 4 PASSED: Short response detected")


def test_product_mismatch():
    """Test 5: Detect product mismatch in response"""
    print("\n" + "="*80)
    print("TEST 5: Product Mismatch Detection")
    print("="*80)
    
    validator = ResponseValidator()
    
    # Response mentions wrong product
    response = "This issue is common with EPB tools. Check the EPB controller connection."
    
    # But query is about CVI3
    result = validator.validate_response(
        response=response,
        query="CVI3 connection problem",
        context="CVI3 troubleshooting",
        product_info={'model_name': 'CVI3-1000'}
    )
    
    print(f"✓ Issues Found: {len(result.issues)}")
    for issue in result.issues:
        print(f"  - {issue.type}: {issue.description}")
    
    assert any(i.type == "product_mismatch" for i in result.issues), "Should detect product mismatch"
    print("\n✅ TEST 5 PASSED: Product mismatch detected")


def test_forbidden_wifi_content():
    """Test 6: Detect WiFi suggestions for non-WiFi product"""
    print("\n" + "="*80)
    print("TEST 6: Forbidden WiFi Content Detection")
    print("="*80)
    
    validator = ResponseValidator()
    
    # Response suggests WiFi troubleshooting
    response = """The connection issue might be related to WiFi signal strength.
    Try moving closer to the access point or check the network settings."""
    
    # Product has NO WiFi capability
    result = validator.validate_response(
        response=response,
        query="tool not connecting",
        context="Connection troubleshooting",
        product_info={
            'model_name': 'EPB8-1800',  # Non-WiFi product
            'wireless': False,
            'battery_powered': True
        }
    )
    
    print(f"✓ Issues Found: {len(result.issues)}")
    for issue in result.issues:
        print(f"  - {issue.type}: {issue.description}")
    
    assert any(i.type == "forbidden_content" for i in result.issues), "Should detect forbidden WiFi content"
    assert result.severity == "high", "Forbidden content should be high severity"
    print("\n✅ TEST 6 PASSED: Forbidden WiFi content detected")


def test_forbidden_battery_content():
    """Test 7: Detect battery suggestions for non-battery product"""
    print("\n" + "="*80)
    print("TEST 7: Forbidden Battery Content Detection")
    print("="*80)
    
    validator = ResponseValidator()
    
    # Response suggests battery troubleshooting
    response = "Check the battery charge level and ensure the battery is properly seated."
    
    # Product is CORDED (not battery-powered)
    result = validator.validate_response(
        response=response,
        query="tool not starting",
        context="Power troubleshooting",
        product_info={
            'model_name': 'EAD-3000',  # Corded product
            'wireless': False,
            'battery_powered': False
        }
    )
    
    print(f"✓ Issues Found: {len(result.issues)}")
    for issue in result.issues:
        print(f"  - {issue.type}: {issue.description}")
    
    assert any(i.type == "forbidden_content" for i in result.issues), "Should detect forbidden battery content"
    print("\n✅ TEST 7 PASSED: Forbidden battery content detected")


def test_valid_response():
    """Test 8: Valid response should pass all checks"""
    print("\n" + "="*80)
    print("TEST 8: Valid Response (Should Pass)")
    print("="*80)
    
    validator = ResponseValidator()
    
    # Good response with specific steps, no uncertainty
    response = """Based on the troubleshooting guide, follow these steps:
    
    1. Check the power cable connection to the controller
    2. Verify the controller LED status (should be green)
    3. Inspect the tool cable for damage
    4. Test with a different port on the controller
    
    Required tools: None (visual inspection only)
    Safety: Disconnect power before inspecting cables
    Source: CVIC Troubleshooting Guide Section 4.2"""
    
    context = """CVIC Connection Troubleshooting
    Section 4.2: Power Connection Issues
    - Check power cable connections
    - Verify LED status (green = normal)
    - Inspect tool cable for damage
    - Try different ports"""
    
    result = validator.validate_response(
        response=response,
        query="CVIC not connecting",
        context=context,
        product_info={
            'model_name': 'CVIC-2000',  # Matches product in response
            'wireless': False,
            'battery_powered': False
        }
    )
    
    print(f"✓ Issues Found: {len(result.issues)}")
    print(f"✓ Severity: {result.severity}")
    print(f"✓ Is Valid: {result.is_valid}")
    print(f"✓ Should Flag: {result.should_flag}")
    
    # Valid response - may have 1-2 low-severity uncertainty phrases like "should be"  
    # but overall should be low severity and not flagged
    assert result.severity in ["none", "low"], f"Should be low/no severity, got {result.severity}"
    assert not result.should_flag, "Should not flag valid response"
    print("\n✅ TEST 8 PASSED: Valid response passed validation")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("RESPONSE VALIDATION SYSTEM TEST SUITE")
    print("="*80)
    print("Testing hallucination detection and response quality validation")
    
    tests = [
        test_uncertainty_detection,
        test_number_verification,
        test_number_verification_valid,
        test_short_response,
        test_product_mismatch,
        test_forbidden_wifi_content,
        test_forbidden_battery_content,
        test_valid_response
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
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total: {passed + failed} tests")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nResponse validation system working correctly:")
        print("  • Uncertainty phrase detection operational")
        print("  • Numerical value verification working")
        print("  • Short response detection functional")
        print("  • Product mismatch detection working")
        print("  • Forbidden content detection (WiFi/battery) operational")
        print("\nReady for integration testing with RAG engine.")
        return 0
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
