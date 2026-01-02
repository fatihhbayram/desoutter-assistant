#!/usr/bin/env python3
"""
RAG Stability Test Suite
========================
Validates RAG system against standard test queries.
Measures pass rate, response time, and quality metrics.

Usage:
    python -m pytest tests/test_rag_stability.py -v
    python tests/test_rag_stability.py  # Direct execution
    
Environment Variables:
    RAG_TEST_API_URL: API base URL (default: http://localhost:8000)
    RAG_TEST_TIMEOUT: Request timeout in seconds (default: 60)
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests

# Import test queries
from tests.fixtures.standard_queries import (
    STANDARD_TEST_QUERIES,
    get_query_summary,
    get_idk_queries
)

# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = os.getenv("RAG_TEST_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.getenv("RAG_TEST_TIMEOUT", "60"))
DIAGNOSE_ENDPOINT = f"{API_BASE_URL}/diagnose"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# WARM-UP FUNCTION
# =============================================================================

def warm_up_system(verbose: bool = True) -> bool:
    """
    Warm up the system before running actual tests.
    
    This runs a dummy query to:
    1. Load the LLM model into GPU memory
    2. Initialize ChromaDB connections
    3. Prime any caches
    
    Args:
        verbose: Print warm-up status
        
    Returns:
        True if warm-up successful, False otherwise
    """
    if verbose:
        print("\n" + "=" * 70)
        print("🔥 WARMING UP SYSTEM")
        print("=" * 70)
        print("Loading LLM model and initializing connections...")
    
    # Dummy warm-up query
    warmup_payload = {
        "part_number": "6151659770",
        "fault_description": "system warm-up test query",
        "language": "en"
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            DIAGNOSE_ENDPOINT,
            json=warmup_payload,
            timeout=120  # Longer timeout for cold start
        )
        warmup_time = time.time() - start_time
        
        if response.status_code == 200:
            if verbose:
                print(f"✅ Warm-up complete ({warmup_time:.1f}s)")
                print("   Waiting 2 seconds for system to stabilize...")
            time.sleep(2)
            if verbose:
                print("=" * 70 + "\n")
            return True
        else:
            if verbose:
                print(f"⚠️  Warm-up returned status {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        if verbose:
            print("⚠️  Warm-up timed out (120s) - system may be slow")
        return False
    except requests.exceptions.ConnectionError as e:
        if verbose:
            print(f"❌ Warm-up failed - cannot connect to API")
            print(f"   Error: {str(e)[:100]}")
        return False
    except Exception as e:
        if verbose:
            print(f"❌ Warm-up failed: {str(e)[:100]}")
        return False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TestResult:
    """Result of a single test query"""
    query_id: str
    query: str
    product: str
    language: str
    
    # Execution results
    success: bool
    response_time_ms: int
    error_message: Optional[str] = None
    
    # Response data
    confidence: Optional[str] = None
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    sufficiency_score: Optional[float] = None
    sources_count: int = 0
    response_length: int = 0
    from_cache: bool = False
    
    # Validation results
    intent_match: bool = False
    confidence_adequate: bool = False
    contains_required: bool = False
    excludes_forbidden: bool = False
    response_time_ok: bool = False
    idk_correct: bool = True  # True if IDK expectation matches reality
    
    # Overall
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class TestSuiteResult:
    """Aggregated results of the test suite"""
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    errors: int
    pass_rate: float
    
    # Timing metrics
    total_time_ms: int
    avg_response_time_ms: float
    min_response_time_ms: int
    max_response_time_ms: int
    
    # Breakdown
    by_intent: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_category: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Details
    results: List[TestResult] = field(default_factory=list)
    failed_tests: List[str] = field(default_factory=list)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def run_single_test(test_query: Dict) -> TestResult:
    """
    Execute a single test query against the RAG API
    
    Args:
        test_query: Test query definition from standard_queries.py
        
    Returns:
        TestResult with all validation results
    """
    query_id = test_query["id"]
    
    result = TestResult(
        query_id=query_id,
        query=test_query["query"],
        product=test_query["product"],
        language=test_query["language"],
        success=False,
        response_time_ms=0
    )
    
    # Prepare request
    payload = {
        "part_number": test_query["product"],
        "fault_description": test_query["query"],
        "language": test_query["language"]
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            DIAGNOSE_ENDPOINT,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        
        result.response_time_ms = int((time.time() - start_time) * 1000)
        
        if response.status_code != 200:
            result.error_message = f"HTTP {response.status_code}: {response.text[:200]}"
            return result
        
        data = response.json()
        result.success = True
        
        # Extract response data
        result.confidence = data.get("confidence", "unknown")
        result.intent = data.get("intent", "unknown")
        result.intent_confidence = data.get("intent_confidence", 0.0)
        result.sufficiency_score = data.get("sufficiency_score")
        result.sources_count = len(data.get("sources", []))
        result.response_length = len(data.get("suggestion", ""))
        result.from_cache = data.get("from_cache", False)
        
        suggestion = data.get("suggestion", "").lower()
        
        # =================================================================
        # VALIDATION CHECKS
        # =================================================================
        
        # 1. Intent Match
        expected_intent = test_query["expected_intent"]
        result.intent_match = (result.intent == expected_intent)
        if not result.intent_match:
            result.failure_reasons.append(
                f"Intent mismatch: expected '{expected_intent}', got '{result.intent}'"
            )
        
        # 2. Confidence Adequate
        min_conf = test_query.get("min_confidence", 0.5)
        conf_value = {"high": 0.8, "medium": 0.5, "low": 0.3}.get(result.confidence, 0.0)
        result.confidence_adequate = (conf_value >= min_conf)
        if not result.confidence_adequate and not test_query.get("expect_idk", False):
            result.failure_reasons.append(
                f"Confidence too low: expected >={min_conf}, got {result.confidence} ({conf_value})"
            )
        
        # 3. Contains Required Terms
        must_contain = test_query.get("must_contain", [])
        missing_terms = [term for term in must_contain if term.lower() not in suggestion]
        result.contains_required = (len(missing_terms) == 0)
        if not result.contains_required:
            result.failure_reasons.append(
                f"Missing required terms: {missing_terms}"
            )
        
        # 4. Excludes Forbidden Terms
        must_not_contain = test_query.get("must_not_contain", [])
        found_forbidden = [term for term in must_not_contain if term.lower() in suggestion]
        result.excludes_forbidden = (len(found_forbidden) == 0)
        if not result.excludes_forbidden:
            result.failure_reasons.append(
                f"Contains forbidden terms: {found_forbidden}"
            )
        
        # 5. Response Time OK
        max_time = test_query.get("max_response_time_ms", 30000)
        result.response_time_ok = (result.response_time_ms <= max_time)
        if not result.response_time_ok:
            result.failure_reasons.append(
                f"Response too slow: {result.response_time_ms}ms > {max_time}ms"
            )
        
        # 6. IDK Expectation Check
        expect_idk = test_query.get("expect_idk", False)
        is_idk_response = any(phrase in suggestion.lower() for phrase in [
            "don't know", "don't have", "cannot", "no information", "bilmiyorum", 
            "insufficient", "unable to", "i'm sorry", "off-topic", "not related"
        ])
        result.idk_correct = (expect_idk == is_idk_response)
        if not result.idk_correct:
            if expect_idk:
                result.failure_reasons.append(
                    "Expected 'I don't know' response but got answer"
                )
            else:
                result.failure_reasons.append(
                    "Got 'I don't know' response but expected answer"
                )
        
        # =================================================================
        # OVERALL PASS/FAIL
        # =================================================================
        
        # For IDK queries: pass if IDK correct
        if expect_idk:
            result.passed = result.idk_correct and result.excludes_forbidden
        else:
            # For normal queries: all checks must pass
            result.passed = (
                result.intent_match and
                result.confidence_adequate and
                result.contains_required and
                result.excludes_forbidden and
                result.response_time_ok and
                result.idk_correct
            )
        
    except requests.exceptions.Timeout:
        result.error_message = f"Request timeout after {REQUEST_TIMEOUT}s"
        result.failure_reasons.append("Timeout")
    except requests.exceptions.ConnectionError as e:
        result.error_message = f"Connection error: {str(e)[:100]}"
        result.failure_reasons.append("Connection failed")
    except Exception as e:
        result.error_message = f"Unexpected error: {str(e)[:100]}"
        result.failure_reasons.append(f"Error: {type(e).__name__}")
    
    return result


def run_test_suite(
    queries: List[Dict] = None,
    verbose: bool = True
) -> TestSuiteResult:
    """
    Run the complete test suite
    
    Args:
        queries: List of test queries (default: STANDARD_TEST_QUERIES)
        verbose: Print progress during execution
        
    Returns:
        TestSuiteResult with all metrics
    """
    if queries is None:
        queries = STANDARD_TEST_QUERIES
    
    if verbose:
        print("\n" + "=" * 70)
        print("🧪 RAG STABILITY TEST SUITE")
        print("=" * 70)
        print(f"API: {API_BASE_URL}")
        print(f"Tests: {len(queries)}")
        print(f"Timeout: {REQUEST_TIMEOUT}s")
        print("=" * 70 + "\n")
    
    results: List[TestResult] = []
    response_times: List[int] = []
    suite_start = time.time()
    
    # Initialize counters
    by_intent: Dict[str, Dict[str, int]] = {}
    by_category: Dict[str, Dict[str, int]] = {}
    
    for i, test_query in enumerate(queries, 1):
        query_id = test_query["id"]
        
        if verbose:
            print(f"[{i:02d}/{len(queries)}] {query_id}: ", end="", flush=True)
        
        result = run_single_test(test_query)
        results.append(result)
        
        if result.success:
            response_times.append(result.response_time_ms)
        
        # Update intent stats
        intent = test_query["expected_intent"]
        if intent not in by_intent:
            by_intent[intent] = {"total": 0, "passed": 0, "failed": 0}
        by_intent[intent]["total"] += 1
        if result.passed:
            by_intent[intent]["passed"] += 1
        else:
            by_intent[intent]["failed"] += 1
        
        # Update category stats
        category = test_query.get("category", "unknown")
        if category not in by_category:
            by_category[category] = {"total": 0, "passed": 0, "failed": 0}
        by_category[category]["total"] += 1
        if result.passed:
            by_category[category]["passed"] += 1
        else:
            by_category[category]["failed"] += 1
        
        # Print result
        if verbose:
            if result.passed:
                print(f"✅ PASS ({result.response_time_ms}ms)")
            elif result.success:
                print(f"❌ FAIL ({result.response_time_ms}ms)")
                for reason in result.failure_reasons[:2]:
                    print(f"      └─ {reason}")
            else:
                print(f"⚠️  ERROR: {result.error_message[:50]}")
    
    total_time = int((time.time() - suite_start) * 1000)
    
    # Calculate metrics
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if r.success and not r.passed)
    errors = sum(1 for r in results if not r.success)
    
    suite_result = TestSuiteResult(
        timestamp=datetime.now().isoformat(),
        total_tests=len(queries),
        passed=passed,
        failed=failed,
        errors=errors,
        pass_rate=(passed / len(queries)) * 100 if queries else 0,
        total_time_ms=total_time,
        avg_response_time_ms=sum(response_times) / len(response_times) if response_times else 0,
        min_response_time_ms=min(response_times) if response_times else 0,
        max_response_time_ms=max(response_times) if response_times else 0,
        by_intent=by_intent,
        by_category=by_category,
        results=results,
        failed_tests=[r.query_id for r in results if not r.passed]
    )
    
    return suite_result


def print_summary(result: TestSuiteResult):
    """Print a formatted summary of test results"""
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    # Overall stats
    status = "✅ PASS" if result.pass_rate >= 80 else "❌ NEEDS WORK"
    print(f"\nOverall: {status}")
    print(f"  Total:  {result.total_tests}")
    print(f"  Passed: {result.passed} ({result.pass_rate:.1f}%)")
    print(f"  Failed: {result.failed}")
    print(f"  Errors: {result.errors}")
    
    # Timing
    print(f"\nTiming:")
    print(f"  Total:   {result.total_time_ms / 1000:.1f}s")
    print(f"  Average: {result.avg_response_time_ms:.0f}ms")
    print(f"  Min:     {result.min_response_time_ms}ms")
    print(f"  Max:     {result.max_response_time_ms}ms")
    
    # By Intent
    print(f"\nBy Intent:")
    for intent, stats in sorted(result.by_intent.items()):
        rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        icon = "✅" if rate >= 80 else "⚠️" if rate >= 50 else "❌"
        print(f"  {icon} {intent}: {stats['passed']}/{stats['total']} ({rate:.0f}%)")
    
    # By Category
    print(f"\nBy Category:")
    for category, stats in sorted(result.by_category.items()):
        rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        icon = "✅" if rate >= 80 else "⚠️" if rate >= 50 else "❌"
        print(f"  {icon} {category}: {stats['passed']}/{stats['total']} ({rate:.0f}%)")
    
    # Failed tests
    if result.failed_tests:
        print(f"\nFailed Tests ({len(result.failed_tests)}):")
        for test_id in result.failed_tests[:10]:
            # Find the result for this test
            test_result = next((r for r in result.results if r.query_id == test_id), None)
            if test_result:
                reasons = ", ".join(test_result.failure_reasons[:2])
                print(f"  ❌ {test_id}: {reasons[:60]}")
        if len(result.failed_tests) > 10:
            print(f"  ... and {len(result.failed_tests) - 10} more")
    
    print("\n" + "=" * 70)
    
    # Recommendation
    if result.pass_rate >= 80:
        print("🎉 System is STABLE (pass rate ≥80%)")
        print("   Safe to proceed with optimizations.")
    elif result.pass_rate >= 60:
        print("⚠️  System needs IMPROVEMENT (60-80% pass rate)")
        print("   Focus on fixing failed test cases before adding features.")
    else:
        print("🚨 System is UNSTABLE (pass rate <60%)")
        print("   Critical fixes needed before any other work.")
    
    print("=" * 70 + "\n")


def save_results(result: TestSuiteResult, output_path: str = None):
    """Save test results to JSON file"""
    if output_path is None:
        output_path = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert to serializable format
    data = {
        "timestamp": result.timestamp,
        "summary": {
            "total_tests": result.total_tests,
            "passed": result.passed,
            "failed": result.failed,
            "errors": result.errors,
            "pass_rate": result.pass_rate,
        },
        "timing": {
            "total_time_ms": result.total_time_ms,
            "avg_response_time_ms": result.avg_response_time_ms,
            "min_response_time_ms": result.min_response_time_ms,
            "max_response_time_ms": result.max_response_time_ms,
        },
        "by_intent": result.by_intent,
        "by_category": result.by_category,
        "failed_tests": result.failed_tests,
        "results": [
            {
                "query_id": r.query_id,
                "passed": r.passed,
                "response_time_ms": r.response_time_ms,
                "confidence": r.confidence,
                "intent": r.intent,
                "failure_reasons": r.failure_reasons
            }
            for r in result.results
        ]
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"📁 Results saved to: {output_path}")
    return output_path


# =============================================================================
# PYTEST COMPATIBLE TESTS
# =============================================================================

def test_api_health():
    """Test that API is reachable"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
    except requests.exceptions.ConnectionError:
        assert False, f"Cannot connect to API at {API_BASE_URL}"


def test_diagnose_endpoint_basic():
    """Test basic diagnose endpoint functionality"""
    payload = {
        "part_number": "6151659770",
        "fault_description": "Motor won't start",
        "language": "en"
    }
    
    response = requests.post(DIAGNOSE_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
    assert response.status_code == 200, f"Diagnose failed: {response.status_code}"
    
    data = response.json()
    assert "suggestion" in data, "Missing 'suggestion' in response"
    assert "confidence" in data, "Missing 'confidence' in response"
    assert "sources" in data, "Missing 'sources' in response"


def test_pass_rate_minimum():
    """Test that pass rate meets minimum threshold (80%)"""
    result = run_test_suite(verbose=False)
    assert result.pass_rate >= 80, f"Pass rate {result.pass_rate:.1f}% is below 80% threshold"


def test_response_time_average():
    """Test that average response time is acceptable (<10s)"""
    result = run_test_suite(verbose=False)
    assert result.avg_response_time_ms <= 10000, \
        f"Average response time {result.avg_response_time_ms}ms exceeds 10s threshold"


def test_idk_responses():
    """Test that IDK queries correctly return 'I don't know' responses"""
    idk_queries = get_idk_queries()
    
    for query in idk_queries:
        result = run_single_test(query)
        assert result.idk_correct, \
            f"Query {query['id']} should return IDK response but didn't"


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point for command-line execution"""
    import argparse
    
    # Default values for argparse
    default_api_url = os.getenv("RAG_TEST_API_URL", "http://localhost:8000")
    default_timeout = int(os.getenv("RAG_TEST_TIMEOUT", "60"))
    
    parser = argparse.ArgumentParser(description="RAG Stability Test Suite")
    parser.add_argument("--api", default=default_api_url, help="API base URL")
    parser.add_argument("--timeout", type=int, default=default_timeout, help="Request timeout")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument("--output", help="Output file path for results")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--category", help="Run only specific category (basic, edge_case, language)")
    parser.add_argument("--intent", help="Run only specific intent")
    
    args = parser.parse_args()
    
    # Update module-level variables
    global API_BASE_URL, REQUEST_TIMEOUT, DIAGNOSE_ENDPOINT
    API_BASE_URL = args.api
    REQUEST_TIMEOUT = args.timeout
    DIAGNOSE_ENDPOINT = f"{args.api}/diagnose"
    
    # Filter queries if needed
    queries = STANDARD_TEST_QUERIES
    if args.category:
        queries = [q for q in queries if q.get("category") == args.category]
        print(f"Running {len(queries)} tests for category: {args.category}")
    if args.intent:
        queries = [q for q in queries if q.get("expected_intent") == args.intent]
        print(f"Running {len(queries)} tests for intent: {args.intent}")
    
    # Warm-up system before running actual tests
    if not args.quiet:
        warmup_success = warm_up_system(verbose=True)
        if not warmup_success:
            print("⚠️  Warm-up failed, proceeding with tests anyway...")
    
    # Run tests
    result = run_test_suite(queries=queries, verbose=not args.quiet)
    
    # Print summary
    if not args.quiet:
        print_summary(result)
    
    # Save if requested
    if args.save:
        save_results(result, args.output)
    
    # Exit code based on pass rate
    if result.pass_rate >= 80:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
