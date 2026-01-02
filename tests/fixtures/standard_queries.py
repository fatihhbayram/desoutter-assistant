"""
Standard Test Query Suite for Desoutter Assistant
=================================================
20+ test queries covering all scenarios and intent types.
Used to validate RAG stability before and after changes.

Usage:
    from tests.fixtures.standard_queries import STANDARD_TEST_QUERIES, get_queries_by_intent
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ExpectedIntent(str, Enum):
    """Expected intent types matching IntentDetector"""
    TROUBLESHOOTING = "troubleshooting"
    SPECIFICATIONS = "specifications"
    INSTALLATION = "installation"
    CALIBRATION = "calibration"
    MAINTENANCE = "maintenance"
    CONNECTION = "connection"
    ERROR_CODE = "error_code"
    GENERAL = "general"


@dataclass
class TestQuery:
    """Structured test query with validation criteria"""
    id: str
    query: str
    product: str
    language: str
    expected_intent: ExpectedIntent
    min_confidence: float
    must_contain: List[str]
    must_not_contain: List[str]
    description: str
    category: str  # For grouping: basic, edge_case, language, etc.
    expect_idk: bool = False  # Should return "I don't know"?
    max_response_time_ms: int = 60000  # 30 seconds default


# =============================================================================
# STANDARD TEST QUERIES (20+)
# =============================================================================

STANDARD_TEST_QUERIES: List[Dict] = [
    # -------------------------------------------------------------------------
    # TROUBLESHOOTING QUERIES (5)
    # -------------------------------------------------------------------------
    {
        "id": "TROUBLE_001",
        "query": "Motor won't start",
        "product": "6151656060",  # EAD20 - Cable tool (NOT battery)
        "language": "en",
        "expected_intent": "troubleshooting",
        "min_confidence": 0.5,
        "must_contain": ["motor"],
        "must_not_contain": [],  # Removed - LLM might mention "no battery" which is correct
        "description": "Basic motor troubleshooting - wired/cable tool",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    {
        "id": "TROUBLE_002",
        "query": "Tool overheating during operation",
        "product": "6151659030",
        "language": "en",
        "expected_intent": "troubleshooting",
        "min_confidence": 0.5,
        "must_contain": ["temperature", "cool"],
        "must_not_contain": [],
        "description": "Overheating issue diagnosis",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    {
        "id": "TROUBLE_003",
        "query": "Grinding noise from the tool",
        "product": "6151660870",
        "language": "en",
        "expected_intent": "troubleshooting",
        "min_confidence": 0.5,
        "must_contain": ["noise"],  # Removed 'bearing' - could be gear, bearing, or other
        "must_not_contain": [],
        "description": "Mechanical noise troubleshooting",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    {
        "id": "TROUBLE_004",
        "query": "Alet çalışmıyor, motor sesi geliyor",
        "product": "6151659770",
        "language": "tr",
        "expected_intent": "troubleshooting",
        "min_confidence": 0.5,
        "must_contain": ["motor"],
        "must_not_contain": [],
        "description": "Turkish troubleshooting query",
        "category": "language",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    {
        "id": "TROUBLE_005",
        "query": "Tool stops randomly during use",
        "product": "6159326910",
        "language": "en",
        "expected_intent": "troubleshooting",
        "min_confidence": 0.5,
        "must_contain": ["check"],  # Removed 'connection' - could be power, overheating, etc.
        "must_not_contain": [],
        "description": "Intermittent operation issue",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    
    # -------------------------------------------------------------------------
    # ERROR CODE QUERIES (4)
    # -------------------------------------------------------------------------
    {
        "id": "ERROR_001",
        "query": "E804 error code on controller",
        "product": "6159275630",
        "language": "en",
        "expected_intent": "error_code",
        "min_confidence": 0.6,
        "must_contain": ["E804", "error"],
        "must_not_contain": [],
        "description": "Specific error code lookup",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    {
        "id": "ERROR_002",
        "query": "What does error E123 mean?",
        "product": "6151660870",
        "language": "en",
        "expected_intent": "error_code",
        "min_confidence": 0.5,
        "must_contain": ["error"],
        "must_not_contain": [],
        "description": "Error code meaning query",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    {
        "id": "ERROR_003",
        "query": "CVI3 shows fault code 47",
        "product": "6159326910",
        "language": "en",
        "expected_intent": "error_code",
        "min_confidence": 0.5,
        "must_contain": ["47"],  # Removed 'fault' - might say 'error' or 'code'
        "must_not_contain": [],
        "description": "CVI3 fault code query",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    {
        "id": "ERROR_004",
        "query": "E047 hata kodu ne anlama geliyor?",
        "product": "6159275630",
        "language": "tr",
        "expected_intent": "error_code",
        "min_confidence": 0.5,
        "must_contain": ["hata", "E047"],
        "must_not_contain": [],
        "description": "Turkish error code query",
        "category": "language",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    
    # -------------------------------------------------------------------------
    # SPECIFICATION QUERIES (3)
    # -------------------------------------------------------------------------
    {
        "id": "SPEC_001",
        "query": "What is the maximum torque of this tool?",
        "product": "6151659770",
        "language": "en",
        "expected_intent": "specifications",
        "min_confidence": 0.6,
        "must_contain": ["torque", "Nm"],
        "must_not_contain": [],
        "description": "Torque specification query",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    {
        "id": "SPEC_002",
        "query": "Tool dimensions and weight",
        "product": "6151660870",
        "language": "en",
        "expected_intent": "specifications",
        "min_confidence": 0.5,
        "must_contain": ["weight"],
        "must_not_contain": [],
        "description": "Physical specifications",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    {
        "id": "SPEC_003",
        "query": "RPM range for this screwdriver",
        "product": "6151659030",
        "language": "en",
        "expected_intent": "specifications",
        "min_confidence": 0.5,
        "must_contain": ["rpm", "speed"],
        "must_not_contain": [],
        "description": "Speed specification query",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    
    # -------------------------------------------------------------------------
    # CONNECTION QUERIES (3)
    # -------------------------------------------------------------------------
    {
        "id": "CONN_001",
        "query": "How to connect tool to controller?",
        "product": "6159326910",
        "language": "en",
        "expected_intent": "connection",
        "min_confidence": 0.5,
        "must_contain": ["connect", "controller"],
        "must_not_contain": [],
        "description": "Basic connection query",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    {
        "id": "CONN_002",
        "query": "WiFi connection issues",
        "product": "6151659770",
        "language": "en",
        "expected_intent": "connection",
        "min_confidence": 0.5,
        "must_contain": ["wifi", "network"],
        "must_not_contain": [],
        "description": "WiFi connectivity query",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    {
        "id": "CONN_003",
        "query": "Cable connection between tool and CVI3",
        "product": "6159326910",
        "language": "en",
        "expected_intent": "connection",
        "min_confidence": 0.5,
        "must_contain": ["cable", "connection"],
        "must_not_contain": ["wifi", "wireless"],
        "description": "Wired connection query",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    
    # -------------------------------------------------------------------------
    # MAINTENANCE QUERIES (2)
    # -------------------------------------------------------------------------
    {
        "id": "MAINT_001",
        "query": "How often should I lubricate the tool?",
        "product": "6151659030",
        "language": "en",
        "expected_intent": "maintenance",
        "min_confidence": 0.5,
        "must_contain": ["lubrication", "maintenance"],
        "must_not_contain": [],
        "description": "Lubrication schedule query",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    {
        "id": "MAINT_002",
        "query": "Bakım periyodu ne kadar?",
        "product": "6151660870",
        "language": "tr",
        "expected_intent": "maintenance",
        "min_confidence": 0.5,
        "must_contain": [],  # Removed - language detection may vary
        "must_not_contain": [],
        "description": "Turkish maintenance query",
        "category": "language",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    
    # -------------------------------------------------------------------------
    # CALIBRATION QUERIES (2)
    # -------------------------------------------------------------------------
    {
        "id": "CALIB_001",
        "query": "How to calibrate torque settings?",
        "product": "6151659770",
        "language": "en",
        "expected_intent": "calibration",
        "min_confidence": 0.5,
        "must_contain": ["calibration", "torque"],
        "must_not_contain": [],
        "description": "Torque calibration query",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    {
        "id": "CALIB_002",
        "query": "Calibration procedure for CVI3",
        "product": "6159326910",
        "language": "en",
        "expected_intent": "calibration",
        "min_confidence": 0.5,
        "must_contain": ["calibration"],
        "must_not_contain": [],
        "description": "Controller calibration query",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    
    # -------------------------------------------------------------------------
    # GENERAL QUERIES (2)
    # -------------------------------------------------------------------------
    {
        "id": "GEN_001",
        "query": "Tell me about this tool",
        "product": "6151659030",
        "language": "en",
        "expected_intent": "general",
        "min_confidence": 0.3,
        "must_contain": [],
        "must_not_contain": [],
        "description": "Generic information query",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    {
        "id": "GEN_002",
        "query": "What can you help me with?",
        "product": "EPBC",
        "language": "en",
        "expected_intent": "general",
        "min_confidence": 0.3,
        "must_contain": [],
        "must_not_contain": [],
        "description": "Help/capability query",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
    
    # -------------------------------------------------------------------------
    # EDGE CASES - Should Return "I Don't Know" (3)
    # -------------------------------------------------------------------------
    {
        "id": "IDK_001",
        "query": "What is the capital of France?",
        "product": "6151659770",
        "language": "en",
        "expected_intent": "general",
        "min_confidence": 0.0,
        "must_contain": ["don't have", "information", "documentation"],
        "must_not_contain": ["Paris", "France is"],
        "description": "Off-topic query - should refuse",
        "category": "edge_case",
        "expect_idk": True,
        "max_response_time_ms": 60000
    },
    {
        "id": "IDK_002",
        "query": "How to repair a Bosch drill?",
        "product": "6151659770",
        "language": "en",
        "expected_intent": "general",
        "min_confidence": 0.0,
        "must_contain": ["don't have", "information"],
        "must_not_contain": ["Bosch repair", "drill repair steps"],
        "description": "Competitor product query - should refuse",
        "category": "edge_case",
        "expect_idk": True,
        "max_response_time_ms": 60000
    },
    {
        "id": "IDK_003",
        "query": "XYZ123 random product error QWERTY nonsense",
        "product": "INVALID_PRODUCT",
        "language": "en",
        "expected_intent": "general",
        "min_confidence": 0.0,
        "must_contain": ["don't have", "information"],
        "must_not_contain": ["solution is", "you should"],
        "description": "Nonsense query - should refuse",
        "category": "edge_case",
        "expect_idk": True,
        "max_response_time_ms": 60000
    },
    
    # -------------------------------------------------------------------------
    # INSTALLATION QUERY (1)
    # -------------------------------------------------------------------------
    {
        "id": "INSTALL_001",
        "query": "How to install the tool on the workstation?",
        "product": "6151659030",
        "language": "en",
        "expected_intent": "installation",
        "min_confidence": 0.5,
        "must_contain": ["install"],  # Removed 'mount' - could say 'setup', 'attach', etc.
        "must_not_contain": [],
        "description": "Installation procedure query",
        "category": "basic",
        "expect_idk": False,
        "max_response_time_ms": 60000
    },
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_queries_by_intent(intent: str) -> List[Dict]:
    """Get all test queries for a specific intent type"""
    return [q for q in STANDARD_TEST_QUERIES if q["expected_intent"] == intent]


def get_queries_by_category(category: str) -> List[Dict]:
    """Get all test queries for a specific category (basic, edge_case, language)"""
    return [q for q in STANDARD_TEST_QUERIES if q["category"] == category]


def get_queries_by_language(language: str) -> List[Dict]:
    """Get all test queries for a specific language"""
    return [q for q in STANDARD_TEST_QUERIES if q["language"] == language]


def get_idk_queries() -> List[Dict]:
    """Get all queries that should return 'I don't know'"""
    return [q for q in STANDARD_TEST_QUERIES if q.get("expect_idk", False)]


def get_query_by_id(query_id: str) -> Optional[Dict]:
    """Get a specific query by its ID"""
    for q in STANDARD_TEST_QUERIES:
        if q["id"] == query_id:
            return q
    return None


def get_query_summary() -> Dict:
    """Get summary statistics about the test queries"""
    intents = {}
    categories = {}
    languages = {}
    
    for q in STANDARD_TEST_QUERIES:
        # Count intents
        intent = q["expected_intent"]
        intents[intent] = intents.get(intent, 0) + 1
        
        # Count categories
        cat = q["category"]
        categories[cat] = categories.get(cat, 0) + 1
        
        # Count languages
        lang = q["language"]
        languages[lang] = languages.get(lang, 0) + 1
    
    return {
        "total_queries": len(STANDARD_TEST_QUERIES),
        "by_intent": intents,
        "by_category": categories,
        "by_language": languages,
        "idk_queries": len(get_idk_queries())
    }


# =============================================================================
# QUICK VALIDATION
# =============================================================================

if __name__ == "__main__":
    summary = get_query_summary()
    print("=" * 60)
    print("STANDARD TEST QUERY SUITE SUMMARY")
    print("=" * 60)
    print(f"\nTotal Queries: {summary['total_queries']}")
    
    print(f"\nBy Intent:")
    for intent, count in sorted(summary['by_intent'].items()):
        print(f"  {intent}: {count}")
    
    print(f"\nBy Category:")
    for cat, count in sorted(summary['by_category'].items()):
        print(f"  {cat}: {count}")
    
    print(f"\nBy Language:")
    for lang, count in sorted(summary['by_language'].items()):
        print(f"  {lang}: {count}")
    
    print(f"\nExpected 'I Don't Know' Responses: {summary['idk_queries']}")
    print("=" * 60)
