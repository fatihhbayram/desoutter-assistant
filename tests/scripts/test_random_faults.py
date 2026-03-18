"""
Random 10 fault test script - tests real bulletin scenarios.
"""
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger

logger = setup_logger("test_random_faults")

# Real fault scenarios from different bulletins
RANDOM_FAULT_TESTS = [
    # From ESDE23007 - EABS Boost failure
    {
        "name": "EABS E047 Boost Mode Failure",
        "query": "EABS tool shows E047 error boost mode failure",
        "product": "EABS8-1500-4Q",
        "expected_bulletin_keywords": ["boost", "e047", "speaker", "mainboard"],
    },
    # From ESDE25013 - ERS Trigger issue
    {
        "name": "ERS Trigger Not Starting Motor",
        "query": "ERS tool trigger pressed but motor does not start",
        "product": "ERS6-50",
        "expected_bulletin_keywords": ["trigger", "motor", "hall", "start"],
    },
    # From ESDE25004 - ERS/EPB8 Transducer
    {
        "name": "ERS Transducer Fault",
        "query": "ERS transducer fault error during tightening",
        "product": "ERS8",
        "expected_bulletin_keywords": ["transducer", "fault", "calibration"],
    },
    # From ESDE23008 - Connect firmware upgrade
    {
        "name": "Connect Firmware Upgrade Failure",
        "query": "CONNECT controller firmware upgrade failed stuck",
        "product": "CONNECT",
        "expected_bulletin_keywords": ["firmware", "upgrade", "update", "flash"],
    },
    # From ESDE21004 - ODIN E064 pairing
    {
        "name": "Battery Tool E064 Pairing Error",
        "query": "EABS E064 error pairing with controller",
        "product": "EABS8",
        "expected_bulletin_keywords": ["e064", "pairing", "odin", "wifi"],
    },
    # From ESDE23010 - CVI3 touchscreen
    {
        "name": "CVI3 Touchscreen Not Working",
        "query": "CVI3 touchscreen display not responding",
        "product": "CVI3",
        "expected_bulletin_keywords": ["touchscreen", "display", "screen"],
    },
    # From ESDE21017 - ExBC WIFI module
    {
        "name": "EPBC WiFi Module Issue",
        "query": "EPBC wifi module communication problem",
        "product": "EPBC17",
        "expected_bulletin_keywords": ["wifi", "module", "communication"],
    },
    # From ESDE17016 - ESD sensitivity
    {
        "name": "ESD Damage to Tool",
        "query": "Battery tool ESD electrostatic damage not working",
        "product": "EABS",
        "expected_bulletin_keywords": ["esd", "electrostatic", "damage", "sensitivity"],
    },
    # General CVI3 error
    {
        "name": "CVI3 Error Code Display",
        "query": "CVI3 controller showing error code on display",
        "product": "CVI3",
        "expected_bulletin_keywords": ["error", "fault", "code"],
    },
    # EPB calibration issue similar to I004
    {
        "name": "EPB Calibration Span Issue",
        "query": "EPB tool calibration failed span error",
        "product": "EPB8-1800",
        "expected_bulletin_keywords": ["calibration", "span", "motor", "align"],
    },
]


def test_random_faults():
    """Test random fault scenarios."""
    print("\n" + "=" * 70)
    print("🧪 RANDOM FAULT TEST - 10 Real Scenarios")
    print("=" * 70)
    
    try:
        from src.llm.rag_engine import RAGEngine
        rag = RAGEngine()
    except Exception as e:
        print(f"⚠️  RAG Engine initialization failed: {e}")
        return 0, len(RANDOM_FAULT_TESTS)
    
    passed = 0
    
    for i, test in enumerate(RANDOM_FAULT_TESTS):
        print(f"\n{'─' * 60}")
        print(f"📋 Test {i+1}/10: {test['name']}")
        print(f"   Query: {test['query']}")
        print(f"   Product: {test['product']}")
        
        try:
            result = rag.retrieve_context(
                test['query'],
                part_number=test['product'],
                top_k=5
            )
            
            documents = result.get('documents', [])
            
            if not documents:
                print(f"   ❌ No documents retrieved")
                continue
            
            print(f"   📄 Retrieved {len(documents)} documents:")
            
            # Check sources and content
            found_esde = False
            found_keywords = []
            
            for doc in documents[:3]:  # Check top 3
                source = doc.get('metadata', {}).get('source', '')
                content = doc.get('content', doc.get('text', ''))[:200].lower()
                
                # Check if ESDE bulletin
                if source.upper().startswith('ESDE'):
                    found_esde = True
                    print(f"      ✅ {source[:50]}")
                else:
                    print(f"      📄 {source[:50]}")
                
                # Check for expected keywords
                for kw in test['expected_bulletin_keywords']:
                    if kw.lower() in content or kw.lower() in source.lower():
                        if kw not in found_keywords:
                            found_keywords.append(kw)
            
            # Evaluate results
            keyword_match = len(found_keywords) >= 2
            
            if found_esde and keyword_match:
                print(f"   ✅ PASS - ESDE bulletin found, keywords: {found_keywords}")
                passed += 1
            elif found_esde:
                print(f"   ⚠️  PARTIAL - ESDE found but limited keyword match: {found_keywords}")
                passed += 0.5
            elif keyword_match:
                print(f"   ⚠️  PARTIAL - Keywords found but no ESDE: {found_keywords}")
                passed += 0.5
            else:
                print(f"   ❌ FAIL - No ESDE, keywords found: {found_keywords}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return passed, len(RANDOM_FAULT_TESTS)


def main():
    """Run tests."""
    passed, total = test_random_faults()
    
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    pct = (passed / total) * 100 if total > 0 else 0
    print(f"   Passed: {passed}/{total} ({pct:.1f}%)")
    
    if pct >= 80:
        print("\n🎉 Excellent! Over 80% pass rate!")
        return 0
    elif pct >= 60:
        print("\n⚠️  Good, but room for improvement")
        return 0
    else:
        print("\n❌ Needs improvement")
        return 1


if __name__ == "__main__":
    sys.exit(main())
