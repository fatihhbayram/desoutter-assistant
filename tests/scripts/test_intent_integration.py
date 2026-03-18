
import requests
import json
import sys
import time

API_URL = "http://localhost:8000/diagnose"

def test_intent(query, expected_intent, description):
    print(f"\nTesting: {description}")
    print(f"Query: '{query}'")
    
    payload = {
        "part_number": "EPBC",  # Generic product
        "fault_description": query,
        "language": "en"
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code != 200:
            print(f"FAILED: API returned {response.status_code}")
            print(response.text)
            return False
            
        data = response.json()
        
        # Check intent fields
        intent = data.get("intent")
        confidence = data.get("intent_confidence")
        
        print(f"Detected Intent: {intent} (Confidence: {confidence})")
        
        if intent == expected_intent:
            print(f"✅ PASSED: Expected '{expected_intent}' and got '{intent}'")
            return True
        else:
            print(f"❌ FAILED: Expected '{expected_intent}' but got '{intent}'")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    print("=== Testing Intent Integration ===")
    
    tests = [
        {
            "query": "E804 error code on controller",
            "expected": "error_code",
            "desc": "Error Code Intent"
        },
        {
            "query": "What is the max torque of this tool?",
            "expected": "specifications",
            "desc": "Specifications Intent"
        },
        {
            "query": "How to connect to WiFi network?",
            "expected": "connection",
            "desc": "Connection Intent"
        },
        {
            "query": "Tool is overheating and stopped working",
            "expected": "troubleshooting",
            "desc": "Troubleshooting Intent"
        },
        {
            "query": "Tell me about this tool",
            "expected": "general",
            "desc": "General/Default Intent"
        }
    ]
    
    passed = 0
    for test in tests:
        if test_intent(test["query"], test["expected"], test["desc"]):
            passed += 1
            
    print(f"\n=== Result: {passed}/{len(tests)} Tests Passed ===")
    
    if passed == len(tests):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
