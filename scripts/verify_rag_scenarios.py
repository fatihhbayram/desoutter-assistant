import requests
import json

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTgwMDAwMDAwMH0.bwYHsQfHLsHJuixoYqVKPb7EI7M_XvKd26oEP4g4X-U"
BASE_URL = "http://localhost:8000/diagnose"

tests = [
    {
        "name": "Test 1: EPBC8 - WiFi",
        "part_number": "6151659000",
        "fault_description": "Cannot connect to CONNECT hub via WiFi, pairing failed"
    },
    {
        "name": "Test 2: EABC80 - Torque",
        "part_number": "6151658400",
        "fault_description": "E06 error code, torque unbalance NOK"
    },
    {
        "name": "Test 3: ERS6 - Motor Cal",
        "part_number": "6151656890",
        "fault_description": "Motor calibration failed, NOT READY error on display"
    },
    {
        "name": "Test 4: EABC32 - Battery",
        "part_number": "6151658430",
        "fault_description": "Battery not charging, LED blinking red"
    },
    {
        "name": "Test 5: EPBC14 - Transducer",
        "part_number": "6151659800",
        "fault_description": "E018 Torque out of range, suspected transducer fault"
    }
]

def run_tests():
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKEN}"
    }
    
    for t in tests:
        print(f"\n{'='*20} {t['name']} {'='*20}")
        data = {
            "part_number": t["part_number"],
            "fault_description": t["fault_description"],
            "language": "en",
            "is_retry": True
        }
        
        try:
            response = requests.post(BASE_URL, headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                print(f"Product: {result.get('product_model')} ({t['part_number']})")
                print(f"Confidence: {result.get('confidence')} ({result.get('confidence_numeric', 0):.2f})")
                print("Top Sources:")
                for i, src in enumerate(result.get('sources', [])[:3]):
                    print(f"  {i+1}. {src.get('source')} (sim: {src.get('similarity')})")
            else:
                print(f"Error: HTTP {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Failed to run test: {e}")

if __name__ == "__main__":
    run_tests()
