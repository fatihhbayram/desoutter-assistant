#!/usr/bin/env python3
"""
Test script for CONNECT-W screen freeze issue
Tests both Turkish and English queries to verify:
1. ESDE23028 bulletin retrieval and ranking
2. Language detection
3. Query understanding
"""
import requests
import json

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTgwMDAwMDAwMH0.bwYHsQfHLsHJuixoYqVKPb7EI7M_XvKd26oEP4g4X-U"
BASE_URL = "http://localhost:8000/diagnose"

tests = [
    {
        "name": "Turkish Query - Ekran Donuyor",
        "part_number": "6159327230",  # CONNECT-W
        "fault_description": "ünite ekranı donuyor",
        "language": "auto"  # Let system detect Turkish
    },
    {
        "name": "English Query - Screen Freeze",
        "part_number": "6159327230",  # CONNECT-W
        "fault_description": "The screen freezes on the tool",
        "language": "en"
    },
    {
        "name": "Turkish Query - Ekran Yanıt Vermiyor",
        "part_number": "6159327230",  # CONNECT-W
        "fault_description": "ekran yanıt vermiyor dondu kaldı",
        "language": "tr"
    },
    {
        "name": "English Query - Infinite Reboot",
        "part_number": "6159327230",  # CONNECT-W
        "fault_description": "controller keeps rebooting infinitely",
        "language": "en"
    }
]

def run_test(test_case):
    """Run a single test case and display results"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_case['name']}")
    print(f"{'='*80}")
    print(f"Part Number: {test_case['part_number']}")
    print(f"Query: {test_case['fault_description']}")
    print(f"Language: {test_case['language']}")
    print("-" * 80)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKEN}"
    }
    
    data = {
        "part_number": test_case["part_number"],
        "fault_description": test_case["fault_description"],
        "language": test_case["language"]
    }
    
    try:
        response = requests.post(BASE_URL, headers=headers, json=data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ SUCCESS")
            print(f"Product: {result.get('product_model')}")
            print(f"Detected Language: {result.get('language')}")
            print(f"Confidence: {result.get('confidence')} ({result.get('confidence_numeric', 0):.2f})")
            print(f"Response Time: {result.get('response_time_ms')}ms")
            
            # Display sources
            sources = result.get('sources', [])
            print(f"\n📚 Retrieved Sources ({len(sources)} total):")
            for i, src in enumerate(sources[:10], 1):
                source_name = src.get('source', 'Unknown')
                similarity = src.get('similarity', 0)
                boosted = src.get('boosted_score', similarity)
                
                # Convert to float if string
                try:
                    similarity = float(similarity) if similarity else 0.0
                    boosted = float(boosted) if boosted else similarity
                except (ValueError, TypeError):
                    similarity = 0.0
                    boosted = 0.0
                
                # Highlight ESDE23028
                marker = "⭐" if "ESDE23028" in source_name else "  "
                boost_info = f" (boosted: {boosted:.3f})" if abs(boosted - similarity) > 0.001 else ""
                
                print(f"{marker} {i:2d}. [{similarity:.3f}{boost_info}] {source_name}")
            
            # Check if ESDE23028 is in top 3
            top_3_sources = [s.get('source', '') for s in sources[:3]]
            has_esde23028 = any('ESDE23028' in s for s in top_3_sources)
            
            if has_esde23028:
                print(f"\n✅ ESDE23028 is in TOP 3 sources!")
            else:
                print(f"\n⚠️  ESDE23028 is NOT in top 3 - needs boosting!")
                # Find position
                for i, src in enumerate(sources, 1):
                    if 'ESDE23028' in src.get('source', ''):
                        print(f"   Found at position {i}")
                        break
            
            # Display suggestion preview
            suggestion = result.get('suggestion', '')
            print(f"\n📝 Suggestion Preview:")
            print(suggestion[:500] + "..." if len(suggestion) > 500 else suggestion)
            
        else:
            print(f"\n❌ ERROR: HTTP {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ FAILED: {e}")

def main():
    print("="*80)
    print("CONNECT-W Screen Freeze Test Suite")
    print("Testing ESDE23028 bulletin retrieval and ranking")
    print("="*80)
    
    for test_case in tests:
        run_test(test_case)
        print("\n")
    
    print("="*80)
    print("Test suite completed!")
    print("="*80)

if __name__ == "__main__":
    main()
