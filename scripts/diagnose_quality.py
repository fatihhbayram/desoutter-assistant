#!/usr/bin/env python3
"""
Compare response quality before/after improvements
Test with a known good query
"""
import requests
import json

# Test query
payload = {
    "part_number": "6151660870",
    "fault_description": "error code E123",
    "language": "en"
}

print("=" * 60)
print("RESPONSE QUALITY ANALYSIS")
print("=" * 60)

try:
    response = requests.post(
        "http://localhost:8000/diagnose",
        json=payload,
        timeout=60
    )
    
    data = response.json()
    
    print(f"\n📊 Metrics:")
    print(f"   Confidence: {data.get('confidence')}")
    print(f"   Sufficiency Score: {data.get('sufficiency_score', 'N/A')}")
    print(f"   Sources: {len(data.get('sources', []))}")
    print(f"   Intent: {data.get('intent')}")
    print(f"   Response Length: {len(data.get('suggestion', ''))} chars")
    
    print(f"\n📚 Sources:")
    for idx, src in enumerate(data.get('sources', [])[:3], 1):
        print(f"   {idx}. {src.get('source', 'N/A')[:40]}")
        print(f"      Similarity: {src.get('similarity')}")
        print(f"      Section: {src.get('section', 'N/A')[:40]}")
    
    print(f"\n📝 Response Preview:")
    suggestion = data.get('suggestion', '')
    print(suggestion[:500])
    
    # Check for issues
    print(f"\n⚠️  Potential Issues:")
    issues = []
    
    if data.get('confidence') == 'low':
        issues.append("❌ Confidence is LOW (should use sufficiency_score)")
    
    if len(data.get('sources', [])) == 0:
        issues.append("❌ No sources returned")
    
    if len(suggestion) < 200:
        issues.append("❌ Response too short")
    
    if "I don't have" in suggestion or "I don't know" in suggestion:
        issues.append("⚠️  'I don't know' response")
    
    if issues:
        for issue in issues:
            print(f"   {issue}")
    else:
        print("   ✅ No obvious issues detected")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
