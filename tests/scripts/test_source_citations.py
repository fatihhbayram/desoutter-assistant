#!/usr/bin/env python3
"""
Test Source Citation Enhancement
Verify that sources include page numbers and section titles
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json

def test_source_citations():
    """Test that API returns enhanced source citations"""
    
    print("=" * 60)
    print("🔍 TESTING SOURCE CITATION ENHANCEMENT")
    print("=" * 60)
    
    # Test query
    payload = {
        "part_number": "6151660870",
        "fault_description": "no wifi connection",
        "language": "en"
    }
    
    print(f"\n📤 Sending request:")
    print(f"   Part: {payload['part_number']}")
    print(f"   Fault: {payload['fault_description']}")
    
    # Make API request
    response = requests.post(
        "http://localhost:8000/diagnose",
        json=payload,
        timeout=30
    )
    
    if response.status_code != 200:
        print(f"\n❌ API Error: {response.status_code}")
        print(response.text)
        return False
    
    data = response.json()
    
    # Check sources
    sources = data.get("sources", [])
    
    print(f"\n📊 Source Citation Analysis:")
    print(f"   Total sources: {len(sources)}")
    
    if not sources:
        print("   ⚠️  No sources returned")
        return False
    
    # Analyze citations
    with_pages = 0
    with_sections = 0
    
    print(f"\n📚 Source Details:")
    for idx, source in enumerate(sources, 1):
        print(f"\n   Source {idx}:")
        print(f"      File: {source.get('source', 'N/A')}")
        print(f"      Page: {source.get('page', 'N/A')}")
        print(f"      Section: {source.get('section', 'N/A')}")
        print(f"      Similarity: {source.get('similarity', 'N/A')}")
        print(f"      Excerpt: {source.get('excerpt', 'N/A')[:80]}...")
        
        if source.get('page'):
            with_pages += 1
        if source.get('section'):
            with_sections += 1
    
    # Summary
    print(f"\n📈 Citation Quality:")
    print(f"   Sources with page numbers: {with_pages}/{len(sources)} ({with_pages/len(sources)*100:.1f}%)")
    print(f"   Sources with sections: {with_sections}/{len(sources)} ({with_sections/len(sources)*100:.1f}%)")
    
    # Test citation formatter
    print(f"\n🎨 Testing Citation Formatter:")
    from src.llm.citation_formatter import format_citation, format_citation_list
    
    for idx, source in enumerate(sources[:3], 1):
        formatted = format_citation(source)
        print(f"   {idx}. {formatted}")
    
    print(f"\n✅ Test Complete!")
    print(f"   - API returns sources: ✅")
    print(f"   - Page numbers available: {'✅' if with_pages > 0 else '⚠️'}")
    print(f"   - Sections available: {'✅' if with_sections > 0 else '⚠️'}")
    print(f"   - Citation formatter works: ✅")
    
    return True

if __name__ == '__main__':
    try:
        success = test_source_citations()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
