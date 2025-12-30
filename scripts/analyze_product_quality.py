#!/usr/bin/env python3
"""
Product Data Quality Analysis
Check completeness and accuracy of product information in MongoDB
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.mongo_client import MongoDBClient
from collections import defaultdict

def analyze_product_quality():
    """Analyze product data quality"""
    
    db = MongoDBClient()
    db.connect()
    
    products = list(db.collection.find({}))
    
    print("=" * 80)
    print(f"📊 PRODUCT DATA QUALITY ANALYSIS")
    print("=" * 80)
    print(f"\nTotal Products: {len(products)}\n")
    
    # 1. Field Completeness
    print("--- FIELD COMPLETENESS ---")
    fields_to_check = ['name', 'series', 'part_number', 'tool_category', 'wireless', 'specs', 'url']
    
    for field in fields_to_check:
        missing = 0
        empty = 0
        filled = 0
        
        for p in products:
            value = p.get(field)
            if value is None or value == 'N/A':
                missing += 1
            elif isinstance(value, (dict, list)) and not value:
                empty += 1
            elif isinstance(value, str) and not value.strip():
                empty += 1
            else:
                filled += 1
        
        total = len(products)
        fill_rate = (filled / total * 100) if total > 0 else 0
        print(f"{field:20} | Filled: {filled:4} ({fill_rate:5.1f}%) | Missing: {missing:4} | Empty: {empty:4}")
    
    # 2. Wireless Capability Distribution
    print("\n--- WIRELESS CAPABILITY ---")
    wireless_stats = defaultdict(int)
    
    for p in products:
        wireless = p.get('wireless', {})
        capable = wireless.get('capable')
        
        if capable is True:
            wireless_stats['Wireless'] += 1
        elif capable is False:
            wireless_stats['Not Wireless'] += 1
        else:
            wireless_stats['Unknown'] += 1
    
    for status, count in wireless_stats.items():
        pct = (count / len(products) * 100) if len(products) > 0 else 0
        print(f"{status:20} | {count:4} ({pct:5.1f}%)")
    
    # 3. Tool Category Distribution
    print("\n--- TOOL CATEGORY ---")
    category_stats = defaultdict(int)
    
    for p in products:
        cat = p.get('tool_category', 'Unknown')
        category_stats[cat] += 1
    
    for cat, count in sorted(category_stats.items(), key=lambda x: -x[1]):
        pct = (count / len(products) * 100) if len(products) > 0 else 0
        print(f"{cat:30} | {count:4} ({pct:5.1f}%)")
    
    # 4. Specs Completeness
    print("\n--- SPECS DETAIL ---")
    specs_filled = sum(1 for p in products if p.get('specs') and len(p.get('specs', {})) > 0)
    specs_empty = len(products) - specs_filled
    
    print(f"Products with specs:    {specs_filled:4} ({specs_filled/len(products)*100:5.1f}%)")
    print(f"Products without specs: {specs_empty:4} ({specs_empty/len(products)*100:5.1f}%)")
    
    # 5. Sample problematic products
    print("\n--- SAMPLE: PRODUCTS WITH MISSING DATA ---")
    problematic = []
    
    for p in products:
        issues = []
        if not p.get('name') or p.get('name') == 'N/A':
            issues.append('no_name')
        if not p.get('specs') or len(p.get('specs', {})) == 0:
            issues.append('no_specs')
        if p.get('wireless', {}).get('capable') is None:
            issues.append('wireless_unknown')
        
        if issues:
            problematic.append({
                'part_number': p.get('part_number', 'N/A'),
                'name': p.get('name', 'N/A'),
                'issues': issues
            })
    
    for item in problematic[:10]:
        print(f"{item['part_number']:15} | {item['name'][:30]:30} | Issues: {', '.join(item['issues'])}")
    
    print(f"\nTotal problematic products: {len(problematic)}")
    
    print("\n" + "=" * 80)
    
    db.close()

if __name__ == '__main__':
    analyze_product_quality()
