#!/usr/bin/env python3
"""
Fix wireless capability for battery tools
Re-run wireless detection with corrected logic
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymongo import MongoClient
from config.settings import MONGO_URI, MONGO_DATABASE
from scripts.migrate_products_v2 import detect_wireless_info, extract_product_family

def fix_wireless_detection():
    """Fix wireless detection for all battery tools"""
    
    print("=" * 60)
    print("🔧 FIXING WIRELESS DETECTION FOR BATTERY TOOLS")
    print("=" * 60)
    
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DATABASE]
    products = db.products
    
    # Find all battery tools
    battery_tools = list(products.find({"tool_category": "battery_tightening"}))
    
    print(f"\nFound {len(battery_tools)} battery tools\n")
    
    fixed = 0
    unchanged = 0
    
    for product in battery_tools:
        part_number = product.get('part_number', '')
        model_name = product.get('model_name', '')
        family = extract_product_family(part_number, model_name)
        
        # Get NEW wireless info with corrected logic
        new_wireless = detect_wireless_info(product, family)
        old_wireless = product.get('wireless', {})
        
        old_capable = old_wireless.get('capable')
        new_capable = new_wireless.get('capable')
        
        # Only update if changed
        if old_capable != new_capable:
            products.update_one(
                {'_id': product['_id']},
                {'$set': {'wireless': new_wireless}}
            )
            
            status = "✅ FIXED" if new_capable else "⚠️  CHANGED"
            print(f"{status} {part_number:15} | {model_name[:30]:30} | {old_capable} → {new_capable}")
            fixed += 1
        else:
            unchanged += 1
    
    print(f"\n📊 Summary:")
    print(f"   Fixed: {fixed}")
    print(f"   Unchanged: {unchanged}")
    print(f"   Total: {len(battery_tools)}")
    
    client.close()

if __name__ == '__main__':
    fix_wireless_detection()
