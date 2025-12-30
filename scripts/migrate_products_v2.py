#!/usr/bin/env python3
"""
Product Schema Migration Script - v1 to v2
Migrates existing products to enhanced schema with:
- tool_category, tool_type, product_family
- wireless info (battery tools)
- platform_connection (cable tools)
- modular_system (drilling tools)

Usage:
    python scripts/migrate_products_v2.py
    python scripts/migrate_products_v2.py --dry-run
"""
import sys
import re
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pymongo import MongoClient
from config.settings import MONGO_URI, MONGO_DATABASE

# =============================================================================
# PLATFORM MAPPING (Cable tools → Compatible platforms)
# =============================================================================
CABLE_TOOL_PLATFORMS = {
    # CVI3 compatible tools
    "EPD": ["CVI3"],
    "EAD": ["CVI3"],
    "EID": ["CVI3"],
    "EFD": ["CVI3"],
    "EFM": ["CVI3"],
    "ERF": ["CVI3"],
    "EFBCIT": ["CVI3"],
    "EFBCA": ["CVI3"],
    "E-PULSE": ["CVI3", "ESP"],
    "EPULSE": ["CVI3", "ESP"],
    
    # CVIR II / ESP compatible tools
    "ERP": ["CVIR II", "ESP"],
    "ERS": ["CVIR II", "CVIXS", "ESP-C"],
    "ERXS": ["CVIXS"],
    
    # ESP-C compatible tools
    "ECS": ["ESP-C"],
    "SLC": ["ESP-C"],
    "SLBN": ["ESP-C"],
    
    # CVI3LT compatible tools (low torque)
    "EAD": ["CVI3", "CVI3LT"],
}

# Battery tool platforms (WiFi capable)
BATTERY_TOOL_PLATFORMS = {
    "EPB": ["CVI3", "Connect"],
    "EAB": ["CVI3", "Connect"],
    "EABS": ["CVI3", "Connect"],
    "EPBC": ["CVI3", "Connect"],
    "EABC": ["CVI3", "Connect"],
    "EBP": ["CVI3", "Connect"],
    "BLRT": ["CVI3", "Connect"],
}


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def extract_product_family(part_number: str, model_name: str = "") -> str:
    """
    Extract product family from part number or model name.
    Examples:
        - EPBC14-T4000 → EPB (C is wireless indicator)
        - EPD-8-900 → EPD
        - 6151659030 (with model EPB 8-1800-10S) → EPB
    """
    # Try from model name first (more reliable)
    if model_name:
        # Match pattern like "EPB 8-1800" or "EABS17-800"
        match = re.match(r'^([A-Z]+)', model_name.replace(" ", ""))
        if match:
            family = match.group(1)
            # Remove trailing 'C' if it's wireless indicator
            if family.endswith('C') and len(family) > 2 and family not in ['ECS', 'SLC']:
                return family[:-1]
            return family
    
    # Try from part number (numeric ones don't have family in them)
    if not part_number.isdigit():
        match = re.match(r'^([A-Z]+)', part_number)
        if match:
            family = match.group(1)
            if family.endswith('C') and len(family) > 2 and family not in ['ECS', 'SLC']:
                return family[:-1]
            return family
    
    # Extract from series name if available
    return "Unknown"


def detect_tool_category(product: dict) -> str:
    """
    Detect tool category from URL or existing category field.
    Categories: battery_tightening, cable_tightening, electric_drilling, platform
    """
    url = product.get('product_url', '').lower()
    category = product.get('category', '').lower()
    series = product.get('series_name', '').lower()
    
    # From URL (most reliable)
    if '/battery-tightening-tools' in url or '/battery-tools' in url:
        return 'battery_tightening'
    elif '/cable-tightening-tools' in url or '/corded-tools' in url or '/cable-tools' in url:
        return 'cable_tightening'
    elif '/electric-drilling-tools' in url or '/drilling' in url:
        return 'electric_drilling'
    elif '/corded-platforms' in url or '/platform' in url or '/controller' in url:
        return 'platform'
    
    # From category text
    if 'battery' in category:
        return 'battery_tightening'
    elif 'cable' in category or 'corded' in category:
        return 'cable_tightening'
    elif 'drill' in category:
        return 'electric_drilling'
    elif 'platform' in category or 'controller' in category:
        return 'platform'
    
    # From series name
    if 'battery' in series:
        return 'battery_tightening'
    elif 'electric' in series and 'drill' in series:
        return 'electric_drilling'
    
    return 'unknown'


def detect_tool_type(series_name: str, model_name: str) -> str:
    """
    Detect tool type from series/model name.
    Types: pistol, angle_head, inline, screwdriver, drill, fixtured, crowfoot
    """
    text = (series_name + " " + model_name).lower()
    
    if 'pistol' in text:
        return 'pistol'
    elif 'angle' in text or 'angle-head' in text:
        return 'angle_head'
    elif 'inline' in text or 'in-line' in text or 'straight' in text:
        return 'inline'
    elif 'screwdriver' in text:
        return 'screwdriver'
    elif 'drill' in text:
        return 'drill'
    elif 'fixtured' in text or 'fixture' in text:
        return 'fixtured'
    elif 'crowfoot' in text:
        return 'crowfoot'
    elif 'nutrunner' in text:
        return 'nutrunner'
    
    return None


def detect_wireless_info(product: dict, family: str) -> dict:
    """
    Detect wireless capability for battery tools.
    ONLY trust wireless_communication field from scraper.
    Returns WirelessInfo dict or None.
    """
    part_number = product.get('part_number', '')
    model_name = product.get('model_name', '')
    wireless_comm = product.get('wireless_communication', 'No')
    
    # Method 1: TRUST wireless_communication field (from scraper)
    # This is the MOST RELIABLE source
    if wireless_comm and wireless_comm.lower() == "yes":
        platforms = BATTERY_TOOL_PLATFORMS.get(family, ["CVI3", "Connect"])
        
        return {
            "capable": True,
            "detection_method": "wireless_communication_field",
            "compatible_platforms": platforms,
            "compatible_platform_ids": []
        }
    
    # Method 2: Check for "C" suffix in model (e.g., EPBC, EABC)
    # This is a secondary indicator
    if family and re.search(rf'{family}C', model_name.replace(" ", "")):
        return {
            "capable": True,
            "detection_method": "model_name_C_suffix",
            "compatible_platforms": BATTERY_TOOL_PLATFORMS.get(family, ["CVI3", "Connect"]),
            "compatible_platform_ids": []
        }
    
    # Default: NO WiFi
    # Do NOT check description text - it's unreliable
    return {
        "capable": False,
        "detection_method": "no_wireless_indicator",
        "compatible_platforms": [],
        "compatible_platform_ids": []
    }


def detect_platform_connection(family: str) -> dict:
    """
    Detect compatible platforms for cable tools.
    Returns PlatformConnection dict.
    """
    platforms = CABLE_TOOL_PLATFORMS.get(family, [])
    
    # If no specific mapping, try to infer from family prefix
    if not platforms:
        if family.startswith('E') and family not in ['ECS']:
            platforms = ["CVI3"]  # Most E-prefix tools are CVI3 compatible
    
    return {
        "required": True,
        "compatible_platforms": platforms,
        "compatible_platform_ids": []
    }


def detect_modular_system(product: dict, family: str) -> dict:
    """
    Detect modular system info for drilling tools.
    Returns ModularSystem dict.
    """
    model_name = product.get('model_name', '').lower()
    series = product.get('series_name', '').lower()
    
    # Base tools (XPB series)
    if 'xpb' in model_name or 'xpb' in series or family == 'XPB':
        return {
            "is_base_tool": True,
            "is_attachment": False,
            "attachment_type": None,
            "compatible_base_tools": []
        }
    
    # Attachments (tightening/drilling heads)
    if 'tightening head' in series or 'drilling head' in series:
        attachment_type = 'tightening' if 'tightening' in series else 'drilling'
        return {
            "is_base_tool": False,
            "is_attachment": True,
            "attachment_type": attachment_type,
            "compatible_base_tools": ["XPB-Modular", "XPB-One"]
        }
    
    # Default
    return {
        "is_base_tool": False,
        "is_attachment": False,
        "attachment_type": None,
        "compatible_base_tools": []
    }


# =============================================================================
# MIGRATION FUNCTION
# =============================================================================

def migrate_products(dry_run: bool = False):
    """
    Migrate all products to schema v2.
    
    Args:
        dry_run: If True, only show what would be changed without modifying DB
    """
    print("=" * 60)
    print("🔄 PRODUCT SCHEMA MIGRATION v1 → v2")
    print("=" * 60)
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE MIGRATION'}")
    print()
    
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DATABASE]
    products_collection = db.products
    
    # Get counts
    total = products_collection.count_documents({})
    already_migrated = products_collection.count_documents({"schema_version": 2})
    to_migrate = total - already_migrated
    
    print(f"📊 Database Status:")
    print(f"   Total products: {total}")
    print(f"   Already v2: {already_migrated}")
    print(f"   To migrate: {to_migrate}")
    print()
    
    if to_migrate == 0:
        print("✅ All products already migrated!")
        client.close()
        return
    
    # Migration counters
    migrated = 0
    skipped = 0
    errors = 0
    
    # Category stats
    category_stats = {
        'battery_tightening': 0,
        'cable_tightening': 0,
        'electric_drilling': 0,
        'platform': 0,
        'unknown': 0
    }
    
    # Process each product
    print("🔄 Migrating products...")
    print("-" * 60)
    
    for product in products_collection.find({"schema_version": {"$ne": 2}}):
        try:
            # Skip if already migrated
            if product.get('schema_version') == 2:
                skipped += 1
                continue
            
            part_number = product.get('part_number', '')
            model_name = product.get('model_name', '')
            series_name = product.get('series_name', '')
            
            # Extract data
            family = extract_product_family(part_number, model_name)
            category = detect_tool_category(product)
            tool_type = detect_tool_type(series_name, model_name)
            
            # Prepare updates
            updates = {
                'tool_category': category,
                'tool_type': tool_type,
                'product_family': family,
                'schema_version': 2,
                'updated_at': datetime.now().isoformat()
            }
            
            # Add category-specific fields
            if category == 'battery_tightening':
                updates['wireless'] = detect_wireless_info(product, family)
            
            elif category == 'cable_tightening':
                updates['platform_connection'] = detect_platform_connection(family)
            
            elif category == 'electric_drilling':
                updates['modular_system'] = detect_modular_system(product, family)
            
            # Update stats
            category_stats[category] = category_stats.get(category, 0) + 1
            
            if dry_run:
                # Just print what would happen
                wireless_info = ""
                if updates.get('wireless', {}).get('capable'):
                    wireless_info = " [WiFi ✓]"
                print(f"  {part_number} → {category} ({family}) {tool_type or ''}{wireless_info}")
                migrated += 1
            else:
                # Actually update the document
                result = products_collection.update_one(
                    {'_id': product['_id']},
                    {'$set': updates}
                )
                
                if result.modified_count > 0:
                    migrated += 1
                    wireless_info = ""
                    if updates.get('wireless', {}).get('capable'):
                        wireless_info = " [WiFi ✓]"
                    print(f"  ✅ {part_number} → {category} ({family}){wireless_info}")
                else:
                    skipped += 1
            
        except Exception as e:
            errors += 1
            print(f"  ❌ Error migrating {product.get('part_number', 'unknown')}: {e}")
    
    # Create indexes (not in dry run)
    if not dry_run and migrated > 0:
        print()
        print("📇 Creating indexes...")
        products_collection.create_index('tool_category')
        products_collection.create_index('product_family')
        products_collection.create_index('wireless.capable')
        products_collection.create_index('schema_version')
        print("  ✅ Indexes created")
    
    # Print summary
    print()
    print("=" * 60)
    print("📊 MIGRATION SUMMARY")
    print("=" * 60)
    print(f"   Total processed: {migrated + skipped + errors}")
    print(f"   Migrated: {migrated}")
    print(f"   Skipped: {skipped}")
    print(f"   Errors: {errors}")
    print()
    print("📈 Category Breakdown:")
    for cat, count in sorted(category_stats.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"   {cat}: {count}")
    
    if dry_run:
        print()
        print("⚠️  DRY RUN - No changes were made!")
        print("   Run without --dry-run to apply changes.")
    else:
        print()
        print("✅ Migration complete!")
    
    client.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate products to schema v2")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying the database'
    )
    
    args = parser.parse_args()
    migrate_products(dry_run=args.dry_run)
