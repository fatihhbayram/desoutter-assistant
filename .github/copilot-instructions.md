# Desoutter Product Schema Update - Specific Implementation

@workspace I need to update the existing `ProductModel` in `src/database/models.py` to add proper categorization and platform relationships.

## 📊 CURRENT SCHEMA (ProductModel)

```python
class ProductModel(BaseModel):
    # === EXISTING FIELDS (KEEP AS-IS) ===
    product_id: str                    # Unique ID (part number)
    model_name: str                    # Product model name
    part_number: str                   # Manufacturer part number
    series_name: str = ""              # Product series
    category: str = ""                 # Product category (currently generic)
    product_url: str                   # Product page URL
    image_url: str = "-"               # Product image URL
    description: str = "-"             # Product description
    
    # Technical specs
    min_torque: str = "-"
    max_torque: str = "-"
    speed: str = "-"
    output_drive: str = "-"
    wireless_communication: str = "No"  # Currently just "Yes"/"No"
    weight: str = "-"
    
    # Metadata
    scraped_date: str = datetime.now().strftime("%Y-%m-%d")
    updated_at: str = datetime.now().isoformat()
    status: str = "active"
```

**Current example:**
```json
{
  "product_id": "6151659030",
  "model_name": "EPB 8-1800-10S",
  "part_number": "6151659030",
  "series_name": "EPB - Transducerized Pistol Battery Tool",
  "category": "Battery Tightening Tools",
  "wireless_communication": "Yes"
}
```

---

## 🎯 REQUIRED UPDATES

### Problem 1: Generic Category
- ❌ Current: `category: "Battery Tightening Tools"` (just a string)
- ✅ Needed: Structured categorization with tool type

### Problem 2: Wireless Field Too Simple
- ❌ Current: `wireless_communication: "Yes"/"No"`
- ✅ Needed: Wireless capability + platform compatibility

### Problem 3: No Platform Relationships
- ❌ Current: No link to tool_units
- ✅ Needed: Compatible platforms (CVI3, Connect, ESP, etc.)

### Problem 4: No Product Family Extraction
- ❌ Current: `series_name` is full text
- ✅ Needed: Extract family code (EPB, EAD, XPB, etc.)

---

## 🗄️ UPDATED SCHEMA (Backward Compatible)

```python
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

class WirelessInfo(BaseModel):
    """Wireless capability information (battery tools only)"""
    capable: bool = False
    detection_method: str = "not_applicable"  # "model_name_C" | "no_standalone_text" | "standalone_text_found" | "not_applicable"
    compatible_platforms: List[str] = []       # ["CVI3", "Connect"]
    compatible_platform_ids: List[str] = []    # MongoDB ObjectId references (as strings)

class PlatformConnection(BaseModel):
    """Platform connection info (cable tools only)"""
    required: bool = True
    compatible_platforms: List[str] = []       # ["CVI3", "CVIR II", "ESP-C"]
    compatible_platform_ids: List[str] = []    # MongoDB ObjectId references

class ModularSystem(BaseModel):
    """Modular system info (drilling tools only)"""
    is_base_tool: bool = False                 # XPB-Modular, XPB-One
    is_attachment: bool = False                # Tightening Head, Drilling Head
    attachment_type: Optional[str] = None      # "tightening" | "drilling"
    compatible_base_tools: List[str] = []      # ["XPB-Modular", "XPB-One"]

class ProductModel(BaseModel):
    """Enhanced Product data model"""
    
    # === EXISTING FIELDS (NO CHANGES) ===
    product_id: str
    model_name: str
    part_number: str
    series_name: str = ""
    category: str = ""                         # Keep for backward compatibility
    product_url: str
    image_url: str = "-"
    description: str = "-"
    min_torque: str = "-"
    max_torque: str = "-"
    speed: str = "-"
    output_drive: str = "-"
    wireless_communication: str = "No"         # Keep for backward compatibility
    weight: str = "-"
    scraped_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"
    
    # === NEW FIELDS (ADDED) ===
    
    # Enhanced categorization
    tool_category: str = "unknown"             # "battery_tightening" | "cable_tightening" | "electric_drilling" | "platform"
    tool_type: Optional[str] = None            # "pistol" | "angle_head" | "inline" | "screwdriver" | "drill"
    product_family: str = ""                   # "EPB" | "EAD" | "XPB" (extracted from part_number)
    
    # Connection/compatibility info (conditional based on tool_category)
    wireless: Optional[WirelessInfo] = None    # Only for battery_tightening
    platform_connection: Optional[PlatformConnection] = None  # Only for cable_tightening
    modular_system: Optional[ModularSystem] = None  # Only for electric_drilling
    
    # Schema version tracking
    schema_version: int = 2                    # Track migrations
    
    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "6151659030",
                "model_name": "EPBC14-T4000-S6S4-T",
                "part_number": "6151659030",
                "series_name": "EPB - Transducerized Pistol Battery Tool",
                "category": "Battery Tightening Tools",
                "wireless_communication": "Yes",
                "tool_category": "battery_tightening",
                "tool_type": "pistol",
                "product_family": "EPB",
                "wireless": {
                    "capable": True,
                    "detection_method": "model_name_C",
                    "compatible_platforms": ["CVI3", "Connect"],
                    "compatible_platform_ids": []
                },
                "schema_version": 2
            }
        }
    
    def to_dict(self) -> dict:
        """Convert model to dictionary"""
        return self.model_dump(exclude_none=True)
```

---

## 🛠️ IMPLEMENTATION PLAN

### Phase 1: Update Models (IMMEDIATE)

**File:** `src/database/models.py`

**Changes:**
1. ✅ Add three new sub-models: `WirelessInfo`, `PlatformConnection`, `ModularSystem`
2. ✅ Add new fields to `ProductModel`
3. ✅ Keep ALL existing fields (backward compatible)
4. ✅ Update `json_schema_extra` example

**Risk:** LOW - Only adding fields, not removing

---

### Phase 2: Create Migration Script

**File:** `scripts/migrate_products_v2.py`

```python
import asyncio
import re
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

# Platform mapping based on product family
CABLE_TOOL_PLATFORMS = {
    "EPD": ["CVI3"],
    "EAD": ["CVI3"],
    "EID": ["CVI3"],
    "ERP": ["CVIR II", "ESP"],
    "ERS": ["CVIR II", "CVIXS", "ESP-C"],
    "ECS": ["ESP-C"],
    "ERXS": ["CVIXS"],
    "SLC": ["ESP-C"],
    "SLBN": ["ESP-C"],
    "E-PULSE": ["CVI3", "ESP"],
    "EFD": ["CVI3"],
    "EFM": ["CVI3"],
    "ERF": ["CVI3"],
    "EFBCIT": ["CVI3"],
    "EFBCA": ["CVI3"]
}

def extract_product_family(part_number: str) -> str:
    """Extract family from part number (e.g., EPBC14 → EPB)"""
    # Remove digits and trailing characters
    match = re.match(r'^([A-Z]+)', part_number)
    if match:
        family = match.group(1)
        # If ends with C and longer than 2 chars, it's wireless indicator
        if family.endswith('C') and len(family) > 2:
            return family[:-1]  # EPBC → EPB
        return family
    return "Unknown"

def detect_tool_category(product: dict) -> str:
    """Detect category from URL or existing category field"""
    url = product.get('product_url', '')
    category = product.get('category', '').lower()
    
    # From URL
    if '/battery-tightening-tools' in url:
        return 'battery_tightening'
    elif '/cable-tightening-tools' in url:
        return 'cable_tightening'
    elif '/electric-drilling-tools' in url:
        return 'electric_drilling'
    elif '/corded-platforms' in url:
        return 'platform'
    
    # From category text
    if 'battery' in category:
        return 'battery_tightening'
    elif 'cable' in category or 'corded' in category:
        return 'cable_tightening'
    elif 'drill' in category:
        return 'electric_drilling'
    elif 'platform' in category:
        return 'platform'
    
    return 'unknown'

def detect_tool_type(series_name: str, model_name: str) -> str:
    """Detect tool type from series/model name"""
    text = (series_name + " " + model_name).lower()
    
    if 'pistol' in text:
        return 'pistol'
    elif 'angle' in text:
        return 'angle_head'
    elif 'inline' in text or 'in-line' in text:
        return 'inline'
    elif 'screwdriver' in text:
        return 'screwdriver'
    elif 'drill' in text:
        return 'drill'
    elif 'fixtured' in text:
        return 'fixtured'
    
    return None

def detect_wireless_info(product: dict, family: str) -> dict:
    """Detect wireless capability for battery tools"""
    part_number = product['part_number']
    wireless_comm = product.get('wireless_communication', 'No')
    description = product.get('description', '').lower()
    
    # Method 1: Existing wireless_communication field
    if wireless_comm == "Yes":
        # Check if model has "C" indicator (e.g., EPBC14)
        if re.search(r'[A-Z]+C\d+', part_number):
            method = "model_name_C"
        else:
            method = "existing_field"
        
        return {
            "capable": True,
            "detection_method": method,
            "compatible_platforms": ["CVI3", "Connect"],
            "compatible_platform_ids": []
        }
    
    # Method 2: Check for "standalone" in description
    if 'standalone' in description:
        return {
            "capable": False,
            "detection_method": "standalone_text_found",
            "compatible_platforms": [],
            "compatible_platform_ids": []
        }
    
    # Default: No wireless
    return {
        "capable": False,
        "detection_method": "no_wireless_field",
        "compatible_platforms": [],
        "compatible_platform_ids": []
    }

def detect_platform_connection(family: str) -> dict:
    """Detect compatible platforms for cable tools"""
    platforms = CABLE_TOOL_PLATFORMS.get(family, [])
    
    return {
        "required": True,
        "compatible_platforms": platforms,
        "compatible_platform_ids": []
    }

def detect_modular_system(product: dict, family: str) -> dict:
    """Detect modular system info for drilling tools"""
    model_name = product.get('model_name', '').lower()
    series = product.get('series_name', '').lower()
    
    # Base tools
    if 'xpb' in model_name or 'xpb' in series:
        return {
            "is_base_tool": True,
            "is_attachment": False,
            "attachment_type": None,
            "compatible_base_tools": []
        }
    
    # Attachments
    if 'tightening head' in series or 'drilling head' in series:
        attachment_type = 'tightening' if 'tightening' in series else 'drilling'
        return {
            "is_base_tool": False,
            "is_attachment": True,
            "attachment_type": attachment_type,
            "compatible_base_tools": ["XPB-Modular", "XPB-One"]
        }
    
    return {
        "is_base_tool": False,
        "is_attachment": False,
        "attachment_type": None,
        "compatible_base_tools": []
    }

async def migrate_products():
    """Migrate all products to new schema"""
    client = AsyncIOMotorClient("mongodb://192.168.1.125:27017")
    db = client.desoutter
    
    total = await db.products.count_documents({})
    print(f"🔄 Migrating {total} products...")
    
    migrated = 0
    skipped = 0
    errors = 0
    
    async for product in db.products.find({}):
        try:
            # Skip if already migrated
            if product.get('schema_version') == 2:
                skipped += 1
                continue
            
            # Extract data
            family = extract_product_family(product['part_number'])
            category = detect_tool_category(product)
            tool_type = detect_tool_type(
                product.get('series_name', ''),
                product.get('model_name', '')
            )
            
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
            
            # Update document
            result = await db.products.update_one(
                {'_id': product['_id']},
                {'$set': updates}
            )
            
            if result.modified_count > 0:
                migrated += 1
                print(f"✅ {product['part_number']} → {category} ({family})")
            
        except Exception as e:
            errors += 1
            print(f"❌ Error migrating {product.get('part_number', 'unknown')}: {e}")
    
    print(f"\n📊 MIGRATION SUMMARY:")
    print(f"   Total: {total}")
    print(f"   Migrated: {migrated}")
    print(f"   Skipped (already v2): {skipped}")
    print(f"   Errors: {errors}")
    
    # Create indexes
    print(f"\n📇 Creating indexes...")
    await db.products.create_index('tool_category')
    await db.products.create_index('product_family')
    await db.products.create_index('wireless.capable')
    await db.products.create_index('schema_version')
    
    print(f"✅ Migration complete!")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(migrate_products())
```

**Run:**
```bash
python scripts/migrate_products_v2.py
```

---

### Phase 3: Update Scraper (FUTURE)

**File:** `src/scraper/desoutter_scraper.py`

When scraping NEW products, populate new fields directly:

```python
# In scraper's product extraction method
product = ProductModel(
    product_id=part_number,
    model_name=model_name,
    part_number=part_number,
    # ... existing fields ...
    
    # NEW: Add enhanced fields during scraping
    tool_category=detect_tool_category(product_url),
    product_family=extract_product_family(part_number),
    tool_type=detect_tool_type(series_name, model_name),
    
    # Conditional fields
    wireless=detect_wireless_info(...) if category == 'battery' else None,
    platform_connection=detect_platform_connection(...) if category == 'cable' else None,
    
    schema_version=2
)
```

---

### Phase 4: Update API Endpoints

**File:** `src/api/routes.py`

Add new filter parameters:

```python
@router.get("/products")
async def get_products(
    category: Optional[str] = None,          # NEW: battery_tightening, cable_tightening
    family: Optional[str] = None,            # NEW: EPB, EAD, XPB
    wireless: Optional[bool] = None,         # NEW: true/false
    platform: Optional[str] = None,          # NEW: CVI3, Connect, ESP
    # ... existing params ...
):
    query = {}
    
    if category:
        query['tool_category'] = category
    
    if family:
        query['product_family'] = family
    
    if wireless is not None:
        query['wireless.capable'] = wireless
    
    if platform:
        query['$or'] = [
            {'wireless.compatible_platforms': platform},
            {'platform_connection.compatible_platforms': platform}
        ]
    
    products = await db.products.find(query).to_list(length=1000)
    return products
```

---

## 🧪 TESTING CHECKLIST

After migration:

```bash
# 1. Check migration success
mongosh mongodb://192.168.1.125:27017/desoutter

# Count migrated products
db.products.count({schema_version: 2})

# Check battery tool with wireless
db.products.findOne({
  part_number: /EPBC/,
  "wireless.capable": true
})

# Check cable tool platforms
db.products.findOne({
  product_family: "EPD",
  "platform_connection.required": true
})

# Check drilling tools
db.products.findOne({
  tool_category: "electric_drilling"
})

# Test API filters
curl "http://localhost:8000/products?category=battery_tightening&wireless=true"
curl "http://localhost:8000/products?family=EPB"
curl "http://localhost:8000/products?platform=CVI3"
```

---

## 📝 IMPLEMENTATION ORDER

**Do this NOW:**

1. ✅ Update `src/database/models.py` (add new fields)
2. ✅ Run `python scripts/migrate_products_v2.py` (migrate 237 products)
3. ✅ Test queries in MongoDB
4. ✅ Update API endpoints (add filters)
5. ✅ Test API responses
6. ⏸️ Update scraper (later, for NEW products)

**Estimated time:** 30 minutes for steps 1-5

---

## ❓ FINAL QUESTIONS

1. **Platform mapping**: Is `CABLE_TOOL_PLATFORMS` dictionary correct?
2. **Wireless detection**: Should I scrape product pages for "standalone battery" text or trust existing `wireless_communication` field?
3. **tool_units relationship**: Should I also update `tool_units` collection with compatible product lists?

**Reply "proceed" to start, or provide corrections!** 🚀