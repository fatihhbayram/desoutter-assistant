#!/usr/bin/env python3
"""
Fix wireless detection in MongoDB based on model codes
Standalone battery tools (EPB, EPBA, EAB) should NOT be wireless
Only connected models (EPBC, EABC, ELC, etc.) should be wireless
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import MongoDBClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def fix_wireless_detection():
    """Fix wireless detection based on model codes"""
    
    # Wireless model patterns (C = Connected/Communication)
    wireless_patterns = [
        r'^EPBC',      # Electric Pistol Battery Connected
        r'^EABC',      # Electric Angle Battery Connected
        r'^EABS',      # Electric Angle Battery Straight (wireless)
        r'^EIBS',      # Electric Inline Battery Straight (wireless)
        r'^ELC',       # Electric Lit Connected
        r'^ELS',       # Electric Lit Straight (wireless)
        r'^BLRTC',     # Battery Low Reaction Torque Connected
        r'^EPBCHT',    # Electric Pistol Battery Connected High Torque
        r'^EABCHT',    # Electric Angle Battery Connected High Torque
    ]
    
    # Standalone battery patterns (NOT wireless)
    standalone_patterns = [
        r'^EPB[^C]',    # EPB but not EPBC
        r'^EPBA',       # Electric Pistol Battery Angle
        r'^EAB[^CS]',   # EAB but not EABC or EABS
        r'^EABA',       # Electric Angle Battery Angle
        r'^BLRTA',      # Battery Low Reaction Torque Angle
        r'^EPBHT',      # Electric Pistol Battery High Torque (not EPBCHT)
        r'^EABHT',      # Electric Angle Battery High Torque (not EABCHT)
        r'^EABAHT',     # Electric Angle Battery Angle High Torque
    ]
    
    with MongoDBClient() as db:
        logger.info("=" * 80)
        logger.info("🔧 Fixing wireless detection in MongoDB")
        logger.info("=" * 80)
        
        # Step 1: Set wireless=True for connected models
        logger.info("\n📶 Setting wireless=True for connected models...")
        wireless_count = 0
        for pattern in wireless_patterns:
            result = db.products.update_many(
                {"part_number": {"$regex": pattern, "$options": "i"}},
                {
                    "$set": {
                        "wireless.capable": True,
                        "specifications.wireless_communication": "Yes"
                    }
                }
            )
            if result.modified_count > 0:
                logger.info(f"   ✅ {pattern}: {result.modified_count} products updated")
                wireless_count += result.modified_count
        
        logger.info(f"\n   Total wireless products: {wireless_count}")
        
        # Step 2: Set wireless=False for standalone battery models
        logger.info("\n🔋 Setting wireless=False for standalone battery models...")
        standalone_count = 0
        for pattern in standalone_patterns:
            result = db.products.update_many(
                {"part_number": {"$regex": pattern, "$options": "i"}},
                {
                    "$set": {
                        "wireless.capable": False,
                        "specifications.wireless_communication": "No"
                    }
                }
            )
            if result.modified_count > 0:
                logger.info(f"   ✅ {pattern}: {result.modified_count} products updated")
                standalone_count += result.modified_count
        
        logger.info(f"\n   Total standalone products: {standalone_count}")
        
        # Step 3: Verify specific examples
        logger.info("\n🔍 Verifying specific examples...")
        
        # EPB 17-700-4Q should be standalone (NOT wireless)
        epb_example = db.products.find_one(
            {"part_number": {"$regex": r"EPB.*17-700-4Q", "$options": "i"}},
            {"part_number": 1, "model_name": 1, "wireless.capable": 1}
        )
        if epb_example:
            wireless_status = "✅ Standalone" if not epb_example.get("wireless", {}).get("capable") else "❌ STILL WIRELESS"
            logger.info(f"   EPB 17-700-4Q: {wireless_status}")
        
        # EPBC should be wireless
        epbc_example = db.products.find_one(
            {"part_number": {"$regex": r"^EPBC", "$options": "i"}},
            {"part_number": 1, "model_name": 1, "wireless.capable": 1}
        )
        if epbc_example:
            wireless_status = "✅ Wireless" if epbc_example.get("wireless", {}).get("capable") else "❌ NOT WIRELESS"
            logger.info(f"   EPBC (example): {wireless_status}")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ Wireless detection fix completed!")
        logger.info(f"   Wireless products: {wireless_count}")
        logger.info(f"   Standalone products: {standalone_count}")
        logger.info(f"   Total updated: {wireless_count + standalone_count}")
        logger.info("=" * 80)


if __name__ == "__main__":
    try:
        fix_wireless_detection()
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)
