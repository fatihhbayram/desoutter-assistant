#!/usr/bin/env python3
"""
Export product data from MongoDB to JSON or CSV
"""
import sys
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EXPORTS_DIR
from src.database import MongoDBClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def export_to_json(products: list, output_file: Path):
    """Export products to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Exported {len(products)} products to {output_file}")


def export_to_csv(products: list, output_file: Path):
    """Export products to CSV file"""
    if not products:
        logger.warning("No products to export")
        return
    
    # Get all field names from first product
    fieldnames = list(products[0].keys())
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for product in products:
            # Remove _id field if present
            if '_id' in product:
                del product['_id']
            writer.writerow(product)
    
    logger.info(f"✅ Exported {len(products)} products to {output_file}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Export Desoutter products from MongoDB')
    parser.add_argument(
        '--format',
        choices=['json', 'csv'],
        default='json',
        help='Export format (default: json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: auto-generated in data/exports/)'
    )
    parser.add_argument(
        '--filter',
        type=str,
        help='MongoDB filter as JSON string (e.g., \'{"category": "Battery Tools"}\')'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Limit number of products (0 = no limit)'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("📤 Starting data export...")
        
        # Parse filter if provided
        filter_dict = {}
        if args.filter:
            try:
                filter_dict = json.loads(args.filter)
                logger.info(f"Filter: {filter_dict}")
            except json.JSONDecodeError:
                logger.error("Invalid filter JSON")
                sys.exit(1)
        
        # Connect to MongoDB
        with MongoDBClient() as db:
            # Get products
            logger.info(f"Fetching products from MongoDB...")
            products = db.get_products(filter_dict, limit=args.limit)
            
            if not products:
                logger.warning("No products found")
                return
            
            logger.info(f"Found {len(products)} products")
            
            # Convert ObjectId to string
            for product in products:
                if '_id' in product:
                    product['_id'] = str(product['_id'])
            
            # Generate output filename if not provided
            if args.output:
                output_file = Path(args.output)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"desoutter_products_{timestamp}.{args.format}"
                output_file = EXPORTS_DIR / filename
            
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Export
            if args.format == 'json':
                export_to_json(products, output_file)
            else:
                export_to_csv(products, output_file)
            
            logger.info(f"\n✅ Export completed successfully!")
            logger.info(f"   File: {output_file}")
            logger.info(f"   Size: {output_file.stat().st_size / 1024:.2f} KB")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
