"""
MongoDB client and operations - Schema v2 Support
"""
from datetime import datetime
from typing import List, Dict, Optional, Any
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError, ConnectionFailure
from config import MONGO_URI, MONGO_DATABASE
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# Fields that should NOT be overwritten if empty in new data
PROTECTED_FIELDS = [
    "description", "image_url", "min_torque", "max_torque",
    "speed", "output_drive", "weight"
]

# Fields that should always be updated
FORCE_UPDATE_FIELDS = [
    "schema_version", "tool_category", "tool_type", "product_family",
    "wireless", "platform_connection", "modular_system", "updated_at"
]


class MongoDBClient:
    """MongoDB client wrapper"""
    
    def __init__(self, uri: str = MONGO_URI, db_name: str = MONGO_DATABASE, collection_name: str = "products"):
        """
        Initialize MongoDB client
        
        Args:
            uri: MongoDB connection URI
            db_name: Database name
            collection_name: Collection name to use
        """
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client: Optional[MongoClient] = None
        self.db = None
        self.collection = None
        
    def connect(self, collection_name: str = "products"):
        """
        Connect to MongoDB
        
        Args:
            collection_name: Collection name to use
        """
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.collection = self.db[collection_name]
            logger.info(f"✅ Connected to MongoDB: {self.db_name}.{collection_name}")
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect(self.collection_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def bulk_upsert(self, products: List[Dict]) -> Dict:
        """
        Bulk upsert products
        
        Args:
            products: List of product dictionaries
            
        Returns:
            Result statistics
        """
        if not products:
            logger.warning("No products to upsert")
            return {"inserted": 0, "modified": 0, "total": 0}
        
        operations = [
            UpdateOne(
                {"product_id": product["product_id"]},
                {"$set": product},
                upsert=True
            )
            for product in products
        ]
        
        try:
            result = self.collection.bulk_write(operations)
            stats = {
                "inserted": result.upserted_count,
                "modified": result.modified_count,
                "total": len(products)
            }
            logger.info(f"✅ Bulk upsert: {stats['inserted']} inserted, {stats['modified']} modified")
            return stats
        except BulkWriteError as e:
            logger.error(f"❌ Bulk write error: {e.details}")
            raise

    def smart_upsert_product(self, product: Dict) -> Dict:
        """
        Smart upsert that preserves existing non-empty values.
        
        Business Rules:
        - Never overwrite existing non-empty values with empty ones
        - Always update Schema v2 fields (tool_category, wireless, etc.)
        - Set schema_version = 2
        - Update updated_at timestamp
        
        Args:
            product: Product dictionary with new data
            
        Returns:
            Result: {"action": "inserted"|"updated", "product_id": str}
        """
        product_id = product.get("product_id") or product.get("part_number")
        
        if not product_id:
            raise ValueError("Product must have product_id or part_number")
        
        # Find existing document
        existing = self.collection.find_one({
            "$or": [
                {"product_id": product_id},
                {"part_number": product_id}
            ]
        })
        
        if existing:
            # Build update document - smart merge
            update_doc = self._build_smart_update(existing, product)
            
            self.collection.update_one(
                {"_id": existing["_id"]},
                {"$set": update_doc}
            )
            
            logger.debug(f"✏️  Updated: {product_id}")
            return {"action": "updated", "product_id": product_id}
        else:
            # Insert new document
            product["schema_version"] = 2
            product["updated_at"] = datetime.now().isoformat()
            self.collection.insert_one(product)
            
            logger.debug(f"➕ Inserted: {product_id}")
            return {"action": "inserted", "product_id": product_id}
    
    def _build_smart_update(self, existing: Dict, new_data: Dict) -> Dict:
        """
        Build smart update document that preserves non-empty values.
        
        Args:
            existing: Existing document from MongoDB
            new_data: New product data
            
        Returns:
            Update document
        """
        update_doc = {}
        
        for key, new_value in new_data.items():
            # Skip MongoDB internal fields
            if key == "_id":
                continue
            
            # Always update force-update fields
            if key in FORCE_UPDATE_FIELDS:
                update_doc[key] = new_value
                continue
            
            # Get existing value
            existing_value = existing.get(key)
            
            # Protected fields: don't overwrite non-empty with empty
            if key in PROTECTED_FIELDS:
                if self._is_empty_value(new_value) and not self._is_empty_value(existing_value):
                    # Keep existing value
                    continue
            
            # Update if new value is meaningful or field doesn't exist
            if not self._is_empty_value(new_value) or existing_value is None:
                update_doc[key] = new_value
        
        # Always set schema version and timestamp
        update_doc["schema_version"] = 2
        update_doc["updated_at"] = datetime.now().isoformat()
        
        return update_doc
    
    def _is_empty_value(self, value: Any) -> bool:
        """Check if value is considered empty."""
        if value is None:
            return True
        if isinstance(value, str) and value.strip() in ("", "-", "N/A", "Unknown"):
            return True
        if isinstance(value, list) and len(value) == 0:
            return True
        if isinstance(value, dict) and len(value) == 0:
            return True
        return False

    def bulk_smart_upsert(self, products: List[Dict]) -> Dict:
        """
        Bulk smart upsert with Schema v2 support.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            Result statistics
        """
        if not products:
            logger.warning("No products to upsert")
            return {"inserted": 0, "updated": 0, "errors": 0, "total": 0}
        
        stats = {"inserted": 0, "updated": 0, "errors": 0, "total": len(products)}
        
        for product in products:
            try:
                result = self.smart_upsert_product(product)
                if result["action"] == "inserted":
                    stats["inserted"] += 1
                else:
                    stats["updated"] += 1
            except Exception as e:
                logger.error(f"❌ Error upserting {product.get('product_id', 'unknown')}: {e}")
                stats["errors"] += 1
        
        logger.info(
            f"✅ Smart bulk upsert: {stats['inserted']} inserted, "
            f"{stats['updated']} updated, {stats['errors']} errors"
        )
        return stats
    
    def insert_many(self, products: List[Dict]) -> int:
        """
        Insert many products (will fail on duplicates)
        
        Args:
            products: List of product dictionaries
            
        Returns:
            Number of inserted products
        """
        if not products:
            return 0
        
        try:
            result = self.collection.insert_many(products)
            logger.info(f"✅ Inserted {len(result.inserted_ids)} products")
            return len(result.inserted_ids)
        except Exception as e:
            logger.error(f"❌ Insert error: {e}")
            raise
    
    def clear_collection(self) -> int:
        """
        Clear all products from collection
        
        Returns:
            Number of deleted products
        """
        result = self.collection.delete_many({})
        logger.info(f"🗑️  Deleted {result.deleted_count} products")
        return result.deleted_count
    
    def count_products(self) -> int:
        """
        Count total products in collection
        
        Returns:
            Product count
        """
        count = self.collection.count_documents({})
        logger.info(f"📊 Total products: {count}")
        return count
    
    def get_products(self, filter_dict: Dict = None, limit: int = 0) -> List[Dict]:
        """
        Get products from collection
        
        Args:
            filter_dict: MongoDB filter
            limit: Max number of products to return (0 = no limit)
            
        Returns:
            List of products
        """
        filter_dict = filter_dict or {}
        cursor = self.collection.find(filter_dict).limit(limit)
        return list(cursor)

    def get_collection(self, name: str):
        """Get a specific collection handle"""
        if self.db is None:
            raise ConnectionFailure("Database not connected")
        return self.db[name]
    
    def create_indexes(self):
        """Create useful indexes including Schema v2 fields"""
        try:
            self.collection.create_index("product_id", unique=True)
            self.collection.create_index("part_number")
            self.collection.create_index("category")
            self.collection.create_index("scraped_date")
            # Schema v2 indexes
            self.collection.create_index("tool_category")
            self.collection.create_index("product_family")
            self.collection.create_index("wireless.capable")
            self.collection.create_index("schema_version")
            logger.info("✅ Indexes created (including Schema v2)")
        except Exception as e:
            logger.error(f"❌ Error creating indexes: {e}")
