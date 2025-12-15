"""
MongoDB client and operations
"""
from typing import List, Dict, Optional
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError, ConnectionFailure
from config import MONGO_URI, MONGO_DATABASE
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


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
        """Create useful indexes"""
        try:
            self.collection.create_index("product_id", unique=True)
            self.collection.create_index("part_number")
            self.collection.create_index("category")
            self.collection.create_index("scraped_date")
            logger.info("✅ Indexes created")
        except Exception as e:
            logger.error(f"❌ Error creating indexes: {e}")
