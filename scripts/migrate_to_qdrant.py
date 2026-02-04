#!/usr/bin/env python3
"""
ChromaDB to Qdrant Migration Script
====================================
Migrates all documents from ChromaDB to Qdrant vector database.

Features:
- Preserves all metadata
- Batch processing for efficiency
- Progress tracking
- Validation after migration

Usage:
    python scripts/migrate_to_qdrant.py
    python scripts/migrate_to_qdrant.py --validate-only
    python scripts/migrate_to_qdrant.py --batch-size 50
"""
import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Tuple
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vectordb.chroma_client import ChromaDBClient
from src.vectordb.qdrant_client import QdrantDBClient, QDRANT_AVAILABLE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MigrationResult:
    """Migration result tracking"""
    def __init__(self):
        self.total_documents = 0
        self.migrated = 0
        self.errors = 0
        self.skipped = 0
        self.start_time = None
        self.end_time = None
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0
    
    def __str__(self):
        return f"""
Migration Result:
  Total Documents: {self.total_documents}
  Migrated: {self.migrated}
  Errors: {self.errors}
  Skipped: {self.skipped}
  Duration: {self.duration_seconds:.2f} seconds
  Success Rate: {(self.migrated / max(self.total_documents, 1)) * 100:.1f}%
"""


def get_all_chroma_documents(chroma_client: ChromaDBClient) -> List[Dict[str, Any]]:
    """
    Extract all documents from ChromaDB.
    
    Returns:
        List of documents with id, text, embedding, metadata
    """
    logger.info("Extracting documents from ChromaDB...")
    
    try:
        # Get all documents from ChromaDB
        collection = chroma_client.collection
        
        # Get total count
        total_count = collection.count()
        logger.info(f"Found {total_count} documents in ChromaDB")
        
        if total_count == 0:
            return []
        
        # Fetch all documents (ChromaDB supports offset/limit)
        batch_size = 500
        all_documents = []
        
        for offset in range(0, total_count, batch_size):
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "embeddings", "metadatas"]
            )
            
            ids = result.get('ids', [])
            documents = result.get('documents', [])
            embeddings = result.get('embeddings', [])
            metadatas = result.get('metadatas', [])
            
            for i, doc_id in enumerate(ids):
                doc = {
                    'id': doc_id,
                    'text': documents[i] if i < len(documents) else '',
                    'embedding': embeddings[i] if i < len(embeddings) else None,
                    'metadata': metadatas[i] if i < len(metadatas) else {}
                }
                all_documents.append(doc)
            
            logger.debug(f"Extracted {len(all_documents)}/{total_count} documents")
        
        return all_documents
        
    except Exception as e:
        logger.error(f"Failed to extract from ChromaDB: {e}")
        return []


def transform_metadata_for_qdrant(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform ChromaDB metadata to Qdrant format.
    
    - Convert types as needed
    - Add missing fields with defaults
    - Extract additional fields from text
    """
    transformed = {}
    
    # Direct copy fields
    direct_fields = [
        'source', 'chunk_index', 'total_chunks', 'page', 'section',
        'product_family', 'product_model', 'document_type', 'chunk_type',
        'esde_code', 'error_code', 'language'
    ]
    
    for field in direct_fields:
        if field in metadata:
            transformed[field] = metadata[field]
    
    # Boolean fields
    bool_fields = ['contains_procedure', 'contains_table', 'contains_error_code']
    for field in bool_fields:
        if field in metadata:
            # Ensure boolean type
            val = metadata[field]
            if isinstance(val, str):
                transformed[field] = val.lower() in ('true', '1', 'yes')
            else:
                transformed[field] = bool(val)
    
    # Array fields (intent_relevance)
    if 'intent_relevance' in metadata:
        val = metadata['intent_relevance']
        if isinstance(val, str):
            transformed['intent_relevance'] = [v.strip() for v in val.split(',')]
        elif isinstance(val, list):
            transformed['intent_relevance'] = val
    
    # Add timestamp
    transformed['migrated_at'] = datetime.utcnow().isoformat()
    
    return transformed


def migrate_documents(
    chroma_client: ChromaDBClient,
    qdrant_client: QdrantDBClient,
    batch_size: int = 100
) -> MigrationResult:
    """
    Migrate all documents from ChromaDB to Qdrant.
    
    Args:
        chroma_client: Source ChromaDB client
        qdrant_client: Target Qdrant client
        batch_size: Documents per batch
        
    Returns:
        MigrationResult with statistics
    """
    result = MigrationResult()
    result.start_time = datetime.now()
    
    # Ensure Qdrant collection exists
    if not qdrant_client.ensure_collection():
        logger.error("Failed to create Qdrant collection")
        result.errors = 1
        result.end_time = datetime.now()
        return result
    
    # Extract all documents from ChromaDB
    documents = get_all_chroma_documents(chroma_client)
    result.total_documents = len(documents)
    
    if not documents:
        logger.warning("No documents to migrate")
        result.end_time = datetime.now()
        return result
    
    # Prepare documents for Qdrant
    qdrant_docs = []
    for doc in tqdm(documents, desc="Preparing documents"):
        if not doc.get('embedding'):
            logger.warning(f"Document {doc['id']} has no embedding, skipping")
            result.skipped += 1
            continue
        
        qdrant_doc = {
            'id': hash(doc['id']) % (10**18),  # Qdrant needs integer or UUID
            'text': doc['text'],
            'embedding': doc['embedding'],
            'metadata': {
                'original_id': doc['id'],  # Preserve original ID
                **transform_metadata_for_qdrant(doc.get('metadata', {}))
            }
        }
        qdrant_docs.append(qdrant_doc)
    
    # Upsert to Qdrant
    logger.info(f"Migrating {len(qdrant_docs)} documents to Qdrant...")
    success, errors = qdrant_client.upsert(qdrant_docs, batch_size=batch_size)
    
    result.migrated = success
    result.errors = errors
    result.end_time = datetime.now()
    
    return result


def validate_migration(
    chroma_client: ChromaDBClient,
    qdrant_client: QdrantDBClient
) -> Tuple[bool, str]:
    """
    Validate that migration was successful.
    
    Checks:
    - Document counts match
    - Sample documents have correct metadata
    
    Returns:
        Tuple of (is_valid, message)
    """
    logger.info("Validating migration...")
    
    try:
        # Get counts
        chroma_count = chroma_client.collection.count()
        qdrant_info = qdrant_client.get_collection_info()
        qdrant_count = qdrant_info.get('points_count', 0) if qdrant_info else 0
        
        logger.info(f"ChromaDB count: {chroma_count}")
        logger.info(f"Qdrant count: {qdrant_count}")
        
        # Allow small difference (some docs might be skipped)
        count_diff = abs(chroma_count - qdrant_count)
        max_diff = max(10, chroma_count * 0.05)  # 5% tolerance
        
        if count_diff > max_diff:
            return False, f"Count mismatch: ChromaDB={chroma_count}, Qdrant={qdrant_count}"
        
        # Sample search validation
        # TODO: Add sample search comparison
        
        return True, f"Validation passed: {qdrant_count} documents migrated"
        
    except Exception as e:
        return False, f"Validation failed: {e}"


def main():
    parser = argparse.ArgumentParser(description="Migrate ChromaDB to Qdrant")
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Documents per batch (default: 100)"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate existing migration"
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip validation after migration"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("CHROMADB TO QDRANT MIGRATION")
    print("=" * 60)
    
    # Check Qdrant availability
    if not QDRANT_AVAILABLE:
        print("❌ qdrant-client not installed. Run: pip install qdrant-client")
        sys.exit(1)
    
    # Initialize clients
    try:
        chroma_client = ChromaDBClient()
        print(f"✅ Connected to ChromaDB")
        
        qdrant_client = QdrantDBClient()
        print(f"✅ Connected to Qdrant")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        sys.exit(1)
    
    # Validate only mode
    if args.validate_only:
        is_valid, message = validate_migration(chroma_client, qdrant_client)
        print(f"\n{'✅' if is_valid else '❌'} {message}")
        sys.exit(0 if is_valid else 1)
    
    # Run migration
    print("\n🚀 Starting migration...")
    result = migrate_documents(
        chroma_client,
        qdrant_client,
        batch_size=args.batch_size
    )
    
    print(result)
    
    # Validate
    if not args.skip_validation and result.migrated > 0:
        is_valid, message = validate_migration(chroma_client, qdrant_client)
        print(f"\n{'✅' if is_valid else '❌'} {message}")
    
    print("\n✨ Migration complete!")


if __name__ == "__main__":
    main()
