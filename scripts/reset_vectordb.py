#!/usr/bin/env python3
"""
Reset ChromaDB - Clear all chunks for fresh re-ingestion
Run this ONCE before re-ingesting documents with new product metadata
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.config import Settings
from config.ai_settings import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def reset_vectordb():
    """Delete and recreate ChromaDB collection"""
    
    print("=" * 60)
    print("🗑️  RESETTING CHROMADB - ALL CHUNKS WILL BE DELETED")
    print("=" * 60)
    
    # Confirm
    confirm = input("\nType 'YES' to confirm deletion: ")
    if confirm.upper() != "YES":
        print("Aborted.")
        return
    
    try:
        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Check current state
        try:
            collection = client.get_collection(CHROMA_COLLECTION_NAME)
            count = collection.count()
            print(f"\n📊 Current chunks: {count}")
        except Exception:
            count = 0
            print("\n📊 No existing collection found")
        
        # Delete collection
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
            print(f"✅ Deleted collection '{CHROMA_COLLECTION_NAME}'")
        except Exception as e:
            print(f"⚠️ Could not delete (may not exist): {e}")
        
        # Recreate empty collection with proper settings
        client.create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Created fresh collection '{CHROMA_COLLECTION_NAME}'")
        
        print("\n" + "=" * 60)
        print("🎉 RESET COMPLETE - Ready for re-ingestion")
        print("=" * 60)
        print("\nNext step: python scripts/reingest_documents.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    reset_vectordb()
