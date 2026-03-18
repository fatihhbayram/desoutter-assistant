
import sys
import os
import time
import hashlib
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectordb.chroma_client import ChromaDBClient
from src.documents.semantic_chunker import SemanticChunker

def test_deduplication():
    print("=== Testing Content Deduplication ===")
    
    # 1. Initialize DB client
    # Use a test collection explicitly to avoid messing with production data
    client = ChromaDBClient(collection_name="test_deduplication_v1")
    client.clear_collection()
    
    print("Initialized test collection (empty)")
    
    # 2. Create chunks manually with content hash
    content = "This is a unique troubleshooting procedure for Error E804."
    normalized = content.lower()
    content_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    chunk1 = {
        "chunk_id": "chunk_1",
        "text": content,
        "metadata": {
            "source": "manual_v1.pdf",
            "page": 10,
            "content_hash": content_hash
        }
    }
    
    chunk2 = {
        "chunk_id": "chunk_2", # Different ID
        "text": content,       # Same content
        "metadata": {
            "source": "manual_v1.pdf", # Same source
            "page": 25, # Different page (maybe repeated content)
            "content_hash": content_hash # Same hash
        }
    }
    
    # Dummy embeddings
    embedding = [0.1] * 384
    
    # 3. Add first chunk
    print("\nAdding Chunk 1...")
    client.add_documents([chunk1], [embedding], check_duplicates=True)
    count1 = client.get_count()
    print(f"DB Count: {count1}")
    
    if count1 != 1:
        print("❌ Failed: Chunk 1 not added")
        return False
        
    # 4. Add second duplicate chunk
    print("\nAdding Chunk 2 (Duplicate)...")
    client.add_documents([chunk2], [embedding], check_duplicates=True)
    count2 = client.get_count()
    print(f"DB Count: {count2}")
    
    if count2 == 1:
        print("✅ PASSED: Duplicate chunk skipped")
        return True
    else:
        print(f"❌ Failed: Duplicate chunk added (Count: {count2})")
        return False

if __name__ == "__main__":
    try:
        if test_deduplication():
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
