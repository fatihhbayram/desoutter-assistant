#!/usr/bin/env python3
"""
Re-ingest all documents with improved chunking and product metadata.
Run after resetting the vector database.

Usage:
    python scripts/reingest_documents.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from src.documents.document_processor import DocumentProcessor
from src.documents.embeddings import EmbeddingsGenerator
from src.vectordb.chroma_client import ChromaDBClient
from config.ai_settings import DOCUMENTS_DIR, MANUALS_DIR, BULLETINS_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def reingest_all_documents():
    """Re-ingest all documents with new product metadata processing"""
    
    print("=" * 60)
    print("📄 RE-INGESTING ALL DOCUMENTS")
    print("=" * 60)
    
    # Initialize components
    processor = DocumentProcessor()
    embeddings = EmbeddingsGenerator()
    chroma = ChromaDBClient()
    
    # Find all document directories
    doc_dirs = []
    
    if MANUALS_DIR.exists() and list(MANUALS_DIR.glob("*")):
        doc_dirs.append(MANUALS_DIR)
        print(f"📁 Found manuals: {MANUALS_DIR}")
    
    if BULLETINS_DIR.exists() and list(BULLETINS_DIR.glob("*")):
        doc_dirs.append(BULLETINS_DIR)
        print(f"📁 Found bulletins: {BULLETINS_DIR}")
    
    # Also check for documents in root documents folder
    if DOCUMENTS_DIR.exists():
        for item in DOCUMENTS_DIR.iterdir():
            if item.is_dir() and item not in doc_dirs and item.name not in ['manuals', 'bulletins']:
                if list(item.glob("*")):
                    doc_dirs.append(item)
                    print(f"📁 Found additional: {item}")
    
    if not doc_dirs:
        print("\n❌ No document directories found!")
        print(f"   Please add documents to:")
        print(f"   - {MANUALS_DIR}")
        print(f"   - {BULLETINS_DIR}")
        return
    
    print(f"\n📂 Processing {len(doc_dirs)} document directories...")
    
    # Process all documents
    all_chunks = []
    total_docs = 0
    failed_docs = 0
    family_counts = {}
    
    for doc_dir in doc_dirs:
        print(f"\n{'='*40}")
        print(f"📂 Processing: {doc_dir.name}")
        print(f"{'='*40}")
        
        documents = processor.process_directory(
            doc_dir,
            extract_tables=True,
            enable_semantic_chunking=True
        )
        
        for doc in documents:
            total_docs += 1
            
            if doc.get('chunks'):
                # Get product family for tracking
                product_family = doc.get('product_metadata', {}).get('product_family', 'UNKNOWN')
                family_counts[product_family] = family_counts.get(product_family, 0) + len(doc['chunks'])
                
                # Add chunks
                all_chunks.extend(doc['chunks'])
                
                print(f"   ✅ {doc['filename']}: {len(doc['chunks'])} chunks [{product_family}]")
            else:
                failed_docs += 1
                print(f"   ❌ {doc['filename']}: No chunks created")
    
    print(f"\n{'='*60}")
    print(f"📊 Processing Summary")
    print(f"{'='*60}")
    print(f"   Documents processed: {total_docs}")
    print(f"   Documents failed: {failed_docs}")
    print(f"   Total chunks created: {len(all_chunks)}")
    
    if not all_chunks:
        print("\n❌ No chunks to ingest!")
        return
    
    # Generate embeddings
    print(f"\n🧮 Generating embeddings for {len(all_chunks)} chunks...")
    
    texts = [c['text'] for c in all_chunks]
    embeddings_list = embeddings.generate_embeddings(texts, show_progress=True)
    
    # Prepare metadata for ChromaDB
    print(f"\n📦 Preparing metadata for ChromaDB...")
    
    metadatas = []
    for chunk in all_chunks:
        meta = chunk.get('metadata', {})
        
        # Convert list fields to comma-separated strings (ChromaDB requirement)
        clean_meta = {}
        for key, value in meta.items():
            if isinstance(value, list):
                clean_meta[key] = ", ".join(str(v) for v in value) if value else ""
            elif isinstance(value, bool):
                clean_meta[key] = value  # ChromaDB supports booleans
            elif isinstance(value, (int, float)):
                clean_meta[key] = value  # ChromaDB supports numbers
            elif value is None:
                clean_meta[key] = ""
            else:
                clean_meta[key] = str(value)
        
        metadatas.append(clean_meta)
    
    # Add to ChromaDB with deduplication
    print(f"\n💾 Adding chunks to ChromaDB with deduplication...")
    
    # Build chunks in format expected by chroma_client (needs chunk_id)
    prepared_chunks = []
    for idx, (text, meta) in enumerate(zip(texts, metadatas)):
        source = meta.get('source', 'unknown')
        chunk_id = f"chunk_{idx}_{hash(text) % 100000}_{source[:20]}"
        prepared_chunks.append({
            "chunk_id": chunk_id,
            "text": text,
            "metadata": meta
        })
    
    added_count = chroma.add_documents(
        chunks=prepared_chunks,
        embeddings=embeddings_list,
        check_duplicates=True
    )
    
    # Final verification
    final_count = chroma.get_count()
    
    print("\n" + "=" * 60)
    print("🎉 RE-INGESTION COMPLETE")
    print("=" * 60)
    print(f"\n📊 Final Statistics:")
    print(f"   Documents processed: {total_docs}")
    print(f"   Chunks created: {len(all_chunks)}")
    print(f"   Chunks added: {added_count}")
    print(f"   Total in database: {final_count}")
    
    # Show product distribution
    print(f"\n📦 Product Family Distribution:")
    for family, count in sorted(family_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(all_chunks)) * 100
        print(f"   {family}: {count} chunks ({pct:.1f}%)")
    
    print("\n✅ Next step: Run test_product_filtering.py to verify")


if __name__ == "__main__":
    try:
        reingest_all_documents()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
