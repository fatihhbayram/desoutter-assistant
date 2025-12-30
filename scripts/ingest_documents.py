#!/usr/bin/env python3
"""
Ingest PDF documents into vector database
Process repair manuals and bulletins

Phase 2 Update: Uses SemanticChunker for improved RAG retrieval
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.ai_settings import MANUALS_DIR, BULLETINS_DIR
from src.documents.document_processor import DocumentProcessor
from src.documents.embeddings import EmbeddingsGenerator
from src.vectordb.chroma_client import ChromaDBClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main ingestion workflow with semantic chunking"""
    logger.info("=" * 80)
    logger.info("📄 Starting Document Ingestion (Phase 2 - Semantic Chunking)")
    logger.info("=" * 80)
    
    # Initialize components
    logger.info("\n🔧 Initializing components...")
    doc_processor = DocumentProcessor()  # Includes SemanticChunker
    embeddings_gen = EmbeddingsGenerator()
    vectordb = ChromaDBClient()
    
    # Check document directories
    manuals_exist = MANUALS_DIR.exists() and list(MANUALS_DIR.glob("*.pdf"))
    bulletins_exist = BULLETINS_DIR.exists() and list(BULLETINS_DIR.glob("*.pdf"))
    
    if not manuals_exist and not bulletins_exist:
        logger.error("❌ No PDF files found!")
        logger.error(f"   Please add PDFs to:")
        logger.error(f"   - {MANUALS_DIR}")
        logger.error(f"   - {BULLETINS_DIR}")
        return
    
    all_documents = []
    all_chunks = []
    
    # Process manuals with semantic chunking
    if manuals_exist:
        logger.info(f"\n📖 Processing Repair Manuals from {MANUALS_DIR}")
        manuals = doc_processor.process_directory(
            MANUALS_DIR, 
            enable_semantic_chunking=True
        )
        for doc in manuals:
            all_documents.append(doc)
            if doc.get("chunks"):
                all_chunks.extend(doc["chunks"])
        logger.info(f"   Processed {len(manuals)} manuals → {sum(d.get('chunk_count', 0) for d in manuals)} chunks")
    
    # Process bulletins with semantic chunking
    if bulletins_exist:
        logger.info(f"\n📰 Processing Bulletins from {BULLETINS_DIR}")
        bulletins = doc_processor.process_directory(
            BULLETINS_DIR,
            enable_semantic_chunking=True
        )
        for doc in bulletins:
            all_documents.append(doc)
            if doc.get("chunks"):
                all_chunks.extend(doc["chunks"])
        logger.info(f"   Processed {len(bulletins)} bulletins → {sum(d.get('chunk_count', 0) for d in bulletins)} chunks")
    
    if not all_documents:
        logger.error("❌ No documents were successfully processed")
        return
    
    logger.info(f"\n✅ Total documents: {len(all_documents)}")
    logger.info(f"✅ Total semantic chunks: {len(all_chunks)}")
    
    # Prepare chunks for ChromaDB
    logger.info("\n📦 Preparing chunks for vector database...")
    prepared_chunks = []
    for idx, chunk in enumerate(all_chunks):
        chunk_id = f"chunk_{idx}_{chunk.get('metadata', {}).get('source', 'unknown')}"
        prepared_chunks.append({
            "chunk_id": chunk_id,
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {})
        })
    
    # Generate embeddings
    logger.info("\n🧮 Generating embeddings...")
    texts = [chunk["text"] for chunk in prepared_chunks]
    embeddings = embeddings_gen.generate_embeddings(texts)
    logger.info(f"   Generated {len(embeddings)} embeddings")
    
    # Add to vector database
    logger.info("\n💾 Storing in vector database with deduplication...")
    vectordb.add_documents(prepared_chunks, embeddings, check_duplicates=True)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ INGESTION COMPLETED (Phase 2 - Semantic Chunking)!")
    logger.info(f"   Documents: {len(all_documents)}")
    logger.info(f"   Semantic Chunks: {len(prepared_chunks)}")
    logger.info(f"   Total in DB: {vectordb.get_count()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)
