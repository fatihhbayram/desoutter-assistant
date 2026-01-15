#!/usr/bin/env python3
"""
Re-ingest specific ESDE bulletin with corrected CONNECT family pattern
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.documents.document_processor import DocumentProcessor
from src.documents.embeddings import EmbeddingsGenerator
from src.vectordb.chroma_client import ChromaDBClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """Re-ingest ESDE23028 with corrected product family"""
    logger.info("=" * 80)
    logger.info("🔄 Re-ingesting ESDE23028 with corrected CONNECT family pattern")
    logger.info("=" * 80)
    
    # Initialize components
    doc_processor = DocumentProcessor()
    embeddings_gen = EmbeddingsGenerator()
    vectordb = ChromaDBClient()
    
    # Target bulletin (Docker container path)
    bulletin_path = Path("/app/documents/bulletins/ESDE23028 - CONNECT - Infinite rebooting of the controller on the DESOUTTER screen.docx")
    
    if not bulletin_path.exists():
        logger.error(f"❌ Bulletin not found: {bulletin_path}")
        return
    
    logger.info(f"\n📰 Processing: {bulletin_path.name}")
    
    # Delete old chunks from this bulletin
    logger.info(f"\n🗑️  Deleting old chunks from ESDE23028...")
    try:
        vectordb.delete_by_source(bulletin_path.name)
        logger.info(f"   ✅ Old chunks deleted")
    except Exception as e:
        logger.warning(f"   ⚠️  Could not delete old chunks: {e}")
    
    # Process bulletin
    logger.info(f"\n📄 Processing bulletin with corrected pattern...")
    doc = doc_processor.process_document(
        bulletin_path,
        enable_semantic_chunking=True
    )
    
    if not doc:
        logger.error("❌ Failed to process bulletin")
        return
    
    chunks = doc.get("chunks", [])

    
    logger.info(f"   ✅ Processed: {len(chunks)} chunks")
    
    # Show detected product family
    if chunks:
        product_family = chunks[0].get("metadata", {}).get("product_family", "UNKNOWN")
        logger.info(f"   📦 Detected product_family: {product_family}")
    
    # Prepare chunks
    logger.info(f"\n📦 Preparing chunks...")
    prepared_chunks = []
    for idx, chunk in enumerate(chunks):
        chunk_id = f"chunk_{idx}_{chunk.get('metadata', {}).get('source', 'unknown')}"
        prepared_chunks.append({
            "chunk_id": chunk_id,
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {})
        })
    
    # Generate embeddings
    logger.info(f"\n🧮 Generating embeddings...")
    texts = [chunk["text"] for chunk in prepared_chunks]
    embeddings = embeddings_gen.generate_embeddings(texts)
    logger.info(f"   ✅ Generated {len(embeddings)} embeddings")
    
    # Add to database
    logger.info(f"\n💾 Storing in vector database...")
    vectordb.add_documents(prepared_chunks, embeddings, check_duplicates=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ RE-INGESTION COMPLETED!")
    logger.info(f"   Bulletin: ESDE23028")
    logger.info(f"   Chunks: {len(prepared_chunks)}")
    logger.info(f"   Product Family: {product_family}")
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
