#!/usr/bin/env python3
"""
Ingest PDF documents into vector database
Process repair manuals and bulletins
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.ai_settings import MANUALS_DIR, BULLETINS_DIR
from src.documents.pdf_processor import PDFProcessor
from src.documents.chunker import TextChunker
from src.documents.embeddings import EmbeddingsGenerator
from src.vectordb.chroma_client import ChromaDBClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main ingestion workflow"""
    logger.info("=" * 80)
    logger.info("📄 Starting Document Ingestion")
    logger.info("=" * 80)
    
    # Initialize components
    logger.info("\n🔧 Initializing components...")
    pdf_processor = PDFProcessor()
    chunker = TextChunker()
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
    
    # Process manuals
    if manuals_exist:
        logger.info(f"\n📖 Processing Repair Manuals from {MANUALS_DIR}")
        manuals = pdf_processor.process_directory(MANUALS_DIR)
        all_documents.extend(manuals)
        logger.info(f"   Processed {len(manuals)} manuals")
    
    # Process bulletins
    if bulletins_exist:
        logger.info(f"\n📰 Processing Bulletins from {BULLETINS_DIR}")
        bulletins = pdf_processor.process_directory(BULLETINS_DIR)
        all_documents.extend(bulletins)
        logger.info(f"   Processed {len(bulletins)} bulletins")
    
    if not all_documents:
        logger.error("❌ No documents were successfully processed")
        return
    
    logger.info(f"\n✅ Total documents processed: {len(all_documents)}")
    
    # Chunk documents
    logger.info("\n✂️  Chunking documents...")
    chunks = chunker.chunk_documents(all_documents)
    logger.info(f"   Created {len(chunks)} chunks")
    
    # Generate embeddings
    logger.info("\n🧮 Generating embeddings...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embeddings_gen.generate_embeddings(texts)
    logger.info(f"   Generated {len(embeddings)} embeddings")
    
    # Add to vector database
    logger.info("\n💾 Storing in vector database...")
    vectordb.add_documents(chunks, embeddings)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ INGESTION COMPLETED!")
    logger.info(f"   Documents: {len(all_documents)}")
    logger.info(f"   Chunks: {len(chunks)}")
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
