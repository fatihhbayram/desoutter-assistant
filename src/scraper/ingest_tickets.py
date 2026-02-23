#!/usr/bin/env python3
"""
Ingest Freshdesk Support Tickets into Vector Database for RAG

This script processes scraped tickets and adds them to Qdrant
for RAG retrieval. Tickets provide valuable Q&A pairs from real
support cases.

Usage:
    # Ingest all tickets from MongoDB
    python scripts/ingest_tickets.py
    
    # Ingest from JSON file
    python scripts/ingest_tickets.py --json data/tickets/tickets_rag.json
    
    # Only resolved tickets
    python scripts/ingest_tickets.py --resolved-only
    
    # Only tickets with PDF content
    python scripts/ingest_tickets.py --with-pdf
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import MongoDBClient
from src.database.models import TicketModel
from src.documents.embeddings import EmbeddingsGenerator
from src.documents.semantic_chunker import SemanticChunker
from src.vectordb.qdrant_client import QdrantDBClient
from src.utils.logger import setup_logger
from config.settings import DATA_DIR

logger = setup_logger(__name__)


def load_tickets_from_json(filepath: Path) -> list:
    """Load RAG documents from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_tickets_from_mongodb(
    resolved_only: bool = False,
    with_pdf: bool = False,
    limit: int = 0
) -> list:
    """Load tickets from MongoDB and convert to RAG format"""
    with MongoDBClient() as db:
        tickets = db.get_tickets(
            resolved_only=resolved_only,
            with_pdf=with_pdf,
            limit=limit
        )
        
        logger.info(f"Loaded {len(tickets)} tickets from MongoDB")
        
        # Convert to RAG format
        rag_docs = []
        for ticket_dict in tickets:
            # Remove MongoDB _id field
            ticket_dict.pop('_id', None)
            ticket_dict.pop('created_at', None)
            ticket_dict.pop('updated_at', None)
            
            try:
                ticket = TicketModel(**ticket_dict)
                rag_docs.append(ticket.to_rag_document())
            except Exception as e:
                logger.warning(f"Error converting ticket {ticket_dict.get('ticket_id')}: {e}")
        
        return rag_docs


def chunk_ticket_content(
    rag_docs: list,
    chunker: SemanticChunker,
    max_chunk_size: int = 1000
) -> list:
    """
    Chunk ticket content for better RAG retrieval.
    
    Tickets are typically shorter than manuals, but PDF content
    can be quite long. We chunk long content semantically.
    """
    all_chunks = []
    
    for doc in rag_docs:
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        doc_id = doc.get("id", "unknown")
        
        # Short content - keep as single chunk
        if len(content) < max_chunk_size:
            all_chunks.append({
                "chunk_id": f"{doc_id}_0",
                "text": content,
                "metadata": {
                    **metadata,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "doc_type": "support_ticket"
                }
            })
        else:
            # Long content - use semantic chunking
            chunks = chunker.chunk_document(
                text=content,
                source_filename=f"ticket_{metadata.get('ticket_id', 'unknown')}.txt",
                doc_type=None,
                product_categories=metadata.get('product')
            )
            
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "chunk_id": f"{doc_id}_{i}",
                    "text": chunk.get("text", ""),
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "doc_type": "support_ticket"
                    }
                })
    
    return all_chunks


def main():
    """Main ingestion workflow"""
    parser = argparse.ArgumentParser(description="Ingest tickets into vector DB")
    
    # Source selection
    parser.add_argument("--json", type=str, help="Load from JSON file instead of MongoDB")
    
    # Filters
    parser.add_argument("--resolved-only", action="store_true", help="Only resolved tickets")
    parser.add_argument("--with-pdf", action="store_true", help="Only tickets with PDF content")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tickets")
    
    # Options
    parser.add_argument("--no-chunk", action="store_true", help="Don't chunk long content")
    parser.add_argument("--clear", action="store_true", help="Clear existing ticket embeddings first")
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("🎫 Ticket Ingestion for RAG")
    logger.info("=" * 70)
    
    # Load tickets
    if args.json:
        json_path = Path(args.json)
        if not json_path.exists():
            logger.error(f"❌ JSON file not found: {json_path}")
            sys.exit(1)
        
        logger.info(f"📂 Loading tickets from {json_path}")
        rag_docs = load_tickets_from_json(json_path)
    else:
        logger.info("📂 Loading tickets from MongoDB...")
        rag_docs = load_tickets_from_mongodb(
            resolved_only=args.resolved_only,
            with_pdf=args.with_pdf,
            limit=args.limit
        )
    
    if not rag_docs:
        logger.error("❌ No tickets to ingest")
        sys.exit(1)
    
    logger.info(f"✅ Loaded {len(rag_docs)} tickets")
    
    # Statistics
    resolved_count = sum(1 for d in rag_docs if d.get("metadata", {}).get("is_resolved"))
    pdf_count = sum(1 for d in rag_docs if d.get("metadata", {}).get("has_pdf_content"))
    logger.info(f"   Resolved: {resolved_count}")
    logger.info(f"   With PDF: {pdf_count}")
    
    # Initialize components
    logger.info("\n🔧 Initializing components...")
    embeddings_gen = EmbeddingsGenerator()
    vectordb = QdrantDBClient()  # Qdrant (migrated from ChromaDB)
    
    # Optional: Clear existing ticket embeddings
    if args.clear:
        logger.info("🗑️  Clearing existing ticket embeddings...")
        # Note: Use reingest_adaptive.py --delete-first for full Qdrant cleanup
    
    # Chunk content
    if args.no_chunk:
        logger.info("\n📦 Preparing documents (no chunking)...")
        prepared_chunks = [
            {
                "chunk_id": doc["id"],
                "text": doc["content"],
                "metadata": {
                    **doc.get("metadata", {}),
                    "doc_type": "support_ticket"
                }
            }
            for doc in rag_docs
        ]
    else:
        logger.info("\n📦 Chunking ticket content...")
        chunker = SemanticChunker(
            chunk_size=400,
            chunk_overlap=50
        )
        prepared_chunks = chunk_ticket_content(rag_docs, chunker)
    
    logger.info(f"   Total chunks: {len(prepared_chunks)}")
    
    # Generate embeddings
    logger.info("\n🧮 Generating embeddings...")
    texts = [chunk["text"] for chunk in prepared_chunks]
    embeddings = embeddings_gen.generate_embeddings(texts)
    logger.info(f"   Generated {len(embeddings)} embeddings")
    
    # Add to vector database
    logger.info("\n💾 Storing in vector database...")
    docs_to_upsert = []
    for chunk, emb in zip(prepared_chunks, embeddings):
        docs_to_upsert.append({
            'id': chunk['chunk_id'],
            'text': chunk['text'],
            'embedding': emb,
            'metadata': chunk['metadata']
        })
    vectordb.upsert(docs_to_upsert)
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    info = vectordb.get_collection_info() or {}
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ TICKET INGESTION COMPLETE!")
    logger.info(f"   Time: {elapsed:.1f} seconds")
    logger.info(f"   Tickets processed: {len(rag_docs)}")
    logger.info(f"   Chunks created: {len(prepared_chunks)}")
    logger.info(f"   Total in DB: {info.get('points_count', 0)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)
