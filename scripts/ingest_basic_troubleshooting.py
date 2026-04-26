#!/usr/bin/env python3
"""
Targeted ingestion for basic_troubleshooting_*.docx files.
Forces document_type = procedure_guide.
"""
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.documents.document_processor import DocumentProcessor
from src.documents.chunkers import ChunkerFactory
from src.documents.embeddings import EmbeddingsGenerator
from src.vectordb.qdrant_client import QdrantDBClient

try:
    from src.documents.document_classifier import DocumentType
except ImportError:
    from enum import Enum
    class DocumentType(str, Enum):
        PROCEDURE_GUIDE = "procedure_guide"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BULLETINS_DIR = Path('/app/documents/bulletins')
COLLECTION = 'desoutter_docs_v2'
QDRANT_HOST = 'qdrant'
QDRANT_PORT = 6333

FILES = [
    'basic_troubleshooting_motor.docx',
    'basic_troubleshooting_battery.docx',
    'basic_troubleshooting_connectivity.docx',
    'basic_troubleshooting_memory.docx',
    'basic_troubleshooting_drive.docx',
    'basic_troubleshooting_software.docx',
]


def main():
    doc_processor = DocumentProcessor()
    chunker_factory = ChunkerFactory()
    embedder = EmbeddingsGenerator(device='cpu')
    qdrant = QdrantDBClient(host=QDRANT_HOST, port=QDRANT_PORT, collection_name=COLLECTION)
    qdrant.ensure_collection()

    chunker = chunker_factory.get_chunker(DocumentType.PROCEDURE_GUIDE)
    logger.info(f"Using chunker: {type(chunker).__name__}")

    total_chunks = 0

    for filename in FILES:
        file_path = BULLETINS_DIR / filename
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            continue

        logger.info(f"Processing: {filename}")

        # Read raw text from docx
        text = doc_processor.process_docx(file_path)
        if not text or len(text.strip()) < 50:
            logger.warning(f"No content extracted from {filename}")
            continue

        text = doc_processor.clean_text(text)
        logger.info(f"  Text length: {len(text)} chars")

        # Chunk
        chunks = chunker.chunk(text, metadata={'source': filename})
        if not chunks:
            logger.warning(f"No chunks generated for {filename}")
            continue

        logger.info(f"  {len(chunks)} chunks generated")

        # Embed
        texts = [c.text for c in chunks]
        embeddings = embedder.generate_embeddings(texts)

        if embeddings is None or len(embeddings) != len(chunks):
            logger.error(f"Embedding mismatch for {filename}: {len(embeddings) if embeddings is not None else 0} vs {len(chunks)}")
            continue

        # Build documents
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = abs(hash(f"{filename}_{i}")) % (10**12)
            documents.append({
                'id': doc_id,
                'text': chunk.text,
                'embedding': embedding if isinstance(embedding, list) else embedding.tolist(),
                'metadata': {
                    'source': filename,
                    'document_type': 'procedure_guide',
                    'product_family': None,
                    'is_generic': True,
                    'chunk_index': i,
                    'chunk_type': chunk.chunk_type,
                }
            })

        success, errors = qdrant.upsert(documents)
        logger.info(f"  Upserted {success} chunks ({errors} errors)")
        total_chunks += success

    logger.info(f"\nDone. Total chunks uploaded: {total_chunks}")


if __name__ == '__main__':
    main()
