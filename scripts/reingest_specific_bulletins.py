#!/usr/bin/env python3
"""
Targeted re-ingestion for specific service bulletins that are under-chunked.
Deletes existing chunks for each file before re-ingesting.
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
from qdrant_client.models import Filter, FieldCondition, MatchText

try:
    from src.documents.document_classifier import DocumentType
except ImportError:
    from enum import Enum
    class DocumentType(str, Enum):
        SERVICE_BULLETIN = "service_bulletin"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BULLETINS_DIR = Path('/app/documents/bulletins')
COLLECTION = 'desoutter_docs_v2'
QDRANT_HOST = 'qdrant'
QDRANT_PORT = 6333

# Bulletins to re-ingest — (filename, product_family or None, is_generic)
FILES = [
    ('ESDE16008 - ERxL-TRD fault _technical news template - Important.docx', None, True),
    ('ESDE21010 - Pset reinitialization on Battery Tools CVI3 (1).docx',      None, True),
    ('ESDE22002 - CVI3 - Additional sensor management issue for Pset with Jog step.docx', None, True),
    ('Rework procedure TRD fault ERS (1).pptx',                                None, True),
]

# Known product family keywords for these bulletins
FAMILY_HINTS = {
    'ESDE16008': ['ERDL', 'ERAL', 'ERxL', 'ERS'],
    'ESDE21010': ['EPB', 'EPBC', 'EABC', 'EABS'],
    'ESDE22002': ['CVI3'],
    'Rework procedure TRD': ['ERS'],
}


def delete_existing_chunks(qdrant_client, source_name: str):
    """Delete all existing chunks for a given source file."""
    try:
        qdrant_client.client.delete(
            collection_name=COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key='source', match=MatchText(text=source_name))]
            ),
        )
        logger.info(f"  Deleted existing chunks for: {source_name}")
    except Exception as e:
        logger.warning(f"  Could not delete existing chunks: {e}")


def main():
    doc_processor = DocumentProcessor()
    chunker_factory = ChunkerFactory()
    embedder = EmbeddingsGenerator(device='cpu')
    qdrant = QdrantDBClient(host=QDRANT_HOST, port=QDRANT_PORT, collection_name=COLLECTION)
    qdrant.ensure_collection()

    chunker = chunker_factory.get_chunker(DocumentType.SERVICE_BULLETIN)
    logger.info(f"Using chunker: {type(chunker).__name__}")

    total_chunks = 0

    for filename, product_family, is_generic in FILES:
        file_path = BULLETINS_DIR / filename
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            continue

        logger.info(f"\nProcessing: {filename}")

        # Delete old chunks
        delete_existing_chunks(qdrant, filename)

        # Read content
        suffix = file_path.suffix.lower()
        if suffix == '.docx':
            text = doc_processor.process_docx(file_path)
        elif suffix == '.pptx':
            text = doc_processor.process_pptx(file_path)
        else:
            logger.error(f"Unsupported format: {suffix}")
            continue

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
            logger.error(f"Embedding mismatch: {len(embeddings) if embeddings else 0} vs {len(chunks)}")
            continue

        # Build documents
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = abs(hash(f"{filename}_{i}_v2")) % (10**12)
            documents.append({
                'id': doc_id,
                'text': chunk.text,
                'embedding': embedding if isinstance(embedding, list) else embedding.tolist(),
                'metadata': {
                    'source': filename,
                    'document_type': 'service_bulletin',
                    'product_family': product_family,
                    'is_generic': is_generic,
                    'chunk_index': i,
                    'chunk_type': getattr(chunk, 'chunk_type', 'text'),
                }
            })

        success, errors = qdrant.upsert(documents)
        logger.info(f"  Upserted {success} chunks ({errors} errors)")
        total_chunks += success

    logger.info(f"\nDone. Total chunks uploaded: {total_chunks}")


if __name__ == '__main__':
    main()
