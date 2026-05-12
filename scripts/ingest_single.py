#!/usr/bin/env python3
"""Ingest a single file into Qdrant using the same logic as reingest_adaptive.py"""
import os
import sys
import hashlib
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.documents.chunkers import ChunkerFactory
from src.documents.document_classifier import DocumentClassifier
from src.documents.document_processor import DocumentProcessor
from src.documents.embeddings import EmbeddingsGenerator
from src.vectordb.qdrant_client import QdrantDBClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

FILENAME_TYPE_MAP = {
    'ESDE': 'service_bulletin',
    'esde': 'service_bulletin',
    'Service_Bulletin': 'service_bulletin',
    'Installation': 'procedure_guide',
    'Maintenance': 'procedure_guide',
    'How_to': 'procedure_guide',
    'How to': 'procedure_guide',
    'Product_Manual': 'technical_manual',
    'User_Manual': 'technical_manual',
}

def classify_by_filename(name: str) -> str:
    for pattern, doc_type in FILENAME_TYPE_MAP.items():
        if pattern in name:
            return doc_type
    return 'technical_manual'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Path to document file')
    parser.add_argument('--qdrant-url', default='qdrant')
    parser.add_argument('--qdrant-port', type=int, default=6333)
    parser.add_argument('--collection', default='desoutter_docs_v2')
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)

    logger.info(f"Processing: {file_path.name}")

    # Load content
    processor = DocumentProcessor()
    result = processor.process_document(file_path, extract_tables=True, enable_semantic_chunking=False)
    content = result.get('text', '') if result else ''
    product_metadata = result.get('product_metadata', {}) if result else {}
    if not content:
        logger.error("No content extracted from file")
        sys.exit(1)
    logger.info(f"Extracted {len(content)} chars")

    # Classify
    doc_type = classify_by_filename(file_path.name)
    logger.info(f"Document type: {doc_type}")

    # Chunk
    chunker = ChunkerFactory.get_chunker(doc_type)
    chunks = chunker.chunk(content, metadata={'source': str(file_path), 'document_type': doc_type})
    logger.info(f"Produced {len(chunks)} chunks")

    # Embed
    logger.info("Generating embeddings (CPU)...")
    embedder = EmbeddingsGenerator(device='cpu')
    texts = [c.text for c in chunks]
    embeddings = embedder.generate_embeddings(texts, show_progress=True)
    logger.info(f"Generated {len(embeddings)} embeddings")

    # Build documents
    documents = []
    for chunk, embedding in zip(chunks, embeddings):
        chunk_id_str = f"{file_path}_{chunk.chunk_index}"
        chunk_id = int(hashlib.md5(chunk_id_str.encode()).hexdigest()[:16], 16)
        documents.append({
            'id': chunk_id,
            'text': chunk.text,
            'embedding': embedding,
            'metadata': {
                'document_id': str(file_path),
                'source': str(file_path),
                'document_type': doc_type,
                'chunk_type': chunk.chunk_type,
                'chunk_index': chunk.chunk_index,
                **chunk.metadata
            }
        })

    # Upsert
    qdrant = QdrantDBClient(
        host=args.qdrant_url,
        port=args.qdrant_port,
        collection_name=args.collection
    )
    success, errors = qdrant.upsert(documents)
    logger.info(f"Upserted {success} chunks, {errors} errors")


if __name__ == '__main__':
    main()
