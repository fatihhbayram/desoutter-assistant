#!/usr/bin/env python3
"""
Targeted ingestion for EABC, ELS/ELB/ELC, and EFD documents.
Evaluation results (2026-05-07): EABC 56% fail, ELS 0 chunks, EFD 29% fail.
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
        TECHNICAL_MANUAL = "technical_manual"
        PROCEDURE_GUIDE = "procedure_guide"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BULLETINS_DIR = Path('/app/documents/bulletins')
COLLECTION = 'desoutter_docs_v2'
QDRANT_HOST = 'qdrant'
QDRANT_PORT = 6333

# (filename, document_type, product_family, is_generic)
FILES = [
    # ── EABC how-to guides → procedure_guide, 2.0x boost ────────────────────
    ('CVI3 Battery tools - How to get EABC tool logs (1).docx',
     'procedure_guide', 'EABC', False),
    ('CVI3 Battery tools - How to manage EABC communication settings - rev1.2 (1).docx',
     'procedure_guide', 'EABC', False),
    ('CVI3 Battery tools - How to upgrade EABC firmware - rev1 (1).docx',
     'procedure_guide', 'EABC', False),
    ('CVI3T-TUTORIAL-How to calibrate EABC tool - rev1.0 (1).docx',
     'procedure_guide', 'EABC', False),
    ('EABC - Tool Maintenance - Wi-Fi board (1).pdf',
     'procedure_guide', 'EABC', False),

    # ── EABC service bulletins → service_bulletin, 2.5x boost ───────────────
    ('ESDE24016 - EABCLRT - Technical changes_transducer wires (1).docx',
     'service_bulletin', 'EABC', True),
    ('ESDE24016 - EABCLRT - Technical changes_transducer wires_rework procedure (1).docx',
     'service_bulletin', 'EABC', True),

    # ── ELS/ELB/ELC product manual → technical_manual, 1.5x boost ───────────
    ('ELB-ELS-ELC_Pistol_Product Instructions_EN_6159929240_EN.pdf',
     'technical_manual', None, True),

    # ── EFD service bulletins → service_bulletin, 2.5x boost ────────────────
    ('ESDE20014 - EFDS _ Removal the defect E013.docx',
     'service_bulletin', 'EFD', True),
    ('ESDE23018 - EFDx TA minimum CVI Monitor version.docx',
     'service_bulletin', 'EFD', True),

    # ── EFD changelog → technical_manual, 1.5x boost ────────────────────────
    ('CHANGELOG_EFMx_EFDx_9.2.2.docx',
     'technical_manual', 'EFD', True),
]

CHUNKER_MAP = {
    'service_bulletin': DocumentType.SERVICE_BULLETIN,
    'technical_manual': DocumentType.TECHNICAL_MANUAL,
    'procedure_guide':  DocumentType.PROCEDURE_GUIDE,
}


def delete_existing(qdrant_client, source_name: str):
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

    total_chunks = 0

    for filename, doc_type, product_family, is_generic in FILES:
        file_path = BULLETINS_DIR / filename
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            continue

        logger.info(f"\nProcessing: {filename}")
        logger.info(f"  doc_type={doc_type} family={product_family} generic={is_generic}")

        delete_existing(qdrant, filename)

        suffix = file_path.suffix.lower()
        try:
            if suffix == '.docx':
                text = doc_processor.process_docx(file_path)
            elif suffix == '.pdf':
                text = doc_processor.process_pdf(file_path)
            elif suffix == '.pptx':
                text = doc_processor.process_pptx(file_path)
            else:
                logger.error(f"Unsupported format: {suffix}")
                continue
        except Exception as e:
            logger.error(f"  Failed to read {filename}: {e}")
            continue

        if not text or len(text.strip()) < 50:
            logger.warning(f"  No content extracted")
            continue

        text = doc_processor.clean_text(text)
        logger.info(f"  Text length: {len(text)} chars")

        chunker_type = CHUNKER_MAP.get(doc_type, DocumentType.SERVICE_BULLETIN)
        chunker = chunker_factory.get_chunker(chunker_type)
        chunks = chunker.chunk(text, metadata={'source': filename})

        if not chunks:
            logger.warning(f"  No chunks generated")
            continue

        logger.info(f"  {len(chunks)} chunks generated")

        texts = [c.text for c in chunks]
        embeddings = embedder.generate_embeddings(texts)

        if embeddings is None or len(embeddings) != len(chunks):
            logger.error(f"  Embedding mismatch")
            continue

        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = abs(hash(f"{filename}_{i}_eabc_els_efd_v1")) % (10**12)
            documents.append({
                'id': doc_id,
                'text': chunk.text,
                'embedding': embedding if isinstance(embedding, list) else embedding.tolist(),
                'metadata': {
                    'source': filename,
                    'document_type': doc_type,
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
