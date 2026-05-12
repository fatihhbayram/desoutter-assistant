#!/usr/bin/env python3
"""Manual enriched ingest for ESDE22012 — WiFi performance impact on EABS tools.
Table data (SN ranges, spare P/N) not extracted by docx processor, so inserted manually.
"""
import os
import sys
import hashlib
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.documents.embeddings import EmbeddingsGenerator
from src.vectordb.qdrant_client import QdrantDBClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SOURCE = "documents/bulletins/ESDE22012- Wi-Fi performance impact on EABS (3).docx"
DOC_TYPE = "service_bulletin"

CHUNKS = [
    {
        "index": 0,
        "text": (
            "ESDE-22012 — PRODUCT IMPACT: Wi-Fi performance on EABS tools\n\n"
            "Products: EABS tools range\n\n"
            "Description of the issue:\n"
            "An incorrect material, including carbon fiber, was used to produce some EABS housings in 2022. "
            "100 tools in a batch of 300, manufactured between 23rd of February and 4th of July 2022, "
            "have been equipped with the incorrect housing.\n\n"
            "Consequences:\n"
            "Carbon material disrupts the Wi-Fi signal. As a result, Wi-Fi performance is reduced. "
            "In applications where maximum Wi-Fi performance is required, a reduction of the maximum "
            "connection distance between the tool and the access point can be observed. "
            "Random disconnection can also be observed. "
            "This has no effect on the mechanical strength and quality of the handle.\n\n"
            "Cause of issue:\n"
            "Incorrect plastic material was accidentally used in production. "
            "This was not detected in production because the parts look exactly the same "
            "and only a simple WiFi function test is done.\n\n"
            "Realized corrective actions in production:\n"
            "Housing stock has been sorted out and incorrect parts eliminated since July 2022."
        ),
    },
    {
        "index": 1,
        "text": (
            "EABS WiFi weak signal. WiFi signal strength reduced. WiFi disconnection. EABS housing WiFi problem. "
            "Poor WiFi performance. WiFi signal loss. WiFi not working. WiFi range reduced.\n\n"
            "ESDE-22012 — EABS WiFi weak signal: How to identify and fix affected tools\n\n"
            "Root cause: Incorrect carbon fiber housing material blocks WiFi signal in affected EABS tools.\n\n"
            "Affected serial number ranges (EABS tools with incorrect carbon fiber housing):\n"
            "- 22D09458 to 22D14346 — manufactured end of February to April 2022\n"
            "- 22E13130 to 22E14026 — manufactured May 2022\n"
            "- 22F12128 to 22F18327 — manufactured June to beginning of July 2022\n\n"
            "How to identify the impacted tool:\n"
            "Check the serial number (S/N) on the tool label. "
            "If the S/N falls within the ranges above, the tool has the incorrect carbon fiber housing "
            "which reduces WiFi signal strength and may cause random disconnections.\n\n"
            "Required action on the field:\n"
            "If the date code is strictly identical to the affected range, "
            "order spare part P/N 6153995960 from PTD and replace the housing. "
            "Follow the EABS maintenance guide to disassemble the handle.\n\n"
            "Source: ESDE-22012"
        ),
    },
]


def main():
    logger.info("Deleting old ESDE22012 chunks from Qdrant...")
    qdrant = QdrantDBClient(host='qdrant', port=6333, collection_name='desoutter_docs_v2')

    from qdrant_client import QdrantClient
    client = QdrantClient(host='qdrant', port=6333)
    results = client.scroll(
        collection_name='desoutter_docs_v2',
        scroll_filter=None,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    old_ids = [
        p.id for p in results[0]
        if 'ESDE22012' in p.payload.get('source', '') or 'ESDE22012' in p.payload.get('document_id', '')
    ]
    if old_ids:
        client.delete(collection_name='desoutter_docs_v2', points_selector=old_ids)
        logger.info(f"Deleted {len(old_ids)} old chunks")

    logger.info("Generating embeddings (CPU)...")
    embedder = EmbeddingsGenerator(device='cpu')
    texts = [c['text'] for c in CHUNKS]
    embeddings = embedder.generate_embeddings(texts, show_progress=True)

    documents = []
    for chunk, embedding in zip(CHUNKS, embeddings):
        chunk_id_str = f"{SOURCE}_{chunk['index']}"
        chunk_id = int(hashlib.md5(chunk_id_str.encode()).hexdigest()[:16], 16)
        documents.append({
            'id': chunk_id,
            'text': chunk['text'],
            'embedding': embedding,
            'metadata': {
                'document_id': SOURCE,
                'source': SOURCE,
                'document_type': DOC_TYPE,
                'chunk_type': 'problem_solution',
                'chunk_index': chunk['index'],
                'product_family': 'EABS',
                'is_generic': False,
            }
        })

    success, errors = qdrant.upsert(documents)
    logger.info(f"Upserted {success} chunks, {errors} errors")


if __name__ == '__main__':
    main()
