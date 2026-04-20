#!/usr/bin/env python3
"""
Adaptive Re-ingestion Script
============================
Re-ingests all documents using adaptive chunking strategies.

Features:
- Detects document type automatically
- Applies appropriate chunking strategy
- Uploads to Qdrant with rich metadata
- Progress tracking and error handling

Usage:
    python scripts/reingest_adaptive.py --source manuals
    python scripts/reingest_adaptive.py --source all
    python scripts/reingest_adaptive.py --source bulletins --dry-run
"""
import os
import re
import sys
import json
import time
import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.documents.chunkers import (
    ChunkerFactory,
    Chunk
)
from src.documents.document_classifier import DocumentClassifier
from src.documents.document_processor import DocumentProcessor
from src.documents.embeddings import EmbeddingsGenerator
from src.vectordb.qdrant_client import QdrantDBClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# LANGUAGE FILTER — Keep English pages only
# =============================================================================

# Common English function words — high frequency, rarely appear in other languages
_EN_WORDS = frozenset([
    # Function words
    'the', 'this', 'that', 'these', 'those', 'with', 'from', 'have', 'will',
    'are', 'was', 'were', 'been', 'not', 'for', 'and', 'but', 'when', 'also',
    'must', 'may', 'should', 'ensure', 'check', 'use', 'note', 'warning',
    # Technical English — common in Desoutter manuals
    'data', 'dimensions', 'torque', 'speed', 'weight', 'model', 'number',
    'output', 'min', 'max', 'type', 'issue', 'part', 'operating', 'mode',
    'tool', 'cable', 'connector', 'system', 'error', 'fault', 'step',
    'procedure', 'assembly', 'maintenance', 'installation', 'manual',
    'product', 'specification', 'technical', 'page', 'instructions',
])

_PAGE_SPLIT = re.compile(r'(--- Page \d+ ---)')


def _is_english_text(text: str) -> bool:
    """Return True if text is likely English."""
    if not text.strip():
        return True  # empty → don't filter

    # Non-ASCII ratio: Greek/Russian/Chinese/Arabic → very high → reject fast
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / len(text) > 0.25:
        return False

    # English word match: count how many EN function words appear
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return True  # only numbers/symbols → keep (spec tables, part numbers)

    # Few words = likely a data/spec table (dimensions, torque values) → keep
    if len(words) < 15:
        return True

    en_hits = sum(1 for w in words if w in _EN_WORDS)
    en_ratio = en_hits / len(words)
    return en_ratio >= 0.04  # ≥4% function word density → English


def _filter_english_pages(text: str) -> str:
    """
    Split text on --- Page N --- markers, keep only English pages.
    Returns filtered text with original page markers preserved.
    """
    parts = _PAGE_SPLIT.split(text)
    # parts alternates: [pre, marker, content, marker, content, ...]
    result = []
    kept = 0
    dropped = 0

    i = 0
    while i < len(parts):
        part = parts[i]
        if _PAGE_SPLIT.match(part):
            # This is a page marker — peek at next part (page content)
            marker = part
            content = parts[i + 1] if i + 1 < len(parts) else ''
            if _is_english_text(content):
                result.append(marker)
                result.append(content)
                kept += 1
            else:
                dropped += 1
            i += 2
        else:
            # Pre-first-page text
            result.append(part)
            i += 1

    if dropped:
        logger.debug(f"    Pages: kept {kept}, dropped {dropped} non-English")

    return ''.join(result)


# Source configurations
SOURCE_CONFIGS = {
    'manuals': {
        'path': 'documents/manuals',
        'extensions': ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.txt', '.md'],
        'default_type': 'TECHNICAL_MANUAL'
    },
    'bulletins': {
        'path': 'documents/bulletins',
        'extensions': ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xlsm', '.xlt', '.txt', '.md'],
        'default_type': 'SERVICE_BULLETIN'
    },
    'tickets': {
        'path': 'data/ticket_pdfs',
        'extensions': ['.pdf', '.txt'],
        'default_type': 'FRESHDESK_TICKET'
    },
    'specs': {
        'path': 'data/documents',
        'extensions': ['.json'],
        'default_type': 'SPEC_SHEET'
    }
}


class AdaptiveIngestionPipeline:
    """Pipeline for ingesting documents with adaptive chunking"""
    
    def __init__(
        self,
        qdrant_url: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "desoutter_docs_v2",
        dry_run: bool = False
    ):
        self.dry_run = dry_run
        self.collection_name = collection_name
        
        # Initialize components
        self.classifier = DocumentClassifier()
        self.chunker_factory = ChunkerFactory()
        self.doc_processor = DocumentProcessor()
        self.embedder = EmbeddingsGenerator(device='cpu')

        # Initialize Qdrant (only if not dry run)
        if not dry_run:
            self.qdrant = QdrantDBClient(
                host=qdrant_url,
                port=qdrant_port,
                collection_name=collection_name
            )
            self.qdrant.ensure_collection()
        else:
            self.qdrant = None
        
        # Stats
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'errors': [],
            'by_type': {}
        }
    
    async def ingest_directory(
        self,
        source: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Ingest all documents from a source directory"""
        if source not in SOURCE_CONFIGS:
            raise ValueError(f"Unknown source: {source}. Options: {list(SOURCE_CONFIGS.keys())}")
        
        config = SOURCE_CONFIGS[source]
        base_path = Path(config['path'])
        
        if not base_path.exists():
            logger.warning(f"Source path does not exist: {base_path}")
            return self.stats
        
        # Find all files
        files = []
        for ext in config['extensions']:
            files.extend(base_path.glob(f"**/*{ext}"))
        
        if limit:
            files = files[:limit]
        
        logger.info(f"Found {len(files)} files in {source}")
        
        # Process each file
        for idx, file_path in enumerate(files, 1):
            logger.info(f"[{idx}/{len(files)}] Processing: {file_path.name}")
            
            try:
                await self.ingest_file(file_path, config['default_type'])
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                self.stats['errors'].append({
                    'file': str(file_path),
                    'error': str(e)
                })
        
        return self.stats
    
    async def ingest_file(
        self,
        file_path: Path,
        default_type: str
    ) -> List[Chunk]:
        """Ingest a single file"""
        # Read file content + product metadata
        content, product_metadata = self._read_file(file_path)
        if not content:
            logger.warning(f"No content extracted: {file_path.name}")
            return []

        # Detect document type — filename rules first, then classifier
        doc_type = self._classify_by_filename(file_path.name)
        if not doc_type:
            doc_type_result = self.classifier.classify(content, file_path.name)
            doc_type = doc_type_result.document_type.value if hasattr(doc_type_result, 'document_type') else str(doc_type_result)
            if not doc_type or doc_type == 'UNKNOWN':
                doc_type = 'technical_manual'

        logger.info(f"  Type: {doc_type}")

        # Track by type
        if doc_type not in self.stats['by_type']:
            self.stats['by_type'][doc_type] = 0

        # Build metadata — merge basic + product extractor results
        metadata = self._extract_metadata(file_path, content, doc_type, product_metadata)

        # Filter non-English pages (multilingual manuals: DE, FR, ES, ZH, RU, ...)
        content = _filter_english_pages(content)

        # Get appropriate chunker via factory
        chunker = self.chunker_factory.get_chunker(doc_type)
        logger.info(f"  Chunker: {chunker.__class__.__name__}")

        # Chunk the document
        chunks = chunker.chunk(content, metadata)
        logger.info(f"  Chunks: {len(chunks)}")

        # Upload to Qdrant
        if not self.dry_run and self.qdrant and chunks:
            await self._upload_chunks(chunks, file_path, doc_type)

        # Update stats
        self.stats['documents_processed'] += 1
        self.stats['chunks_created'] += len(chunks)
        self.stats['by_type'][doc_type] += len(chunks)

        return chunks
    
    @staticmethod
    def _classify_by_filename(filename: str) -> Optional[str]:
        """
        Classify document type by filename patterns.
        Returns doc_type string or None if no rule matches.
        """
        import re
        name = filename.upper()
        stem = Path(filename).stem.upper()

        # --- Service Bulletin ---
        if stem.startswith('ESDE'):
            return 'service_bulletin'

        # --- Compatibility Matrix ---
        if re.search(r'COMPATIBILITY|BATTERY.AND.TOOL.RANGE', name):
            return 'compatibility_matrix'

        # --- Error Code List ---
        if re.search(r'ERROR.CODE|ERROR_CODE|FAULT.CODE', name):
            return 'error_code_list'

        # --- Procedure Guide ---
        if re.search(
            r'HOW.TO|HOW-TO|PROCEDURE|REWORK|INSTALLATION|SET.?UP|UPGRADE|'
            r'MAINTENANCE|CALIBRAT|WIRESHARK|NETWORK.CAPTURE|REPLACE|REPAIR|'
            r'DISASSEMBLY|ASSEMBLY|CHECKLIST|CHECK.LIST',
            name
        ):
            return 'procedure_guide'

        # --- Technical Manual ---
        if re.search(
            r'CHANGELOG|MANUAL|MANUEL|PRODUCT.INSTRUCTIONS|PRODUCT.INFORMATION|'
            r'TRAINING|PRESENTATION|OVERVIEW|BACK.TO.BASICS|CERTIFICATE|'
            r'SPECIFICATION|USER.?GUIDE|SPARE.PARTS|NETWORK|DATASHEET|'
            r'NOTICE|INFORMATION.NOTE|WIRING|ARCHITECTURE',
            name
        ):
            return 'technical_manual'

        # Part number based files: 6159925220_EN.pdf, 8920200000_EN.pdf
        if re.match(r'^\d{7,10}', stem):
            return 'technical_manual'

        # No match → let classifier decide
        return None

    def _read_file(self, file_path: Path):
        """
        Read file content for all supported formats.
        Returns (content: str, product_metadata: dict) tuple.
        - JSON: parsed and formatted directly
        - All other formats (PDF, DOCX, PPTX, XLSX, TXT, MD): via DocumentProcessor
        """
        suffix = file_path.suffix.lower()
        try:
            if suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                content = self._format_product_spec(data) if isinstance(data, dict) else json.dumps(data, indent=2)
                return content, {}

            # DocumentProcessor handles PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, TXT, MD
            result = self.doc_processor.process_document(
                file_path,
                extract_tables=True,
                enable_semantic_chunking=False
            )
            if result and result.get('text'):
                return result['text'], result.get('product_metadata', {})

            logger.warning(f"DocumentProcessor returned no content for: {file_path.name}")
            return None, {}

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None, {}
    
    def _format_product_spec(self, data: Dict) -> str:
        """Format product spec JSON as readable text"""
        parts = []
        
        if 'name' in data:
            parts.append(f"Product: {data['name']}")
        if 'product_id' in data:
            parts.append(f"Product ID: {data['product_id']}")
        if 'family' in data:
            parts.append(f"Product Family: {data['family']}")
        
        if 'specifications' in data:
            parts.append("\nSpecifications:")
            for key, value in data['specifications'].items():
                parts.append(f"  {key}: {value}")
        
        if 'features' in data:
            parts.append("\nFeatures:")
            for feature in data['features']:
                parts.append(f"  - {feature}")
        
        if 'capabilities' in data:
            parts.append("\nCapabilities:")
            for cap in data['capabilities']:
                parts.append(f"  - {cap}")
        
        return '\n'.join(parts)
    
    # Known Desoutter product family tokens — used for bulletin family detection
    _KNOWN_FAMILIES = {
        'EPBC', 'EABC', 'EABS', 'EIBS', 'BLRTC', 'BLRTA',
        'EPB', 'EPBA', 'EAB', 'EABA',
        'EAD', 'EPD', 'EFD', 'EIDS',
        'ERS', 'ERSA', 'ERSF',
        'ECS', 'ELS', 'ELB', 'ELC',
        'EM', 'ERAL', 'EME', 'EMEL',
        'XPB', 'QSHIELD',
        'CVI3', 'CVIC', 'CVIR', 'CVIL',
        'CONNECT', 'DVT',
    }

    # Bulletin filename shorthands → canonical family name
    _FAMILY_NAME_MAP = {
        'EXBC': 'EPBC',   # "ExBC" in bulletin titles = EPBC/EABC family
        'EXBD': 'EPD',
        'EXAD': 'EAD',
        'AXON': 'CONNECT',
    }

    # Non-product words ProductExtractor sometimes misidentifies as product families
    _NON_FAMILY_WORDS = {
        'WIFI', 'WIRELESS', 'BLUETOOTH', 'ETHERNET', 'FIRMWARE',
        'SOFTWARE', 'UPDATE', 'CALIBRATION', 'BATTERY', 'CONNECTOR',
        'CABLE', 'MODULE', 'BOARD', 'ERROR', 'FAULT', 'ISSUE',
        'CONNECTION', 'INFORMATION', 'NOTICE', 'ERP',
    }

    def _extract_metadata(
        self,
        file_path: Path,
        content: str,
        doc_type: str,
        product_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Extract metadata from file, content, and ProductExtractor results"""
        import re

        metadata = {
            'source': file_path.name,
            'filename': file_path.name,
            'document_type': doc_type
        }

        # ESDE bulletin code (filename takes priority over content)
        esde_match = re.search(r'(ESDE[-\s]?\d{4,5})', file_path.name + ' ' + content[:200], re.IGNORECASE)
        if esde_match:
            metadata['esde_code'] = esde_match.group(1).upper().replace(' ', '')

        # --- Product family detection ---
        if doc_type == 'service_bulletin':
            # Scan filename + first 500 chars of content for known product families
            scan_text = (file_path.stem + ' ' + content[:500]).upper()

            # Resolve shorthands first (ExBC → EPBC, etc.)
            for alias, canonical in self._FAMILY_NAME_MAP.items():
                scan_text = scan_text.replace(alias, canonical)

            found_families = [f for f in self._KNOWN_FAMILIES if re.search(r'\b' + re.escape(f) + r'\b', scan_text)]

            if len(found_families) > 1:
                # Multi-product bulletin → mark generic so it passes all Qdrant filters
                metadata['is_generic'] = True
                metadata['product_family'] = found_families[0]
            elif len(found_families) == 1:
                metadata['product_family'] = found_families[0]
                metadata['is_generic'] = False
            else:
                # No known family found → generic (don't silently drop the bulletin)
                metadata['is_generic'] = True
        else:
            # Non-bulletins: use ProductExtractor if it detected a real family name
            if product_metadata:
                detected = (product_metadata.get('product_family') or '').upper()
                if detected and detected not in self._NON_FAMILY_WORDS:
                    metadata['product_family'] = detected
                if 'is_generic' in product_metadata:
                    metadata['is_generic'] = product_metadata['is_generic']

        return metadata
    
    async def _upload_chunks(
        self,
        chunks: List[Chunk],
        file_path: Path,
        doc_type: str
    ):
        """Upload chunks to Qdrant"""
        import hashlib

        texts = [chunk.text for chunk in chunks]

        # Generate embeddings in batch
        embeddings = self.embedder.generate_embeddings(texts, show_progress=False)

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
                    'document_type': doc_type,
                    'chunk_type': chunk.chunk_type,
                    'chunk_index': chunk.chunk_index,
                    **chunk.metadata
                }
            })

        # Batch upsert
        try:
            success, errors = self.qdrant.upsert(documents)
            logger.debug(f"Uploaded {success} chunks, {errors} errors")
            if errors:
                logger.warning(f"{errors} chunks failed to upload for {file_path.name}")
        except Exception as e:
            logger.error(f"Error uploading to Qdrant: {e}")
            raise


async def main():
    parser = argparse.ArgumentParser(description='Adaptive document ingestion')
    parser.add_argument(
        '--source',
        choices=list(SOURCE_CONFIGS.keys()) + ['all'],
        default='all',
        help='Document source to ingest'
    )
    parser.add_argument(
        '--qdrant-url',
        default='localhost',
        help='Qdrant server URL'
    )
    parser.add_argument(
        '--qdrant-port',
        type=int,
        default=6333,
        help='Qdrant server port'
    )
    parser.add_argument(
        '--collection',
        default='desoutter_docs_v2',
        help='Qdrant collection name'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of documents to process'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without uploading to Qdrant'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AdaptiveIngestionPipeline(
        qdrant_url=args.qdrant_url,
        qdrant_port=args.qdrant_port,
        collection_name=args.collection,
        dry_run=args.dry_run
    )
    
    # Determine sources to process
    if args.source == 'all':
        sources = list(SOURCE_CONFIGS.keys())
    else:
        sources = [args.source]
    
    # Process each source
    start_time = time.time()
    
    for source in sources:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing source: {source}")
        logger.info(f"{'='*50}")
        
        await pipeline.ingest_directory(source, args.limit)
    
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "="*50)
    print("INGESTION SUMMARY")
    print("="*50)
    print(f"Documents processed: {pipeline.stats['documents_processed']}")
    print(f"Chunks created: {pipeline.stats['chunks_created']}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print("\nBy document type:")
    for doc_type, count in pipeline.stats['by_type'].items():
        print(f"  {doc_type}: {count} chunks")
    
    if pipeline.stats['errors']:
        print(f"\nErrors: {len(pipeline.stats['errors'])}")
        for error in pipeline.stats['errors'][:5]:
            print(f"  - {error['file']}: {error['error']}")
    
    if args.dry_run:
        print("\n[DRY RUN - No data uploaded to Qdrant]")


if __name__ == '__main__':
    asyncio.run(main())
