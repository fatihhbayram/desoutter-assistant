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
from src.vectordb.qdrant_client import QdrantVectorDB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Source configurations
SOURCE_CONFIGS = {
    'manuals': {
        'path': 'documents/manuals',
        'extensions': ['.pdf', '.txt', '.md'],
        'default_type': 'TECHNICAL_MANUAL'
    },
    'bulletins': {
        'path': 'documents/bulletins',
        'extensions': ['.pdf', '.txt', '.md'],
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
        embedding_model: str = "all-MiniLM-L6-v2",
        dry_run: bool = False
    ):
        self.dry_run = dry_run
        self.collection_name = collection_name
        
        # Initialize components
        self.classifier = DocumentClassifier()
        self.chunker_factory = ChunkerFactory()
        
        # Initialize Qdrant (only if not dry run)
        if not dry_run:
            self.qdrant = QdrantVectorDB(
                host=qdrant_url,
                port=qdrant_port,
                collection_name=collection_name,
                embedding_model=embedding_model
            )
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
        # Read file content
        content = self._read_file(file_path)
        if not content:
            logger.warning(f"Empty or unreadable file: {file_path}")
            return []
        
        # Detect document type
        doc_type = self.classifier.classify(content)
        if doc_type == 'UNKNOWN':
            doc_type = default_type
        
        logger.debug(f"Document type: {doc_type}")
        
        # Track by type
        if doc_type not in self.stats['by_type']:
            self.stats['by_type'][doc_type] = 0
        
        # Extract metadata from content/filename
        metadata = self._extract_metadata(file_path, content, doc_type)
        
        # Get appropriate chunker
        chunker = self.chunker_factory.get_chunker(doc_type)
        
        # Chunk the document
        chunks = chunker.chunk(content, metadata)
        
        logger.debug(f"Created {len(chunks)} chunks")
        
        # Upload to Qdrant
        if not self.dry_run and self.qdrant and chunks:
            await self._upload_chunks(chunks, file_path, doc_type)
        
        # Update stats
        self.stats['documents_processed'] += 1
        self.stats['chunks_created'] += len(chunks)
        self.stats['by_type'][doc_type] += len(chunks)
        
        return chunks
    
    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read file content based on type"""
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle product spec format
                    if isinstance(data, dict):
                        return self._format_product_spec(data)
                    return json.dumps(data, indent=2)
            elif file_path.suffix == '.pdf':
                # For PDF, we'd need PyPDF2 or similar
                # Placeholder - assume text extraction done elsewhere
                txt_path = file_path.with_suffix('.txt')
                if txt_path.exists():
                    return txt_path.read_text(encoding='utf-8')
                return None
            else:
                return file_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None
    
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
    
    def _extract_metadata(
        self,
        file_path: Path,
        content: str,
        doc_type: str
    ) -> Dict[str, Any]:
        """Extract metadata from file and content"""
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'document_type': doc_type
        }
        
        # Extract product info from filename or content
        import re
        
        # Product patterns
        product_patterns = [
            r'(E[A-Z]{2,4}[-\s]?\d{3,4}[A-Z]*)',  # EABC-3000
            r'(CV[A-Z]{1,2}[-\s]?\d*)',  # CVI3, CVIR
            r'(SPD[-\s]?\d{3,4})',  # SPD1200
        ]
        
        for pattern in product_patterns:
            match = re.search(pattern, file_path.name + ' ' + content[:500], re.IGNORECASE)
            if match:
                product = match.group(1).upper().replace(' ', '-')
                metadata['product_model'] = product
                
                # Determine product family
                if product.startswith('EABC'):
                    metadata['product_family'] = 'EABC'
                elif product.startswith('EFD'):
                    metadata['product_family'] = 'EFD'
                elif product.startswith('ERSF'):
                    metadata['product_family'] = 'ERSF'
                elif product.startswith('CVI') or product.startswith('CVIR'):
                    metadata['product_family'] = 'CONTROLLER'
                break
        
        # ESDE code
        esde_match = re.search(r'ESDE[-\s]?(\d{4,5})', content, re.IGNORECASE)
        if esde_match:
            metadata['esde_code'] = f"ESDE-{esde_match.group(1)}"
        
        return metadata
    
    async def _upload_chunks(
        self,
        chunks: List[Chunk],
        file_path: Path,
        doc_type: str
    ):
        """Upload chunks to Qdrant"""
        points = []
        
        for chunk in chunks:
            # Prepare payload
            payload = {
                'text': chunk.text,
                'document_id': str(file_path),
                'document_type': doc_type,
                'chunk_type': chunk.chunk_type,
                'chunk_index': chunk.chunk_index,
                **chunk.metadata
            }
            
            # Generate ID
            import hashlib
            chunk_id = hashlib.md5(
                f"{file_path}_{chunk.chunk_index}".encode()
            ).hexdigest()
            
            points.append({
                'id': chunk_id,
                'text': chunk.text,
                'payload': payload
            })
        
        # Batch upload
        try:
            self.qdrant.add_documents(
                texts=[p['text'] for p in points],
                metadatas=[p['payload'] for p in points],
                ids=[p['id'] for p in points]
            )
            logger.debug(f"Uploaded {len(points)} chunks to Qdrant")
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
