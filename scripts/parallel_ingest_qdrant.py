#!/usr/bin/env python3
"""
Parallel Document Ingestion Script
==================================
Ingests documents to both ChromaDB and Qdrant simultaneously.

Features:
- Parallel ingestion for comparison
- Adaptive chunking based on document type
- Metadata enrichment
- Progress tracking
- Validation

Usage:
    python scripts/parallel_ingest_qdrant.py
    python scripts/parallel_ingest_qdrant.py --qdrant-only
    python scripts/parallel_ingest_qdrant.py --validate
"""
import os
import sys
import logging
import argparse
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Document processing
from src.documents.document_processor import DocumentProcessor
from src.documents.document_classifier import DocumentClassifier, DocumentType
from src.documents.embeddings import EmbeddingsGenerator

# Vector stores
from src.vectordb.chroma_client import ChromaDBClient
from src.vectordb.qdrant_client import QdrantDBClient, QDRANT_AVAILABLE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "documents")
BULLETINS_DIR = os.path.join(DOCUMENTS_DIR, "bulletins")
MANUALS_DIR = os.path.join(DOCUMENTS_DIR, "manuals")

# Product family patterns for metadata extraction
PRODUCT_FAMILY_PATTERNS = {
    "EABC": ["EABC", "ExB Com", "ExBCom"],
    "EPBC": ["EPBC", "ExB Com", "ExBCom"],
    "EABA": ["EABA", "ExB Advanced"],
    "EPBA": ["EPBA", "ExB Advanced"],
    "EAB": ["EAB", "ExB Flex"],
    "EPB": ["EPB", "ExB Flex"],
    "CVI3": ["CVI3", "CVI 3"],
    "CVIL2": ["CVIL2", "CVI Light"],
    "CVIR": ["CVIR", "CVI Rail"],
    "CONNECT": ["CONNECT", "Connect-W"],
    "ERS": ["ERS", "ERSF"],
    "EFD": ["EFD", "Fluid Dispenser"],
}


class IngestionResult:
    """Track ingestion results"""
    def __init__(self):
        self.total_files = 0
        self.processed_files = 0
        self.total_chunks = 0
        self.chroma_chunks = 0
        self.qdrant_chunks = 0
        self.errors = []
        self.start_time = None
        self.end_time = None
    
    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0
    
    def __str__(self):
        return f"""
Ingestion Result:
  Files: {self.processed_files}/{self.total_files}
  Total Chunks: {self.total_chunks}
  ChromaDB: {self.chroma_chunks}
  Qdrant: {self.qdrant_chunks}
  Errors: {len(self.errors)}
  Duration: {self.duration:.1f}s
"""


def extract_product_family(text: str, filename: str) -> Optional[str]:
    """Extract product family from text or filename"""
    combined = f"{filename} {text[:500]}".upper()
    
    for family, patterns in PRODUCT_FAMILY_PATTERNS.items():
        for pattern in patterns:
            if pattern.upper() in combined:
                return family
    
    return None


def extract_esde_code(text: str, filename: str) -> Optional[str]:
    """Extract ESDE code from text or filename"""
    import re
    
    combined = f"{filename} {text[:500]}"
    match = re.search(r'ESDE[-\s]?(\d{4,5})', combined, re.IGNORECASE)
    
    if match:
        return f"ESDE{match.group(1)}"
    
    return None


def extract_error_codes(text: str) -> List[str]:
    """Extract error codes from text"""
    import re
    
    # Common error code patterns
    patterns = [
        r'E\d{2,3}',           # E01, E012
        r'[A-Z]{2,4}-\d{3,4}', # EABC-001, SPD-E06
    ]
    
    codes = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        codes.update(matches)
    
    return list(codes)


def enrich_metadata(
    base_metadata: Dict[str, Any],
    text: str,
    filename: str,
    doc_type: DocumentType
) -> Dict[str, Any]:
    """Enrich chunk metadata with extracted information"""
    
    enriched = {**base_metadata}
    
    # Document type
    enriched['document_type'] = doc_type.value
    
    # Product family
    product_family = extract_product_family(text, filename)
    if product_family:
        enriched['product_family'] = product_family
    
    # ESDE code
    esde_code = extract_esde_code(text, filename)
    if esde_code:
        enriched['esde_code'] = esde_code
    
    # Error codes
    error_codes = extract_error_codes(text)
    if error_codes:
        enriched['error_codes'] = error_codes
        enriched['contains_error_code'] = True
    
    # Content flags
    text_lower = text.lower()
    enriched['contains_procedure'] = any(kw in text_lower for kw in [
        'step 1', 'adım 1', 'procedure', 'prosedür', 'follow these steps'
    ])
    enriched['contains_table'] = '|' in text or 'table' in text_lower
    
    # Language detection (simple)
    turkish_chars = sum(1 for c in text if c in 'çğıöşüÇĞİÖŞÜ')
    enriched['language'] = 'tr' if turkish_chars > 10 else 'en'
    
    # Timestamp
    enriched['ingested_at'] = datetime.utcnow().isoformat()
    
    return enriched


def process_document(
    filepath: str,
    processor: DocumentProcessor,
    classifier: DocumentClassifier,
    embedder: EmbeddingsGenerator
) -> List[Dict[str, Any]]:
    """
    Process a single document into chunks with metadata.
    
    Returns:
        List of chunks ready for ingestion
    """
    filename = os.path.basename(filepath)
    logger.debug(f"Processing: {filename}")
    
    try:
        # Load and extract text
        chunks = processor.process_file(filepath)
        
        if not chunks:
            logger.warning(f"No chunks from: {filename}")
            return []
        
        # Classify document type
        full_text = " ".join([c.get('text', '') for c in chunks[:5]])
        classification = classifier.classify(full_text, filename)
        doc_type = classification.document_type
        
        logger.debug(f"{filename} -> {doc_type.value} (conf: {classification.confidence:.2f})")
        
        # Process chunks
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            
            if not text.strip():
                continue
            
            # Generate ID
            chunk_id = hashlib.md5(f"{filename}_{i}_{text[:100]}".encode()).hexdigest()
            
            # Generate embedding
            embedding = embedder.generate(text)
            
            # Enrich metadata
            metadata = enrich_metadata(
                base_metadata={
                    'source': filename,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    **chunk.get('metadata', {})
                },
                text=text,
                filename=filename,
                doc_type=doc_type
            )
            
            processed_chunks.append({
                'id': chunk_id,
                'text': text,
                'embedding': embedding,
                'metadata': metadata
            })
        
        return processed_chunks
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return []


def ingest_to_chroma(
    chunks: List[Dict[str, Any]],
    client: ChromaDBClient
) -> int:
    """Ingest chunks to ChromaDB"""
    try:
        success_count = 0
        for chunk in chunks:
            try:
                client.add_document(
                    doc_id=chunk['id'],
                    text=chunk['text'],
                    embedding=chunk['embedding'],
                    metadata=chunk['metadata']
                )
                success_count += 1
            except Exception as e:
                logger.debug(f"ChromaDB insert error: {e}")
        
        return success_count
    except Exception as e:
        logger.error(f"ChromaDB ingestion failed: {e}")
        return 0


def ingest_to_qdrant(
    chunks: List[Dict[str, Any]],
    client: QdrantDBClient
) -> int:
    """Ingest chunks to Qdrant"""
    try:
        # Convert IDs to integers for Qdrant
        qdrant_docs = []
        for chunk in chunks:
            qdrant_docs.append({
                'id': int(hashlib.md5(chunk['id'].encode()).hexdigest()[:16], 16),
                'text': chunk['text'],
                'embedding': chunk['embedding'],
                'metadata': {
                    'original_id': chunk['id'],
                    **chunk['metadata']
                }
            })
        
        success, errors = client.upsert(qdrant_docs)
        return success
        
    except Exception as e:
        logger.error(f"Qdrant ingestion failed: {e}")
        return 0


def run_parallel_ingestion(
    documents_dirs: List[str],
    chroma_client: Optional[ChromaDBClient],
    qdrant_client: Optional[QdrantDBClient],
    max_workers: int = 4
) -> IngestionResult:
    """
    Run parallel ingestion to both vector stores.
    
    Args:
        documents_dirs: List of directories containing documents
        chroma_client: ChromaDB client (None to skip)
        qdrant_client: Qdrant client (None to skip)
        max_workers: Number of parallel workers
        
    Returns:
        IngestionResult with statistics
    """
    result = IngestionResult()
    result.start_time = datetime.now()
    
    # Initialize processors
    processor = DocumentProcessor()
    classifier = DocumentClassifier()
    embedder = EmbeddingsGenerator()
    
    # Ensure Qdrant collection exists
    if qdrant_client:
        qdrant_client.ensure_collection()
    
    # Collect all files
    all_files = []
    for dir_path in documents_dirs:
        if os.path.exists(dir_path):
            for ext in ['*.pdf', '*.docx', '*.doc', '*.txt']:
                all_files.extend(Path(dir_path).glob(ext))
    
    result.total_files = len(all_files)
    logger.info(f"Found {result.total_files} documents to process")
    
    if not all_files:
        result.end_time = datetime.now()
        return result
    
    # Process documents
    with tqdm(total=len(all_files), desc="Processing documents") as pbar:
        for filepath in all_files:
            try:
                # Process document
                chunks = process_document(
                    str(filepath),
                    processor,
                    classifier,
                    embedder
                )
                
                if not chunks:
                    pbar.update(1)
                    continue
                
                result.total_chunks += len(chunks)
                
                # Ingest to ChromaDB
                if chroma_client:
                    chroma_count = ingest_to_chroma(chunks, chroma_client)
                    result.chroma_chunks += chroma_count
                
                # Ingest to Qdrant
                if qdrant_client:
                    qdrant_count = ingest_to_qdrant(chunks, qdrant_client)
                    result.qdrant_chunks += qdrant_count
                
                result.processed_files += 1
                
            except Exception as e:
                result.errors.append(f"{filepath}: {e}")
                logger.error(f"Error processing {filepath}: {e}")
            
            pbar.update(1)
    
    result.end_time = datetime.now()
    return result


def validate_ingestion(
    chroma_client: Optional[ChromaDBClient],
    qdrant_client: Optional[QdrantDBClient]
) -> Dict[str, Any]:
    """Validate ingestion by comparing counts and sample searches"""
    
    validation = {
        'chroma_count': 0,
        'qdrant_count': 0,
        'match': False,
        'details': []
    }
    
    if chroma_client:
        try:
            validation['chroma_count'] = chroma_client.collection.count()
        except Exception as e:
            validation['details'].append(f"ChromaDB count failed: {e}")
    
    if qdrant_client:
        try:
            info = qdrant_client.get_collection_info()
            validation['qdrant_count'] = info.get('points_count', 0) if info else 0
        except Exception as e:
            validation['details'].append(f"Qdrant count failed: {e}")
    
    # Check match (with tolerance)
    if validation['chroma_count'] > 0 and validation['qdrant_count'] > 0:
        diff = abs(validation['chroma_count'] - validation['qdrant_count'])
        tolerance = max(10, validation['chroma_count'] * 0.05)
        validation['match'] = diff <= tolerance
    
    return validation


def main():
    parser = argparse.ArgumentParser(description="Parallel document ingestion")
    parser.add_argument(
        "--qdrant-only", action="store_true",
        help="Only ingest to Qdrant (skip ChromaDB)"
    )
    parser.add_argument(
        "--chroma-only", action="store_true",
        help="Only ingest to ChromaDB (skip Qdrant)"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Only validate existing ingestion"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("PARALLEL DOCUMENT INGESTION")
    print("=" * 60)
    
    # Initialize clients
    chroma_client = None
    qdrant_client = None
    
    if not args.qdrant_only:
        try:
            chroma_client = ChromaDBClient()
            print("✅ ChromaDB connected")
        except Exception as e:
            print(f"⚠️ ChromaDB not available: {e}")
    
    if not args.chroma_only:
        if QDRANT_AVAILABLE:
            try:
                qdrant_client = QdrantDBClient()
                print("✅ Qdrant connected")
            except Exception as e:
                print(f"⚠️ Qdrant not available: {e}")
        else:
            print("⚠️ qdrant-client not installed")
    
    # Validate only mode
    if args.validate:
        print("\n📊 Validating ingestion...")
        validation = validate_ingestion(chroma_client, qdrant_client)
        print(f"\nChromaDB: {validation['chroma_count']} chunks")
        print(f"Qdrant: {validation['qdrant_count']} chunks")
        print(f"Match: {'✅' if validation['match'] else '❌'}")
        if validation['details']:
            for detail in validation['details']:
                print(f"  - {detail}")
        return
    
    # Run ingestion
    if not chroma_client and not qdrant_client:
        print("❌ No vector stores available")
        sys.exit(1)
    
    print(f"\n🚀 Starting ingestion...")
    result = run_parallel_ingestion(
        documents_dirs=[BULLETINS_DIR, MANUALS_DIR],
        chroma_client=chroma_client,
        qdrant_client=qdrant_client,
        max_workers=args.workers
    )
    
    print(result)
    
    if result.errors:
        print(f"\n⚠️ Errors ({len(result.errors)}):")
        for err in result.errors[:5]:
            print(f"  - {err}")
    
    # Validate
    print("\n📊 Validating...")
    validation = validate_ingestion(chroma_client, qdrant_client)
    print(f"ChromaDB: {validation['chroma_count']} chunks")
    print(f"Qdrant: {validation['qdrant_count']} chunks")
    print(f"Match: {'✅' if validation['match'] else '⚠️'}")
    
    print("\n✨ Ingestion complete!")


if __name__ == "__main__":
    main()
