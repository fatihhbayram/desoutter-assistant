#!/usr/bin/env python3
"""
Qdrant Metadata Enrichment Script
==================================
Enriches existing Qdrant points with missing metadata fields.

This script reads all 26K+ points from the desoutter_docs_v2 collection,
analyzes their text content, and updates metadata in-place using Qdrant's
set_payload API — NO re-embedding required.

Fields enriched:
  - document_type: Classified document type (8 types)
  - chunk_type: Type of chunk (semantic_section, table_row, error_code, etc.)
  - product_model: Specific product model (EABC-3000, EFD-1500, etc.)
  - contains_procedure: Whether chunk contains step-by-step instructions
  - contains_table: Whether chunk contains tabular data
  - contains_error_code: Whether chunk contains error codes
  - esde_code: ESDE bulletin code if present
  - error_code: Specific error code if present
  - intent_relevance: List of relevant intent types

Usage:
    python scripts/enrich_qdrant_metadata.py
    python scripts/enrich_qdrant_metadata.py --dry-run
    python scripts/enrich_qdrant_metadata.py --batch-size 50 --limit 100
"""
import os
import sys
import re
import time
import argparse
import logging
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, Filter, FieldCondition,
    MatchValue, ScrollRequest
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "desoutter_docs_v2")

# =============================================================================
# DOCUMENT TYPE DETECTION
# =============================================================================

DOCUMENT_TYPE_PATTERNS = {
    "service_bulletin": {
        "required": [r"ESDE[-\s]?\d{4,5}", r"Service\s+Bulletin"],
        "optional": [r"Affected\s+Models", r"Manufacturing\s+Defect", r"Corrective\s+Action"],
        "min_required": 1
    },
    "configuration_guide": {
        "required": [r"[Cc]onfigur", r"[Ss]etup|[Pp]arameter", r"[Pp]set"],
        "optional": [r"Torque\s+Settings", r"Angle\s+Control", r"Menu\s+Navigation"],
        "min_required": 2
    },
    "compatibility_matrix": {
        "required": [r"[Cc]ompatibil", r"\|\s*Tool\s*\|.*Controller"],
        "optional": [r"Supported\s+Versions", r"Firmware\s+Require"],
        "min_required": 1
    },
    "error_code_list": {
        "required": [r"[A-Z]{1,4}[-\s]?\d{2,4}\s*:"],
        "optional": [r"Error\s+Code", r"Diagnostic", r"Fault"],
        "min_required": 3  # At least 3 error code patterns
    },
    "procedure_guide": {
        "required": [r"^\s*\d+\.\s+[A-Z]", r"[Pp]rocedure|[Ss]tep"],
        "optional": [r"Installation", r"Firmware\s+Update", r"Calibration", r"Prerequisite"],
        "min_required": 2
    },
    "freshdesk_ticket": {
        "required": [r"[Tt]icket", r"[Cc]ustomer|[Mm]üşteri"],
        "optional": [r"[Rr]esolution", r"[Rr]eply", r"Subject:"],
        "min_required": 1
    },
    "spec_sheet": {
        "required": [r"[Ss]pecification", r"\d+\s*(Nm|kg|mm|rpm|bar|V|W)"],
        "optional": [r"[Dd]imension", r"[Ww]eight", r"[Pp]ower"],
        "min_required": 2
    }
}

# =============================================================================
# PRODUCT MODEL EXTRACTION
# =============================================================================

PRODUCT_PATTERNS = [
    # Battery tools
    r'\b(EABC[-\s]?\d{3,5}[A-Z0-9]*)\b',
    r'\b(EPBC[-\s]?\d{1,5}[A-Z0-9-]*)\b',
    r'\b(EPB[-\s]?\d{1,5}[A-Z0-9-]*)\b',
    r'\b(EPBA[-\s]?\d{1,5}[A-Z0-9-]*)\b',
    r'\b(EPBAHT[-\s]?\d{1,5}[A-Z0-9-]*)\b',
    r'\b(EABS[-\s]?\d{1,5}[A-Z0-9-]*)\b',
    r'\b(EAB[-\s]?\d{1,5}[A-Z0-9-]*)\b',
    # Cable tools
    r'\b(EFD[-\s]?\d{1,5}[A-Z0-9-]*)\b',
    r'\b(ERSF[-\s]?\d{1,5}[A-Z0-9-]*)\b',
    r'\b(ERS[-\s]?\d{1,5}[A-Z0-9-]*)\b',
    r'\b(ERXS[-\s]?\d{1,5}[A-Z0-9-]*)\b',
    r'\b(EIBS[-\s]?\d{1,5}[A-Z0-9-]*)\b',
    r'\b(ELC[-\s]?\d{1,5}[A-Z0-9-]*)\b',
    r'\b(BLRTC[-\s]?\d{1,5}[A-Z0-9-]*)\b',
    r'\b(SLC[-\s]?\d{1,5}[A-Z0-9-]*)\b',
    r'\b(ECS[-\s]?\d{1,5}[A-Z0-9-]*)\b',
    # Controllers
    r'\b(CVI3[-\s]?\d*[A-Z]*)\b',
    r'\b(CVIR[-\s]?II?[A-Z0-9]*)\b',
    r'\b(CVIC[-\s]?II?[A-Z0-9]*)\b',
    r'\b(CVIL[-\s]?II?[A-Z0-9]*)\b',
    r'\b(CVIXS[-\s]?\d*)\b',
    r'\b(CONNECT[-\s]?[WX]?)\b',
    r'\b(AXON[-\s]?\d*)\b',
    r'\b(ESP[-\s]?\d*)\b',
]

PRODUCT_FAMILY_MAP = {
    'EABC': 'EABC', 'EPBC': 'EPBC', 'EPB': 'EPB', 'EPBA': 'EPBA',
    'EPBAHT': 'EPBAHT', 'EABS': 'EABS', 'EAB': 'EAB',
    'EFD': 'EFD', 'ERSF': 'ERSF', 'ERS': 'ERS', 'ERXS': 'ERXS',
    'EIBS': 'EIBS', 'ELC': 'ELC', 'BLRTC': 'BLRTC', 'SLC': 'SLC', 'ECS': 'ECS',
    'CVI3': 'CONTROLLER', 'CVIR': 'CONTROLLER', 'CVIC': 'CONTROLLER',
    'CVIL': 'CONTROLLER', 'CVIXS': 'CONTROLLER',
    'CONNECT': 'CONTROLLER', 'AXON': 'CONTROLLER', 'ESP': 'CONTROLLER',
}

# Intent relevance mapping per document type
INTENT_RELEVANCE_MAP = {
    "service_bulletin": ["TROUBLESHOOT", "ERROR_CODE"],
    "configuration_guide": ["CONFIGURATION", "PROCEDURE", "HOW_TO"],
    "compatibility_matrix": ["COMPATIBILITY", "SPECIFICATION"],
    "error_code_list": ["ERROR_CODE", "TROUBLESHOOT"],
    "procedure_guide": ["PROCEDURE", "HOW_TO", "INSTALLATION", "FIRMWARE"],
    "freshdesk_ticket": ["TROUBLESHOOT", "HOW_TO", "GENERAL"],
    "spec_sheet": ["SPECIFICATION", "CAPABILITY_QUERY"],
    "technical_manual": ["HOW_TO", "CONFIGURATION", "MAINTENANCE", "GENERAL"],
}


# =============================================================================
# ENRICHMENT FUNCTIONS
# =============================================================================

def detect_document_type(text: str, source: str = "") -> str:
    """Detect document type from text content and source path."""
    combined = text + " " + source
    
    # Check source path first for quick classification
    source_lower = source.lower()
    if 'esde' in source_lower or 'esb' in source_lower or 'bulletin' in source_lower:
        return "service_bulletin"
    if 'ticket' in source_lower or 'freshdesk' in source_lower:
        return "freshdesk_ticket"
    
    best_type = "technical_manual"  # default fallback
    best_score = 0
    
    for doc_type, patterns in DOCUMENT_TYPE_PATTERNS.items():
        required_matches = 0
        optional_matches = 0
        
        for pattern in patterns["required"]:
            if doc_type == "error_code_list":
                # Count occurrences for error codes
                matches = re.findall(pattern, combined[:3000], re.MULTILINE)
                required_matches += len(matches)
            else:
                if re.search(pattern, combined[:3000], re.IGNORECASE | re.MULTILINE):
                    required_matches += 1
        
        for pattern in patterns["optional"]:
            if re.search(pattern, combined[:3000], re.IGNORECASE):
                optional_matches += 1
        
        if required_matches >= patterns["min_required"]:
            score = required_matches * 2 + optional_matches
            if score > best_score:
                best_score = score
                best_type = doc_type
    
    return best_type


def detect_chunk_type(text: str) -> str:
    """Determine chunk type from content analysis."""
    # Check for table content
    if re.search(r'\|.*\|.*\|', text) or text.count('\t') > 5:
        return "table_row"
    
    # Check for error code listing
    if re.search(r'[A-Z]{1,4}[-\s]?\d{2,4}\s*:', text) and len(text) < 500:
        return "error_code"
    
    # Check for procedure steps
    if re.search(r'^\s*\d+\.\s+[A-Z]', text, re.MULTILINE):
        step_count = len(re.findall(r'^\s*\d+\.\s+', text, re.MULTILINE))
        if step_count >= 2:
            return "procedure"
    
    # Check for problem-solution pair (ESDE)
    if re.search(r'ESDE[-\s]?\d{4,5}', text) and re.search(r'(solution|remedy|fix|corrective)', text, re.IGNORECASE):
        return "problem_solution_pair"
    
    # Default
    return "semantic_section"


def extract_product_model(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract product model and family from text."""
    for pattern in PRODUCT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            model = match.group(1).upper().replace(' ', '-')
            # Skip generic controller mentions
            if model in ('CONNECT', 'ESP'):
                continue
            # Determine family
            for prefix, family in sorted(PRODUCT_FAMILY_MAP.items(), key=lambda x: -len(x[0])):
                if model.startswith(prefix):
                    return model, family
            return model, None
    return None, None


def extract_error_codes(text: str) -> List[str]:
    """Extract error codes from text."""
    codes = []
    # E-code patterns (E06, E018, E004, etc.)
    codes.extend(re.findall(r'\b(E\d{2,4})\b', text))
    # I-code patterns
    codes.extend(re.findall(r'\b(I\d{3,4})\b', text))
    # Tool-specific codes (TRD-E06, etc.)
    codes.extend(re.findall(r'\b([A-Z]{2,4}-[EI]\d{2,4})\b', text))
    return list(set(codes))


def extract_esde_codes(text: str) -> List[str]:
    """Extract ESDE bulletin codes."""
    return list(set(re.findall(r'(ESDE[-\s]?\d{4,5})', text, re.IGNORECASE)))


def detect_features(text: str) -> Dict[str, bool]:
    """Detect content features for metadata enrichment."""
    return {
        "contains_procedure": bool(re.search(
            r"(Step\s+\d+|^\s*\d+\.\s+[A-Z])", text, re.IGNORECASE | re.MULTILINE
        )),
        "contains_table": bool(re.search(
            r"\|.*\|.*\||\t.*\t", text
        )),
        "contains_error_code": bool(re.search(
            r"[A-Z]{1,4}[-\s]?\d{2,4}\s*:", text
        )),
        "contains_compatibility_info": bool(re.search(
            r"[Cc]ompatib|[Uu]yumlu|[Ss]upported\s+[Vv]ersion", text
        )),
    }


def build_enriched_metadata(text: str, existing_payload: Dict) -> Dict[str, Any]:
    """Build enriched metadata from text content and existing payload."""
    source = existing_payload.get("source", "")
    
    # Detect document type
    document_type = detect_document_type(text, source)
    
    # Detect chunk type
    chunk_type = detect_chunk_type(text)
    
    # Extract product info
    product_model, product_family = extract_product_model(text)
    
    # Use existing product_family if we didn't find one
    if not product_family and existing_payload.get("product_family"):
        product_family = existing_payload["product_family"]
    
    # Extract codes
    error_codes = extract_error_codes(text)
    esde_codes = extract_esde_codes(text)
    
    # Detect features
    features = detect_features(text)
    
    # Build intent relevance
    intent_relevance = INTENT_RELEVANCE_MAP.get(document_type, ["GENERAL"])
    # Add intent based on content features
    if features["contains_error_code"] and "ERROR_CODE" not in intent_relevance:
        intent_relevance.append("ERROR_CODE")
    if features["contains_procedure"] and "PROCEDURE" not in intent_relevance:
        intent_relevance.append("PROCEDURE")
    if features["contains_compatibility_info"] and "COMPATIBILITY" not in intent_relevance:
        intent_relevance.append("COMPATIBILITY")
    
    # Build enrichment payload
    enrichment = {
        "document_type": document_type,
        "chunk_type": chunk_type,
        "intent_relevance": intent_relevance,
        **features,
    }
    
    # Add optional fields only if found
    if product_model:
        enrichment["product_model"] = product_model
    if product_family:
        enrichment["product_family"] = product_family
    if error_codes:
        enrichment["error_code"] = error_codes[0]  # Primary error code
    if esde_codes:
        enrichment["esde_code"] = esde_codes[0].replace(" ", "-").upper()
    
    return enrichment


# =============================================================================
# MAIN ENRICHMENT PIPELINE
# =============================================================================

def enrich_collection(
    host: str = QDRANT_HOST,
    port: int = QDRANT_PORT,
    collection: str = COLLECTION_NAME,
    batch_size: int = 100,
    limit: Optional[int] = None,
    dry_run: bool = False
):
    """
    Enrich all points in Qdrant collection with missing metadata.
    
    Uses scroll + set_payload to update metadata in-place — no re-embedding.
    """
    client = QdrantClient(host=host, port=port, timeout=60)
    
    # Verify collection exists
    collections = [c.name for c in client.get_collections().collections]
    if collection not in collections:
        logger.error(f"Collection '{collection}' not found. Available: {collections}")
        return
    
    # Get collection info
    info = client.get_collection(collection)
    total_points = info.points_count
    logger.info(f"Collection '{collection}': {total_points} points")
    
    if limit:
        total_points = min(total_points, limit)
        logger.info(f"Processing limited to {total_points} points")
    
    # Stats
    stats = {
        "total": 0,
        "enriched": 0,
        "skipped": 0,
        "errors": 0,
        "by_doc_type": {},
        "by_chunk_type": {},
    }
    
    # Scroll through all points
    offset = None
    processed = 0
    start_time = time.time()
    
    while True:
        # Read batch
        results, next_offset = client.scroll(
            collection_name=collection,
            offset=offset,
            limit=batch_size,
            with_payload=True,
            with_vectors=False  # Skip vectors — no re-embedding needed
        )
        
        if not results:
            break
        
        # Process batch
        for point in results:
            stats["total"] += 1
            processed += 1
            
            if limit and processed > limit:
                break
            
            try:
                payload = point.payload or {}
                text = payload.get("text", "")
                
                if not text:
                    stats["skipped"] += 1
                    continue
                
                # Build enriched metadata
                enrichment = build_enriched_metadata(text, payload)
                
                # Track stats
                doc_type = enrichment["document_type"]
                chunk_type = enrichment["chunk_type"]
                stats["by_doc_type"][doc_type] = stats["by_doc_type"].get(doc_type, 0) + 1
                stats["by_chunk_type"][chunk_type] = stats["by_chunk_type"].get(chunk_type, 0) + 1
                
                # Apply enrichment
                if not dry_run:
                    client.set_payload(
                        collection_name=collection,
                        payload=enrichment,
                        points=[point.id],
                    )
                
                stats["enriched"] += 1
                
            except Exception as e:
                stats["errors"] += 1
                if stats["errors"] <= 5:
                    logger.error(f"Error processing point {point.id}: {e}")
        
        # Progress logging
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total_points - processed) / rate if rate > 0 else 0
        
        logger.info(
            f"Progress: {processed}/{total_points} "
            f"({processed/total_points*100:.1f}%) "
            f"Rate: {rate:.0f} pts/s "
            f"ETA: {eta:.0f}s"
        )
        
        if limit and processed >= limit:
            break
        
        offset = next_offset
        if offset is None:
            break
    
    # Print summary
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("METADATA ENRICHMENT SUMMARY")
    print("=" * 60)
    print(f"Total points:    {stats['total']}")
    print(f"Enriched:        {stats['enriched']}")
    print(f"Skipped (empty): {stats['skipped']}")
    print(f"Errors:          {stats['errors']}")
    print(f"Time elapsed:    {elapsed:.1f}s")
    print(f"Rate:            {stats['enriched']/elapsed:.0f} pts/s" if elapsed > 0 else "")
    
    print(f"\nBy document type:")
    for doc_type, count in sorted(stats["by_doc_type"].items(), key=lambda x: -x[1]):
        print(f"  {doc_type:30s} {count:6d}")
    
    print(f"\nBy chunk type:")
    for chunk_type, count in sorted(stats["by_chunk_type"].items(), key=lambda x: -x[1]):
        print(f"  {chunk_type:30s} {count:6d}")
    
    if dry_run:
        print("\n🔸 DRY RUN — No changes applied to Qdrant")
    else:
        print("\n✅ Metadata enrichment complete!")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich Qdrant metadata in-place")
    parser.add_argument("--host", default=QDRANT_HOST, help="Qdrant host")
    parser.add_argument("--port", type=int, default=QDRANT_PORT, help="Qdrant port")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Collection name")
    parser.add_argument("--batch-size", type=int, default=100, help="Scroll batch size")
    parser.add_argument("--limit", type=int, help="Limit number of points to process")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without writing")
    
    args = parser.parse_args()
    
    enrich_collection(
        host=args.host,
        port=args.port,
        collection=args.collection,
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run
    )
