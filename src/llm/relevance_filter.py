"""
Relevance filtering for RAG retrieval results.
Filters out documents that don't match query intent.

PRODUCTION-SAFE:
- Optional post-processing filter (can be disabled via config)
- Does not modify hybrid search logic
- Backward compatible - returns original results if disabled
- Logs all filtering decisions for monitoring
"""

from typing import List, Dict, Optional
from config.relevance_filters import (
    ENABLE_RELEVANCE_FILTERING,
    RELEVANCE_FILTER_RULES,
    MIN_SIMILARITY_AFTER_FILTER,
    MAX_DOCUMENTS_TO_EXCLUDE
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def filter_irrelevant_results(
    query: str,
    results: List[Dict],
    enable_filtering: Optional[bool] = None
) -> List[Dict]:
    """
    Filter out irrelevant documents based on query intent.
    
    SAFETY: Returns original results if:
    - Filtering is disabled
    - No intent detected
    - Too many documents would be excluded
    
    Args:
        query: User's fault description
        results: List of retrieved documents with metadata
        enable_filtering: Override config setting (for testing)
        
    Returns:
        Filtered list of results (or original if filtering disabled/failed)
    """
    # Check if filtering is enabled
    if enable_filtering is None:
        enable_filtering = ENABLE_RELEVANCE_FILTERING
    
    if not enable_filtering:
        logger.info("Relevance filtering disabled via config")
        return results
    
    if not results:
        logger.info("No results to filter")
        return results
    
    # Detect query intent
    intent = detect_query_intent(query)
    
    if not intent:
        logger.info("No specific intent detected, skipping filter")
        return results
    
    logger.info(f"Detected query intent: {intent}")
    
    # Get filter rules for this intent
    rules = RELEVANCE_FILTER_RULES.get(intent)
    if not rules:
        logger.warning(f"No filter rules found for intent: {intent}")
        return results
    
    # Filter results
    filtered = []
    excluded = []
    
    for result in results:
        content = result.get('content', '').lower()
        source = result.get('source', '')
        
        # Check exclude keywords
        is_excluded = False
        exclude_reason = None
        
        for keyword in rules.get('exclude_if_contains', []):
            if keyword.lower() in content:
                is_excluded = True
                exclude_reason = f"contains '{keyword}'"
                break
        
        if is_excluded:
            excluded.append({
                'source': source,
                'reason': exclude_reason
            })
            continue
        
        # Check required keywords
        required_keywords = rules.get('require_at_least_one', [])
        if required_keywords:
            has_required = any(kw.lower() in content for kw in required_keywords)
            if not has_required:
                excluded.append({
                    'source': source,
                    'reason': 'missing required keywords'
                })
                continue
        
        # Check minimum similarity
        similarity = float(result.get('similarity', 0))
        if similarity < MIN_SIMILARITY_AFTER_FILTER:
            excluded.append({
                'source': source,
                'reason': f'similarity {similarity:.2f} below threshold'
            })
            continue
        
        filtered.append(result)
    
    # Safety check: Don't exclude too many documents
    excluded_count = len(excluded)
    if excluded_count > MAX_DOCUMENTS_TO_EXCLUDE:
        logger.warning(
            f"Would exclude {excluded_count} documents (>{MAX_DOCUMENTS_TO_EXCLUDE}), "
            f"skipping filter for safety"
        )
        return results
    
    # Log filtering results
    logger.info(f"Filtered {excluded_count}/{len(results)} documents")
    logger.info(f"Remaining: {len(filtered)} documents")
    
    if excluded:
        logger.debug("Excluded documents:")
        for exc in excluded[:5]:  # Log first 5
            logger.debug(f"  - {exc['source']}: {exc['reason']}")
    
    # Safety check: Ensure we have at least some results
    if not filtered:
        logger.warning("All documents filtered out, returning original results")
        return results
    
    return filtered


def detect_query_intent(query: str) -> Optional[str]:
    """
    Detect the primary intent of a fault description query.
    
    Uses trigger keywords to identify fault category.
    Returns the first matching category (order matters).
    Uses word boundary matching to avoid false positives (e.g., 'led' in 'failed').
    
    Args:
        query: User's fault description
        
    Returns:
        Intent category or None if no match
    """
    import re
    query_lower = query.lower()
    
    # Check each intent category
    # Order matters - more specific intents first
    intent_order = [
        "error_codes",  # Error codes are very specific (E018, E01, etc.)
        "torque_calibration",
        "pset_configuration",
        "sensor",
        "touchscreen",  # More specific than display_screen
        "display_screen",
        "battery_power",
        "motor_mechanical",
        "sound_noise",
        "led_indicators",
        "button_controls",
        "cable_connector",
        "communication_protocol",  # More specific than wifi_network
        "wifi_network",
        "software_firmware"  # Most general
    ]
    
    for intent in intent_order:
        if intent not in RELEVANCE_FILTER_RULES:
            continue
            
        rules = RELEVANCE_FILTER_RULES[intent]
        trigger_keywords = rules.get('trigger_keywords', [])
        
        # Use word boundary regex to avoid false matches (e.g., 'led' in 'failed')
        for keyword in trigger_keywords:
            # Escape special regex characters in keyword
            escaped_keyword = re.escape(keyword)
            # Match whole word or phrase
            pattern = r'\b' + escaped_keyword + r'\b'
            if re.search(pattern, query_lower):
                return intent
    
    return None


def get_filter_stats(query: str, original_results: List[Dict], filtered_results: List[Dict]) -> Dict:
    """
    Get statistics about filtering for monitoring/debugging.
    
    Args:
        query: Original query
        original_results: Results before filtering
        filtered_results: Results after filtering
        
    Returns:
        Dict with filtering statistics
    """
    intent = detect_query_intent(query)
    
    return {
        "query": query,
        "intent": intent,
        "original_count": len(original_results),
        "filtered_count": len(filtered_results),
        "excluded_count": len(original_results) - len(filtered_results),
        "exclusion_rate": (len(original_results) - len(filtered_results)) / len(original_results) if original_results else 0
    }
