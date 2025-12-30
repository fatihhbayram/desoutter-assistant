"""
Citation Formatter for RAG Sources
Provides standardized formatting for source citations in responses.
"""
from typing import List, Dict, Optional


def format_citation(source: Dict) -> str:
    """
    Format a single source citation for display.
    
    Args:
        source: Source dict with 'source', 'page', 'section', 'similarity'
        
    Returns:
        Formatted citation string
        
    Examples:
        >>> format_citation({"source": "Manual.pdf", "page": 25, "section": "3.2 Troubleshooting"})
        "Manual.pdf, Page 25, Section 3.2 Troubleshooting"
        
        >>> format_citation({"source": "Bulletin.pdf", "page": None, "section": "Safety"})
        "Bulletin.pdf, Section Safety"
    """
    parts = [source.get("source", "Unknown")]
    
    if source.get("page"):
        parts.append(f"Page {source['page']}")
    
    if source.get("section"):
        parts.append(f"Section {source['section']}")
    
    return ", ".join(parts)


def format_citation_list(sources: List[Dict]) -> str:
    """
    Format multiple sources for inclusion in LLM prompt or display.
    
    Args:
        sources: List of source dicts
        
    Returns:
        Formatted citation list string
        
    Example:
        >>> sources = [
        ...     {"source": "Manual.pdf", "page": 25, "section": "Troubleshooting"},
        ...     {"source": "Bulletin.pdf", "page": 3, "section": None}
        ... ]
        >>> print(format_citation_list(sources))
        Sources:
        1. Manual.pdf, Page 25, Section Troubleshooting
        2. Bulletin.pdf, Page 3
    """
    if not sources:
        return "No sources available"
    
    formatted = ["Sources:"]
    for idx, source in enumerate(sources, 1):
        formatted.append(f"{idx}. {format_citation(source)}")
    
    return "\n".join(formatted)


def format_context_with_citations(chunks: List[Dict]) -> str:
    """
    Format context chunks with citation headers for LLM prompt.
    
    Args:
        chunks: List of chunk dicts with 'text' and 'metadata'
        
    Returns:
        Formatted context string with citations
        
    Example:
        Context 1 (Source: Manual.pdf, Page 25, Section 3.2):
        [chunk text]
        
        Context 2 (Source: Bulletin.pdf, Page 3):
        [chunk text]
    """
    if not chunks:
        return ""
    
    formatted_chunks = []
    
    for idx, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        
        # Build citation header
        citation_parts = []
        if metadata.get("source"):
            citation_parts.append(f"Source: {metadata['source']}")
        if metadata.get("page_number"):
            citation_parts.append(f"Page {metadata['page_number']}")
        if metadata.get("section"):
            citation_parts.append(f"Section {metadata['section']}")
        
        citation_header = ", ".join(citation_parts) if citation_parts else "Unknown Source"
        
        # Format chunk
        formatted_chunks.append(
            f"Context {idx} ({citation_header}):\n{chunk.get('text', '')}"
        )
    
    return "\n\n".join(formatted_chunks)


def get_citation_summary(sources: List[Dict]) -> Dict[str, int]:
    """
    Get summary statistics about citations.
    
    Args:
        sources: List of source dicts
        
    Returns:
        Dict with citation statistics
    """
    total = len(sources)
    with_pages = sum(1 for s in sources if s.get("page"))
    with_sections = sum(1 for s in sources if s.get("section"))
    
    unique_sources = len(set(s.get("source", "") for s in sources))
    
    return {
        "total_citations": total,
        "with_page_numbers": with_pages,
        "with_sections": with_sections,
        "unique_documents": unique_sources,
        "page_coverage": f"{(with_pages/total*100):.1f}%" if total > 0 else "0%"
    }
