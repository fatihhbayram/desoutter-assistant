"""
Context Window Optimizer for RAG Engine
Optimizes retrieved chunks for better LLM responses

Features:
- Importance-based chunk prioritization
- Duplicate/similar content deduplication
- Token budget management
- Metadata-enriched context formatting
- Smart truncation preserving key information
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re
import hashlib

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class OptimizedChunk:
    """Represents an optimized chunk for context window"""
    text: str
    source: str
    similarity: float
    importance_score: float
    section_type: str
    heading_text: str
    is_procedure: bool
    is_warning: bool
    token_estimate: int
    
    def to_context_string(self, include_metadata: bool = True) -> str:
        """
        Format chunk for context window

        Priority 2.2: Reduced metadata overhead
        - OLD: source + heading + section_type = ~50-100 tokens
        - NEW: source + doc_type only = ~10-20 tokens
        - Result: ~50% token reduction in metadata
        """
        parts = []

        # Priority 2.2: Minimal metadata (only essential fields)
        if include_metadata:
            # Only include: source document name (no heading, no detailed section type)
            header = f"[{self.source}]"
            parts.append(header)

        # Keep warning marker (critical for safety)
        if self.is_warning:
            parts.append("⚠️ SAFETY WARNING:")

        # Add the actual content
        parts.append(self.text.strip())

        return "\n".join(parts)


class ContextOptimizer:
    """
    Optimizes retrieved chunks for better context window usage
    
    Strategies:
    1. Prioritize by importance score + similarity
    2. Deduplicate similar content
    3. Group by source document
    4. Respect token budget
    5. Preserve critical content (warnings, procedures)
    """
    
    # Token estimation: ~4 characters per token (conservative for multilingual)
    CHARS_PER_TOKEN = 4
    
    # Default token budget (leaving room for prompt + response)
    # qwen2.5:7b has 32K context, we use ~8K for context
    DEFAULT_TOKEN_BUDGET = 8000
    
    # Minimum similarity threshold for deduplication
    DEDUP_THRESHOLD = 0.85
    
    def __init__(
        self,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        dedup_threshold: float = DEDUP_THRESHOLD,
        prioritize_warnings: bool = True,
        prioritize_procedures: bool = True
    ):
        self.token_budget = token_budget
        self.dedup_threshold = dedup_threshold
        self.prioritize_warnings = prioritize_warnings
        self.prioritize_procedures = prioritize_procedures
        
        logger.info(f"ContextOptimizer initialized: budget={token_budget} tokens")
    
    def optimize(
        self,
        retrieved_docs: List[Dict],
        query: str,
        max_chunks: int = 10
    ) -> Tuple[List[OptimizedChunk], Dict]:
        """
        Optimize retrieved documents for context window
        
        Args:
            retrieved_docs: List of retrieved documents with metadata
            query: Original user query (for relevance scoring)
            max_chunks: Maximum number of chunks to include
            
        Returns:
            Tuple of (optimized chunks, optimization stats)
        """
        if not retrieved_docs:
            return [], {"status": "empty", "chunks_in": 0, "chunks_out": 0}
        
        stats = {
            "chunks_in": len(retrieved_docs),
            "chunks_out": 0,
            "tokens_used": 0,
            "duplicates_removed": 0,
            "truncated_chunks": 0,
            "warnings_prioritized": 0,
            "procedures_prioritized": 0
        }
        
        # Step 1: Convert to OptimizedChunk objects
        chunks = self._convert_to_optimized_chunks(retrieved_docs)
        
        # Step 2: Deduplicate similar content
        chunks, dedup_count = self._deduplicate_chunks(chunks)
        stats["duplicates_removed"] = dedup_count
        
        # Step 3: Score and sort chunks
        chunks = self._score_and_sort(chunks, query)
        
        # Count prioritized items
        stats["warnings_prioritized"] = sum(1 for c in chunks if c.is_warning)
        stats["procedures_prioritized"] = sum(1 for c in chunks if c.is_procedure)
        
        # Step 4: Apply token budget
        selected_chunks, tokens_used, truncated = self._apply_token_budget(
            chunks, 
            max_chunks
        )
        stats["tokens_used"] = tokens_used
        stats["truncated_chunks"] = truncated
        stats["chunks_out"] = len(selected_chunks)
        
        logger.info(
            f"Context optimized: {stats['chunks_in']}→{stats['chunks_out']} chunks, "
            f"{stats['tokens_used']} tokens, {stats['duplicates_removed']} duplicates removed"
        )
        
        return selected_chunks, stats
    
    def _convert_to_optimized_chunks(self, docs: List[Dict]) -> List[OptimizedChunk]:
        """Convert raw documents to OptimizedChunk objects"""
        chunks = []
        
        for doc in docs:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            similarity = doc.get("similarity", 0.0)
            
            # Handle similarity that might be a string
            if isinstance(similarity, str):
                try:
                    similarity = float(similarity)
                except:
                    similarity = 0.5
            
            chunk = OptimizedChunk(
                text=text,
                source=metadata.get("source", "Unknown"),
                similarity=similarity,
                importance_score=metadata.get("importance_score", 0.5),
                section_type=metadata.get("section_type", "general"),
                heading_text=metadata.get("heading_text", ""),
                is_procedure=metadata.get("is_procedure", False),
                is_warning=metadata.get("is_warning", False),
                token_estimate=self._estimate_tokens(text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        return max(1, len(text) // self.CHARS_PER_TOKEN)
    
    def _deduplicate_chunks(
        self, 
        chunks: List[OptimizedChunk]
    ) -> Tuple[List[OptimizedChunk], int]:
        """
        Remove duplicate or near-duplicate chunks
        Uses content hashing + Jaccard similarity
        """
        if len(chunks) <= 1:
            return chunks, 0
        
        # Track seen content
        seen_hashes = set()
        seen_texts = []
        unique_chunks = []
        duplicates = 0
        
        for chunk in chunks:
            # Quick hash check
            text_clean = self._normalize_text(chunk.text)
            text_hash = hashlib.md5(text_clean.encode()).hexdigest()[:16]
            
            if text_hash in seen_hashes:
                duplicates += 1
                continue
            
            # Jaccard similarity check against seen texts
            is_duplicate = False
            for seen_text in seen_texts:
                similarity = self._jaccard_similarity(text_clean, seen_text)
                if similarity > self.dedup_threshold:
                    is_duplicate = True
                    duplicates += 1
                    break
            
            if not is_duplicate:
                seen_hashes.add(text_hash)
                seen_texts.append(text_clean)
                unique_chunks.append(chunk)
        
        return unique_chunks, duplicates
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Lowercase, remove extra whitespace
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _score_and_sort(
        self, 
        chunks: List[OptimizedChunk],
        query: str
    ) -> List[OptimizedChunk]:
        """
        Score chunks based on multiple factors and sort
        
        Scoring factors:
        - Similarity score (from retrieval)
        - Importance score (from semantic chunking)
        - Is warning (critical safety info)
        - Is procedure (actionable steps)
        - Query term overlap
        """
        query_terms = set(self._normalize_text(query).split())
        
        def compute_score(chunk: OptimizedChunk) -> float:
            score = 0.0
            
            # Base: similarity score (0-1), weight: 0.4
            score += chunk.similarity * 0.4
            
            # Importance score (0-1), weight: 0.3
            score += chunk.importance_score * 0.3
            
            # Warning bonus (critical safety), weight: 0.15
            if self.prioritize_warnings and chunk.is_warning:
                score += 0.15
            
            # Procedure bonus (actionable), weight: 0.1
            if self.prioritize_procedures and chunk.is_procedure:
                score += 0.1
            
            # Query term overlap bonus, weight: 0.05
            chunk_terms = set(self._normalize_text(chunk.text).split())
            if query_terms and chunk_terms:
                overlap = len(query_terms & chunk_terms) / len(query_terms)
                score += overlap * 0.05
            
            return score
        
        # Sort by computed score (descending)
        chunks_with_scores = [(chunk, compute_score(chunk)) for chunk in chunks]
        chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, _ in chunks_with_scores]
    
    def _apply_token_budget(
        self,
        chunks: List[OptimizedChunk],
        max_chunks: int
    ) -> Tuple[List[OptimizedChunk], int, int]:
        """
        Select chunks within token budget
        
        Returns:
            Tuple of (selected chunks, total tokens, truncated count)
        """
        selected = []
        total_tokens = 0
        truncated = 0
        
        # Reserve tokens for formatting overhead (headers, separators)
        format_overhead_per_chunk = 20  # ~80 chars for headers
        
        for chunk in chunks:
            if len(selected) >= max_chunks:
                break
            
            chunk_tokens = chunk.token_estimate + format_overhead_per_chunk
            
            # Check if adding this chunk exceeds budget
            if total_tokens + chunk_tokens > self.token_budget:
                # Try truncating the chunk
                remaining_tokens = self.token_budget - total_tokens - format_overhead_per_chunk
                if remaining_tokens > 100:  # Minimum useful content
                    truncated_text = chunk.text[:remaining_tokens * self.CHARS_PER_TOKEN]
                    # Try to break at sentence boundary
                    last_period = truncated_text.rfind('.')
                    if last_period > len(truncated_text) * 0.5:
                        truncated_text = truncated_text[:last_period + 1]
                    
                    chunk.text = truncated_text + "..."
                    chunk.token_estimate = self._estimate_tokens(truncated_text)
                    selected.append(chunk)
                    total_tokens += chunk.token_estimate + format_overhead_per_chunk
                    truncated += 1
                break
            
            selected.append(chunk)
            total_tokens += chunk_tokens
        
        return selected, total_tokens, truncated
    
    def build_context_string(
        self,
        chunks: List[OptimizedChunk],
        include_metadata: bool = True,
        group_by_source: bool = False
    ) -> str:
        """
        Build formatted context string from optimized chunks
        
        Args:
            chunks: Optimized chunks to format
            include_metadata: Include source/section info
            group_by_source: Group chunks from same source
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        if group_by_source:
            # Group by source document
            groups = defaultdict(list)
            for chunk in chunks:
                groups[chunk.source].append(chunk)
            
            parts = []
            for source, source_chunks in groups.items():
                source_parts = [f"\n=== {source} ==="]
                for chunk in source_chunks:
                    source_parts.append(chunk.to_context_string(include_metadata=False))
                parts.append("\n".join(source_parts))
            
            return "\n\n".join(parts)
        else:
            # Simple sequential formatting
            return "\n\n---\n\n".join([
                chunk.to_context_string(include_metadata=include_metadata)
                for chunk in chunks
            ])


def optimize_context_for_rag(
    retrieved_docs: List[Dict],
    query: str,
    token_budget: int = 8000,
    max_chunks: int = 10
) -> Tuple[str, List[Dict], Dict]:
    """
    Convenience function to optimize context for RAG
    
    Args:
        retrieved_docs: Retrieved documents from vector search
        query: User query
        token_budget: Maximum tokens for context
        max_chunks: Maximum number of chunks
        
    Returns:
        Tuple of (context_string, source_list, stats)
    """
    optimizer = ContextOptimizer(token_budget=token_budget)
    
    optimized_chunks, stats = optimizer.optimize(
        retrieved_docs=retrieved_docs,
        query=query,
        max_chunks=max_chunks
    )
    
    context_string = optimizer.build_context_string(
        optimized_chunks,
        include_metadata=True,
        group_by_source=False
    )
    
    # Build source list for response
    sources = [
        {
            "source": chunk.source,
            "similarity": f"{chunk.similarity:.2f}",
            "section_type": chunk.section_type,
            "is_warning": chunk.is_warning,
            "is_procedure": chunk.is_procedure,
            "excerpt": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
        }
        for chunk in optimized_chunks
    ]
    
    return context_string, sources, stats
