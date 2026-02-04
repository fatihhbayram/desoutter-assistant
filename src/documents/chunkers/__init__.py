"""
Adaptive Chunking Strategies
============================
Document-type specific chunking for optimal retrieval.

Strategies:
- SemanticChunker: Section-based chunking for configuration guides
- TableAwareChunker: Preserves table structure for compatibility matrices
- EntityChunker: Groups error codes with descriptions
- ProblemSolutionChunker: Preserves problem-solution pairs in ESDE bulletins
- StepPreservingChunker: Keeps procedure steps together
- HybridChunker: Fallback for mixed content
"""

from .base_chunker import BaseChunker, Chunk
from .chunker_factory import ChunkerFactory
from .semantic_chunker import SemanticChunker
from .table_aware_chunker import TableAwareChunker
from .entity_chunker import EntityChunker
from .problem_solution_chunker import ProblemSolutionChunker
from .step_preserving_chunker import StepPreservingChunker
from .hybrid_chunker import HybridChunker

__all__ = [
    'BaseChunker',
    'Chunk',
    'ChunkerFactory',
    'SemanticChunker',
    'TableAwareChunker',
    'EntityChunker',
    'ProblemSolutionChunker',
    'StepPreservingChunker',
    'HybridChunker',
]
