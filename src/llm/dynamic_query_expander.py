"""
Dynamic Query Expander
Extracts expansion terms from Vector DB instead of hardcoded lists.

Phase 2 of RAG Performance Enhancement v2.0
"""

from typing import List, Set, Optional, Dict
from collections import Counter
import re


class DynamicQueryExpander:
    """
    Expands queries using terms extracted from vector database.
    More accurate than hardcoded synonyms because expansions
    come from actual indexed content.
    """
    
    # Stop words to exclude from expansion
    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
        'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
        'that', 'this', 'these', 'those', 'it', 'its', 'they', 'them',
        'their', 'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how',
        # Desoutter-specific common words to skip
        'tool', 'tools', 'desoutter', 'manual', 'section', 'page', 'see',
        'refer', 'check', 'ensure', 'make', 'sure', 'following', 'step',
    }
    
    # Minimum word length for expansion terms
    MIN_WORD_LENGTH = 3
    
    # Technical terms to always include if found
    PRIORITY_TERMS = {
        'esde', 'bulletin', 'error', 'fault', 'failure', 'issue',
        'disconnection', 'calibration', 'transducer', 'trigger',
        'mainboard', 'housing', 'firmware', 'esd', 'boost',
    }
    
    def __init__(self, chroma_client, embeddings_generator):
        """
        Initialize with ChromaDB client and embedding model.
        
        Args:
            chroma_client: ChromaDB collection client
            embeddings_generator: EmbeddingsGenerator to create query embeddings
        """
        self.chroma = chroma_client
        self.embedder = embeddings_generator
        
        # Cache for recent expansions (optimization)
        self._cache: Dict[str, str] = {}
        self._cache_max_size = 100
    
    def expand_query(
        self,
        query: str,
        product_filter: dict = None,
        n_expansion_docs: int = 10,
        max_expansion_terms: int = 8
    ) -> str:
        """
        Expand query using terms from similar documents in vector DB.
        
        Args:
            query: Original user query
            product_filter: Optional ChromaDB filter for product family
            n_expansion_docs: Number of docs to sample for expansion
            max_expansion_terms: Maximum terms to add
        
        Returns:
            Expanded query string
        """
        # Check cache first
        cache_key = f"{query}:{str(product_filter)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Step 1: Quick vector search for similar content
        similar_docs = self._get_similar_docs(
            query=query,
            n_results=n_expansion_docs,
            where_filter=product_filter
        )
        
        if not similar_docs:
            return query
        
        # Step 2: Extract frequent terms from results
        expansion_terms = self._extract_expansion_terms(
            docs=similar_docs,
            original_query=query,
            max_terms=max_expansion_terms
        )
        
        # Step 3: Build expanded query
        if expansion_terms:
            expanded = f"{query} {' '.join(expansion_terms)}"
        else:
            expanded = query
        
        # Cache result
        self._add_to_cache(cache_key, expanded)
        
        return expanded
    
    def expand_for_bulletins(
        self,
        query: str,
        product_filter: dict = None
    ) -> str:
        """
        Aggressive expansion specifically for bulletin search.
        Searches only bulletin documents for expansion terms.
        """
        # Filter to only bulletins
        bulletin_filter = {
            "$or": [
                {"source": {"$contains": "ESDE"}},
                {"doc_type": {"$eq": "service_bulletin"}}
            ]
        }
        
        # Combine with product filter if provided
        if product_filter:
            combined_filter = {"$and": [bulletin_filter, product_filter]}
        else:
            combined_filter = bulletin_filter
        
        return self.expand_query(
            query=query,
            product_filter=combined_filter,
            n_expansion_docs=15,  # More docs for bulletins
            max_expansion_terms=10  # More terms allowed
        )
    
    def _get_similar_docs(
        self,
        query: str,
        n_results: int,
        where_filter: dict = None
    ) -> List[dict]:
        """
        Get similar documents from vector DB.
        Only fetches metadata and small text snippets for efficiency.
        """
        try:
            # Generate query embedding
            query_embedding = self.embedder.generate_embedding(query)
            
            # Query ChromaDB
            results = self.chroma.query(
                query_text=query,
                query_embedding=query_embedding,
                n_results=n_results,
                where=where_filter
            )
            
            docs = []
            if results:
                documents = results.get('documents', [[]])[0]
                metadatas = results.get('metadatas', [[]])[0]
                
                for doc, meta in zip(documents, metadatas):
                    docs.append({
                        'text': doc,
                        'metadata': meta or {}
                    })
            
            return docs
            
        except Exception as e:
            # Log but don't fail - fall back to original query
            print(f"Error in vector search for expansion: {e}")
            return []
    
    def _extract_expansion_terms(
        self,
        docs: List[dict],
        original_query: str,
        max_terms: int
    ) -> List[str]:
        """
        Extract most relevant expansion terms from documents.
        
        Strategy:
        1. Tokenize all document text
        2. Count term frequencies
        3. Filter out stop words and query terms
        4. Prioritize technical terms
        5. Return top N terms
        """
        # Tokenize original query
        query_terms = set(self._tokenize(original_query))
        
        # Count terms across all docs
        term_counter = Counter()
        
        for doc in docs:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            # Extract terms from text
            terms = self._tokenize(text)
            term_counter.update(terms)
            
            # Also check metadata fields
            for field in ['symptoms', 'error_codes', 'keywords', 'fault_keywords']:
                if field in metadata and metadata[field]:
                    meta_terms = self._tokenize(str(metadata[field]))
                    # Weight metadata terms higher
                    for t in meta_terms:
                        term_counter[t] += 2
        
        # Filter and score terms
        expansion_candidates = []
        
        for term, count in term_counter.items():
            # Skip if in original query
            if term in query_terms:
                continue
            
            # Skip stop words
            if term in self.STOP_WORDS:
                continue
            
            # Skip short words
            if len(term) < self.MIN_WORD_LENGTH:
                continue
            
            # Calculate score
            score = count
            
            # Boost priority technical terms
            if term in self.PRIORITY_TERMS:
                score *= 3
            
            # Boost terms that look like error codes
            if re.match(r'^[ei]\d{2,4}$', term):
                score *= 5
            
            expansion_candidates.append((term, score))
        
        # Sort by score and take top N
        expansion_candidates.sort(key=lambda x: -x[1])
        top_terms = [term for term, score in expansion_candidates[:max_terms]]
        
        return top_terms
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase, split on non-alphanumeric.
        """
        text_lower = text.lower()
        # Split on non-alphanumeric, keep hyphenated terms
        tokens = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', text_lower)
        return tokens
    
    def _add_to_cache(self, key: str, value: str):
        """Add to cache with size limit."""
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value


class HybridQueryExpander:
    """
    Combines dynamic vector-based expansion with fallback rules.
    Uses vector DB when available, falls back to rules if needed.
    """
    
    # Fallback rules for common patterns (used when vector search fails)
    FALLBACK_RULES = {
        'wifi': ['wireless', 'connection', 'signal'],
        'drops': ['disconnection', 'intermittent', 'lost'],
        'error': ['fault', 'failure', 'issue'],
        'not starting': ['no boot', 'dead', 'wont start'],
        'not detected': ['not found', 'not recognized', 'offline'],
        'weak signal': ['poor performance', 'signal strength', 'low signal'],
        'communication': ['connection', 'link', 'comm'],
        'no power': ['not powering', 'dead', 'no boot'],
    }
    
    def __init__(self, chroma_client=None, embeddings_generator=None):
        """
        Initialize hybrid expander.
        
        Args:
            chroma_client: Optional ChromaDB client (for dynamic expansion)
            embeddings_generator: Optional EmbeddingsGenerator
        """
        self.dynamic_expander = None
        
        if chroma_client and embeddings_generator:
            self.dynamic_expander = DynamicQueryExpander(
                chroma_client=chroma_client,
                embeddings_generator=embeddings_generator
            )
    
    def expand_query(
        self,
        query: str,
        product_filter: dict = None,
        use_dynamic: bool = True
    ) -> str:
        """
        Expand query using best available method.
        
        Priority:
        1. Dynamic vector-based expansion (if available)
        2. Fallback to rule-based expansion
        """
        # Try dynamic expansion first
        if use_dynamic and self.dynamic_expander:
            try:
                expanded = self.dynamic_expander.expand_query(
                    query=query,
                    product_filter=product_filter
                )
                
                # If dynamic expansion added terms, use it
                if len(expanded) > len(query):
                    return expanded
                    
            except Exception as e:
                print(f"Dynamic expansion failed, using fallback: {e}")
        
        # Fallback to rules
        return self._rule_based_expansion(query)
    
    def expand_for_bulletins(
        self,
        query: str,
        product_filter: dict = None
    ) -> str:
        """
        Aggressive expansion for bulletin search.
        """
        if self.dynamic_expander:
            try:
                return self.dynamic_expander.expand_for_bulletins(
                    query=query,
                    product_filter=product_filter
                )
            except Exception as e:
                print(f"Bulletin expansion failed, using fallback: {e}")
        
        # Fallback with bulletin-focused additions
        expanded = self._rule_based_expansion(query)
        return f"{expanded} issue problem failure fault bulletin"
    
    def _rule_based_expansion(self, query: str) -> str:
        """Simple rule-based expansion as fallback."""
        query_lower = query.lower()
        expansions = set()
        
        for trigger, terms in self.FALLBACK_RULES.items():
            if trigger in query_lower:
                expansions.update(terms)
        
        # Remove terms already in query
        expansions = {e for e in expansions if e not in query_lower}
        
        if expansions:
            return f"{query} {' '.join(list(expansions)[:5])}"
        
        return query


def get_query_expander(chroma_client=None, embeddings_generator=None) -> HybridQueryExpander:
    """
    Factory function to create a query expander.
    
    Returns HybridQueryExpander configured with available dependencies.
    """
    return HybridQueryExpander(
        chroma_client=chroma_client,
        embeddings_generator=embeddings_generator
    )
