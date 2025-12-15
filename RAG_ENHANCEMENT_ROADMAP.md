# 🚀 RAG Enhancement Roadmap - Semantic Chunking, Domain Embeddings & Performance Optimization

> **Created:** 15 December 2025  
> **Updated:** 15 December 2025 - **PHASE 1 COMPLETE ✅**
> **Purpose:** Comprehensive enhancement of RAG system with semantic chunking, domain-specific embeddings, and performance monitoring  
> **Target:** Production-ready, high-accuracy repair diagnosis system

---

## 📊 Current State Analysis

### ✅ Existing Implementation
- **RAG Engine**: ChromaDB + Ollama qwen2.5:7b-instruct
- **Embeddings**: HuggingFace all-MiniLM-L6-v2 (384-dim)
- **Chunking**: ~~Basic sentence-based (500 tokens, 50 overlap)~~ → **Semantic chunking LIVE ✅**
- **Retrieval**: Top-K (5 results) with dynamic similarity threshold (0.30)
- **Feedback System**: User feedback collection + learned mappings
- **Dashboard**: Basic analytics (total diagnoses, confidence breakdown, top products)

### ⚠️ Previous Limitations (NOW RESOLVED)
1. ~~Chunking: Naive sentence splitting → loses semantic boundaries~~ → **FIXED: Recursive chunking** ✅
2. **Embeddings**: Generic model → misses domain-specific terminology (pending Phase 2)
3. ~~Retrieval: Simple L2 distance + fixed top-K~~ → **OPTIMIZED: Dynamic threshold 0.30** ✅
4. **Metadata**: ~~Basic~~ → **Rich 14-field metadata** ✅
5. **Caching**: No query/embedding caching → slow repeated searches (Phase 2)
6. **Ingestion**: Synchronous, blocking → UI hangs during bulk uploads (Phase 2)
7. **Performance**: No metrics on latency, accuracy, hit rate (Phase 2)
8. **Feedback Loop**: Basic counting → no learning signal propagation to embeddings (Phase 2)

---

## ✅ 🎯 Phase 1: Semantic Chunking - COMPLETE (15 December 2025)

### 1.1 Implement Recursive Character Chunking ✅
**Status:** COMPLETE  
**File:** `src/documents/semantic_chunker.py` (420+ lines)

**Implementation Details:**
- Recursive character-level chunking (paragraph → sentence → word → character)
- Preserves document structure (headings, procedures, warnings, tables, lists)
- Configuration: chunk_size=400, chunk_overlap=100, max_recursion_depth=3
- Minimum chunk size: 50 characters (prevents empty chunks)

**Key Features:**
- `_split_by_paragraphs()`: Preserve paragraph boundaries
- `_is_heading()` / `_get_heading_level()`: Structure detection (markdown, all-caps, numbered)
- `_chunk_paragraph()`: Size-aware chunking with overlap
- `_split_by_sentences()`: Intelligent segmentation
- `_detect_section_type()`: Content classification (8 types)

**Testing:** ✅ Test 1: Basic Semantic Chunking - PASS

---

### 1.2 Add Document Structure Recognition ✅
**Status:** COMPLETE  
**Class:** `DocumentTypeDetector`

**Implementation:**
```python
class DocumentTypeDetector:
    """Auto-detect document type and apply specialized strategies"""
    
    TYPES = {
        TECHNICAL_MANUAL: "procedure, operation, maintenance, specifications",
        SERVICE_BULLETIN: "bulletin, known issue, update, revision",
        TROUBLESHOOTING_GUIDE: "symptom, problem, solution, diagnosis",
        PARTS_CATALOG: "parts, catalog, component, assembly, specification",
        SAFETY_DOCUMENT: "warning, danger, caution, safety, hazard"
    }
    
    def detect_type(self, text: str) -> DocumentType:
        # Keyword-based detection with scoring
        # Returns most probable type
```

**Document Types:** 5 classifications
- TECHNICAL_MANUAL: Complex structure, procedures
- SERVICE_BULLETIN: Short updates, known issues
- TROUBLESHOOTING_GUIDE: Symptom→solution mappings
- PARTS_CATALOG: Structured parts data
- SAFETY_DOCUMENT: Critical safety content (highest importance)

**Testing:** ✅ Test 2: Document Type Detection - PASS

---

### 1.3 Extract and Classify Content ✅
**Status:** COMPLETE  
**Class:** `FaultKeywordExtractor`

**Domain-Specific Keywords:** 9 categories
```python
CATEGORIES = {
    "motor": [motor, spindle, rotation, rpm, speed, stall, bearing, brush],
    "noise": [noise, grinding, squealing, clicking, humming, vibration],
    "mechanical": [jamming, stuck, resistance, gearbox, wear],
    "electrical": [voltage, current, circuit, short, grounding],
    "calibration": [calibration, adjust, alignment, tolerance, precision],
    "leakage": [leak, seal, gasket, drip, moisture, oil],
    "corrosion": [corrosion, rust, oxidation, tarnish],
    "wear": [wear, worn, erosion, crack, failure],
    "connection": [connection, loose, cable, disconnect],
    "torque": [torque, nm, tightening, wrench, tension]
}
```

**Testing:** ✅ Test 3: Fault Keyword Extraction - PASS

---

### 1.4 Rich Metadata Per Chunk ✅
**Status:** COMPLETE  
**Class:** `ChunkMetadata` dataclass

**14 Metadata Fields:**
1. `source` - Original filename
2. `chunk_index` - Sequential number
3. `document_type` - Type classification
4. `section_type` - Content type (8 options)
5. `heading_level` - 0-6 for hierarchy
6. `heading_text` - Parent heading context
7. `fault_keywords` - Extracted repair keywords
8. `is_procedure` - Step-by-step detection
9. `is_warning` - Safety warning flag
10. `contains_table` - Tabular data flag
11. `importance_score` - 0.0-1.0 scoring
12. `position_ratio` - Relative doc position
13. `page_number` - Optional PDF page
14. `word_count` - Chunk word count

**Importance Scoring Logic:**
- Base: 0.5
- +0.3 for warnings
- +0.2 for procedures
- +0.1 for headings (scaled by level)
- +0.15 for safety documents
- Range: 0.0-1.0

**Testing:** ✅ Test 4: DocumentProcessor Integration - PASS (metadata extraction verified)

---

### 1.5 DocumentProcessor Integration ✅
**Status:** COMPLETE  
**File:** `src/documents/document_processor.py` (UPDATED)

**Changes:**
- SemanticChunker initialized in `__init__()`
- `process_document()` now supports `enable_semantic_chunking` parameter
- `process_directory()` processes all docs with semantic chunking
- Output includes: text, metadata, chunks, chunk_count
- Supports: PDF, DOCX, PPTX, XLSX, XLS

**Integration Method:**
```python
# Detect document type
doc_type = self._map_to_document_type(metadata["type"])

# Apply semantic chunking
chunks = self.semantic_chunker.chunk_document(
    text=text,
    source_filename=file_path.name,
    doc_type=doc_type
)

return {
    "filename": file_path.name,
    "text": text,
    "metadata": metadata,
    "chunks": chunks,  # NEW: Rich semantic chunks
    "chunk_count": len(chunks)
}
```

---

### 1.6 Configuration for Phase 2 ✅
**Status:** COMPLETE  
**File:** `config/ai_settings.py` (UPDATED)

**New Settings:**
```python
# Semantic Chunking
CHUNK_SIZE = 400              # characters
CHUNK_OVERLAP = 100           # characters
MAX_RECURSION_DEPTH = 3       # levels

# Domain Embeddings (Phase 2)
DOMAIN_EMBEDDING_MODEL_PATH = None  # Will be set after fine-tuning
USE_DOMAIN_EMBEDDINGS = False
DOMAIN_EMBEDDING_TRAINING_ENABLED = False
EMBEDDING_DIMENSION = 384
EMBEDDING_POOLING = "mean"
```

---

### 1.7 Comprehensive Test Suite ✅
**Status:** COMPLETE  
**File:** `scripts/test_semantic_chunking.py` (NEW, 355 lines)

**Tests Implemented:**
1. ✅ **Test 1: Basic Semantic Chunking** - PASS
   - Chunks sample manual text
   - Verifies chunk count and metadata
   - Shows importance scoring

2. ✅ **Test 2: Document Type Detection** - PASS
   - Tests 5 document type classifications
   - Detects Service Bulletin, Manual, Troubleshooting, Catalog, Safety

3. ✅ **Test 3: Fault Keyword Extraction** - PASS
   - Tests 9 domain categories
   - Extracts motor, noise, mechanical, electrical, etc.

4. ✅ **Test 4: DocumentProcessor Integration** - PASS
   - End-to-end document processing
   - Chunk generation with metadata
   - Section type distribution
   - Importance score statistics

**Overall Result: 4/4 TESTS PASSED ✅**

---

### Phase 1 Summary

| Component | Status | Lines | Tests |
|-----------|--------|-------|-------|
| SemanticChunker | ✅ Complete | 420+ | 4/4 PASS |
| DocumentTypeDetector | ✅ Complete | 50+ | 1/1 PASS |
| FaultKeywordExtractor | ✅ Complete | 60+ | 1/1 PASS |
| ChunkMetadata | ✅ Complete | 30+ | - |
| DocumentProcessor Integration | ✅ Complete | 50+ | 1/1 PASS |
| Config Updates | ✅ Complete | 20+ | - |
| Test Suite | ✅ Complete | 355 | 4/4 PASS |
| Documentation | ✅ Complete | - | - |

**Deliverables:**
- ✅ SemanticChunker module (recursive chunking)
- ✅ Document type detection (5 types)
- ✅ Metadata extraction (14 fields)
- ✅ Fault keyword extraction (9 categories)
- ✅ Integration with DocumentProcessor
- ✅ Comprehensive test suite (100% passing)
- ✅ Configuration for Phase 2

**Ready for Phase 2:**
- ✅ Semantic chunking pipeline implemented
- ✅ Document structure preservation working
- ✅ Rich metadata available for filtering
- ⏳ Next: Re-ingest 276 documents with semantic chunks
- ⏳ Next: Domain embeddings fine-tuning
- ⏳ Next: ChromaDB refresh with improved retrieval

---

## 🧠 Phase 2: Domain-Specific Embeddings & Retrieval Enhancement (Week 3-4)

### 2.1 Prepare Domain Training Data ⏳
**Goal:** Collect positive feedback pairs for embeddings fine-tuning

**Data Sources:**
- User feedback from successful RAG interactions
- Query-document pairs with high user ratings
- Manual expert-labeled semantic similarities
- Failure pairs from negative feedback

**Target:** 100-200 positive pairs (query → correct document chunk)

### 2.2 Fine-Tune Embeddings on Desoutter Corpus ⏳
**Goal:** Create embeddings optimized for repair domain

**Approach:** Contrastive learning with domain-specific pairs

```python
class DomainEmbeddingTrainer:
    """Fine-tune embedding model on repair domain"""
    
    def __init__(self):
        self.base_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.device = "cuda"
    
    def fine_tune(self, training_pairs: List[Dict]):
        """
        Training pair format:
        {
            "query": "Motor grinding noise and overheating",
            "positive": "Bearing wear causes grinding and temperature rise",
            "similarity": 1.0  # Perfect match
        }
        """
        # Load model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Create training data
        train_examples = [
            InputExample(
                texts=[p["query"], p["positive"]],
                label=p["similarity"]
            )
            for p in training_pairs
        ]
        
        # Fine-tune with contrastive loss
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=32
        )
        
        train_loss = losses.CosineSimilarityLoss(
            model=model,
            sentence_embedding_dimension=384
        )
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=3,
            warmup_steps=100,
            show_progress_bar=True
        )
        
        model.save("/data/embeddings/desoutter-domain-model")
        return model
```

**Configuration:**
```python
DOMAIN_EMBEDDING_MODEL_PATH = "/data/embeddings/desoutter-domain-model"
USE_DOMAIN_EMBEDDINGS = True  # Switch to fine-tuned model
EMBEDDING_DIMENSION = 384     # Maintain compatibility
EMBEDDING_POOLING = "mean"    # Token aggregation method
```

**Expected Impact:**
- Domain terminology better understood
- 15-25% improvement in semantic relevance
- Reduced false positives on generic terms

### 2.3 Document Re-ingestion with Semantic Chunks ⏳
**Goal:** Re-ingest 276 documents with semantic chunking enabled

**Process:**
```bash
# 1. Backup existing data
cp -r /data/chromadb /data/chromadb-backup-phase1

# 2. Clear ChromaDB
python scripts/clear_chromadb.py

# 3. Re-ingest with semantic chunking + domain embeddings
python scripts/ingest_documents.py \
    --input-dir documents/ \
    --enable-semantic-chunking \
    --use-domain-embeddings \
    --embedding-model /data/embeddings/desoutter-domain-model \
    --batch-size 16
```

**Expected Results:**
- Input: 276 documents
- Output: ~3000-5000 semantic chunks (3-5x increase from 1080 basic chunks)
- Metadata: 14 fields per chunk
- ChromaDB: Updated collection with domain embeddings
- Processing time: ~15-20 minutes (GPU-accelerated)

**Quality Checks:**
- Verify chunk count and distribution
- Sample random chunks for semantic validity
- Check metadata field population
- Validate document type detection accuracy

### 2.4 Hybrid Search: Semantic + BM25 ⏳
**Goal:** Combine dense and sparse retrieval for better coverage

```python
class HybridSearcher:
    """Combine semantic search + BM25 keyword matching"""
    
    def __init__(self):
        self.chromadb = ChromaDB(collection="documents")
        self.bm25_index = BM25Retriever(
            documents=self._load_documents(),
            k1=1.5,
            b=0.75
        )
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        1. Get semantic results (dense retrieval)
        2. Get keyword results (sparse retrieval)
        3. Combine using Reciprocal Rank Fusion
        """
        # Semantic search
        semantic_results = self.chromadb.query(
            query_embeddings=self.embed_query(query),
            n_results=top_k * 2,
            where={"importance_score": {"$gte": 0.5}}
        )
        
        # BM25 search
        bm25_results = self.bm25_index.retrieve(query, k=top_k * 2)
        
        # Reciprocal Rank Fusion
        hybrid_results = self._reciprocal_rank_fusion(
            semantic_results,
            bm25_results,
            top_k=top_k
        )
        
        return hybrid_results
    
    def _reciprocal_rank_fusion(self, results1, results2, top_k):
        """Combine rankings: RRF = 1/(60 + rank)"""
        scores = {}
        
        for rank, result in enumerate(results1, 1):
            doc_id = result["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1/(60 + rank)
        
        for rank, result in enumerate(results2, 1):
            doc_id = result["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1/(60 + rank)
        
        # Return top-k by combined score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
```

**Benefits:**
- Better coverage: catches exact matches and semantic similarities
- Reduced false negatives from purely semantic search
- Improved precision for domain-specific terminology

### 2.5 Query Expansion Pipeline ⏳
**Goal:** Automatically expand queries with related domain terms

```python
class QueryExpander:
    """Expand queries with domain-aware synonyms"""
    
    DOMAIN_SYNONYMS = {
        "noise": ["squealing", "grinding", "humming", "clicking", "vibration"],
        "heat": ["overheating", "temperature", "thermal"],
        "motor": ["spindle", "drive", "actuator", "rotation"],
        "bearing": ["ball bearing", "roller bearing", "bushing"],
        "leak": ["leakage", "seepage", "drip", "moisture"]
    }
    
    def expand(self, query: str) -> List[str]:
        """Generate expanded query variations"""
        
        expanded = [query]  # Original query
        
        # 1. Expand with domain synonyms
        for term, synonyms in DOMAIN_SYNONYMS.items():
            if term in query.lower():
                for synonym in synonyms[:2]:  # Top 2 synonyms
                    expanded.append(query.replace(term, synonym, flags=re.IGNORECASE))
        
        # 2. LLM-based expansion for low-confidence queries
        confidence = self._compute_confidence(query)
        if confidence < 0.7:
            llm_expansions = self.llm.expand_query(
                query,
                context="You are a repair technician expert. Expand this query:",
                max_tokens=100
            )
            expanded.extend(llm_expansions)
        
        # Return unique, de-duplicated expansions
        return list(dict.fromkeys(expanded))[:5]  # Max 5 variations
    
    def multi_search(self, query: str, searcher, top_k: int = 5):
        """Search with all query expansions"""
        expanded_queries = self.expand(query)
        all_results = {}
        
        for q in expanded_queries:
            results = searcher.search(q, top_k=top_k)
            for result in results:
                doc_id = result["id"]
                if doc_id not in all_results:
                    all_results[doc_id] = result
                else:
                    all_results[doc_id]["score"] += result["score"] * 0.8  # Decay
        
        # Return re-ranked combined results
        ranked = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]
```

**Query Expansion Examples:**
- "Motor noise" → ["Motor noise", "Spindle squealing", "Actuator grinding"]
- "Battery not charging" → ["Battery not charging", "Power supply failure"]
- "Loose connection" → ["Loose connection", "Cable disconnect", "Poor contact"]

### 2.6 Metadata-Based Filtering ⏳
**Goal:** Use rich metadata to improve retrieval precision

```python
# Advanced filtering options
filters = {
    "document_type": {"$in": ["troubleshooting_guide", "service_bulletin"]},
    "importance_score": {"$gte": 0.7},
    "is_procedure": True,
    "contains_warning": {"$in": [True]},
    "section_type": {"$in": ["PROCEDURE", "WARNING", "TROUBLESHOOTING"]}
}

results = chromadb.query(
    query_embeddings=embedding,
    n_results=10,
    where=filters
)
```

**Filter Dimensions:**
- By document type (5 options)
- By importance score (0.0-1.0)
- By section type (8 options)
- By fault keywords (9 categories)
- By document source
- By heading level
- By presence of procedures/warnings/tables

---

## 🎯 Phase 3: Performance Monitoring & Optimization (Week 5-6)
    def __init__(self):
        self.dense_retriever = ChromaDBClient()
        self.sparse_retriever = BM25Retriever.from_documents(documents)
    
    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.5):
        """
        Combine:
        - Dense: similarity score (0-1)
        - Sparse: BM25 score (0-infinity) → normalize
        """
        dense_results = self.dense_retriever.query(query, n_results=top_k*2)
        sparse_results = self.sparse_retriever.get_relevant_documents(query)
        
        # Normalize and combine with weight α
        combined = {}
        for result in dense_results:
            combined[result['id']] = α * result['score']
        for result in sparse_results:
            combined[result['id']] += (1-α) * result['bm25_score']
        
        # Sort and return top-k
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

**Configuration:**
```python
HYBRID_ALPHA = 0.7  # 70% semantic, 30% keyword
SPARSE_TOP_K = 10   # Retrieve more candidates from BM25
DENSE_TOP_K = 10
```

### 2.3 Implement Query Expansion
**Goal:** Expand user queries with related terms

```python
class QueryExpander:
    """
    Expand "grinding noise" →
    ["grinding sound", "grinding noise", "mechanical noise", "bearing wear indicator"]
    """
    
    def expand(self, query: str) -> List[str]:
        # 1. Get synonyms from domain lexicon
        # 2. Use LLM to generate related terms
        # 3. Keep original query
        
        synonyms = self._get_domain_synonyms(query)
        expansions = self._llm_expand(query)
        
        return [query] + synonyms + expansions[:3]  # Keep top 3
```

**Expected Impact:**
- ✅ Domain accuracy improvement (+25% F1 score)
- ✅ Better handling of technical terminology
- ✅ Hybrid search catches edge cases (+10% recall)

---

## 🔍 Phase 3: Advanced ChromaDB Retrieval (Week 3-4)

### 3.1 Metadata Filtering & ANN Search
**Goal:** Use metadata to filter irrelevant chunks + optimize vector search

```python
class AdvancedChromaRetriever:
    """Enhanced ChromaDB with metadata filtering"""
    
    def query(
        self,
        query_embedding: List[float],
        part_number: Optional[str] = None,
        doc_types: Optional[List[str]] = None,
        importance_min: float = 0.5,
        top_k: int = 5
    ) -> Dict:
        """
        Build WHERE filter from metadata
        """
        where_filter = None
        
        if any([part_number, doc_types, importance_min]):
            where_conditions = []
            
            if part_number:
                where_conditions.append({
                    "product_category": {"$contains": part_number}
                })
            
            if doc_types:
                where_conditions.append({
                    "doc_type": {"$in": doc_types}
                })
            
            if importance_min:
                where_conditions.append({
                    "importance_score": {"$gte": importance_min}
                })
            
            where_filter = {"$and": where_conditions} if len(where_conditions) > 1 else where_conditions[0]
        
        # Query with filter
        results = self.collection.query(
            query_embeddings=[query_embedding],
            where=where_filter,
            n_results=top_k,
            include=["documents", "metadatas", "distances", "embeddings"]
        )
        
        return results
```

### 3.2 Top-K Result Merging & Ranking
**Goal:** Combine multiple retrieval results intelligently

```python
class ResultMerger:
    """
    Merge results from multiple sources:
    1. Semantic search (ChromaDB)
    2. Keyword search (BM25)
    3. Learned mappings (feedback-boosted)
    """
    
    def merge_and_rank(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict],
        learned_boosts: Dict[str, float]
    ) -> List[Dict]:
        """
        Ranking criteria:
        1. Distance/similarity score (base)
        2. Metadata importance score
        3. Learned boost from positive feedback
        4. Document freshness (newer = better)
        """
        
        merged = {}
        
        # Semantic results (60% weight)
        for i, result in enumerate(semantic_results):
            doc_id = result['id']
            score = (1 - result['distance']) * 0.6
            score *= result.get('importance_score', 1.0)
            score *= learned_boosts.get(doc_id, 1.0)
            merged[doc_id] = score
        
        # Keyword results (40% weight)
        for i, result in enumerate(keyword_results):
            doc_id = result['id']
            # Normalize BM25 score
            bm25_norm = result['bm25_score'] / 50  # Cap at 50
            score = min(bm25_norm, 1.0) * 0.4
            merged[doc_id] = merged.get(doc_id, 0) + score
        
        # Sort by combined score
        ranked = sorted(
            merged.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [self.get_document(doc_id) for doc_id, score in ranked[:5]]
```

**Configuration:**
```python
METADATA_FILTERS_ENABLED = true
ANN_DISTANCE_METRIC = "cosine"  # 'l2', 'cosine', 'ip'
RESULT_MERGER_WEIGHTS = {
    "semantic": 0.6,
    "keyword": 0.4,
    "learned_boost": 1.0,
    "importance": 1.0,
    "freshness": 0.1
}
```

**Expected Impact:**
- ✅ 99% faster retrieval (ANN + filtering)
- ✅ Better precision (+15% exact matches)
- ✅ Learned feedback amplifies good sources

---

## 💬 Phase 4: Prompt Optimization & Top-K Merging (Week 4-5)

### 4.1 Context-Aware Prompt Engineering
**Goal:** Generate better prompts based on retrieved context quality

```python
class PromptOptimizer:
    """
    Adapt prompt based on:
    1. Number of documents retrieved
    2. Confidence score of retrieval
    3. Document diversity
    """
    
    def build_rag_prompt(
        self,
        fault_description: str,
        retrieved_docs: List[Dict],
        product_info: Dict,
        language: str = "en"
    ) -> str:
        """
        Adaptive prompt strategy:
        - High confidence (>5 docs): Use multi-source approach
        - Medium confidence (2-5): Synthesize carefully
        - Low confidence (<2): Indicate uncertainty
        """
        
        doc_count = len(retrieved_docs)
        avg_relevance = np.mean([d['similarity'] for d in retrieved_docs])
        
        if doc_count >= 5 and avg_relevance > 0.8:
            # Rich context - can be more detailed
            prompt = f"""Based on multiple technical documents, provide a comprehensive repair suggestion.

Product: {product_info['model_name']}
Fault: {fault_description}

Related documentation:
{self._format_docs(retrieved_docs)}

Provide:
1. Root cause analysis
2. Step-by-step repair procedure
3. Required tools/parts
4. Safety precautions
5. Alternative solutions if primary fails"""
        
        elif doc_count >= 2:
            # Moderate context
            prompt = f"""Based on available documentation:

Fault: {fault_description}

Sources:
{self._format_docs(retrieved_docs)}

Suggest the most likely repair approach."""
        
        else:
            # Low confidence - admit uncertainty
            prompt = f"""Limited documentation available for this issue.

Fault: {fault_description}

Known information:
{self._format_docs(retrieved_docs)}

Provide best-effort suggestion with confidence level."""
        
        return prompt
```

### 4.2 Few-Shot Prompt Engineering
**Goal:** Use successful diagnosis examples to guide LLM

```python
class FewShotPromptBuilder:
    """Add examples of good diagnosis outputs"""
    
    def build_prompt_with_examples(self, query: str, top_k: int = 3):
        # Get successful examples from diagnosis_feedback collection
        good_examples = self.db.diagnosis_feedback.find(
            {"feedback_type": "positive", "is_correct": True}
        ).limit(top_k)
        
        prompt = "You are a Desoutter repair expert. Here are examples of good diagnoses:\n\n"
        
        for example in good_examples:
            prompt += f"""
Example {i}:
Fault: {example['fault_description']}
Expert Response: {example['suggestion']}
Confidence: {example['confidence']}
---
"""
        
        prompt += f"\nNow diagnose:\nFault: {query}"
        return prompt
```

### 4.3 Output Quality Scoring
**Goal:** Score generated responses for quality & confidence

```python
class ResponseScorer:
    """Evaluate response quality automatically"""
    
    def score_response(
        self,
        fault: str,
        response: str,
        retrieved_docs: List[Dict]
    ) -> Dict:
        """
        Score based on:
        1. Coverage: mentions key terms from fault
        2. Structure: has step-by-step, warnings, etc.
        3. Grounding: references sources
        4. Confidence: no hedging/uncertainty phrases
        """
        
        scores = {
            "coverage": self._score_coverage(fault, response),      # 0-1
            "structure": self._score_structure(response),           # 0-1
            "grounding": self._score_grounding(response, retrieved_docs), # 0-1
            "confidence": self._score_confidence(response)          # 0-1
        }
        
        overall = np.mean(list(scores.values()))
        
        return {
            **scores,
            "overall_quality": overall,
            "confidence_level": "high" if overall > 0.8 else "medium" if overall > 0.5 else "low"
        }
```

**Configuration:**
```python
PROMPT_OPTIMIZATION = {
    "use_adaptive_context": true,
    "few_shot_examples": 3,
    "include_warnings": true,
    "quality_threshold": 0.7  # Min score to return
}
```

**Expected Impact:**
- ✅ Better structured responses (+25% user satisfaction)
- ✅ More confident recommendations when data supports
- ✅ Fewer hallucinations (quality check before return)

---

## 📚 Phase 5: Self-Learning Feedback Loop Enhancement (Week 5-6)

### 5.1 Feedback Signal Propagation
**Goal:** Use feedback to improve future diagnoses

```python
class FeedbackPropagation:
    """Learn from feedback and improve embeddings/rankings"""
    
    def process_feedback(
        self,
        diagnosis_id: str,
        feedback_type: str,  # "positive" | "negative"
        actual_solution: Optional[str] = None
    ):
        """
        Positive feedback:
        1. Boost similarity score of used documents
        2. Increase importance score of helpful chunks
        3. Add to learned_mappings if pattern recognized
        
        Negative feedback:
        1. Demote source documents
        2. Extract why it failed (optional)
        3. Flag for manual review if critical
        """
        
        if feedback_type == "positive":
            # Boost documents that helped
            for doc_id in diagnosis.source_doc_ids:
                self._increase_importance(doc_id, boost=0.1)
                self._increase_use_count(doc_id)
            
            # Learn fault→solution mapping
            self._learn_mapping(
                fault=diagnosis.fault_description,
                solution=diagnosis.suggestion,
                confidence=0.9
            )
        
        elif feedback_type == "negative":
            # Demote documents
            for doc_id in diagnosis.source_doc_ids:
                self._decrease_importance(doc_id, penalty=0.2)
            
            # If user provided correct solution, learn it
            if actual_solution:
                self._learn_mapping(
                    fault=diagnosis.fault_description,
                    solution=actual_solution,
                    confidence=0.95  # Higher confidence (user-validated)
                )
                
                # Flag for dataset improvement
                self._flag_for_review(diagnosis_id, reason="negative_feedback_with_solution")
```

### 5.2 Learned Mapping Ranking
**Goal:** Boost sources that align with learned fault-solution pairs

```python
class LearnedMappingBooster:
    """Use feedback to learn and boost relevant sources"""
    
    def get_learned_boost(self, fault: str, retrieved_docs: List[Dict]) -> Dict[str, float]:
        """
        Find learned mappings similar to this fault
        Return boost factor for each document
        """
        
        # Get learned fault-solution pairs
        learned_pairs = self.db.learned_mappings.find(
            {"confidence": {"$gte": 0.7}},
            sort=[("confidence", -1)]
        )
        
        boosts = {}
        for pair in learned_pairs:
            # Calculate similarity of fault to learned fault
            fault_similarity = self._similarity(fault, pair['fault_keywords'])
            
            if fault_similarity > 0.7:
                # Boost documents mentioned in solution
                for doc_id in pair.get('source_doc_ids', []):
                    boost = fault_similarity * pair['confidence']
                    boosts[doc_id] = boost
        
        return boosts
```

### 5.3 Continuous Learning Loop
**Goal:** Periodically re-train embeddings on accumulated positive feedback

```python
class ContinuousFeedbackLearner:
    """Periodically retrain embeddings on positive examples"""
    
    def should_retrain(self) -> bool:
        """Retrain if:
        - >100 new positive feedback received
        - >2 weeks since last retrain
        - Accuracy declined >5%
        """
        
        positive_count = self.db.diagnosis_feedback.count_documents(
            {"feedback_type": "positive", "used_for_training": False}
        )
        
        last_retrain = self._get_last_retrain_date()
        days_since = (datetime.now() - last_retrain).days
        
        return positive_count > 100 or days_since > 14
    
    def retrain(self):
        """Retrain embedding model on successful diagnoses"""
        # Get positive feedback examples
        positive_examples = self.db.diagnosis_feedback.find(
            {"feedback_type": "positive"}
        ).limit(1000)
        
        # Create training pairs
        train_pairs = [
            (example['fault_description'], example['used_doc_text'])
            for example in positive_examples
        ]
        
        # Fine-tune embeddings (on GPU)
        self.embedding_trainer.fine_tune(train_pairs, epochs=1)
        
        # Reload embeddings in ChromaDB
        self.vectordb.reload_embeddings()
        
        # Mark as used
        self.db.diagnosis_feedback.update_many(
            {"feedback_type": "positive"},
            {"$set": {"used_for_training": True}}
        )
```

**Configuration:**
```python
FEEDBACK_LEARNING = {
    "enabled": true,
    "min_feedback_for_retrain": 100,
    "retrain_interval_days": 14,
    "min_accuracy_threshold": 0.85,
    "learning_rate": 0.001
}
```

**Expected Impact:**
- ✅ Continuous improvement over time (+5-10% accuracy per retrain cycle)
- ✅ Learned mappings capture domain patterns
- ✅ Self-correcting system (errors ≠ permanent)

---

## ⚡ Phase 6: Background Ingestion & Caching (Week 6-7)

### 6.1 Asynchronous Document Ingestion
**Goal:** Non-blocking background processing of documents

```python
class BackgroundIngestor:
    """
    Queue-based async ingestion with status tracking
    Uses Celery + Redis for background tasks
    """
    
    @celery.task(bind=True)
    def ingest_document_async(self, file_path: str):
        """Queue document for ingestion"""
        
        try:
            # Start task
            self.update_state(state='PROCESSING', meta={'step': 'extracting_text'})
            
            # Extract text
            text = self.document_processor.extract(file_path)
            self.update_state(meta={'step': 'chunking', 'progress': 0.3})
            
            # Chunk
            chunks = self.semantic_chunker.chunk(text, doc_type='pdf')
            self.update_state(meta={'step': 'embedding', 'progress': 0.6})
            
            # Generate embeddings (batch)
            embeddings = self.embeddings.batch_generate(
                [c['text'] for c in chunks],
                batch_size=32
            )
            self.update_state(meta={'step': 'storing', 'progress': 0.9})
            
            # Store in ChromaDB
            self.vectordb.add_documents(chunks, embeddings)
            
            # Complete
            return {'status': 'success', 'chunks_added': len(chunks)}
        
        except Exception as e:
            self.update_state(state='FAILURE', meta={'error': str(e)})
            raise
```

**Configuration:**
```python
# Celery settings
CELERY_BROKER_URL = "redis://redis:6379/0"
CELERY_RESULT_BACKEND = "redis://redis:6379/0"
INGESTION_BATCH_SIZE = 32
INGESTION_TIMEOUT = 3600  # 1 hour
```

### 6.2 Query & Embedding Caching
**Goal:** Cache frequent queries to avoid redundant computation

```python
class QueryCache:
    """Cache embeddings and retrieval results"""
    
    def __init__(self, ttl: int = 3600):
        self.cache = redis.Redis(host='redis', port=6379)
        self.ttl = ttl  # 1 hour
    
    def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Check cache before generating embedding"""
        key = f"embedding:{hash(text)}"
        cached = self.cache.get(key)
        return json.loads(cached) if cached else None
    
    def cache_embedding(self, text: str, embedding: List[float]):
        """Store embedding for future use"""
        key = f"embedding:{hash(text)}"
        self.cache.setex(key, self.ttl, json.dumps(embedding))
    
    def get_cached_retrieval(self, query: str, top_k: int) -> Optional[List[Dict]]:
        """Check cache before retrieving"""
        key = f"retrieval:{hash(query)}:{top_k}"
        cached = self.cache.get(key)
        return json.loads(cached) if cached else None
    
    def cache_retrieval(self, query: str, top_k: int, results: List[Dict]):
        """Store retrieval results"""
        key = f"retrieval:{hash(query)}:{top_k}"
        self.cache.setex(key, self.ttl, json.dumps(results))
```

### 6.3 Batch Embedding Generation
**Goal:** Process embeddings efficiently on GPU

```python
class BatchEmbeddingGenerator:
    """Generate embeddings in batches for speed"""
    
    def generate_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[float]]:
        """Process texts in batches on GPU"""
        
        embeddings = []
        
        pbar = tqdm(total=len(texts)) if show_progress else None
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Generate on GPU
            batch_embeddings = self.model.encode(
                batch,
                batch_size=batch_size,
                device='cuda',
                convert_to_numpy=True
            )
            
            embeddings.extend(batch_embeddings)
            
            if pbar:
                pbar.update(len(batch))
        
        if pbar:
            pbar.close()
        
        return embeddings
```

**Configuration:**
```python
CACHING = {
    "enabled": true,
    "cache_backend": "redis",
    "cache_ttl": 3600,
    "query_cache_enabled": true,
    "embedding_cache_enabled": true
}

BATCH_PROCESSING = {
    "enabled": true,
    "batch_size": 32,
    "use_gpu": true,
    "show_progress": true
}
```

**Expected Impact:**
- ✅ Ingestion doesn't block UI (async)
- ✅ 50-70% faster repeated queries (caching)
- ✅ 3x faster embedding generation (batch + GPU)

---

## 📈 Phase 7: Performance Dashboard & Monitoring (Week 7-8)

### 7.1 RAG Performance Metrics
**Goal:** Track and display key RAG system metrics

```python
class RAGMetricsCollector:
    """Collect metrics on RAG system performance"""
    
    def record_diagnosis(
        self,
        diagnosis: Dict,
        retrieval_time_ms: float,
        generation_time_ms: float,
        retrieval_count: int
    ):
        """Record diagnosis with metrics"""
        
        metric = {
            "timestamp": datetime.now(),
            "diagnosis_id": diagnosis['id'],
            "part_number": diagnosis['part_number'],
            "fault_keywords": self._extract_keywords(diagnosis['fault']),
            
            # Retrieval metrics
            "retrieval_time_ms": retrieval_time_ms,
            "retrieval_count": retrieval_count,
            "avg_relevance_score": diagnosis['avg_relevance'],
            "max_relevance_score": diagnosis['max_relevance'],
            
            # Generation metrics
            "generation_time_ms": generation_time_ms,
            "output_length": len(diagnosis['suggestion']),
            "confidence": diagnosis['confidence'],
            
            # Quality metrics (computed later when feedback arrives)
            "feedback_received": false,
            "is_correct": None,
            "correction_time_hours": None
        }
        
        self.db.rag_metrics.insert_one(metric)
```

### 7.2 Aggregated Performance Dashboard Endpoint
**Goal:** Provide real-time performance metrics to frontend

```python
@app.get("/admin/rag-performance")
async def get_rag_performance(authorization: str = Header(...)):
    """Get comprehensive RAG performance metrics"""
    verify_admin_token(authorization)
    
    metrics_collector = RAGMetricsCollector()
    
    return {
        "time_period": "24h",
        
        "retrieval": {
            "avg_time_ms": metrics_collector.avg_retrieval_time(hours=24),
            "p95_time_ms": metrics_collector.p95_retrieval_time(hours=24),
            "avg_documents_retrieved": metrics_collector.avg_doc_count(hours=24),
            "cache_hit_rate": metrics_collector.cache_hit_rate(hours=24)
        },
        
        "generation": {
            "avg_time_ms": metrics_collector.avg_generation_time(hours=24),
            "p95_time_ms": metrics_collector.p95_generation_time(hours=24),
            "avg_output_length": metrics_collector.avg_output_length(hours=24)
        },
        
        "accuracy": {
            "positive_feedback_rate": metrics_collector.positive_feedback_pct(hours=24),
            "correction_rate": metrics_collector.correction_rate(hours=24),
            "avg_confidence": metrics_collector.avg_confidence(hours=24)
        },
        
        "system": {
            "total_diagnoses": metrics_collector.total_diagnoses(hours=24),
            "active_users": metrics_collector.active_users(hours=24),
            "documents_in_vectordb": self.vectordb.get_count(),
            "rag_engine_status": "healthy"
        },
        
        "trends": {
            "retrieval_time_trend": metrics_collector.retrieval_time_trend(hours=24),  # 24 hourly points
            "accuracy_trend": metrics_collector.accuracy_trend(hours=24),
            "user_count_trend": metrics_collector.user_count_trend(hours=24)
        }
    }
```

### 7.3 Advanced Analytics
**Goal:** Deep-dive analysis for continuous improvement

```python
class RAGAnalytics:
    """Advanced analytics for RAG optimization"""
    
    def get_poor_performing_documents(self, threshold: float = 0.3) -> List[Dict]:
        """Find documents that rarely appear in correct diagnoses"""
        
        # Get documents with low positive feedback rate
        poor_docs = self.db.rag_metrics.aggregate([
            {
                "$group": {
                    "_id": "$document_id",
                    "usage_count": {"$sum": 1},
                    "correct_count": {
                        "$sum": {"$cond": ["$is_correct", 1, 0]}
                    }
                }
            },
            {
                "$project": {
                    "success_rate": {
                        "$divide": ["$correct_count", "$usage_count"]
                    }
                }
            },
            {"$match": {"success_rate": {"$lt": threshold}}}
        ])
        
        return list(poor_docs)
    
    def get_fault_pattern_analysis(self, hours: int = 168) -> Dict:
        """Analyze common fault patterns and their solutions"""
        
        # Aggregate by fault keywords
        patterns = self.db.rag_metrics.aggregate([
            {"$match": {"timestamp": {"$gte": datetime.now() - timedelta(hours=hours)}}},
            {
                "$group": {
                    "_id": "$fault_keywords",
                    "total": {"$sum": 1},
                    "correct": {"$sum": {"$cond": ["$is_correct", 1, 0]}},
                    "avg_time_hours": {"$avg": "$correction_time_hours"}
                }
            },
            {"$sort": {"total": -1}},
            {"$limit": 20}
        ])
        
        return {
            "top_patterns": list(patterns),
            "analysis": {
                "easiest_to_diagnose": patterns[0] if patterns else None,
                "hardest_to_diagnose": list(patterns)[-1] if patterns else None,
                "total_unique_patterns": len(list(patterns))
            }
        }
```

**Configuration:**
```python
PERFORMANCE_MONITORING = {
    "enabled": true,
    "metrics_retention_days": 90,
    "alert_thresholds": {
        "retrieval_time_ms": 2000,
        "generation_time_ms": 5000,
        "accuracy_rate": 0.75,
        "cache_hit_rate": 0.3
    },
    "dashboards": [
        "system_health",
        "accuracy_trends",
        "performance_benchmarks",
        "document_quality"
    ]
}
```

### 7.4 Frontend Dashboard Updates
**File:** `frontend/src/AdminDashboard.jsx` (UPDATE)

```jsx
// New tabs:
<div className="dashboard-tabs">
  <Tab name="Overview" default={true} />
  <Tab name="RAG Performance" />      {/* NEW */}
  <Tab name="Accuracy Analytics" />   {/* NEW */}
  <Tab name="Document Quality" />     {/* NEW */}
  <Tab name="System Health" />        {/* ENHANCED */}
</div>

// RAG Performance Tab shows:
// - Retrieval time trends (graph)
// - Cache hit rate over time
// - Average relevance scores
// - Generation time distributions
// - Top performing vs poor documents
```

**Expected Impact:**
- ✅ Data-driven optimization decisions
- ✅ Early detection of degrading performance
- ✅ Identify documents needing review/removal
- ✅ Track ROI of improvements

---

## 🏆 Expected Outcomes & Success Metrics

### Performance Targets

| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| **Retrieval Time** | 500ms | 100ms | Phase 6 |
| **Accuracy (F1)** | 0.72 | 0.85 | Phase 5 |
| **Cache Hit Rate** | 0% | 40% | Phase 6 |
| **Positive Feedback %** | 65% | 80% | Phase 7 |
| **Document Coverage** | 237 | 500+ | Phase 6 |
| **Inference Cost/Query** | 2.5s | 0.8s | Phase 6 |

### Quality Improvements

| Dimension | Improvement | Method |
|-----------|-------------|--------|
| **Semantic Understanding** | +25% | Domain-specific embeddings |
| **Precision** | +15% | Metadata filtering + hybrid search |
| **Recall** | +10% | Query expansion + ANN search |
| **Hallucination Reduction** | -20% | Semantic chunking + response scoring |
| **Context Preservation** | +30% | Recursive chunking with structure |

### Learning Benefits

| Loop Component | Benefit |
|---|---|
| **Positive Feedback** | Boost relevant documents (+10% accuracy) |
| **Negative Feedback** | Demote poor sources, learn corrections (+5% accuracy) |
| **Learned Mappings** | Pattern recognition (+8% accuracy) |
| **Periodic Retraining** | Continuous improvement (+5% per cycle) |

---

## 🛠️ Implementation Checklist

### Phase 1: Semantic Chunking
- [ ] Create `SemanticChunker` class
- [ ] Add document structure detection
- [ ] Implement metadata extraction per chunk
- [ ] Write unit tests
- [ ] Update ingestion pipeline
- [ ] Re-ingest existing documents
- [ ] A/B test semantic vs. basic chunking

### Phase 2: Domain Embeddings
- [ ] Prepare training dataset from feedback
- [ ] Fine-tune embeddings on domain
- [ ] Test and validate quality
- [ ] Implement hybrid (sparse + dense) search
- [ ] Add query expansion
- [ ] Deploy and monitor

### Phase 3: Advanced Retrieval
- [ ] Implement metadata filtering
- [ ] Add ANN optimizations
- [ ] Create result merger
- [ ] Implement learned boost system
- [ ] Add ranking pipeline
- [ ] Test retrieval quality

### Phase 4: Prompt Optimization
- [ ] Create `PromptOptimizer` class
- [ ] Implement few-shot examples
- [ ] Add response quality scorer
- [ ] Integrate into diagnosis flow
- [ ] Monitor output quality
- [ ] Adjust weights based on metrics

### Phase 5: Feedback Loop
- [ ] Implement feedback propagation
- [ ] Create learned mapping boosting
- [ ] Add periodic retraining
- [ ] Monitor accuracy improvements
- [ ] Document learned patterns
- [ ] Create user feedback for corrections

### Phase 6: Background Processing
- [ ] Setup Celery + Redis
- [ ] Create async ingestion tasks
- [ ] Implement embedding caching
- [ ] Add query result caching
- [ ] Deploy and test under load
- [ ] Monitor queue performance

### Phase 7: Monitoring & Dashboards
- [ ] Create metrics collection system
- [ ] Build performance dashboard endpoint
- [ ] Create analytics queries
- [ ] Update frontend dashboard
- [ ] Set up alerts for degradation
- [ ] Create reporting queries

---

## 📊 Resource Requirements

### Compute Resources
- **GPU**: NVIDIA GPU recommended for:
  - Embedding generation (batch processing)
  - Fine-tuning embeddings
  - Faster inference (optional for Ollama)
  - Estimated: 6-8GB VRAM

- **CPU**: 4+ cores for:
  - Background task processing (Celery workers)
  - Document chunking
  - LLM inference (Ollama)

- **Memory**: 16GB+ recommended for:
  - ChromaDB + full vector index
  - Redis caching
  - LLM model loading

### External Services
- **Redis**: Caching + task queue
- **Celery**: Background task processing
- **HuggingFace**: Download fine-tuned models
- **Optional GPU Cloud**: For periodic retraining

### Storage
- **ChromaDB**: 500MB-2GB (depending on document volume)
- **Embedding Cache**: 100MB-500MB
- **Metrics Database**: 100MB-1GB/month

---

## 🎯 Success Criteria

### Functional Requirements
- ✅ System processes documents asynchronously without UI blocking
- ✅ Retrieval time <200ms for 99% of queries
- ✅ Accuracy improves by 15%+ from baseline
- ✅ Positive feedback rate reaches 80%+
- ✅ System self-corrects from negative feedback

### Non-Functional Requirements
- ✅ Dashboard updates in real-time
- ✅ No query latency degradation with cache
- ✅ System handles 1000+ documents
- ✅ Supports concurrent ingestion + queries
- ✅ Metrics retention for trend analysis

### Business Outcomes
- ✅ Technicians solve more problems on first try
- ✅ Reduced support tickets for incorrect diagnoses
- ✅ Faster repair turnaround
- ✅ Better documentation coverage insights
- ✅ Data-driven optimization roadmap

---

## 📅 Timeline & Milestones

```
Week 1-2:   Semantic Chunking + Metadata
Week 3:     Domain Embeddings Fine-tuning
Week 4:     Advanced Retrieval (Filtering + ANN)
Week 5:     Prompt Optimization + Quality Scoring
Week 6:     Feedback Loop Enhancement
Week 7:     Async Ingestion + Caching
Week 8:     Performance Dashboard + Monitoring
Week 9-10:  Testing, optimization, documentation
```

**Total Estimated Effort:** 400-500 engineering hours

---

## 🔗 Related Documentation

- [RAG Engine Implementation](./src/llm/rag_engine.py)
- [Feedback Learning System](./src/llm/feedback_engine.py)
- [ChromaDB Client](./src/vectordb/chroma_client.py)
- [Chunking Module](./src/documents/chunker.py)
- [AI Configuration](./config/ai_settings.py)

---

## 📝 Notes & Considerations

### Open Questions
1. **Fine-tuning dataset size**: How many positive feedback examples needed for meaningful improvement?
2. **Model selection**: Best language model for technical domain (current: qwen2.5:7b)?
3. **Retraining frequency**: Weekly/monthly/on-demand?
4. **Multi-language support**: Should embeddings handle Turkish + English?

### Risk Mitigation
- **Embedding retraining**: Keep old models for fallback
- **Performance degradation**: Implement rollback strategy
- **Cache misses**: Design graceful fallback to computation
- **Learned bias**: Monitor for feedback loop amplifying errors

### Future Enhancements (Post-Phase 7)
- Multi-modal embeddings (images + text)
- Active learning (auto-query technicians on low-confidence cases)
- Federated learning (learn from multiple repair centers)
- Reinforcement learning (optimize for business KPIs)

---

**End of Roadmap**

*Last Updated: 15 December 2025*  
*Next Review: 22 December 2025*
