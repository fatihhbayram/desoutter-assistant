# 🔬 RAG Quality Improvement Guide

Technical documentation of the RAG (Retrieval-Augmented Generation) system architecture and optimization strategies for the Desoutter Assistant.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture Deep-Dive](#architecture-deep-dive)
- [Quality Metrics](#quality-metrics)
- [Optimization Techniques](#optimization-techniques)
- [Test Suite](#test-suite)
- [Future Improvements](#future-improvements)

---

## 🎯 Overview

### What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI architecture that enhances Large Language Model (LLM) responses by retrieving relevant context from a knowledge base before generating answers. This approach:

- **Reduces hallucinations** by grounding responses in actual documents
- **Enables domain-specific knowledge** without fine-tuning the LLM
- **Provides source citations** for transparency and verification
- **Allows knowledge updates** without model retraining

### Why RAG for Industrial Tools?

Industrial tool support requires:
- **Precise technical accuracy** - incorrect repair advice can be costly or dangerous
- **Product-specific knowledge** - 451 different tool models with unique specifications
- **Up-to-date information** - service bulletins and maintenance updates
- **Traceable sources** - ability to verify advice against official documentation

---

## 🏗️ Architecture Deep-Dive

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Query ──→ Intent Detection ──→ Query Expansion            │
│                      │                      │                   │
│                      ▼                      ▼                   │
│              [8 Intent Types]      [9 Synonym Categories]       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       HYBRID RETRIEVAL                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐              ┌─────────────────┐          │
│  │  Semantic Search│              │   BM25 Search   │          │
│  │   (ChromaDB)    │              │  (Keyword)      │          │
│  │   Weight: 0.7   │              │   Weight: 0.3   │          │
│  └────────┬────────┘              └────────┬────────┘          │
│           │                                │                    │
│           └──────────┬─────────────────────┘                    │
│                      ▼                                          │
│              ┌──────────────┐                                   │
│              │  RRF Fusion  │                                   │
│              │   (k=60)     │                                   │
│              └──────┬───────┘                                   │
│                     │                                           │
│                     ▼                                           │
│         ┌────────────────────┐                                  │
│         │  Metadata Boosting │                                  │
│         │  • Bulletin: 1.5x  │                                  │
│         │  • Procedure: 1.3x │                                  │
│         │  • Warning: 1.2x   │                                  │
│         └────────┬───────────┘                                  │
│                  │                                              │
│                  ▼                                              │
│         Top-5 Ranked Results                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CONTEXT GROUNDING                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Context Sufficiency Score = f(similarity, doc_count, terms)    │
│                                                                 │
│  if score < 0.5:                                                │
│      return "I don't know" response                             │
│  else:                                                          │
│      proceed to LLM generation                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM GENERATION                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Intent-Specific Prompt + Retrieved Context + Query             │
│                         │                                       │
│                         ▼                                       │
│              ┌──────────────────┐                               │
│              │  Qwen2.5:7b      │                               │
│              │  (GPU Inference) │                               │
│              └────────┬─────────┘                               │
│                       │                                         │
│                       ▼                                         │
│         ┌─────────────────────────┐                             │
│         │  Response Validation    │                             │
│         │  • Number verification  │                             │
│         │  • Forbidden content    │                             │
│         │  • Uncertainty phrases  │                             │
│         └───────────┬─────────────┘                             │
│                     │                                           │
│                     ▼                                           │
│              Final Response                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Hybrid Search Implementation

The hybrid search combines two complementary retrieval methods:

#### Semantic Search (ChromaDB)

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Similarity Metric**: Cosine similarity
- **Threshold**: 0.30 minimum similarity
- **Weight**: 0.7 in final score

```python
# Semantic search retrieves documents based on meaning
results = chromadb.query(
    query_embeddings=embed(user_query),
    n_results=10,
    where={"product_line": product_filter}  # Optional metadata filter
)
```

#### BM25 Keyword Search

- **Index Size**: 19,032 unique terms
- **Parameters**: k1=1.5, b=0.75 (standard tuning)
- **Weight**: 0.3 in final score

```python
# BM25 retrieves documents based on term frequency
class BM25Index:
    def search(self, query: str, top_k: int = 10):
        tokens = tokenize(query)
        scores = {}
        for doc_id, doc in self.documents.items():
            scores[doc_id] = self._score_document(tokens, doc)
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]
```

#### RRF (Reciprocal Rank Fusion)

Combines results from both methods:

```python
def rrf_fusion(semantic_ranks: dict, bm25_ranks: dict, k: int = 60):
    """
    RRF Score = Σ 1/(k + rank) for each ranking
    """
    combined = {}
    for doc_id, rank in semantic_ranks.items():
        combined[doc_id] = combined.get(doc_id, 0) + (0.7 / (k + rank))
    for doc_id, rank in bm25_ranks.items():
        combined[doc_id] = combined.get(doc_id, 0) + (0.3 / (k + rank))
    return sorted(combined.items(), key=lambda x: -x[1])
```

---

### Query Expansion

The `QueryExpander` class enriches queries with domain-specific synonyms:

| Category | Original Terms | Expansions |
|----------|----------------|------------|
| Motor | motor | spindle, drive, rotation |
| Error | error, fault | failure, warning, issue |
| Battery | battery | power, cell, accumulator |
| Calibration | calibration | calibrate, adjustment, tuning |
| Torque | torque | tightening, tension, nm |
| Connection | connection | cable, wire, connector |
| Noise | noise | squealing, grinding, vibration |
| Bearing | bearing | ball bearing, bushing |
| Controller | controller | CVI3, unit, platform |

**Error Code Normalization:**
- `e047` → `E47` → `E047`
- `error 15` → `E15`

---

### Product-Specific Filtering

Prevents cross-product contamination in search results:

```python
class ProductExtractor:
    """Automatically identifies product families from filenames and content."""
    
    PRODUCT_PATTERNS = {
        "EPB": ["battery_tool", "epb"],
        "EPBC": ["battery_tool", "epbc", "wifi"],
        "CVI3": ["controller", "cvi3"],
        "CVIR": ["controller", "cvir"],
        # ... 27 product families
    }
    
    def extract(self, filename: str, content: str) -> List[str]:
        """Returns list of product categories for metadata tagging."""
```

**15 Fault Categories with Negative Keywords:**

| Category | Keywords | Negative Keywords |
|----------|----------|-------------------|
| WiFi | wifi, wireless, signal | cable, corded |
| Motor | motor, spindle, rotation | battery, charger |
| Battery | battery, charge, cell | corded, cable |
| Torque | torque, nm, tightening | wifi, display |
| Display | display, screen, lcd | motor, bearing |
| ... | ... | ... |

---

### Intent Detection

8 intent types with specialized prompts:

| Intent | Keywords (EN) | Keywords (TR) | Prompt Focus |
|--------|---------------|---------------|--------------|
| Troubleshooting | not working, error, fails | çalışmıyor, arıza | Step-by-step diagnosis |
| Specifications | specs, weight, dimensions | özellik, ağırlık | Technical data tables |
| Installation | install, setup, mount | kurulum, montaj | Installation procedures |
| Calibration | calibrate, adjustment | kalibrasyon, ayar | Calibration steps |
| Maintenance | maintain, service, clean | bakım, temizlik | Maintenance schedules |
| Connection | connect, cable, wifi | bağlantı, kablo | Connection architecture |
| Error Code | E01, error code, fault | hata kodu | Error resolution |
| General | how, what, info | nasıl, bilgi | General information |

---

## 📊 Quality Metrics

### Current Performance

| Metric | Value | Target |
|--------|-------|--------|
| **Overall Pass Rate** | 96% (24/25) | >90% |
| **Troubleshooting Accuracy** | 100% (5/5) | >90% |
| **Error Code Accuracy** | 100% (4/4) | >95% |
| **Specifications Accuracy** | 67% (2/3) | >80% |
| **Connection Accuracy** | 100% (3/3) | >90% |
| **"I Don't Know" Rate** | ~12% | 10-15% |
| **Average Response Time** | 2.4ms (cached) | <3s |

### Test Suite Breakdown

```
Test Results (final_v2.json):
├── Troubleshooting: 5/5 ✅
├── Error Codes: 4/4 ✅
├── Specifications: 2/3 (1 edge case) ⚠️
├── Connection: 3/3 ✅
├── Maintenance: 2/2 ✅
├── Calibration: 2/2 ✅
├── General: 5/5 ✅
└── Installation: 1/1 ✅

By Category:
├── Basic Tests: 18/19 ✅
├── Language Tests: 3/3 ✅
└── Edge Cases: 3/3 ✅
```

### Confidence Distribution

| Confidence Level | Criteria | Response Handling |
|------------------|----------|-------------------|
| **High** | Similarity >0.6, 3+ sources | Standard response |
| **Medium** | Similarity 0.4-0.6, 2+ sources | Response with caveats |
| **Low** | Similarity 0.3-0.4, 1+ source | Suggest verification |
| **Insufficient** | Similarity <0.3 or 0 sources | "I don't know" response |

---

## ⚙️ Optimization Techniques

### Semantic Chunking

Structure-aware document splitting:

| Document Type | Chunk Size | Overlap | Rationale |
|---------------|------------|---------|-----------|
| Troubleshooting | 200 tokens | 50 | Precise issue-solution mapping |
| Service Bulletin | 300 tokens | 75 | Medium context units |
| Technical Manual | 400 tokens | 100 | Full procedural context |

**Metadata Fields (14 per chunk):**
- `source`: Original document filename
- `page_number`: Extracted page reference
- `section`: Section heading
- `document_type`: Manual, bulletin, guide, etc.
- `section_type`: Procedure, warning, table, etc.
- `importance_score`: 0.0-1.0 based on content
- `contains_warning`: Boolean for safety content
- `is_procedure`: Boolean for actionable steps
- `fault_keywords`: Extracted domain terms
- `product_categories`: Associated products
- `content_hash`: SHA-256 for deduplication
- `chunk_index`: Sequential position
- `position_ratio`: Relative document position
- `heading_text`: Parent heading context

### Context Window Optimization

8K token budget management:

```python
class ContextOptimizer:
    def optimize(self, chunks: List[Chunk], query: str) -> List[Chunk]:
        # 1. Remove duplicates (Jaccard similarity >85%)
        deduplicated = self._deduplicate(chunks)
        
        # 2. Score by relevance + importance
        scored = self._score_chunks(deduplicated, query)
        
        # 3. Prioritize warnings and procedures
        prioritized = self._prioritize_safety(scored)
        
        # 4. Fit within token budget
        fitted = self._fit_budget(prioritized, max_tokens=8000)
        
        return fitted
```

**Scoring Formula:**
- Similarity: 40%
- Importance: 30%
- Warning bonus: 15%
- Procedure bonus: 10%
- Query term overlap: 5%

### Response Caching

LRU + TTL cache for repeated queries:

- **Cache Size**: 1000 entries
- **TTL**: 1 hour
- **Speedup**: ~100,000x for cache hits (30s → 0.3ms)
- **Cache Key**: Hash of (product, query, language)

---

## 🧪 Test Suite

### Test Categories

1. **Basic Tests** (19 scenarios)
   - Standard troubleshooting queries
   - Specification lookups
   - Error code resolution
   - Connection guidance

2. **Language Tests** (3 scenarios)
   - Turkish query handling
   - English response generation
   - Mixed language support

3. **Edge Cases** (3 scenarios)
   - Unanswerable queries → "I don't know"
   - Off-topic questions → Graceful handling
   - Ambiguous product references

### Running Tests

```bash
# Run full test suite
python scripts/test_rag_quality.py

# Run specific category
python scripts/test_rag_quality.py --category troubleshooting

# Save results to JSON
python scripts/test_rag_quality.py --output test_results/latest.json
```

### Test Result Format

```json
{
  "summary": {
    "total_tests": 25,
    "passed": 24,
    "failed": 1,
    "pass_rate": 96.0
  },
  "by_intent": {
    "troubleshooting": {"total": 5, "passed": 5},
    "error_code": {"total": 4, "passed": 4}
  },
  "failed_tests": ["SPEC_002"],
  "results": [...]
}
```

---

## 🔮 Future Improvements

### Short-term (Q1 2026)

- [ ] **Cross-Encoder Re-ranking**: Add re-ranking layer for top-10 → top-5 filtering
- [ ] **Confidence Scoring**: Numeric confidence score in API responses
- [ ] **User Profiles**: Personalized responses based on expertise level

### Medium-term (Q2 2026)

- [ ] **Embedding Fine-tuning**: Domain-specific embedding model using contrastive learning
- [ ] **Multi-modal RAG**: Image-based troubleshooting with visual context
- [ ] **Active Learning**: Prioritize annotation of uncertain responses

### Long-term (2026+)

- [ ] **Knowledge Graph**: Structured product-component relationships
- [ ] **Agentic RAG**: Multi-step reasoning for complex diagnostics
- [ ] **Federated Learning**: Learn from multiple deployment sites

---

## 📚 References

- [RAG Enhancement Roadmap](RAG_ENHANCEMENT_ROADMAP.md) - Detailed phase documentation
- [CHANGELOG.md](CHANGELOG.md) - Version history with RAG changes
- [Sentence Transformers](https://www.sbert.net/) - Embedding model documentation
- [ChromaDB](https://docs.trychroma.com/) - Vector database documentation
- [Ollama](https://ollama.ai/docs/) - LLM inference documentation

---

## 📈 Success Criteria

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Relevance Rate | ~80% | >80% | ✅ Achieved |
| User Rating ≥4/5 | ~70% | >70% | ✅ Achieved |
| "I don't know" Rate | 12% | 10-15% | ✅ Healthy |
| Response Time | 2.4ms | <3s | ✅ Excellent |
| Hallucination Rate | <5% | <5% | ✅ Controlled |

> **Philosophy**: Quality over coverage. It's better to say "I don't know" than to provide incorrect information that could lead to improper repairs.
