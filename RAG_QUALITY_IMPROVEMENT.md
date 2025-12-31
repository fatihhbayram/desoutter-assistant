# RAG Quality Improvement Roadmap

**Created:** 28 December 2025  
**Based on:** Senior AI/ML Engineer RAG System Audit  
**Goal:** Reliable AI that says "I don't know" when uncertain, cites sources, never invents information

---

## Current System Status

### ✅ Already Implemented

| Feature | Status | Phase | Notes |
|---------|--------|-------|-------|
| Hybrid Retrieval | ✅ Complete | 2.2 | BM25 + Semantic, RRF fusion |
| Metadata Tagging | ✅ Complete | 1.1 | 14 fields per chunk |
| Metadata Enrichment (Automatic Tagging) | ✅ Complete | 1.2 | Automatic product tagging & filtering |
| Semantic Chunking | ✅ Complete | 1.1 | Section-aware, adaptive sizing |
| Re-ranking | ✅ Complete | 2.2 | RRF, top-10 → top-5 |
| Feedback System | ✅ Complete | 6 | Thumbs up/down |
| "I don't know" logic | ✅ Complete | Q1 | Context sufficiency scoring |
| Response Validation | ✅ Complete | Q2 | Hallucination detection |
| Intent Detection | ✅ Complete | Q3 | 8 intent types, dynamic prompts |
| Content Deduplication | ✅ Complete | Q4 | SHA-256 hash-based |
| Adaptive Chunking | ✅ Complete | Q5 | Document type-based sizing |

### 🟡 Partially Implemented

| Feature | Current | Gap | Priority |
|---------|---------|-----|----------|
| User Profiles | None | No personalization | MEDIUM |
| Source Citation | ✅ Complete | 1.2 | Page number & Section extraction fixed |

### ❌ Missing Critical Features

| Feature | Impact | Priority |
|---------|--------|----------|
| Confidence scoring | MEDIUM | 🟡 HIGH |
| Advanced analytics | LOW | 🟢 LOW |

---

## Priority 1: Response Grounding & Validation (Week 1) - ✅ COMPLETE

### 🔴 CRITICAL: "I Don't Know" Logic - ✅ DONE

**Problem:** System hallucinates when context insufficient

**Implementation:**
- [x] Add `context_sufficiency_score` calculation
- [x] Threshold: <0.5 → return "I don't know"
- [x] Update prompts with explicit instruction
- [x] Test with 20 unanswerable queries

---

### 🔴 CRITICAL: Response Validation - ✅ DONE

**Problem:** No post-processing to catch hallucinations

**Implementation:**
- [x] Create `src/llm/response_validator.py`
- [x] Implement uncertainty detection
- [x] Implement number verification
- [x] Add validation to RAG pipeline
- [x] Log flagged responses for review

---

## Priority 2: Intent-Based Dynamic Prompts (Week 2) - ✅ COMPLETE

### 🟡 HIGH: Query Intent Detection - ✅ DONE

**Implementation:**
- [x] Create `IntentDetector` with 8 intent types
- [x] Create intent-specific prompt templates
- [x] Update `build_rag_prompt()` to use intent
- [x] Add intent metadata to API responses
- [x] Test with 5 queries per intent

**Date Completed:** 30 December 2025

---

## Priority 3: Content Deduplication (Week 2) - ✅ COMPLETE

### 🟢 LOW: Content Deduplication - ✅ DONE

**Problem:** Duplicate chunks waste storage and reduce retrieval quality

**Implementation:**
- [x] Add `content_hash` (SHA-256) to chunk metadata
- [x] Check for duplicates before indexing in ChromaDB
- [x] Add `ENABLE_DEDUPLICATION` config flag
- [x] Log duplicate ratio for analytics
- [x] Test deduplication logic

**Date Completed:** 30 December 2025

---

## Priority 4: Adaptive Chunk Sizing (Week 2) - ✅ COMPLETE

### 🟢 LOW: Adaptive Chunk Size - ✅ DONE

**Problem:** Fixed chunk size doesn't suit all document types

**Implementation:**
- [x] Implement document type detection
- [x] Create adaptive sizing logic:
  - Troubleshooting guides: 200 tokens (precision)
  - Service bulletins: 300 tokens (medium)
  - Technical manuals: 400 tokens (context)
- [x] Test chunk size distribution
- [x] Verify retrieval quality improvement

**Date Completed:** 30 December 2025

---

## Priority 6: Source Citation Enhancement (Critical) - ✅ COMPLETE

### 🎯 Source Citation - ✅ DONE

**Problem:** RAG responses were unable to cite specific page numbers because metadata was missing.

**Implementation:**
- [x] Fix `clean_text` in `DocumentProcessor` to preserve paragraph structure
- [x] Update `SemanticChunker` regex for robust page detection
- [x] Full re-ingestion of 541 documents
- [x] Verification: All chunks now have `page_number` and `section`

**Date Completed:** 31 December 2025

---

## Priority 5: Analytics & Monitoring (Ongoing)

### 📊 Quality Metrics Dashboard

**Metrics to Track:**
- Relevance Rate: % of top-5 chunks relevant (Target: >80%)
- Answer Accuracy: User rating ≥4/5 (Target: >70%)
- "I don't know" Rate: Healthy range 10-15%
- Response Time: End-to-end (Target: <3s)

**Implementation:**
- [ ] Create metrics collection in MongoDB
- [ ] Add metrics logging to RAG pipeline
- [ ] Build analytics dashboard
- [ ] Weekly review process

---

## Controller Units Scraping (Immediate)

**New Products to Add:**

1. Axon Terminal - 697650
2. CVIR II - 110918
3. CVIxS - 147522
4. ESPC Controller - 167639
5. CVIC II H2 - 110917
6. CVIC II L2 - 110916
7. ESP Low Voltage - 110928
8. ESP2A Low Voltage - 110927
9. CVIL II - 110919
10. Connect Industrial Smart Hub - 110912

**Implementation:**
- [ ] Update scraper to handle controller pages
- [ ] Scrape all 10 controller units
- [ ] Add to MongoDB products collection
- [ ] Update connection architecture mapping
- [ ] Test controller capability detection

---

## Implementation Timeline

### Week 1: Response Grounding (CRITICAL)
- "I don't know" logic
- Source citation
- Response validation

### Week 2: Intent-Based Prompts
- Intent-specific prompts (15 intents)
- Confidence scoring

### Week 3: User Personalization
- User profile system
- Expertise-based responses

### Week 4: Quality Loop
- Analytics dashboard
- Controller units scraping

---

## Success Criteria

| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| Relevance Rate | ~60% | >80% | Week 2 |
| User Rating | Unknown | >70% ≥4/5 | Week 3 |
| "I don't know" Rate | 0% | 10-15% | Week 1 |
| Response Time | ~2s | <3s | Maintain |

**Remember:** Quality over coverage. Better to say "I don't know" than to hallucinate.
