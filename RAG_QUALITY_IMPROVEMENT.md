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
| Semantic Chunking | ✅ Complete | 1.1 | Section-aware, 400 chars |
| Re-ranking | ✅ Complete | 2.2 | RRF, top-10 → top-5 |
| Feedback System | ✅ Complete | 6 | Thumbs up/down |
| Self-Learning | ✅ Complete | 6 | Wilson score ranking |
| Relevance Filtering | ✅ Complete | 0.1 | 15 fault categories |
| Product Capability | ✅ Complete | 0.2 | Wireless/battery detection |
| Error Code Detection | ✅ Complete | - | E01-E25, transducer |

### 🟡 Partially Implemented

| Feature | Current | Gap | Priority |
|---------|---------|-----|----------|
| System Prompt | Static | Not intent-based | HIGH |
| Chunk Size | 400 chars | Should be 600-1200 tokens | MEDIUM |
| Deduplication | None | No content hash | LOW |
| User Profiles | None | No personalization | MEDIUM |

### ❌ Missing Critical Features

| Feature | Impact | Priority |
|---------|--------|----------|
| "I don't know" logic | HIGH | 🔴 CRITICAL |
| Source citation | HIGH | 🔴 CRITICAL |
| Response validation | HIGH | 🔴 CRITICAL |
| Intent-based prompts | MEDIUM | 🟡 HIGH |
| Confidence scoring | MEDIUM | 🟡 HIGH |

---

## Priority 1: Response Grounding & Validation (Week 1)

### 🔴 CRITICAL: "I Don't Know" Logic

**Problem:** System hallucinates when context insufficient

**Implementation:**
- [ ] Add `context_sufficiency_score` calculation
- [ ] Threshold: <0.5 → return "I don't know"
- [ ] Update prompts with explicit instruction
- [ ] Test with 20 unanswerable queries

---

### 🔴 CRITICAL: Source Citation

**Problem:** Responses don't cite which document was used

**Implementation:**
- [ ] Update `build_rag_prompt()` to include citations
- [ ] Add citation instruction to system prompt
- [ ] Parse LLM response for citation format
- [ ] Test citation accuracy

---

### 🔴 CRITICAL: Response Validation

**Problem:** No post-processing to catch hallucinations

**Implementation:**
- [ ] Create `src/llm/response_validator.py`
- [ ] Implement uncertainty detection
- [ ] Implement number verification
- [ ] Add validation to RAG pipeline
- [ ] Log flagged responses for review

---

## Priority 2: Intent-Based Dynamic Prompts (Week 2)

### 🟡 HIGH: Query Intent Detection

**Implementation:**
- [ ] Create intent-specific prompt templates
- [ ] Update `build_rag_prompt()` to use intent
- [ ] Add intent-specific instructions
- [ ] Test with 5 queries per intent

---

### 🟡 HIGH: Confidence Scoring

**Implementation:**
- [ ] Add confidence calculation
- [ ] Return confidence with response
- [ ] Display confidence in UI
- [ ] Log low-confidence responses (<0.5)

---

## Priority 3: User Personalization (Week 3)

### 🟡 MEDIUM: User Profile System

**Implementation:**
- [ ] Add user_profiles collection to MongoDB
- [ ] Create profile management API
- [ ] Update prompts based on expertise_level
- [ ] Filter chunks by primary_products
- [ ] Add profile UI in frontend

---

## Priority 4: Document Processing Improvements (Week 4)

### 🟢 LOW: Adaptive Chunk Size

**Implementation:**
- [ ] Analyze current chunk distribution
- [ ] Implement adaptive chunking
- [ ] Re-ingest documents
- [ ] Compare retrieval quality

---

### 🟢 LOW: Content Deduplication

**Implementation:**
- [ ] Add content_hash to metadata
- [ ] Check before indexing
- [ ] Log duplicate ratio
- [ ] Skip duplicates

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
