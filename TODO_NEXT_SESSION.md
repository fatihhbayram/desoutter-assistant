# TODO: Next Session

## Status: Source Citation Enhancement COMPLETE ✅

We have successfully fixed the page number extraction issue and re-ingested all documents. The RAG system now has accurate metadata.

## Next Priorities

### 1. Phase 2: Retrieval Enhancement (Week 1)
- [ ] **Hybrid Retrieval:** Implement `rank_bm25` to combine keyword search with vector search.
- [ ] **Re-Ranking:** Implement a re-ranker (e.g., Cross-Encoder) to filter top-k results for better relevance.
- [ ] **Metadata Filtering:** Add filters for `product_line` and `doc_type` in the API query.

### 2. Phase 3: Dynamic Prompts (Week 2)
- [ ] **Intent Detection:** Refine the `IntentDetector` to better classify queries (Troubleshooting vs. Specs).
- [ ] **Prompt Templates:** Create specific system prompts for each intent type.

## Technical Debt / Maintenance
- [ ] **Unit Tests:** Add more comprehensive unit tests for `DocumentProcessor`.
- [ ] **CI/CD:** Consider setting up a GitHub Action for automated testing.
