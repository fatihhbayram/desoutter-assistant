# 🚀 EL-HAREZMI V2.0: TUNING & ACCURACY ROADMAP

**Status**: 80% Test Pass Rate
**Target**: 95%+ Pass Rate
**Started**: 2026-02-23

The system has reached 80% stability (Qdrant + 15 Intents + El-Harezmi). To resolve the remaining 20% failure gap causing "Timeout", "Missing Terms", and "I don't know" errors, the following priority list will be executed in Phase 6.

To resume development in the next session, you can issue the following prompt:
👉 **"Let's start Phase 6 Tuning with Priority 1 (or 2)."**

---

### 🥇 Priority 1: Retrieval & Confidence Tuning
*Resolves "I don't know" and "Low Confidence" errors.*
- [ ] 1.1. Adjust RRF weights (BM25 vs Dense) in `hybrid_search.py` favoring technical terms.
- [ ] 1.2. Lower the strict `0.5` confidence threshold in `confidence_scorer.py` to `0.4` for narrow-scope intents like CONNECTION/COMPATIBILITY.
- [ ] 1.3. Increase `RAG_TOP_K` limit from `5` to `7` or `10` to prevent loss of procedural steps during retrieval.

### 🥈 Priority 2: LLM Configuration & Timeout Prevention
*Resolves >60s Timeout errors on heavy Configuration queries.*
- [ ] 2.1. Add strict length limits to El-Harezmi Stage 5 (Response) prompts (e.g., Max 4 sentences/bullets).
- [ ] 2.2. Clean up unnecessary metadata overhead sent to the Ollama model context window.

### 🥉 Priority 3: Stage 3 & 4 Prompt Hardening
*Resolves "Missing terms" errors where the LLM skips crucial metrics like "CVI3" or "rpm".*
- [ ] 3.1. Add explicit instructions to `stage3_extraction.py` JSON prompts: "If standard units (Nm, RPM, °C) are present, you MUST write them into the JSON."
- [ ] 3.2. Expand the `COMPATIBILITY_MATRIX` dataset in `stage4_validation.py` to cover failing products.

### 🏅 Priority 4: Adaptive Chunking Refinement
*Improves text chunking boundaries for edge cases.*
- [ ] 4.1. Increase overlap values in `semantic_chunker.py` specifically for Configuration guides.
- [ ] 4.2. Strengthen the regex rules in `step_preserving_chunker.py` to preserve complex sub-bullet points (e.g., 1.a, 1.b).
