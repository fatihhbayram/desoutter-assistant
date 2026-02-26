# 🚀 PHASE 6: TUNING & ACCURACY ROADMAP

**Current Status**: 88% Test Pass Rate (22/25 scenarios) ✅
**Target**: 95%+ Pass Rate
**Last Updated**: 2026-02-25
**System**: Legacy RAG Engine (14-stage) on Qdrant v1.7.4

---

## 📊 CURRENT STATE (February 25, 2026)

### ✅ COMPLETED WORK (Today)

#### 1. ChromaDB → Qdrant Migration Cleanup
- [x] Fixed Qdrant healthcheck (bash TCP socket test)
- [x] Cleaned all ChromaDB ghost references (16 changes, 4 files)
  - `ai-stack.yml`: Healthcheck + comment
  - `src/llm/rag_engine.py`: 9 docstrings/comments
  - `src/llm/hybrid_search.py`: 3 docstrings
  - `README.md`: Vector count + roadmap
- [x] Qdrant container now **healthy** ✅
- [x] README updated: 26,513 vectors (Qdrant v1.7.4)

**Result**: Code is now 100% Qdrant-native, no ChromaDB remnants.

---

### 🎯 CURRENT ARCHITECTURE

```
Production Stack (Active)
━━━━━━━━━━━━━━━━━━━━━━
┌──────────────────────────┐
│  LEGACY RAG (14-stage)   │  ← 100% traffic
│  88% pass rate           │
│  ✅ Qdrant native        │
└──────────────────────────┘
         ↓
┌──────────────────────────┐
│  QDRANT v1.7.4           │
│  26,513 vectors          │
│  Status: GREEN ✅        │
│  Container: HEALTHY ✅   │
└──────────────────────────┘

El-Harezmi Pipeline (Ready)
━━━━━━━━━━━━━━━━━━━━━━━━━
┌──────────────────────────┐
│  EL-HAREZMI (5-stage)    │  ← 0% traffic
│  Stage 1,4,5: Ready ✅   │
│  Stage 2,3: Partial 🔄   │
└──────────────────────────┘
```

---

## 🎯 PHASE 6 PRIORITIES

### 🥇 Priority 1: Retrieval & Confidence Tuning
**Goal**: Fix "I don't know" and low confidence errors
**Estimated Improvement**: +5-7% pass rate
**Duration**: 2-3 hours

#### 1.1. RRF Weights Adjustment
**Files**: `src/llm/hybrid_search.py` + `config/ai_settings.py`

**Current**:
```python
HYBRID_SEMANTIC_WEIGHT = 0.6  # Semantic similarity
HYBRID_BM25_WEIGHT = 0.4      # Keyword matching
```

**Change**:
```python
# Add dynamic weights function in config/ai_settings.py
def get_dynamic_weights(query: str) -> tuple:
    """Adjust weights based on query type"""

    # Error code present → BM25 dominant
    if re.search(r'\b[EI]\d{2,4}\b', query.upper()):
        return (0.6, 0.4)  # (bm25, semantic)

    # Troubleshooting → Balanced
    elif has_troubleshooting_keywords(query):
        return (0.45, 0.55)

    # General → Semantic dominant
    else:
        return (0.4, 0.6)
```

**Test**:
```bash
python scripts/test_hybrid_search.py
# Check BM25 boost for error code queries
```

---

#### 1.2. Confidence Threshold Lowering
**File**: `src/llm/confidence_scorer.py`

**Current**:
```python
confidence_threshold = 0.5  # For all intents
```

**Change**:
```python
def get_confidence_threshold(intent: IntentType) -> float:
    """Adjust threshold based on intent type"""

    narrow_scope_intents = [
        IntentType.COMPATIBILITY,
        IntentType.CONNECTION,
        IntentType.SPECIFICATION
    ]

    if intent in narrow_scope_intents:
        return 0.4  # More tolerant
    else:
        return 0.5  # Default
```

**Reason**: 0.5 is too strict for narrow-scope queries like CONNECTION/COMPATIBILITY.

**Test**:
```bash
# Compatibility query test
python -c "from src.llm.rag_engine import RAGEngine; \
  rag = RAGEngine(); \
  result = rag.generate_repair_suggestion('Is EABC-3000 compatible with CVI3?'); \
  print(result['confidence'])"
```

---

#### 1.3. RAG_TOP_K Increase
**File**: `config/ai_settings.py`

**Current**:
```python
RAG_TOP_K = 5  # 5 chunk retrieval
```

**Change**:
```python
RAG_TOP_K = 7  # Conservative increase
# or
RAG_TOP_K = 10  # Aggressive (more context)
```

**Reason**: Procedure steps can get lost across different chunks (Step 1, 2, 3...).

**Trade-off**: More chunks → More tokens to LLM → Slightly slower (but more complete context).

**Test**:
```bash
python scripts/test_product_filtering.py
# Check chunk coverage for multi-step procedure queries
```

---

### 🥈 Priority 2: LLM Configuration & Timeout Prevention
**Goal**: Fix >60s timeout errors
**Estimated Improvement**: +2-3% pass rate
**Duration**: 2-3 hours

#### 2.1. Response Length Limiting
**File**: `src/llm/rag_engine.py` (prompt generation section)

**Change**: Add to system prompt:
```python
system_prompt = f"""
{get_system_prompt(language)}

CRITICAL RULES:
- Your responses MUST be MAX 4 bullet points or 4 sentences
- Be concise and clear, don't over-explain
- For procedures, provide only main steps

Example ✅ Good:
"1. Check cable connection
2. Set torque to 45 Nm
3. Press reset button
4. Run test"

Example ❌ Bad:
"First, power down the system and wait 5 minutes... [10 lines of detail]"
"""
```

**Test**:
```bash
# Configuration query (usually generates long responses)
curl -X POST http://localhost:8000/diagnose \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"part_number":"EABC-3000","fault_description":"how to configure pset"}'
# Measure response time (target: <30s)
```

---

#### 2.2. Metadata Overhead Cleanup
**File**: `src/llm/rag_engine.py` (context building section)

**Current**: Sending all metadata to context window (15+ fields).

**Change**: Send only essential metadata:
```python
def _build_context_for_llm(self, chunks):
    cleaned_chunks = []
    for chunk in chunks:
        cleaned_chunks.append({
            "text": chunk.text,
            "source": chunk.metadata.get("source"),
            "page": chunk.metadata.get("page"),
            "doc_type": chunk.metadata.get("doc_type")
            # Don't send other 12 fields (created_at, chunk_id, etc.)
        })
    return cleaned_chunks
```

**Result**: Lower token count → Faster LLM processing.

---

### 🥉 Priority 3: Stage 3 & 4 Prompt Hardening
**Goal**: Fix missing terms errors (CVI3, rpm, Nm missing)
**Estimated Improvement**: +1-2% pass rate
**Duration**: 1-2 hours

#### 3.1. Explicit Unit Preservation
**Files**: `src/el_harezmi/stage3_info_extraction.py` (El-Harezmi)
**Files**: `src/llm/prompts.py` (Legacy RAG)

**Change**: Add to prompt:
```python
extraction_prompt = f"""
...

MANDATORY RULE: If standard units are present in context
(Nm, RPM, °C, V, A, bar, psi), you MUST include them in JSON output.

Context example:
"Tighten to 45 Nm at 120 RPM"

JSON output:
{{
  "torque_spec": "45 Nm",
  "speed_spec": "120 RPM",
  "action": "tighten"
}}

NEVER skip units or use generic phrases like "appropriate value".
"""
```

---

#### 3.2. COMPATIBILITY_MATRIX Expansion
**File**: `src/el_harezmi/stage4_kg_validation.py`

**Current**: ~50 product compatibility mappings.

**Change**: Add mappings for failing products:
```python
COMPATIBILITY_MATRIX = {
    "EPBC8-1800": ["CVI3", "CONNECT"],
    "EABC-3000": ["CVI3", "CONNECT"],
    "ERS6-2000": ["CVI3"],  # ← New
    # ... add 20+ new products
}
```

**Source**: Extract product list from failing test scenarios.

---

### 🏅 Priority 4: Adaptive Chunking Refinement
**Goal**: Fix edge case chunking issues
**Estimated Improvement**: +1% pass rate
**Duration**: 1 hour

#### 4.1. Chunk Overlap Increase
**File**: `src/documents/chunkers/semantic_chunker.py`

**Current**:
```python
CHUNK_OVERLAP = 50  # tokens
```

**Change**: Increase for configuration guides:
```python
def get_chunk_overlap(doc_type: str) -> int:
    if doc_type == "configuration_guide":
        return 100  # 50 → 100
    elif doc_type == "procedure":
        return 80   # 50 → 80
    else:
        return 50   # Default
```

---

#### 4.2. Step-Preserving Regex Strengthening
**File**: `src/documents/chunkers/step_preserving_chunker.py`

**Current**: Only detects "1.", "2.", "3."

**Change**: Also capture sub-bullets:
```python
STEP_PATTERNS = [
    r'^\d+\.',           # "1.", "2."
    r'^\d+\.\d+',        # "1.1", "2.3"
    r'^\d+\.[a-z]\)',    # "1.a)", "2.b)"
    r'^\([a-z]\)',       # "(a)", "(b)"
    r'^[A-Z]\.',         # "A.", "B."
]
```

---

## 📋 NEXT SESSION STARTUP PROMPT (UPDATED: Feb 26, 2026)

**CURRENT STATUS**: ✅ Priority 1 & 2 COMPLETE (Commit: 626313f)

When you open this file tomorrow, just tell me:

```
Phase 6 devam et
```

I will automatically:
1. Run full baseline test to validate Priority 2 improvements (~10-15 min)
2. Analyze results: Did CONFIG timeout tests pass? (target: 87.5% pass rate)
3. If successful: Move to Priority 3 (Missing terms fix)
4. If not: Debug and adjust Priority 2 settings
5. Continue with Priority 3 & 4 as planned

---

## 🎯 EXPECTED RESULTS

| Phase | Pass Rate | Improvement | Duration |
|-------|-----------|-------------|----------|
| **Baseline** | 88% | - | - |
| Priority 1 complete | **93-95%** | +5-7% | 2-3 hours |
| Priority 2 complete | **95-97%** | +2-3% | 2-3 hours |
| Priority 3 complete | **96-98%** | +1-2% | 1-2 hours |
| Priority 4 complete | **97-99%** | +1% | 1 hour |
| **TOTAL** | **95-99%** | +7-11% | **6-9 hours** |

---

## 📊 TEST PLAN

After each priority:
```bash
# Full regression test
./scripts/run_baseline_test.sh

# Save results
cp test_results/baseline_test_*.json test_results/phase6_priority1_results.json

# Calculate pass rate
python -c "import json; \
  results = json.load(open('test_results/phase6_priority1_results.json')); \
  total = len(results); \
  passed = sum(1 for r in results if r['passed']); \
  print(f'Pass Rate: {passed}/{total} = {passed/total*100:.1f}%')"
```

---

## 🚨 COMMIT PLAN

After each priority:
```bash
git add .
git commit -m "feat(phase6): Priority X complete - Y% pass rate

- Changed: ...
- Test results: X/25 passing
- Performance: ...

✅ All tests passing"
```

---

## 💡 NOTES

1. **Commit**: Commit after each priority (easy rollback)
2. **Test coverage**: Run relevant tests after each change
3. **Backup**: Backup database and vectordb before Priority 1
4. **Performance**: Measure response time with each change (target: <20s)

---

---

## 📝 SESSION SUMMARY: February 26, 2026

### ✅ COMPLETED TODAY

#### Priority 1: Retrieval & Confidence Tuning ✅
- **Duration**: ~2 hours
- **Commit**: 626313f
- **Test Result**: 80% pass rate (32/40)

**Changes**:
1. ✅ Dynamic RRF Weights ([hybrid_search.py](src/llm/hybrid_search.py#L506-L516))
   - Error codes → BM25 dominant (0.6/0.4)
   - Troubleshooting → Balanced (0.45/0.55)
   - General → Semantic dominant (0.4/0.6)

2. ✅ Intent-Based Confidence Thresholds ([confidence_scorer.py](src/llm/confidence_scorer.py#L44-L70))
   - Narrow-scope intents: 0.5 → 0.4 threshold
   - Affects: compatibility, connection, specification, calibration, configuration

3. ✅ RAG_TOP_K Increase ([ai_settings.py](config/ai_settings.py#L161))
   - Chunk retrieval: 5 → 7 chunks
   - Better multi-step procedure coverage

**Results**:
- ✅ Error code: 100% (4/4)
- ✅ Calibration: 100% (2/2)
- ✅ Procedure: 100% (2/2)
- ❌ Configuration: 0% (0/3) - TIMEOUT (>60s)
- ⚠️  Compatibility: 67% (2/3)
- ⚠️  Connection: 75% (3/4)

---

#### Priority 2: LLM Configuration & Timeout Prevention ✅
- **Duration**: ~1.5 hours
- **Commit**: 626313f (same commit)
- **Manual Test**: SUCCESSFUL (27s, no timeout)

**Changes**:
1. ✅ Response Length Limiting ([prompts.py](src/llm/prompts.py#L232-L270))
   - New CONFIGURATION_SYSTEM_PROMPT_EN
   - "Maximum 4 steps, be concise" rules
   - All prompts updated with brevity requirements

2. ✅ Metadata Overhead Cleanup ([context_optimizer.py](src/llm/context_optimizer.py#L36-L60))
   - Context metadata: 50% reduction
   - Removed: heading_text, section_type details
   - Kept: source name, safety warnings

**Manual Test Results**:
```
Query: "How to configure pset parameters for torque range 5-10 Nm?"
- Response time: 27s (was >60s) ✅
- Intent: "configuration" ✅
- Confidence: "high" ✅
- Timeout: NO ✅
- Response length: 353 words (acceptable, not timeout risk)
```

**Expected Impact**:
- CONFIG tests should now pass (0/3 → 3/3)
- Target pass rate: 80% → 87.5% (35/40)
- NEEDS VALIDATION: Full baseline test tomorrow

---

### 🎯 NEXT SESSION TASKS (February 27, 2026)

#### Task 1: Validate Priority 2 Improvements ⏱️ 15-20 min
```bash
# Run full baseline test
./scripts/run_baseline_test.sh --save

# Expected results:
# - Pass rate: 80% → 87.5% (35/40)
# - CONFIG tests: 0/3 → 3/3 ✅
# - Avg response time: 9.3s → ~7-8s

# If successful → Move to Priority 3
# If not → Debug CONFIG tests, check logs
```

#### Task 2: Priority 3 - Prompt Hardening ⏱️ 1-2 hours
**Target**: Fix 4 "missing terms" tests
- TROUBLE_002: Missing 'temperature'
- SPEC_003: Missing 'rpm', 'speed'
- CONN_002: Missing 'network'
- MAINT_001: Missing 'lubrication'

**Changes Needed**:
1. Explicit unit preservation in prompts
2. COMPATIBILITY_MATRIX expansion

**Expected Impact**: +2-3% (4 tests fixed)

#### Task 3: Priority 4 - Adaptive Chunking ⏱️ 1 hour
**Target**: Fix remaining edge cases
- Chunk overlap increase
- Step-preserving regex strengthening

**Expected Impact**: +1% (1-2 tests)

---

### 📊 PROGRESS TRACKER

| Priority | Status | Pass Rate | Duration | Commit |
|----------|--------|-----------|----------|--------|
| **Priority 1** | ✅ DONE | 80% (32/40) | 2h | 626313f |
| **Priority 2** | ✅ DONE | 80%* (needs validation) | 1.5h | 626313f |
| **Priority 3** | ⏳ TODO | Target: 90% (36/40) | 1-2h | - |
| **Priority 4** | ⏳ TODO | Target: 92.5% (37/40) | 1h | - |
| **TOTAL** | 50% | **80% → 92.5% (target)** | **5.5h / 6-9h** | - |

*Priority 2 needs full baseline test validation tomorrow

---

### 🚀 QUICK START COMMAND (Tomorrow)

```bash
# Just say this:
Phase 6 devam et

# I will:
# 1. Run baseline test (15 min)
# 2. Show results comparison
# 3. Move to Priority 3 or debug if needed
```

---

**Last Updated**: February 26, 2026, 19:50
**Session by**: Claude (AI Assistant)
**Status**: ✅ Priority 1 & 2 Complete, Ready for validation test
**Next Session**: Priority 2 validation → Priority 3 → Priority 4
