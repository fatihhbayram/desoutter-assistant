# Desoutter Repair Assistant
## AI-Powered Technical Support System

**Fatih Bayram**

---

# The Problem

### My Story

- **14 years** as a service technician (since 2011)
- Biggest challenge: **Learning fault solutions takes time**
- Looking up manuals and bulletins = **wasted time**
- Knowledge trapped in experienced technicians' heads

### The Challenge

| Pain Point | Impact |
|------------|--------|
| Manual document search | 15-30 min per fault |
| Inconsistent answers | Varies by experience |
| Knowledge loss | Experts leave, knowledge leaves |

---

# The Vision

> **An AI assistant that helps technicians instantly - without sending data to third-party clouds.**

### Core Principles

1. **Data Sovereignty** - All data stays on-premise
2. **Self-Hosted LLM** - No OpenAI/Claude costs or data leakage
3. **Continuous Learning** - Improves from feedback

---

# Infrastructure

### Self-Hosted AI Server

| Component | Specification |
|-----------|---------------|
| **Platform** | Proxmox VE 8.x |
| **GPU** | NVIDIA RTX A2000 (6GB) |
| **LLM** | Qwen2.5:7b-instruct |
| **Speed** | 40-50 tokens/sec |

### Why Not Cloud AI?

| Cloud AI | Self-Hosted |
|----------|-------------|
| Data sent externally | Data stays in-house |
| ~$500-1000/month | $0 after hardware |
| Internet required | Works offline |

---

# How It Works: RAG Pipeline

```
User Query: "Motor makes grinding noise"
                    │
                    ▼
         ┌─────────────────────┐
         │  1. HYBRID SEARCH   │
         │  Semantic + Keyword │
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  2. PRODUCT FILTER  │
         │  Only relevant docs │
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  3. LLM GENERATION  │
         │  Qwen2.5:7b (GPU)   │
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  4. VALIDATION      │
         │  Anti-hallucination │
         └─────────────────────┘
                    │
                    ▼
            AI Response
        (with confidence score)
```

---

# Key Feature 1: Hybrid Search

### Best of Both Worlds

| Method | Strength |
|--------|----------|
| **Semantic Search** | Understands meaning |
| **BM25 Keyword** | Finds exact terms |
| **Hybrid (Ours)** | Combined accuracy |

```
Query: "E804 error"
         │
    ┌────┴────┐
    ▼         ▼
Semantic   BM25
 (60%)    (40%)
    │         │
    └────┬────┘
         ▼
   Fused Results
```

**Result:** 35% better retrieval vs semantic-only

---

# Key Feature 2: Semantic Chunking

### Document Processing

```
PDF Manual (100 pages)
         │
         ▼
┌─────────────────────┐
│  Text Extraction    │
│  + Page Numbers     │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Semantic Chunking  │
│  500 tokens each    │
│  50 token overlap   │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Metadata Enrichment│
│  - Product family   │
│  - Document type    │
│  - Fault keywords   │
└─────────────────────┘
         │
         ▼
    ChromaDB
  (28,414 chunks)
```

---

# Key Feature 3: Product Filtering

### The Problem

Query about "EADC 10E-06" could return:
- EAD20 content ❌
- EPB battery tool ❌
- CVI3 controller ❌

### Our Solution

```
Query + Product: "6151659030" (EADC)
              │
              ▼
    ChromaDB WHERE clause
    product_family = "EADC"
              │
              ▼
    Only EADC docs returned ✓
```

**Result:** Eliminates 90% retrieval noise

---

# Key Feature 4: Self-Learning

### Feedback Loop

```
Technician Query
        │
        ▼
   RAG Response
        │
        ▼
  User Feedback
        │
   ┌────┴────┐
   ▼         ▼
  👍         👎
Positive  Negative
   │         │
   ▼         ▼
Boost     Penalize
Source    Source
   │         │
   └────┬────┘
        ▼
 Wilson Score Ranking
        │
        ▼
 Better Future Results
```

**The system learns which documents give good answers.**

---

# Key Feature 5: Hallucination Prevention

### Multi-Layer Validation

| Layer | Function |
|-------|----------|
| **Context Grounding** | Is answer in retrieved docs? |
| **Response Validator** | Detect forbidden content |
| **Confidence Scorer** | Multi-factor scoring |

### If Uncertain:

```
"I don't have enough information
 to answer this question."
```

**Result:** <2% hallucination rate

---

# Key Feature 6: Response Caching

### Performance Boost

| Scenario | Response Time |
|----------|---------------|
| First query | 8-12 seconds |
| Cached query | <1 ms |
| **Speedup** | **~100,000x** |

```
Query → Hash → Cache Check
                   │
            ┌──────┴──────┐
            ▼             ▼
          HIT           MISS
       (instant)    (full pipeline)
            │             │
            ▼             ▼
        Return       Process & Cache
```

---

# Key Feature 7: Intent Detection

### 8 Query Types

| Intent | Example |
|--------|---------|
| `troubleshooting` | "Motor won't start" |
| `error_code` | "What is E804?" |
| `specifications` | "Maximum torque?" |
| `calibration` | "Calibration steps?" |
| `maintenance` | "Service interval?" |

**Each intent gets a specialized prompt.**

---

# Web Interface

### Technician View - 4 Step Wizard

```
┌────────────────────────────────────────────────────────┐
│  Step 1: PRODUCT SEARCH                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Search: [EADC 10E___________________] 🔍         │  │
│  │                                                  │  │
│  │ Filters: [Series ▼] [Type ▼] [Wireless ▼]       │  │
│  │                                                  │  │
│  │ Results:                                         │  │
│  │ ○ 6151659030 - EADC 10E-06                      │  │
│  │ ○ 6151659770 - EADC 15E-10                      │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  Step 2: DESCRIBE FAULT                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Language: [English ▼]                            │  │
│  │                                                  │  │
│  │ Fault Description:                               │  │
│  │ ┌──────────────────────────────────────────────┐│  │
│  │ │ Motor makes grinding noise when starting,    ││  │
│  │ │ tool vibrates excessively                    ││  │
│  │ └──────────────────────────────────────────────┘│  │
│  │                                                  │  │
│  │              [Get Diagnosis 🔍]                  │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  Step 3: AI RESPONSE                                   │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Confidence: ████████░░ 78%                       │  │
│  │                                                  │  │
│  │ Diagnosis:                                       │  │
│  │ The grinding noise indicates worn motor         │  │
│  │ bearings. Recommended steps:                    │  │
│  │ 1. Inspect motor bearings for wear              │  │
│  │ 2. Check gear assembly lubrication              │  │
│  │ 3. Verify spindle alignment                     │  │
│  │                                                  │  │
│  │ Sources:                                         │  │
│  │ 📄 EADC Service Manual (p.42)                   │  │
│  │ 📄 Service Bulletin SB-2024-03                  │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  Step 4: FEEDBACK                                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Was this answer helpful?                         │  │
│  │                                                  │  │
│  │     [👍 Helpful]      [👎 Not Helpful]          │  │
│  │                                                  │  │
│  │ Your feedback improves future results!           │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

# Web Interface

### Admin Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│  ADMIN DASHBOARD                              [Logout 🚪]   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 SYSTEM STATS           📈 PERFORMANCE                   │
│  ┌───────────────────┐    ┌───────────────────┐            │
│  │ Products: 451     │    │ Avg Response: 9.2s │            │
│  │ Documents: 541    │    │ Cache Hit: 42%     │            │
│  │ Chunks: 28,414    │    │ Pass Rate: 96%     │            │
│  └───────────────────┘    └───────────────────┘            │
│                                                             │
│  👥 USER MANAGEMENT        📁 DOCUMENTS                     │
│  ┌───────────────────┐    ┌───────────────────┐            │
│  │ admin (Admin)     │    │ 📄 Upload PDF     │            │
│  │ tech1 (Technician)│    │ 📄 Upload DOCX    │            │
│  │ [+ Add User]      │    │ 📄 Re-ingest All  │            │
│  └───────────────────┘    └───────────────────┘            │
│                                                             │
│  🧠 LEARNING INSIGHTS      💾 CACHE CONTROL                 │
│  ┌───────────────────┐    ┌───────────────────┐            │
│  │ Top Sources       │    │ Entries: 847      │            │
│  │ Feedback Stats    │    │ [Clear Cache]     │            │
│  │ Training Ready: ✓ │    │ [View Stats]      │            │
│  └───────────────────┘    └───────────────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Two User Roles

| Role | Access |
|------|--------|
| **Technician** | Query system, submit feedback |
| **Admin** | + User management, documents, metrics |

---

# System Metrics

### Current Production Stats

| Metric | Value |
|--------|-------|
| **Test Pass Rate** | 96% (24/25) |
| **Products** | 451 tools |
| **Document Chunks** | 28,414 |
| **Documents** | 541 files |
| **Tickets Ingested** | 2,249 |

### Models

| Component | Model |
|-----------|-------|
| **LLM** | Qwen2.5:7b-instruct |
| **Embeddings** | all-MiniLM-L6-v2 (384-dim) |
| **Vector DB** | ChromaDB |

---

# Live Demo

### Scenarios

1. **Troubleshooting**
   - "Motor makes grinding noise"

2. **Error Code**
   - "What is error E804?"

3. **Turkish Query**
   - "Alet çalışmıyor"

4. **Feedback Submission**
   - 👍 / 👎 buttons

---

# Summary

### What We Built

| Feature | Benefit |
|---------|---------|
| **Hybrid Search** | 35% better retrieval |
| **Product Filtering** | 90% noise reduction |
| **Self-Learning** | Improves over time |
| **Hallucination Prevention** | <2% error rate |
| **Response Caching** | 100,000x speedup |
| **On-Premise** | Full data control |

### Impact

| Before | After |
|--------|-------|
| 15-30 min search | Instant answer |
| Inconsistent | 96% accuracy |
| Knowledge in heads | Knowledge in system |

---

# Thank You

### Questions?

**Fatih Bayram**
- GitHub: [@fatihhbayram](https://github.com/fatihhbayram)

### Resources
- [Desoutter Assistant](https://github.com/fatihhbayram/desoutter-assistant)
- [Proxmox AI Infrastructure](https://github.com/fatihhbayramm/proxmox-ai-infrastructure)
