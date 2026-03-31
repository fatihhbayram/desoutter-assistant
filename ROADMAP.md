# Desoutter Assistant — Project Roadmap

Strategic development plan for the Desoutter Assistant project.

---

## Vision

Build an **enterprise-grade, safety-critical AI support system** that enables industrial tool technicians to diagnose faults, access procedures, and verify compatibility — with verified accuracy and continuous learning from real-world feedback.

> **Core principle:** Safety first. Completeness second. Performance third.

---

## Current Status

**Version:** 2.0.1 — Production Ready  
**Last Updated:** March 31, 2026  
**Active Development:** Service Management System (Q3 2026)

| Metric | Value |
|--------|-------|
| Test Pass Rate | 85% (34/40 scenarios) |
| Timeout Rate | 0% |
| Products Indexed | 451 |
| Documents Processed | 541 (121 PDF + 420 DOCX) |
| Freshdesk Tickets Ingested | 2,249 |
| Vector DB Chunks | 26,513 (Qdrant, 384-dim) |
| BM25 Unique Terms | 19,032 |
| Intent Types | 15 (multi-label) |
| Supported Languages | Turkish, English |

---

## Completed Milestones

### 🏆 Phase 7 — El-Harezmi Pipeline + Qdrant (v2.0.0) — February 2026

A complete architectural overhaul: the monolithic 14-stage RAG pipeline was replaced with the **5-stage El-Harezmi architecture**, and ChromaDB was migrated to **Qdrant**.

| Date | Milestone | Description |
|------|-----------|-------------|
| Feb 18 | **El-Harezmi 5-Stage Pipeline** | `stage1_intent_classifier`, `stage2_retrieval_strategy`, `stage3_info_extraction`, `stage4_kg_validation`, `stage5_response_formatter` |
| Feb 18 | **Qdrant Migration** | ChromaDB → Qdrant v1.7.4; `desoutter_docs_v2` collection; dense + sparse hybrid vectors |
| Feb 18 | **Adaptive Chunking** | 8 document types → 6 strategies: SemanticChunker, TableAwareChunker, EntityChunker, ProblemSolutionChunker, StepPreservingChunker, HybridChunker |
| Feb 18 | **15 Intent Types** | Expanded from 8 → 15 multi-label intents with entity extraction |
| Feb 18 | **Metadata Enrichment** | In-place enrichment of 26,513 Qdrant points (document_type, chunk_type, intent_relevance, error_code, esde_code…) |
| Feb 18 | **Test Suite Expansion** | 25 → 40 test scenarios, 13 Turkish queries |

---

### ✅ Phase 6 — Tuning & Accuracy (v2.0.1) — March 2026

| Date | Milestone | Description |
|------|-----------|-------------|
| Mar 15 | **Timeout Elimination** | All 3 timeout scenarios resolved (7.5% → 0%) |
| Mar 15 | **Prompt Hardening** | KEY TERMS REPETITION rule added to EN/TR system prompts |
| Mar 15 | **Test Quality Improvements** | Fake error codes removed, realistic expectations set |
| Mar 15 | **85% Pass Rate Accepted** | 9 categories at 100%; phase declared production-ready |
| Mar 24 | **Security Audit** | White-hat Red Team assessment; vulnerabilities documented |

---

### ✅ Phase 5 — Production Features (v1.7.0 – v1.8.0) — January 2026

| Date | Milestone | Description |
|------|-----------|-------------|
| Jan 7 | Triple-Path Retrieval | Dedicated bulletin path + general + controller docs |
| Jan 7 | Dynamic Query Expander | Vector DB-based term expansion (replaced hardcoded synonyms) |
| Jan 6 | General RAG Quality | Error code boost, phrase/bigram matching, bulletin deduplication |
| Jan 5 | **Intelligent Product Filtering** | 40+ regex patterns, Qdrant `where` clause filtering |
| Jan 5 | **Intent Expansion (8 → 15)** | Config, Compat, Procedure, Firmware, Comparison, Capability, Accessory |
| Jan 4 | **Freshdesk Integration** | 2,249 tickets scraped, processed, and indexed |

---

### ✅ Phase 1–4 — Core RAG System (v0.1.0 – v1.6.0) — Nov–Dec 2025

| Version | Date | Highlights |
|---------|------|------------|
| v1.6.0 | Dec 31 | Source citation, product metadata enrichment |
| v1.5.0 | Dec 30 | Intent detection (8 types), content deduplication |
| v1.4.0 | Dec 29 | Context grounding ("I don't know" logic), response validation |
| v1.3.0 | Dec 27 | Relevance filtering, connection architecture mapping |
| v1.2.0 | Dec 22 | Domain embeddings, self-learning, performance metrics |
| v1.1.0 | Dec 17 | Hybrid Search (BM25 + Semantic + RRF), response caching |
| v1.0.0 | Dec 15 | Semantic chunker, TechWizard UI, Admin dashboard, GPU |
| v0.1.0 | Nov 22 | Initial release: FastAPI, MongoDB, Ollama, RAG, JWT auth |

---

## In Progress

### 🔄 Q2 2026 — Pipeline Stabilisation & Optimisation

| Priority | Task | Status |
|----------|------|--------|
| 🔴 High | A/B test El-Harezmi vs legacy RAG side-by-side | Pending |
| 🔴 High | Turkish prompt improvements (COMPAT_001 intent mismatch) | In Progress |
| 🟡 Medium | Cross-Encoder re-ranking (Top-20 → Top-5 semantic re-ranking) | Planned |
| 🟡 Medium | Controller unit documentation scraping (10 remaining series) | In Progress |
| 🟢 Low | Admin UI redesign — simplified document management | Planned |

---

## Planned Features

### Short-term (Q2 2026)

| Priority | Feature | Description | Effort |
|----------|---------|-------------|--------|
| 🔴 High | Cross-Encoder Re-ranking | ColBERT / bi-encoder re-ranking on top-K results | 2 days |
| 🔴 High | Confidence Score in API | Numeric confidence exposed on every response | 1 day |
| 🔴 High | Rate Limiting | Nginx or FastAPI middleware request throttling | 1 day |
| 🟡 Medium | TechWizard Production Deploy | Ship wizard UI to production environment | 1 day |
| 🟡 Medium | CORS Hardening | Restrict `allow_origins` to known domains | 0.5 day |
| 🟡 Medium | Controller Units Scrape | Remaining 10 controller product series | 1 day |
| 🟢 Low | Admin UI Redesign | Cleaner document management interface | 2 days |

### Medium-term (Q3–Q4 2026)

| Priority | Feature | Description | Effort |
|----------|---------|-------------|--------|
| 🔴 High | **Service Management System** | End-to-end service request lifecycle — see detail below | 2 weeks |
| 🔴 High | **KPI Dashboard** | Supervisor/manager analytics with charts | 1 week |
| 🔴 High | Field Installation Support | Extend AI guidance to on-site installation procedures | 1 week |
| 🟡 Medium | User Profiles | Expertise-level customisation per user role | 3 days |
| 🟡 Medium | Embedding Fine-tuning | Domain-specific model training on Desoutter corpus | 1 week |
| 🟡 Medium | Vision AI | Photo-based fault detection (camera upload) | 2 weeks |
| 🟡 Medium | Async Ingestion Queue | Celery + Redis background document processing | 1 week |
| 🟢 Low | Prompt Caching | 40% latency reduction via response memoisation | 3 days |

### Long-term (2027+)

| Priority | Feature | Description |
|----------|---------|-------------|
| 🔴 High | Mobile PWA | Progressive Web App for field technicians |
| 🔴 High | Qdrant Horizontal Scaling | Scale to 100 M+ vectors for multi-tenant deployment |
| 🟡 Medium | SAP Integration | Automatic spare parts ordering from diagnosis results |
| 🟡 Medium | Predictive Maintenance | Failure prediction from historical service patterns |
| 🟢 Low | Voice Assistant | Hands-free fault reporting for workshop use |
| 🟢 Low | Offline Mode | Local inference for disconnected / air-gapped environments |

---

## Planned: Service Management System (Q3 2026)

A comprehensive module to track repairs, manage customer devices, and generate KPI reports.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  SERVICE MANAGEMENT SYSTEM                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │   Device     │  │   Service    │  │    Customer       │ │
│  │  Registry    │──│   Requests   │──│   Management      │ │
│  └──────────────┘  └──────────────┘  └───────────────────┘ │
│         │                │                    │             │
│         └────────────────┼────────────────────┘             │
│                          │                                  │
│                          ▼                                  │
│                  ┌───────────────┐                          │
│                  │ KPI Dashboard │                          │
│                  │ • Metrics     │                          │
│                  │ • Reports     │                          │
│                  │ • Analytics   │                          │
│                  └───────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### Database Schema (Planned)

#### `devices` Collection

```javascript
{
  serial_number: String,       // Unique device identifier
  part_number: String,         // Product part number
  model_name: String,          // Model name
  customer: { id, name, contact },
  purchase_date: Date,
  warranty_end_date: Date,
  service_history: [String],   // Service request IDs
  status: "active" | "retired" | "lost"
}
```

#### `service_requests` Collection

```javascript
{
  request_id: String,          // SR-YYYYMMDD-XXX
  device: { serial_number, part_number, model_name },
  customer: { id, name, reference },
  service_type: "SMART_CARE" | "BASIC_CARE" | "REPAIR" | "CALIBRATION",
  warranty_status: "WARRANTY" | "PAID" | "GOODWILL" | "CONTRACT",
  fault_description: String,
  ai_diagnosis: { suggestion, confidence, sources, intent },
  status: "pending" | "in_progress" | "completed" | "delivered",
  assigned_to: String,
  parts_used: [{ part_number, quantity }],
  calibration: { performed, certificate_number }
}
```

### API Endpoints (Planned)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/devices` | List registered devices |
| POST | `/api/devices` | Register new device |
| GET | `/api/services` | List service requests |
| POST | `/api/services` | Create service request |
| PUT | `/api/services/{id}/status` | Update service status |
| GET | `/api/kpi/overview` | Dashboard KPIs |
| GET | `/api/reports/export` | Export to Excel / PDF |

### Role Matrix (Planned)

| Feature | Admin | Manager | Supervisor | Technician |
|---------|:-----:|:-------:|:----------:|:----------:|
| Create service request | ✅ | ✅ | ✅ | ✅ |
| Edit any service | ✅ | ✅ | ✅ | ❌ |
| Delete service | ✅ | ✅ | ❌ | ❌ |
| View KPI dashboard | ✅ | ✅ | ✅ | ❌ |
| User management | ✅ | ❌ | ❌ | ❌ |
| Revenue reports | ✅ | ✅ | ❌ | ❌ |

---

## Success Metrics

### Technical Targets

| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| Test Pass Rate | 85% | ≥ 90% | Q3 2026 |
| Avg Response Time (non-cached) | 23.6 s | < 15 s | Q3 2026 |
| Cache Hit Rate | High | > 60% | Q2 2026 |
| Hallucination Rate | < 2% | < 1% | Q3 2026 |
| Compatibility Query Accuracy | ~67% | ≥ 90% | Q2 2026 |
| "I don't know" Rate | 12% | 10–15% | Maintain |

### Business Targets

| Metric | Target | Timeline |
|--------|--------|----------|
| User Satisfaction | > 4.5 / 5.0 | Q3 2026 |
| First Contact Resolution | > 70% | Q3 2026 |
| Average Diagnosis Time | < 2 min | Q3 2026 |
| Document Coverage | > 90% of products | Q2 2026 |

---

## Contributing to the Roadmap

We welcome feature suggestions and contributions.

### How to Suggest Features

1. Open a GitHub Issue with the `enhancement` label
2. Describe the use case and expected benefit
3. Propose a rough implementation approach

### How to Contribute

1. Check the "Planned Features" section for available work
2. Comment on the GitHub Issue to claim it
3. Submit a PR following the contribution guidelines in [README.md](README.md)

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Full project overview |
| [CHANGELOG.md](CHANGELOG.md) | Full version history (v0.1.0 → v2.0.1) |
| [QUICKSTART.md](QUICKSTART.md) | Get up and running in < 10 minutes |
| [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) | Mermaid diagrams for presentations |
| [PROXMOX_DEPLOYMENT.md](PROXMOX_DEPLOYMENT.md) | Production infrastructure guide |

---

*Last Updated: March 31, 2026*
