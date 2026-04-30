# Project Roadmap

Strategic development plan for the Desoutter Assistant project.

---

## Vision

Build an **enterprise-grade AI support system** that enables technicians to diagnose and repair industrial tools faster, with verified accuracy, while continuously learning from real-world feedback.

---

## Current Status

**Phase**: Phase 7 — Evaluation Pipeline (April 2026)

| Metric | Value |
|--------|-------|
| Test Pass Rate | 85% (34/40 scenarios) |
| Timeout Rate | 0% (all resolved) |
| Products Indexed | 451 |
| Documents Processed | 547 (121 PDF + 426 Word) |
| Vector DB | Qdrant — 6,195 chunks (384-dim, language-filtered) |
| Q&A Evaluation Dataset | 459 real-world field support Q&A pairs |
| BM25 Terms | 19,032 |
| Intent Types | 15 (expanded from 8) |

---

## Completed Milestones

### Q1 2026 - Phase 6: Tuning & Accuracy

#### March 2026

| Date | Milestone | Description |
|------|-----------|-------------|
| Mar 15 | **Phase 6 Completion** | 85% pass rate accepted as production-ready |
| Mar 15 | Timeout Elimination | All 3 timeouts resolved (7.5% → 0%) |
| Mar 15 | Test Suite Quality | Fake codes fixed, 40 scenarios, realistic expectations |
| Mar 15 | Prompt Hardening | KEY TERMS REPETITION rule added |
| Mar 15 | 9 Perfect Categories | Troubleshooting, Specs, Config, Calib at 100% |

#### January 2026

| Date | Milestone | Description |
|------|-----------|-------------|
| Jan 5 | **Intelligent Product Filtering** | Pattern-based extraction, Qdrant filtering, 91.7% test pass |
| Jan 5 | **Intent Expansion** | 15 intent types (from 8) - added config, compat, procedure, firmware, comparison, capability, accessory |
| Jan 4 | **Freshdesk Integration Complete** | 2,249 tickets scraped, processed, and ingested |

### Q4 2025 - Core RAG System

#### December 2025

| Date | Milestone | Description |
|------|-----------|-------------|
| Dec 31 | Source Citation Enhancement | Page number extraction, product metadata enrichment |
| Dec 30 | Intent Detection | 8 intent types with specialized prompts |
| Dec 30 | Content Deduplication | SHA-256 hash-based duplicate prevention |
| Dec 30 | Adaptive Chunking | Document type-based chunk sizing |
| Dec 29 | Context Grounding | "I don't know" logic with context scoring |
| Dec 29 | Response Validation | Hallucination detection and prevention |
| Dec 27 | Relevance Filtering | 15 fault categories with keyword filtering |
| Dec 22 | Performance Metrics | Query latency, cache hit rates, health monitoring |
| Dec 22 | Multi-turn Conversation | Session management with context preservation |
| Dec 22 | Self-Learning Loop | Wilson score ranking, source learning |
| Dec 17 | Context Optimization | 8K token budget management |
| Dec 17 | Metadata Boosting | Service bulletin prioritization |
| Dec 17 | GPU Activation | NVIDIA RTX A2000 inference |
| Dec 16 | **Hybrid Search** | BM25 + Semantic + RRF fusion |
| Dec 16 | Response Caching | LRU + TTL cache (~100,000x speedup) |
| Dec 15 | Semantic Chunking | Structure-aware document processing |
| Dec 14 | TechWizard UI | 4-step wizard interface |
| Dec 09 | Admin Dashboard | Analytics and feedback statistics |
| Dec 09 | Feedback System | Self-learning RAG with user feedback |

#### November 2025

| Date | Milestone | Description |
|------|-----------|-------------|
| Nov 28 | Authentication | JWT-based login with role management |
| Nov 27 | **RAG Engine** | PDF processing, Qdrant integration |
| Nov 26 | Ollama Integration | Local LLM with GPU support |
| Nov 22 | Project Start | Initial setup, scraper, MongoDB |

---

## In Progress

### Q2 2026 - Knowledge Base Enrichment

**Status**: Active Development

- [x] **Phase 8: Basic Troubleshooting Knowledge Base** *(2026-04-26)*
  - 6 procedure_guide documents ingested (Motor/Battery/Connectivity/Memory/Drive/Software)
  - `is_generic=True` docs bypass cross-product exclusion in retrieval
  - `procedure_guide` type gets 2.0x score boost

- [x] **Phase 7: Evaluation Pipeline** *(2026-04-30)*
  - 459 real-world field Q&A pairs built from field support cases
  - `evaluate_rag.py` — keyword overlap scoring against `/diagnose` endpoint
  - Score boost rebalancing: `service_bulletin` 4.0x → 2.5x, `technical_manual` 1.5x added
  - Early results: Good 60%, Partial 20%, Fail 20% (5-question sample)

- ~~**Faz 4: product_model support in /diagnose**~~ — *Cancelled: frontend already resolves model name to part number via search autocomplete*

---

## Planned Features

### Short-term (Q2 2026)

| Priority | Feature | Description | Effort |
|----------|---------|-------------|--------|
| 🔴 High | Cross-Encoder Re-ranking | Top-10 → Top-5 semantic re-ranking | 2 days |
| 🔴 High | Confidence Score API | Numeric confidence in responses | 1 day |
| 🟡 Medium | TechWizard Integration | Deploy wizard UI to production | 1 day |
| 🟡 Medium | Controller Units Scrape | 10 remaining product series | 1 day |
| 🟢 Low | Admin UI Redesign | Simplified document management | 2 days |

### Medium-term (Q3-Q4 2026)

| Priority | Feature | Description | Effort |
|----------|---------|-------------|--------|
| 🔴 High | Service Management System | Service requests, tracking, workflow | 2 weeks |
| 🔴 High | KPI Dashboard | Supervisor/Manager analytics | 1 week |
| 🔴 High | Field Installation Support | Extend beyond repair to on-site setup | 1 week |
| 🟡 Medium | User Profiles | Expertise-based response customization | 3 days |
| 🟡 Medium | Embedding Fine-tuning | Domain-specific embeddings | 1 week |
| 🟡 Medium | Vision AI | Photo-based fault detection | 2 weeks |
| 🟢 Low | Prompt Caching | 40% latency reduction | 3 days |
| 🟢 Low | Async Ingestion Queue | Celery + Redis background processing | 1 week |

### Long-term (2027+)

| Priority | Feature | Description |
|----------|---------|-------------|
| 🔴 High | Mobile PWA | Progressive Web App for field use |
| 🔴 High | Qdrant Scaling | Scale to 100M+ vectors for multi-tenant |
| 🟡 Medium | SAP Integration | Automatic spare parts ordering |
| 🟡 Medium | Predictive Maintenance | Failure prediction system |
| 🟢 Low | Voice Assistant | Hands-free fault reporting |
| 🟢 Low | Offline Mode | Local inference for disconnected use |

---

## Service Management System (Planned Q3 2026)

### Overview

A comprehensive service management module to track repairs, manage customers, and analyze performance.

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    SERVICE MANAGEMENT SYSTEM                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Device    │  │   Service   │  │       Customer          │ │
│  │  Registry   │──│   Requests  │──│      Management         │ │
│  │             │  │             │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│         │                │                      │               │
│         └────────────────┼──────────────────────┘               │
│                          │                                      │
│                          ▼                                      │
│                  ┌───────────────┐                              │
│                  │ KPI Dashboard │                              │
│                  │  • Metrics    │                              │
│                  │  • Reports    │                              │
│                  │  • Analytics  │                              │
│                  └───────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Database Schema (Planned)

#### devices Collection
```javascript
{
  serial_number: String,      // Unique device identifier
  part_number: String,        // Product part number
  model_name: String,         // Model name
  customer: { id, name, contact },
  purchase_date: Date,
  warranty_end_date: Date,
  service_history: [String],  // Service request IDs
  status: "active" | "retired" | "lost"
}
```

#### service_requests Collection
```javascript
{
  request_id: String,         // SR-YYYYMMDD-XXX
  device: { serial_number, part_number, model_name },
  customer: { id, name, reference },
  service_type: "SMART_CARE" | "BASIC_CARE" | "REPAIR" | "CALIBRATION",
  warranty_status: "WARRANTY" | "PAID" | "GOODWILL" | "CONTRACT",
  fault_description: String,
  ai_diagnosis: { suggestion, confidence, sources },
  status: "pending" | "in_progress" | "completed" | "delivered",
  assigned_to: String,
  parts_used: [{ part_number, quantity }],
  calibration: { performed, certificate_number }
}
```

### API Endpoints (Planned)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/devices | List devices |
| POST | /api/devices | Register device |
| GET | /api/services | List service requests |
| POST | /api/services | Create service request |
| PUT | /api/services/{id}/status | Update status |
| GET | /api/kpi/overview | Dashboard metrics |
| GET | /api/reports/export | Export to Excel/PDF |

### Role Matrix (Planned)

| Feature | Admin | Manager | Supervisor | Technician |
|---------|:-----:|:-------:|:----------:|:----------:|
| Create Service | ✅ | ✅ | ✅ | ✅ |
| Edit Any Service | ✅ | ✅ | ✅ | ❌ |
| Delete Service | ✅ | ✅ | ❌ | ❌ |
| View KPI Dashboard | ✅ | ✅ | ✅ | ❌ |
| User Management | ✅ | ❌ | ❌ | ❌ |
| Revenue Reports | ✅ | ✅ | ❌ | ❌ |

---

## Success Metrics

### Technical Metrics

| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| Test Pass Rate | 85% | >85% | Maintain |
| Response Time | 23.6s (non-cached) | <20s | Q2 2026 |
| Cache Hit Rate | High | >60% | Q2 2026 |
| Hallucination Rate | <2% | <1% | Q3 2026 |
| "I don't know" Rate | 12% | 10-15% | Maintain |

### Business Metrics (Future)

| Metric | Target | Timeline |
|--------|--------|----------|
| User Satisfaction | >4.5/5.0 | Q3 2026 |
| First Contact Resolution | >70% | Q3 2026 |
| Average Diagnosis Time | <2 min | Q3 2026 |
| Document Coverage | >90% products | Q2 2026 |

---

## Contributing to Roadmap

We welcome feature suggestions and contributions!

### How to Suggest Features

1. Open a GitHub Issue with the `enhancement` label
2. Describe the use case and expected benefit
3. Propose a rough implementation approach

### How to Contribute

1. Check the "Planned Features" section for available work
2. Comment on the issue to claim it
3. Submit a PR following our contribution guidelines

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Project overview |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [QUICKSTART.md](QUICKSTART.md) | Quick setup guide |
| [PROXMOX_DEPLOYMENT.md](PROXMOX_DEPLOYMENT.md) | Infrastructure guide |

---

*Last Updated: April 30, 2026*
