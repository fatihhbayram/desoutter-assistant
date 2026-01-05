# 🗺️ Project Roadmap

Strategic development plan for the Desoutter Assistant project.

---

## 🎯 Vision

Build an **enterprise-grade AI support system** that enables technicians to diagnose and repair industrial tools faster, with verified accuracy, while continuously learning from real-world feedback.

---

## 📊 Current Status

**Phase**: Production-Ready RAG System with Self-Learning

| Metric | Value |
|--------|-------|
| Test Pass Rate | 96% (24/25 scenarios) |
| Products Indexed | 451 |
| Documents Processed | 541 |
| Freshdesk Tickets | 2,249 |
| Semantic Chunks | ~26,528 |
| BM25 Terms | 19,032 |
| Domain Terms | 351 |

---

## ✅ Completed Milestones

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
| Dec 27 | Connection Architecture | 6 product family mappings |
| Dec 22 | Domain Embeddings | 351 Desoutter-specific terms |
| Dec 22 | Performance Metrics | Query latency, cache hit rates, health monitoring |
| Dec 22 | Multi-turn Conversation | Session management with context preservation |
| Dec 22 | Self-Learning Loop | Wilson score ranking, source learning |
| Dec 18 | ProductModel Schema v2 | Comprehensive product categorization |
| Dec 17 | Source Relevance Feedback | Per-document relevance UI |
| Dec 17 | Context Optimization | 8K token budget management |
| Dec 17 | Metadata Boosting | Service bulletin prioritization |
| Dec 17 | GPU Activation | NVIDIA RTX A2000 inference |
| Dec 16 | Hybrid Search | BM25 + Semantic + RRF fusion |
| Dec 16 | Response Caching | ~100,000x speedup |
| Dec 15 | Semantic Chunking | Structure-aware document processing |
| Dec 14 | TechWizard UI | 4-step wizard interface |
| Dec 09 | Admin Dashboard | Analytics and feedback statistics |
| Dec 09 | Feedback System | Self-learning RAG with user feedback |
| Dec 04 | Document Viewer | Multi-format document support |

#### November 2025

| Date | Milestone | Description |
|------|-----------|-------------|
| Nov 28 | Authentication | JWT-based login with role management |
| Nov 27 | RAG Engine | PDF processing, ChromaDB integration |
| Nov 26 | Ollama Integration | Local LLM with GPU support |
| Nov 22 | Project Start | Initial setup, scraper, MongoDB |

---

## 🔄 In Progress

### January 2026

- [x] **Intelligent Product Filtering** ✅ (Jan 5)
  - Pattern-based `IntelligentProductExtractor` (40+ regex patterns)
  - ChromaDB `where` clause filtering at query time
  - CVI3 query → Only CVI3 documents returned
  - 91.7% test pass rate

- [x] **Freshdesk Ticket Integration** ✅
  - Scraped 2,249 tickets from support portal
  - Processed PDF attachments
  - Ingested Q&A pairs into RAG
  - Status: Complete

- [ ] **Controller Units Scraping**
  - 10 controller units to add (AXON, CVIR II, CVIxS, etc.)
  - Update connection architecture mapping

---

## 📋 Planned Features

### Short-term (Q1 2026)

| Priority | Feature | Description | Effort |
|----------|---------|-------------|--------|
| 🔴 High | Cross-Encoder Re-ranking | Top-10 → Top-5 semantic re-ranking | 2 days |
| 🔴 High | Confidence Score API | Numeric confidence in responses | 1 day |
| 🟡 Medium | TechWizard Integration | Deploy wizard UI to production | 1 day |
| 🟡 Medium | Missing Series Scrape | 11 remaining product series | 1 day |
| 🟢 Low | Admin UI Redesign | Simplified document management | 2 days |

### Medium-term (Q2 2026)

| Priority | Feature | Description | Effort |
|----------|---------|-------------|--------|
| 🔴 High | Service Management System | Service requests, tracking, workflow | 2 weeks |
| 🔴 High | KPI Dashboard | Supervisor/Manager analytics | 1 week |
| 🟡 Medium | User Profiles | Expertise-based response customization | 3 days |
| 🟡 Medium | Embedding Fine-tuning | Domain-specific embeddings | 1 week |
| 🟢 Low | Vision AI | Photo-based fault detection | 2 weeks |

### Long-term (H2 2026)

| Priority | Feature | Description |
|----------|---------|-------------|
| 🔴 High | Mobile PWA | Progressive Web App for field use |
| 🟡 Medium | SAP Integration | Automatic spare parts ordering |
| 🟡 Medium | Predictive Maintenance | Failure prediction system |
| 🟢 Low | Voice Assistant | Hands-free fault reporting |
| 🟢 Low | Offline Mode | Local inference for disconnected use |

---

## 🏗️ Service Management System (Planned)

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

## 📈 Success Metrics

### Technical Metrics

| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| Test Pass Rate | 96% | >95% | Maintain |
| Response Time | 2.4ms (cached) | <3s | Maintain |
| Hallucination Rate | <5% | <3% | Q1 2026 |
| "I don't know" Rate | 12% | 10-15% | Maintain |

### Business Metrics (Future)

| Metric | Target | Timeline |
|--------|--------|----------|
| User Satisfaction | >4.5/5.0 | Q2 2026 |
| First Contact Resolution | >70% | Q2 2026 |
| Average Diagnosis Time | <2 min | Q2 2026 |
| Document Coverage | >90% products | Q1 2026 |

---

## 🤝 Contributing to Roadmap

We welcome feature suggestions and contributions!

### How to Suggest Features

1. Open a GitHub Issue with the `enhancement` label
2. Describe the use case and expected benefit
3. Propose a rough implementation approach

### How to Contribute

1. Check the "In Progress" section for available work
2. Comment on the issue to claim it
3. Submit a PR following our contribution guidelines

---

## 📚 Related Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Project overview |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [RAG_QUALITY_IMPROVEMENT.md](RAG_QUALITY_IMPROVEMENT.md) | Technical RAG details |
| [QUICKSTART.md](QUICKSTART.md) | Quick setup guide |
| [PROXMOX_DEPLOYMENT.md](PROXMOX_DEPLOYMENT.md) | Infrastructure guide |

---

*Last Updated: January 5, 2026*
