# Desoutter Assistant

> **Enterprise-Grade AI-Powered Technical Support System for Industrial Tool Diagnostics**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2+-61DAFB?logo=react&logoColor=white)](https://reactjs.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-v1.7.4-red?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-85%25%20Passing-success)](test_results/)
[![Version](https://img.shields.io/badge/Version-2.0.2-blueviolet)](CHANGELOG.md)

An intelligent **Retrieval-Augmented Generation (RAG)** system that provides context-aware diagnostics and troubleshooting assistance for Desoutter industrial tools. The system features a **self-learning feedback loop**, a **5-stage El-Harezmi pipeline**, **15 intent types**, **adaptive document chunking**, and a production-grade architecture achieving **85% test pass rate** (34/40 scenarios).

---

## What Does This Project Do?

When a field technician encounters an issue with a Desoutter industrial tool:

1. **Asks a question** — e.g. *"My EPBC8-1800-4Q shows error code E018"*
2. **System classifies intent** — Multi-label classification (15 intent types) with entity extraction
3. **Intent-aware retrieval** — Searches across 26,513 document chunks indexed in Qdrant
4. **Structured extraction** — LLM-based extraction with compatibility validation
5. **Responds** — Structured, source-cited answer with confidence score
6. **Learns** — Continuously improves from explicit user feedback (👍/👎)

```
┌──────────────────┐     ┌──────────────────────────────┐     ┌──────────────────────┐
│    Technician    │────▶│   El-Harezmi 5-Stage RAG      │────▶│  Solution + Sources  │
│   "E018 error"   │     │  (Intent → Retrieve → Extract  │     │  Confidence: 89%     │
│                  │     │   → Validate → Format)         │     │  Stage: TROUBLESHOOT │
└──────────────────┘     └──────────────────────────────┘     └──────────────────────┘
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [System Architecture](#system-architecture)
- [El-Harezmi 5-Stage Pipeline](#el-harezmi-5-stage-pipeline)
- [Technology Stack](#technology-stack)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Performance Metrics](#performance-metrics)
- [Feedback & Self-Learning](#feedback--self-learning)
- [Docker Services](#docker-services)
- [Testing](#testing)
- [Key Files Reference](#key-files-reference)
- [Security Considerations](#security-considerations)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

### Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Docker | 20.10+ | `docker --version` |
| Docker Compose | 2.0+ | `docker compose version` |
| RAM | 16 GB minimum | — |
| NVIDIA GPU | Optional (recommended) | `nvidia-smi` |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/fatihhbayram/desoutter-assistant.git
cd desoutter-assistant

# 2. Configure environment variables
cp .env.example .env
# Edit .env with your settings (MongoDB, Ollama, Qdrant URLs, JWT secret)

# 3. Start all services
docker-compose up -d

# 4. Wait for initialization (~60 seconds on first run)
sleep 60

# 5. Access the application
# → Frontend:  http://localhost:3001
# → API Docs:  http://localhost:8000/docs
# → Health:    http://localhost:8000/health
```

### Default Credentials

| Username | Password | Role | Permissions |
|----------|----------|------|-------------|
| `admin` | `admin123` | Admin | Full system access, user management, document upload |
| `tech` | `tech123` | Technician | Query, conversation, feedback |

> **⚠️ Security**: Always change `JWT_SECRET` and default passwords before any production deployment.

---

## Project Structure

```
desoutter-assistant/
│
├── src/                              # Main application source
│   ├── api/
│   │   ├── main.py                  # FastAPI app — all REST endpoints (~88K)
│   │   └── el_harezmi_router.py     # El-Harezmi dedicated API router
│   │
│   ├── el_harezmi/                  # 5-Stage El-Harezmi Pipeline
│   │   ├── pipeline.py              # Pipeline orchestrator (async)
│   │   ├── stage1_intent_classifier.py   # Multi-label intent + entity extraction
│   │   ├── stage2_retrieval_strategy.py  # Intent-aware Qdrant retrieval
│   │   ├── stage3_info_extraction.py     # LLM-based structured extraction
│   │   ├── stage4_kg_validation.py       # Knowledge graph / compatibility matrix
│   │   ├── stage5_response_formatter.py  # Intent-specific response templates
│   │   └── async_llm_client.py           # Async Ollama LLM wrapper
│   │
│   ├── vectordb/
│   │   └── qdrant_client.py         # Qdrant vector DB operations (hybrid search)
│   │
│   ├── documents/                   # Document ingestion pipeline
│   │   ├── document_classifier.py   # Detects 8 document types
│   │   ├── document_processor.py    # PDF/DOCX/PPTX extraction
│   │   ├── pdf_processor.py         # CID-aware PDF text extraction
│   │   ├── embeddings.py            # all-MiniLM-L6-v2 embedding generation
│   │   ├── product_extractor.py     # 40+ regex pattern product recognition
│   │   └── chunkers/                # 6 adaptive chunking strategies
│   │       ├── base_chunker.py
│   │       ├── semantic_chunker.py
│   │       ├── table_aware_chunker.py
│   │       ├── entity_chunker.py
│   │       ├── problem_solution_chunker.py
│   │       ├── step_preserving_chunker.py
│   │       ├── hybrid_chunker.py
│   │       └── chunker_factory.py
│   │
│   ├── database/
│   │   ├── mongo_client.py          # MongoDB async client
│   │   ├── models.py                # Pydantic models (users, products, diagnoses)
│   │   └── feedback_models.py       # Feedback & learned mapping schemas
│   │
│   ├── scraper/                     # Desoutter product catalog scrapers
│   ├── llm/                         # Legacy RAG engine (14-stage, maintained)
│   └── utils/                       # Shared helpers, logging
│
├── config/
│   ├── ai_settings.py               # All RAG / model parameters
│   ├── feature_flags.py             # Runtime feature toggles
│   ├── relevance_filters.py         # Fault category keyword filters
│   └── tool_controller_compatibility.py  # Hard-coded compatibility matrix
│
├── frontend/                        # React 18 + Vite user interface
│   ├── src/
│   │   ├── App.jsx                  # Main app shell (auth, routing, admin UI)
│   │   └── TechWizard.jsx           # Technician 4-step chat wizard
│   └── package.json
│
├── tests/                           # Test suite (40 scenarios)
│   ├── test_el_harezmi.py           # El-Harezmi pipeline tests
│   ├── test_adaptive_chunking.py    # Chunker strategy tests
│   ├── test_rag_stability.py        # Legacy RAG regression tests
│   ├── test_api_el_harezmi.py       # API-level integration tests
│   └── fixtures/                   # Test queries and expected outcomes
│
├── scripts/                         # Utility and ingestion scripts
│   ├── reingest_adaptive.py         # Adaptive re-ingestion with new chunkers
│   ├── enrich_qdrant_metadata.py    # In-place Qdrant metadata enrichment
│   ├── parallel_ingest_qdrant.py    # Parallel batch upload to Qdrant
│   ├── ingest_tickets.py            # Freshdesk ticket ingestion
│   ├── run_baseline_test.sh         # Full test suite runner
│   └── scrape_*.py                  # Product catalog scrapers
│
├── documents/                       # Source PDFs and DOCX manuals
├── data/                            # Runtime data (logs, cache, exports)
│
├── docker-compose.desoutter.yml     # Service definitions (API + GPU config)
├── Dockerfile                       # Backend container image
├── requirements.txt                 # Python dependencies
└── .env.example                     # Environment variable template
```

---

## System Architecture

### High-Level Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         REACT FRONTEND (Port 3001)                   │
│              TechWizard  │  Admin Dashboard  │  User Management      │
└──────────────────────────────────────────────────────────────────────┘
                                    │ HTTP/REST
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       FASTAPI BACKEND (Port 8000)                    │
│                                                                      │
│  ┌──────────────────┐  ┌─────────────────────────────────────────┐  │
│  │   JWT Auth       │  │         El-Harezmi Router                │  │
│  │   Rate Limiting  │  │   (POST /el-harezmi/diagnose)            │  │
│  │   CORS Control   │  └─────────────────────────────────────────┘  │
│  └──────────────────┘                   │                           │
│                                         ▼                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              El-Harezmi 5-Stage Pipeline                     │   │
│  │  Stage1: Intent  →  Stage2: Retrieve  →  Stage3: Extract    │   │
│  │          →  Stage4: Validate  →  Stage5: Format             │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
         │                    │                          │
         ▼                    ▼                          ▼
┌──────────────┐   ┌─────────────────┐        ┌─────────────────┐
│   MongoDB    │   │     Ollama      │        │     Qdrant      │
│   (27017)    │   │    (11434)      │        │    (6333/6334)  │
│ • users      │   │ Qwen2.5:7b      │        │ 26,513 chunks   │
│ • feedback   │   │ GPU-accelerated │        │ Dense + Sparse  │
│ • diagnoses  │   └─────────────────┘        │ 384-dim vectors │
│ • mappings   │                              │ Payload filters │
└──────────────┘                              └─────────────────┘
```

---

## El-Harezmi 5-Stage Pipeline

The **El-Harezmi pipeline** is the core AI engine introduced in v2.0.0. It replaces the linear 14-stage design with a structured, intent-driven architecture.

| Stage | Module | Description |
|:-----:|--------|-------------|
| **1** | `stage1_intent_classifier.py` | Multi-label intent classification (15 types) + entity extraction (product, controller, firmware, parameters) |
| **2** | `stage2_retrieval_strategy.py` | Intent-aware Qdrant retrieval with dynamic boost factors per document/chunk type |
| **3** | `stage3_info_extraction.py` | LLM-based structured JSON extraction (prerequisites, steps, ranges, warnings) |
| **4** | `stage4_kg_validation.py` | Knowledge graph validation against hard-coded compatibility matrix for 237+ products |
| **5** | `stage5_response_formatter.py` | Intent-specific response templates (CONFIGURATION, COMPATIBILITY, TROUBLESHOOT, PROCEDURE, etc.) |

### 15 Intent Types

| Category | Intent Types |
|----------|-------------|
| **Core** | `TROUBLESHOOT`, `ERROR_CODE`, `HOW_TO`, `MAINTENANCE`, `GENERAL` |
| **Extended** | `CONFIGURATION`, `COMPATIBILITY`, `SPECIFICATION`, `PROCEDURE`, `CALIBRATION` |
| **Advanced** | `FIRMWARE`, `INSTALLATION`, `COMPARISON`, `CAPABILITY_QUERY`, `ACCESSORY_QUERY` |

### Adaptive Document Chunking

The ingestion pipeline detects document type and applies the most appropriate chunking strategy:

| Document Type | Chunking Strategy | Purpose |
|---------------|-------------------|---------|
| `CONFIGURATION_GUIDE` | `SemanticChunker` | Preserves parameter sections |
| `COMPATIBILITY_MATRIX` | `TableAwareChunker` | Preserves table rows + headers |
| `ERROR_CODE_LIST` | `EntityChunker` | One chunk per error code |
| `SERVICE_BULLETIN` | `ProblemSolutionChunker` | Preserves problem+solution pairs |
| `PROCEDURE_GUIDE` | `StepPreservingChunker` | Preserves numbered steps |
| All others | `HybridChunker` | Adaptive fallback |

---

## Technology Stack

### AI / ML

| Component | Technology | Details |
|-----------|-----------|---------|
| **LLM** | Ollama + Qwen2.5:7b-instruct | Local inference, GPU-accelerated |
| **Vector DB** | Qdrant v1.7.4 | Dense + sparse hybrid vectors, payload filtering |
| **Embeddings** | all-MiniLM-L6-v2 | 384-dimensional sentence vectors |
| **Keyword Search** | BM25 (custom) + Qdrant sparse vectors | RRF-fused hybrid retrieval |

### Backend

| Component | Technology | Version |
|-----------|-----------|---------|
| **API Framework** | FastAPI | 0.109+ |
| **Database** | MongoDB | 7.0 |
| **Auth** | PyJWT + Bcrypt | 2.8 / 4.1 |
| **Document Parsing** | PyPDF2, pdfplumber, python-docx, python-pptx | Latest |
| **Deep Learning** | PyTorch | 2.1.2 |

### Frontend

| Component | Technology | Version |
|-----------|-----------|---------|
| **UI Framework** | React | 18.2 |
| **Build Tool** | Vite | 5.0 |
| **HTTP Client** | Axios | 1.6 |

### Infrastructure

| Component | Technology | Details |
|-----------|-----------|---------|
| **Containerisation** | Docker + Docker Compose | Multi-service orchestration |
| **GPU** | NVIDIA RTX A2000 | 6 GB VRAM, model inference |
| **Virtualisation** | Proxmox VM | Ubuntu 22.04 LTS |

---

## API Reference

### Authentication

```http
POST /auth/login           # Obtain JWT token
GET  /auth/me              # Current user profile
```

### El-Harezmi Pipeline (v2)

```http
POST /el-harezmi/diagnose  # 5-stage AI diagnosis (recommended)
GET  /el-harezmi/health    # Pipeline health check
GET  /el-harezmi/intents   # List supported intent types
```

### Legacy Diagnosis (v1 — maintained for compatibility)

```http
POST /diagnose             # 14-stage RAG diagnosis
POST /diagnose/feedback    # Submit feedback
GET  /diagnose/history     # Diagnosis history
```

### Conversation (Multi-turn)

```http
POST   /conversation/start    # Start new conversation session
GET    /conversation/{id}     # Retrieve session history
DELETE /conversation/{id}     # End session
```

### Admin (Admin role required)

```http
GET  /admin/dashboard             # Metrics dashboard
GET  /admin/metrics/health        # System health
GET  /admin/metrics/stats         # Performance statistics
POST /admin/documents/upload      # Upload PDF/DOCX/PPTX
POST /admin/documents/ingest      # Trigger document ingestion
GET  /admin/users                 # List users
POST /admin/users                 # Create user
PUT  /admin/users/{id}            # Update user
DELETE /admin/users/{id}          # Delete user
GET  /admin/learning/status       # Self-learning status
```

**Interactive API Docs:** `http://localhost:8000/docs`

---

## Usage Examples

### Authentication + Diagnosis

```bash
# 1. Obtain JWT token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123"}' | jq -r '.access_token')

# 2. El-Harezmi diagnosis (v2 — recommended)
curl -X POST http://localhost:8000/el-harezmi/diagnose \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "query": "EABC-3000 shows error E018, torque out of range",
    "language": "en"
  }'

# 3. Legacy diagnosis (v1 — still supported)
curl -X POST http://localhost:8000/diagnose \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "part_number": "6151659000",
    "fault_description": "motor not starting, error code E018",
    "language": "en"
  }'
```

### Example Response (El-Harezmi)

```json
{
  "diagnosis_id": "diag_abc123",
  "intent": "ERROR_CODE",
  "secondary_intents": ["TROUBLESHOOT"],
  "product_model": "EABC-3000",
  "suggestion": "**EABC-3000 — Error E018: Torque Out of Range**\n\n✅ **Cause:**\nE018 indicates a transducer fault due to an incorrect cable assembly...\n\n🔧 **Steps:**\n1. Check cable assembly connector type\n2. Verify transducer connection is secure\n...",
  "confidence": 0.89,
  "validation_status": "ALLOW",
  "sources": [
    {
      "document": "ESDE25004_ERS_range_EPB8_Transducer_Issue.pdf",
      "page": 3,
      "chunk_type": "problem_solution_pair",
      "snippet": "E018 indicates transducer fault — check cable assembly connector type..."
    }
  ],
  "pipeline_metrics": {
    "total_time_ms": 3420,
    "stage1_time_ms": 45,
    "stage2_time_ms": 210,
    "stage3_time_ms": 2800,
    "stage4_time_ms": 12,
    "stage5_time_ms": 80,
    "chunks_retrieved": 12
  }
}
```

### Web Interface Workflow

1. Navigate to `http://localhost:3001`
2. Log in with your credentials
3. Use the TechWizard — select product → describe the fault
4. View AI response with confidence score and source citations
5. Submit 👍/👎 feedback to continuously improve the system

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Pass Rate** | 85% (34/40 scenarios) |
| **Timeout Rate** | 0% (resolved) |
| **Products Indexed** | 451 |
| **Indexed Documents** | 541 (121 PDF + 420 DOCX) |
| **Freshdesk Tickets** | 2,249 processed |
| **Vector DB Chunks** | 26,513 chunks (Qdrant, 384-dim) |
| **BM25 Unique Terms** | 19,032 |
| **Cache Speedup** | ~100,000× for repeated queries |
| **Hallucination Rate** | < 2% |
| **Intent Types** | 15 (multi-label) |
| **Avg Response Time** | 23.6 s (non-cached) |
| **Supported Languages** | Turkish, English |

### Category Pass Rates (v2.0.2)

| Test Category | Pass Rate | Notes |
|---------------|-----------|-------|
| Troubleshooting | 100% | ✅ |
| Specifications | 100% | ✅ |
| Configuration | 100% | ✅ |
| Calibration | 100% | ✅ |
| Procedure | 100% | ✅ |
| Firmware | 100% | ✅ |
| Installation | 100% | ✅ |
| General (IDK) | 100% | ✅ |
| Accessory | 100% | ✅ |
| Error Codes | ~67% | E804 not in docs (test issue) |
| Compatibility | ~67% | Turkish prompt refinement |
| Maintenance | ~67% | Lubrication term mismatch |

---

## Feedback & Self-Learning

Every user interaction feeds a self-learning loop to improve future retrieval and ranking.

### Feedback Flow

```
User Query ──▶ RAG Retrieval ──▶ LLM Response ──▶ User Feedback (👍/👎)
                                                          │
                          ┌───────────────────────────────┴───────────────────┐
                          │ 👍 Positive: Reinforce source, record pattern      │
                          │ 👎 Negative: Record anti-pattern, demote source    │
                          └───────────────────────────────┬───────────────────┘
                                                          │
                                              Wilson Score Re-ranking
                                                          │
                                              Improved Future Results
```

### Wilson Score Formula

```
Wilson Score = (p + z²/2n − z√(p(1−p)/n + z²/4n²)) / (1 + z²/n)

Where:
  p = positive feedback ratio
  n = total feedback count
  z = 1.96  (95% confidence interval)
```

This ensures sources with few ratings cannot outrank well-tested sources.

### Learning Components

| Component | Description |
|-----------|-------------|
| `DiagnosisFeedback` | Stores every feedback event with full context |
| `LearnedMapping` | Successful fault → solution patterns |
| `SourceRankingLearner` | Wilson score document prioritisation |
| `ContrastiveLearningManager` | Collects positive/negative pairs for future embedding fine-tuning |

---

## Docker Services

| Service | Port | Description | Resources |
|---------|------|-------------|-----------|
| **mongodb** | 27017 | Primary database (users, feedback, mappings) | 1 core, 2 GB RAM |
| **qdrant** | 6333/6334 | Vector database (dense + sparse) | 2 cores, 4 GB RAM |
| **ollama** | 11434 | LLM inference server | 2 cores, 8 GB RAM + GPU |
| **desoutter-api** | 8000 | FastAPI backend | 3 cores, 12 GB RAM |
| **desoutter-frontend** | 3001 | React frontend (Nginx) | 1 core, 1 GB RAM |

### Key Docker Commands

```bash
# Start all services
docker-compose up -d

# Watch logs (all services)
docker-compose logs -f

# Watch a specific service
docker-compose logs -f desoutter-api

# Stop all services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# Check GPU allocation
docker exec ollama nvidia-smi
```

---

## Testing

### Running the Test Suite

```bash
# Full 40-scenario baseline test
./scripts/run_baseline_test.sh

# El-Harezmi pipeline unit tests
pytest tests/test_el_harezmi.py -v

# Adaptive chunking tests
pytest tests/test_adaptive_chunking.py -v

# RAG stability regression tests
pytest tests/test_rag_stability.py -v

# API integration tests
pytest tests/test_api_el_harezmi.py -v
```

### Test Categories

| File | Scenarios | Description |
|------|-----------|-------------|
| `test_el_harezmi.py` | 40 | Full El-Harezmi pipeline (all 15 intent types) |
| `test_adaptive_chunking.py` | — | Chunker detection and strategy selection |
| `test_rag_stability.py` | — | Legacy 14-stage RAG regression tests |
| `test_api_el_harezmi.py` | — | API-level integration tests |

---

## Key Files Reference

| File | Description |
|------|-------------|
| [src/el_harezmi/pipeline.py](src/el_harezmi/pipeline.py) | 5-stage El-Harezmi orchestrator |
| [src/el_harezmi/stage1_intent_classifier.py](src/el_harezmi/stage1_intent_classifier.py) | Multi-label intent classification (15 types) |
| [src/el_harezmi/stage2_retrieval_strategy.py](src/el_harezmi/stage2_retrieval_strategy.py) | Intent-aware Qdrant retrieval + boost factors |
| [src/el_harezmi/stage3_info_extraction.py](src/el_harezmi/stage3_info_extraction.py) | LLM-based structured information extraction |
| [src/el_harezmi/stage4_kg_validation.py](src/el_harezmi/stage4_kg_validation.py) | Knowledge graph compatibility validation |
| [src/el_harezmi/stage5_response_formatter.py](src/el_harezmi/stage5_response_formatter.py) | Intent-specific response templates |
| [src/vectordb/qdrant_client.py](src/vectordb/qdrant_client.py) | Qdrant operations (hybrid search, filtering) |
| [src/documents/chunkers/](src/documents/chunkers/) | All 6 adaptive chunking strategies |
| [src/documents/document_classifier.py](src/documents/document_classifier.py) | Regex-based 8-type document detection |
| [src/api/main.py](src/api/main.py) | All FastAPI routes and middleware |
| [src/api/el_harezmi_router.py](src/api/el_harezmi_router.py) | El-Harezmi-specific API routes |
| [config/ai_settings.py](config/ai_settings.py) | RAG parameters, thresholds, model settings |
| [config/feature_flags.py](config/feature_flags.py) | Runtime feature toggles |
| [config/tool_controller_compatibility.py](config/tool_controller_compatibility.py) | Hard-coded compatibility matrix |
| [frontend/src/App.jsx](frontend/src/App.jsx) | Main React component (auth, routing, admin) |
| [scripts/enrich_qdrant_metadata.py](scripts/enrich_qdrant_metadata.py) | In-place Qdrant metadata enrichment |
| [scripts/reingest_adaptive.py](scripts/reingest_adaptive.py) | Full adaptive re-ingestion pipeline |

---

## Security Considerations

### Production Hardening Checklist

| # | Action | Priority |
|---|--------|----------|
| 1 | Change `JWT_SECRET` in `.env` — never use the default | 🔴 Critical |
| 2 | Replace default credentials (`admin123`, `tech123`) | 🔴 Critical |
| 3 | Restrict CORS `allow_origins` to specific domains | 🟡 High |
| 4 | Enable MongoDB authentication (`MONGO_URI` with credentials) | 🟡 High |
| 5 | Configure rate limiting (Nginx or API gateway) | 🟡 High |
| 6 | Enable HTTPS via reverse proxy (Nginx + Let's Encrypt) | 🟡 High |
| 7 | Rotate JWT tokens regularly and set short expiry | 🟢 Medium |
| 8 | Restrict admin endpoints to internal network | 🟢 Medium |

### Current Security Posture

| Risk | Status | Mitigation |
|------|--------|------------|
| Default credentials | Medium | Change before deployment |
| Open CORS policy | Medium | Restrict `allow_origins` |
| No built-in rate limiting | Medium | Add Nginx limits |
| Input sanitisation | Low | FastAPI Pydantic validation in place |
| GPU single point of failure | Low | CPU fallback available |

> **Full security assessment:** see [SECURITY_ASSESSMENT_REPORT_TR.md](SECURITY_ASSESSMENT_REPORT_TR.md)

---

## Additional Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Rapid deployment guide (under 10 minutes) |
| [ROADMAP.md](ROADMAP.md) | Development roadmap and planned features |
| [CHANGELOG.md](CHANGELOG.md) | Full version history (v0.1.0 → v2.0.2) |
| [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) | Mermaid diagrams for presentations |
| [PROXMOX_DEPLOYMENT.md](PROXMOX_DEPLOYMENT.md) | Proxmox VM infrastructure guide |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'feat: add amazing feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request against `main`

### Code Standards

- **Python**: PEP 8, type hints everywhere, docstrings on all public methods
- **JavaScript**: ESLint + Prettier
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) format (`feat:`, `fix:`, `docs:`, `refactor:`)
- **Tests**: All new features must include tests in the appropriate `tests/` file

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Author

**Fatih Bayram** — [@fatihhbayram](https://github.com/fatihhbayram)

---

## Acknowledgements

- **[Ollama](https://ollama.com/)** — Local LLM inference infrastructure
- **[Qdrant](https://qdrant.tech/)** — High-performance scalable vector database
- **[HuggingFace](https://huggingface.co/)** — Sentence Transformers and model hub
- **[FastAPI](https://fastapi.tiangolo.com/)** — Modern, high-performance Python web framework
- **[LangChain](https://www.langchain.com/)** — RAG orchestration patterns

---

<p align="center">
  <strong>Powered by</strong> El-Harezmi Pipeline | Qdrant | Ollama | FastAPI | React
  <br><br>
  <em>Enterprise RAG System — v2.0.2 — Production Ready</em>
</p>
