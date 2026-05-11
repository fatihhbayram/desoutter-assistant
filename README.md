# Desoutter Assistant

> **AI-Powered Technical Support System for Industrial Tool Maintenance**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2+-61DAFB?logo=react&logoColor=white)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

A **Retrieval-Augmented Generation (RAG)** system that provides context-aware troubleshooting assistance for Desoutter industrial tools. Built on hybrid search (BM25 + Semantic), product-aware filtering, and a real-world Q&A evaluation pipeline.

---

## What Does This Project Do?

When a technician encounters an issue with a Desoutter tool:

1. **Asks a question**: "My EPBC8-1800-4Q shows error code E018"
2. **System searches**: Hybrid retrieval across 6,200+ document chunks
3. **Generates response**: AI-powered solution with cited sources
4. **Learns**: Continuously improves from user feedback

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│    Technician    │────▶│  Desoutter AI    │────▶│ Solution + Sources│
│   "E018 error"   │     │  (RAG Pipeline)  │     │  Confidence: 89%  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Performance Metrics](#performance-metrics)
- [Feedback System](#feedback-system)
- [Docker Services](#docker-services)
- [Testing](#testing)
- [Key Files Reference](#key-files-reference)
- [Security Considerations](#security-considerations)

---

## Quick Start

### Prerequisites

- Docker & Docker Compose (v2.0+)
- 16GB RAM (minimum 8GB)
- NVIDIA GPU (optional, for faster inference)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/fatihhbayram/desoutter-assistant.git
cd desoutter-assistant

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings (MongoDB, Ollama, Qdrant URLs)

# 3. Start all services
docker compose -f ai-stack.yml up -d

# 4. Wait for services to initialize (~60 seconds)
sleep 60

# 5. Access the application
# Frontend: http://localhost:3001
# API Docs: http://localhost:8000/docs
```

> **Security Note**: Change `JWT_SECRET` in `.env` before deployment.

---

## Project Structure

```
desoutter-assistant/
│
├── src/                          # Main source code
│   ├── api/                      # FastAPI REST endpoints
│   ├── database/                 # MongoDB connection and models
│   ├── documents/                # Document processing pipeline
│   ├── llm/                      # RAG engine and AI components
│   │   ├── rag_engine.py        # Main RAG pipeline
│   │   ├── hybrid_search.py     # Semantic + BM25 search with RRF
│   │   ├── self_learning.py     # Feedback learning system
│   │   ├── intent_detector.py   # 15-category query classification
│   │   ├── response_validator.py # Hallucination detection
│   │   ├── confidence_scorer.py  # Confidence calculation
│   │   ├── context_optimizer.py  # Token budget management
│   │   └── response_cache.py     # LRU + TTL caching
│   ├── vectordb/                 # Qdrant vector database client
│   ├── el_harezmi/               # ⚠️ Experimental 5-stage pipeline (SUSPENDED — do not modify)
│   ├── scraper/                  # Web scraping modules (not in git — contains internal URLs)
│   └── utils/                    # Helper functions, logging
│
├── frontend/                     # React 18 user interface
│   ├── src/
│   │   ├── App.jsx              # Main application (auth, routing)
│   │   └── TechWizard.jsx       # Technician chat interface
│   └── package.json
│
├── scripts/                      # Utility scripts
│   ├── reingest_adaptive.py     # Full knowledge base re-ingestion
│   ├── ingest_eabc_els_efd.py   # Targeted EABC/ELS/EFD ingestion
│   ├── ingest_basic_troubleshooting.py  # Basic troubleshooting docs
│   ├── reingest_specific_bulletins.py   # Single-bulletin re-ingestion
│   ├── build_qa_dataset.py      # Build Q&A evaluation dataset
│   └── evaluate_rag.py          # RAG quality evaluation (keyword overlap)
├── documents/                    # PDF manuals and service bulletins
│   ├── bulletins/               # ESDE bulletins, product guides, how-to docs
│   └── manuals/                 # Product manuals
├── data/                         # Runtime data (eval results, logs, cache)
├── config/                       # Configuration files
│   └── ai_settings.py           # RAG parameters and score boosts
│
├── ai-stack.yml                  # Multi-container orchestration (use this, not docker-compose.yml)
├── Dockerfile                    # Backend container image
└── requirements.txt              # Python dependencies
```

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE                                 │
│                       React 18 + Vite (Port 3001)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            FastAPI BACKEND                               │
│                              (Port 8000)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐  │
│  │  API Routes │  │  Services   │  │         RAG Engine              │  │
│  │  /diagnose  │──│  auth       │──│  • Hybrid Search (BM25+Semantic)│  │
│  │  /feedback  │  │  diagnose   │  │  • Product Family Filtering     │  │
│  │  /admin     │  │  document   │  │  • Score Boosting               │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                           │
         ▼                    ▼                           ▼
┌─────────────────┐  ┌─────────────────┐         ┌─────────────────┐
│    MongoDB      │  │     Ollama      │         │     Qdrant      │
│    (27017)      │  │    (11434)      │         │    (6333)       │
│  • users        │  │  Qwen2.5:7b     │         │  6,200+ chunks  │
│  • feedback     │  │  GPU-accelerated│         │  384-dim vectors│
│  • mappings     │  └─────────────────┘         └─────────────────┘
└─────────────────┘
```

### RAG Pipeline

Each query passes through these stages:

| Stage | Name | Description |
|:-----:|------|-------------|
| 1 | **Off-topic Detection** | Filters irrelevant queries |
| 2 | **Language Detection** | Auto-detects Turkish/English |
| 3 | **Cache Check** | Returns cached response if available |
| 4 | **Self-Learning Context** | Applies learned patterns and boosts |
| 5 | **Hybrid Retrieval** | Semantic + BM25 with RRF fusion |
| 6 | **Product Filtering** | Filters documents by product family |
| 7 | **Capability Filtering** | WiFi/Battery content filtering |
| 8 | **Context Grounding** | Returns "I don't know" if context is insufficient |
| 9 | **Context Optimization** | 8K token budget, semantic deduplication |
| 10 | **Intent Detection** | 15 query categories (troubleshooting, specs, etc.) |
| 11 | **LLM Generation** | Qwen2.5:7b response generation |
| 12 | **Response Validation** | Hallucination detection |
| 13 | **Confidence Scoring** | Multi-factor confidence calculation |
| 14 | **Save & Cache** | MongoDB persistence + LRU cache update |

### Score Boosting

Document type boosts applied during retrieval ranking:

| Document Type | Boost | Note |
|--------------|-------|------|
| `service_bulletin` | 2.5x | ESDE bulletins — primary source |
| `procedure_guide` | 2.0x | How-to guides, basic troubleshooting |
| `technical_manual` | 1.5x | Product manuals, installation guides |

---

## Technology Stack

### AI/ML Layer

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Ollama + Qwen2.5:7b-instruct | Natural language generation |
| **Vector DB** | Qdrant (v1.7.4) | Semantic document storage with metadata filtering |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) | 384-dimensional vector generation |
| **Keyword Search** | BM25 (custom implementation) | Exact term matching |

> **Note:** LangChain is listed in requirements.txt but is not actively used — the RAG pipeline is fully custom.

### Backend

| Component | Technology | Version |
|-----------|------------|---------|
| **Web Framework** | FastAPI | 0.109 |
| **Database** | MongoDB | 7.0 |
| **Authentication** | PyJWT + Bcrypt | 2.8 / 4.1 |
| **Document Processing** | pdfplumber, python-docx, python-pptx | Latest |
| **Deep Learning** | PyTorch | 2.1.2 |

### Frontend

| Component | Technology | Version |
|-----------|------------|---------|
| **UI Framework** | React | 18.2 |
| **Build Tool** | Vite | 5.0 |
| **HTTP Client** | Axios | 1.6 |

### Infrastructure

| Component | Technology | Details |
|-----------|------------|---------|
| **Container** | Docker + Compose | Multi-container via `ai-stack.yml` |
| **GPU** | NVIDIA RTX A2000 | 6GB VRAM, LLM acceleration |
| **Virtualization** | Proxmox VM | Ubuntu 22.04 LTS |

---

## API Reference

### Authentication

```http
POST /auth/login          # Get JWT token
GET  /auth/me             # Get current user info
```

### Diagnosis (Core)

```http
POST /diagnose            # Get AI-powered fault diagnosis
POST /diagnose/feedback   # Submit user feedback
GET  /diagnose/history    # Get diagnosis history
```

### Admin (Requires Admin Role)

```http
GET  /admin/dashboard         # Dashboard metrics
GET  /admin/metrics/health    # System health status
GET  /admin/metrics/stats     # Performance statistics
POST /admin/documents/upload  # Upload document (PDF/DOCX/PPTX)
POST /admin/documents/ingest  # Process documents into RAG
GET  /admin/users             # List users
POST /admin/users             # Create new user
```

**Interactive API Documentation:** http://localhost:8000/docs

---

## Usage Examples

### API Query

```bash
# 1. Get authentication token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123"}' | jq -r '.access_token')

# 2. Request fault diagnosis
curl -X POST http://localhost:8000/diagnose \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "part_number": "6151659000",
    "fault_description": "motor not starting, error code E018",
    "language": "en"
  }'
```

### Example Response

```json
{
  "suggestion": "Error Code E018: Torque Out of Range\n\nCause: Transducer fault due to incorrect cable connector type.\n\nSolution:\n1. Check cable assembly and connector compatibility\n2. Verify transducer connection is secure\n3. Replace cable if connector type is incorrect",
  "confidence": 0.89,
  "sources": [
    {
      "source": "ESDE25004_ERS_range_EPB8_Transducer_Issue.pdf",
      "excerpt": "E018 indicates transducer fault - check cable assembly connector type..."
    }
  ],
  "intent": "error_code"
}
```

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Evaluation Coverage (Good+Partial)** | 87.2% — system returns relevant content (725 Q&A pairs) |
| **Evaluation Good Rate** | 37.5% full pipeline (keyword overlap ≥ 0.15) |
| **Evaluation Partial Rate** | 49.7% (keyword overlap ≥ 0.07) |
| **Evaluation Fail Rate** | 12.8% — no relevant content returned |
| **Retrieval Good@10** | 92.1% (Hybrid BM25+Semantic, isolated retrieval test) |
| **Avg Keyword Overlap** | 0.138 (full pipeline) / 0.302 (retrieval-only) |
| **Avg Response Time** | ~20.5s (non-cached) |
| **Vector DB Chunks** | 6,200+ (384-dim, language-filtered) |
| **Q&A Evaluation Dataset** | 725 real-world field support Q&A pairs (from 4,000 field cases) |
| **Intent Categories** | 15 types (troubleshoot, error_code, spec, config, compat, etc.) |
| **Knowledge Base** | ESDE bulletins, product manuals, EABC/ELS/EFD/CVI3 guides |
| **Hallucination Detection** | Enabled (response validator) |
| **Confidence Score** | Numeric 0.0–1.0 via `confidence_score` field in `/diagnose` |
| **Active LLM** | Qwen3:8b — best Good rate (37.9%) in 3-model comparison |
| **Evaluation Metrics** | Keyword Overlap + ROUGE-L (both in `evaluate_rag.py`) |

### LLM Model Comparison (725 Q&A pairs, full pipeline)

| Model | Good% | Partial% | Fail% | Latency |
|-------|------:|--------:|------:|--------:|
| Qwen2.5:7b-instruct | 37.5% | 49.7% | 12.8% | ~20.5s |
| Llama3:latest | 37.0% | 55.0% | 8.0% | ~23.7s |
| **Qwen3:8b** | **37.9%** | **49.9%** | **12.1%** | **~21.1s** |

### Retrieval Model Comparison (725 Q&A pairs, top-10)

| Method | Good% | Fail% | Avg Overlap |
|--------|------:|------:|------------:|
| Hybrid BM25+Semantic (0.5/0.5) | **92.1%** | 0.4% | 0.302 |
| BM25-only | 91.0% | 1.1% | 0.296 |
| Semantic-only | 88.6% | 1.1% | 0.288 |
| TF-IDF | 83.0% | 2.5% | 0.249 |

> **Evaluation methodology:** Each question from the Q&A dataset is sent to `/diagnose`. The keyword overlap between expected answer (field agent reply) and actual response is measured. Thresholds: good ≥ 0.15, partial ≥ 0.07. Note: expected answers are informal agent emails; actual responses are structured markdown — so overlap naturally skews lower than true quality.

---

## Feedback System

After each AI response, users can submit feedback:

```
┌─────────────────────────────────────────────────────────────┐
│  AI Response: "Error E018 indicates transducer fault..."    │
├─────────────────────────────────────────────────────────────┤
│  Was this helpful?                                          │
│  [ 👍 Helpful ]    [ 👎 Not Helpful ]                       │
│  Reason: Solved my problem / Partially helpful /            │
│          Information was incorrect / Missing details        │
└─────────────────────────────────────────────────────────────┘
```

| Rating | System Action |
|--------|---------------|
| `helpful` | Boost source relevance (Wilson score) |
| `partially_helpful` | Minor boost, flag for review |
| `not_helpful` | Record negative signal |
| `incorrect` | Flag sources, prevent future use |

---

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| **ollama** | 11434 | LLM inference (GPU-accelerated) |
| **qdrant** | 6333 | Vector database |
| **mongodb** | 27017 | Document database |
| **desoutter-api** | 8000 | FastAPI backend |
| **desoutter-frontend** | 3001 | React frontend |

### Docker Commands

```bash
# Start all services
docker compose -f ai-stack.yml up -d

# View API logs
docker compose -f ai-stack.yml logs --tail=50 desoutter-api

# Restart API (src/ is volume-mounted — no rebuild needed for code changes)
docker compose -f ai-stack.yml restart desoutter-api

# Full rebuild (after Dockerfile or dependency changes)
docker compose -f ai-stack.yml up -d --build
```

---

## Testing

### Evaluation Pipeline

The primary quality measurement tool:

```bash
# Build Q&A dataset from field support tickets
python scripts/build_qa_dataset.py

# Run full evaluation (725 questions, ~4 hours)
python scripts/evaluate_rag.py --limit 725 --delay 0.3

# Run quick sample (50 questions)
python scripts/evaluate_rag.py --limit 50 --offset 0

# Results saved to:
# /app/data/eval_results.json   — per-question results
# /app/data/eval_summary.json   — overall + per-model breakdown
```

### Unit Tests

```bash
# RAG stability tests
pytest tests/test_rag_stability.py -v

# Chunking tests
pytest tests/test_adaptive_chunking.py -v
```

> **Note:** `tests/test_el_harezmi.py` and `tests/test_api_el_harezmi.py` relate to the suspended El-Harezmi pipeline and should not be run.

---

## Key Files Reference

| File | Description |
|------|-------------|
| [src/llm/rag_engine.py](src/llm/rag_engine.py) | Main RAG pipeline orchestrator |
| [src/llm/hybrid_search.py](src/llm/hybrid_search.py) | BM25 + Semantic search with RRF fusion |
| [src/llm/self_learning.py](src/llm/self_learning.py) | Feedback learning with Wilson scores |
| [src/llm/intent_detector.py](src/llm/intent_detector.py) | 15-category query classification |
| [src/llm/response_validator.py](src/llm/response_validator.py) | Hallucination detection |
| [src/llm/confidence_scorer.py](src/llm/confidence_scorer.py) | Multi-factor confidence calculation |
| [src/llm/context_optimizer.py](src/llm/context_optimizer.py) | Token budget and deduplication |
| [src/llm/response_cache.py](src/llm/response_cache.py) | LRU + TTL caching layer |
| [src/api/main.py](src/api/main.py) | FastAPI routes and middleware |
| [src/vectordb/qdrant_client.py](src/vectordb/qdrant_client.py) | Qdrant vector database operations |
| [config/ai_settings.py](config/ai_settings.py) | RAG parameters and score boost values |
| [scripts/reingest_adaptive.py](scripts/reingest_adaptive.py) | Full knowledge base re-ingestion tool |
| [scripts/evaluate_rag.py](scripts/evaluate_rag.py) | RAG quality evaluation script |
| [scripts/build_qa_dataset.py](scripts/build_qa_dataset.py) | Q&A dataset builder from field cases |

### Suspended / Archived

| File | Status |
|------|--------|
| `src/el_harezmi/` | ⚠️ SUSPENDED — experimental 5-stage pipeline, abandoned 2026-04-13 |
| `src/api/el_harezmi_router.py` | ⚠️ SUSPENDED — routes `/api/v2/chat`, 0% traffic |

---

## Security Considerations

### Production Checklist

- [ ] Set strong `JWT_SECRET` in `.env`
- [ ] Configure `CORS_ORIGINS` to your domain (currently `*`)
- [ ] Enable MongoDB authentication
- [ ] Use HTTPS via reverse proxy (Cloudflare Tunnel recommended)
- [ ] Restrict rate limiting to protect against DoS

---

## Additional Documentation

| Document | Description |
|----------|-------------|
| [ROADMAP.md](ROADMAP.md) | Development roadmap and planned features |
| [CHANGELOG.md](CHANGELOG.md) | Version history and changes |
| [PROXMOX_DEPLOYMENT.md](PROXMOX_DEPLOYMENT.md) | Infrastructure deployment guide |

---

## Author

**Fatih Bayram** - [@fatihhbayram](https://github.com/fatihhbayram)

---

## Acknowledgments

- **Ollama** — Local LLM serving
- **Qdrant** — High-performance vector database
- **HuggingFace** — Sentence transformers
- **FastAPI** — Modern Python web framework

---

<p align="center">
  <strong>Powered by</strong> Ollama | Qdrant | FastAPI | React | BM25
</p>
