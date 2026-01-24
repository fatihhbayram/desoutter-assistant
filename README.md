# Desoutter Assistant

> **Enterprise-Grade AI-Powered Technical Support System for Industrial Tool Maintenance**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2+-61DAFB?logo=react&logoColor=white)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-96%25%20Passing-success)](test_results/)

An intelligent **Retrieval-Augmented Generation (RAG)** system that provides context-aware repair and troubleshooting assistance for Desoutter industrial tools. Features a self-learning feedback loop, 14-stage quality pipeline, and production-grade architecture achieving **96% test pass rate**.

---

## What Does This Project Do?

When a technician encounters an issue with a Desoutter tool:

1. **Asks a question**: "My EPBC8-1800-4Q shows error code E018"
2. **System searches**: Finds relevant information from 28,000+ document chunks
3. **Generates response**: AI-powered solution with cited sources
4. **Learns**: Continuously improves from user feedback

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│    Technician    │────▶│  Desoutter AI    │────▶│ Solution + Sources│
│   "E018 error"   │     │  (14-stage RAG)  │     │  Confidence: 89%  │
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
- [Self-Learning System](#self-learning-system)
- [Local Development](#local-development)
- [Docker Services](#docker-services)
- [Testing](#testing)
- [Key Files Reference](#key-files-reference)
- [Security Considerations](#security-considerations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

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

# 2. Start all services
docker-compose -f docker-compose.desoutter.yml up -d

# 3. Wait for services to initialize (~60 seconds)
sleep 60

# 4. Access the application
# Frontend: http://localhost:3001
# API Docs: http://localhost:8000/docs
```

### Default Credentials

| Username | Password | Role | Permissions |
|----------|----------|------|-------------|
| `admin` | `admin123` | Admin | Full system access |
| `tech` | `tech123` | Technician | Query and feedback only |

> **Security Note**: Change `JWT_SECRET` in `.env` file for production deployments.

---

## Project Structure

```
desoutter-assistant/
│
├── src/                          # Main source code
│   ├── api/                      # FastAPI REST endpoints
│   │   └── main.py              # API entry point and route definitions
│   ├── database/                 # MongoDB connection and models
│   │   └── mongo_client.py      # Database operations
│   ├── documents/                # Document processing pipeline
│   │   ├── document_processor.py # PDF/DOCX/PPTX extraction
│   │   └── semantic_chunker.py   # Intelligent text segmentation
│   ├── llm/                      # RAG engine and AI components
│   │   ├── rag_engine.py        # 14-stage main pipeline (81KB)
│   │   ├── hybrid_search.py     # Semantic + BM25 search fusion
│   │   ├── self_learning.py     # Feedback learning system
│   │   ├── intent_detector.py   # 8-category query classification
│   │   ├── response_validator.py # Hallucination detection
│   │   ├── confidence_scorer.py  # Confidence score calculation
│   │   ├── context_optimizer.py  # Token budget management
│   │   └── response_cache.py     # LRU + TTL caching
│   ├── scraper/                  # Web scraping modules
│   └── utils/                    # Helper functions, logging
│
├── frontend/                     # React 18 user interface
│   ├── src/
│   │   ├── App.jsx              # Main application (auth, routing)
│   │   ├── TechWizard.jsx       # Technician chat interface
│   │   └── MetricsDashboard.jsx # Admin dashboard
│   └── package.json
│
├── scripts/                      # Utility scripts (42+ scripts)
│   ├── run_api.py               # Start API server
│   ├── ingest_documents.py      # Document processing
│   └── run_baseline_test.sh     # Test suite runner
│
├── documents/                    # PDF manuals and service bulletins
├── data/                         # Runtime data (vectordb, logs, cache)
├── tests/                        # Test files and fixtures
├── config/                       # Configuration files
│   └── ai_settings.py           # RAG parameters
│
├── docker-compose.desoutter.yml  # Multi-container orchestration
├── Dockerfile                    # Backend container image
├── requirements.txt              # Python dependencies (Phase 1)
├── requirements-phase2.txt       # AI/RAG dependencies (Phase 2)
└── .env.example                  # Environment variables template
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
│  │  /diagnose  │──│  auth       │──│  • Hybrid Search                │  │
│  │  /feedback  │  │  diagnose   │  │  • Self-Learning                │  │
│  │  /admin     │  │  document   │  │  • Product Filtering            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                           │
         ▼                    ▼                           ▼
┌─────────────────┐  ┌─────────────────┐         ┌─────────────────┐
│    MongoDB      │  │     Ollama      │         │    ChromaDB     │
│    (27017)      │  │    (11434)      │         │   (Embedded)    │
│  • users        │  │  Qwen2.5:7b     │         │  28,414 chunks  │
│  • feedback     │  │  GPU-accelerated│         │  384-dim vectors│
│  • mappings     │  └─────────────────┘         └─────────────────┘
└─────────────────┘
```

### 14-Stage RAG Pipeline

The heart of the system - each query passes through 14 stages for high-quality responses:

| Stage | Name | Description |
|:-----:|------|-------------|
| 1 | **Off-topic Detection** | Filters irrelevant queries |
| 2 | **Language Detection** | Auto-detects Turkish/English |
| 3 | **Cache Check** | Returns cached response if available (~100,000x speedup) |
| 4 | **Self-Learning Context** | Applies learned patterns and boosts |
| 5 | **Hybrid Retrieval** | Semantic (60%) + BM25 (40%) with RRF fusion |
| 6 | **Product Filtering** | Filters documents by product family |
| 7 | **Capability Filtering** | WiFi/Battery content filtering |
| 8 | **Context Grounding** | Returns "I don't know" if uncertain (threshold < 0.35) |
| 9 | **Context Optimization** | 8K token budget, semantic deduplication |
| 10 | **Intent Detection** | 8 query categories (troubleshooting, specs, etc.) |
| 11 | **LLM Generation** | GPU-accelerated response generation with Qwen2.5:7b |
| 12 | **Response Validation** | Hallucination and forbidden content detection |
| 13 | **Confidence Scoring** | Multi-factor confidence calculation |
| 14 | **Save & Cache** | MongoDB persistence + LRU cache update |

```
┌──────────────────────────────────────────────────────────────┐
│                       USER QUERY                             │
└──────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────────┐
    │                         ▼                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ STAGE 1-4: PRE-PROCESSING                      │  │
    │  │ Off-topic → Language → Cache → Self-Learning   │  │
    │  └────────────────────────────────────────────────┘  │
    │                         │                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ STAGE 5-7: RETRIEVAL                           │  │
    │  │ Hybrid Search → Product Filter → Capability    │  │
    │  └────────────────────────────────────────────────┘  │
    │                         │                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ STAGE 8-10: CONTEXT PROCESSING                 │  │
    │  │ Grounding → Optimization → Intent Detection    │  │
    │  └────────────────────────────────────────────────┘  │
    │                         │                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ STAGE 11-14: GENERATION & VALIDATION           │  │
    │  │ LLM → Validation → Confidence → Cache          │  │
    │  └────────────────────────────────────────────────┘  │
    └─────────────────────────┼─────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                       AI RESPONSE                            │
│              (With confidence score & sources)               │
└──────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### AI/ML Layer

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Ollama + Qwen2.5:7b-instruct | Natural language understanding & generation |
| **Vector DB** | ChromaDB 0.4.22 | Semantic document storage and retrieval |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) | 384-dimensional vector generation |
| **Keyword Search** | BM25 (Custom Implementation) | Fast keyword-based retrieval |
| **Orchestration** | LangChain 0.1 | RAG workflow management |

### Backend

| Component | Technology | Version |
|-----------|------------|---------|
| **Web Framework** | FastAPI | 0.109 |
| **Database** | MongoDB | 7.0 |
| **Authentication** | PyJWT + Bcrypt | 2.8 / 4.1 |
| **Document Processing** | PyPDF2, pdfplumber, python-docx | Latest |
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
| **Container** | Docker + Compose | Multi-container orchestration |
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

### Conversation (Multi-turn)

```http
POST   /conversation/start    # Start new conversation
GET    /conversation/{id}     # Get conversation history
DELETE /conversation/{id}     # End conversation
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
  "suggestion": "Error Code E018: Torque Out of Range!\n\nCause: The error code E018 indicates a transducer fault due to an incorrect cable assembly with the wrong connector.\n\nSolution:\n1. Check the cable assembly and connector compatibility\n2. Verify transducer connection is secure\n3. Replace cable if connector type is incorrect...",
  "confidence": 0.89,
  "sources": [
    {
      "document": "ESDE25004_ERS_range_EPB8_Transducer_Issue.pdf",
      "page": 3,
      "snippet": "E018 indicates transducer fault - check cable assembly connector type..."
    }
  ],
  "intent": "troubleshooting"
}
```

### Web Interface Workflow

1. Navigate to http://localhost:3001
2. Login with credentials (admin/admin123)
3. Select product from dropdown or search
4. Enter your question in the chat interface
5. View AI response with confidence score and source citations
6. Submit feedback to improve future results

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Pass Rate** | 96% (24/25 scenarios) |
| **Total Products** | 451 (71 wireless, 380 cable) |
| **ChromaDB Chunks** | 28,414 semantic chunks |
| **Indexed Documents** | 541 (121 PDF + 420 Word) |
| **Freshdesk Tickets** | 2,249 processed |
| **BM25 Index Terms** | 19,032 unique terms |
| **Domain Terms** | 351 Desoutter-specific |
| **Cache Speedup** | ~100,000x for repeated queries |
| **Hallucination Rate** | <2% |
| **Intent Categories** | 8 specialized types |

---

## Feedback System

The feedback system is the foundation of continuous learning. Every user interaction helps improve future responses.

### Feedback Interface

After each AI response, users can submit feedback:

```
┌─────────────────────────────────────────────────────────────┐
│  AI Response: "Error E018 indicates transducer fault..."    │
├─────────────────────────────────────────────────────────────┤
│  Was this helpful?                                          │
│                                                             │
│  [ 👍 Helpful ]    [ 👎 Not Helpful ]                       │
│                                                             │
│  Reason (optional):                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ○ Solved my problem                                  │   │
│  │ ○ Partially helpful                                  │   │
│  │ ○ Information was incorrect                          │   │
│  │ ○ Missing important details                          │   │
│  │ ○ Wrong product/context                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Additional comments:                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ [Free text input...]                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  [ Submit Feedback ]                                        │
└─────────────────────────────────────────────────────────────┘
```

### Feedback API

```bash
# Submit feedback
curl -X POST http://localhost:8000/diagnose/feedback \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "diagnosis_id": "diag_abc123",
    "rating": "helpful",
    "reason": "solved_problem",
    "comment": "The cable connector was indeed the issue",
    "part_number": "6151659000"
  }'
```

### Feedback Types

| Rating | Description | System Action |
|--------|-------------|---------------|
| `helpful` | Response solved the problem | Boost source relevance, strengthen pattern |
| `partially_helpful` | Some useful information | Minor boost, flag for review |
| `not_helpful` | Response didn't help | Record negative signal, analyze gaps |
| `incorrect` | Information was wrong | Flag sources, prevent future use |

### Feedback Reasons

| Reason Code | Description |
|-------------|-------------|
| `solved_problem` | Completely resolved the issue |
| `partially_helpful` | Some parts were useful |
| `incorrect_info` | Contains factual errors |
| `missing_details` | Needs more specific information |
| `wrong_product` | Information for different product |
| `outdated_info` | Information no longer applicable |
| `too_generic` | Not specific enough for the problem |

---

## Self-Learning System

The system continuously learns from user feedback to improve response quality:

```
User Query ──▶ RAG Retrieval ──▶ LLM Response ──▶ User Feedback
                                                        │
                                         ┌──────────────┴──────────────┐
                                         │ 👍 Positive: Reinforce      │
                                         │ 👎 Negative: Record & avoid │
                                         └──────────────┬──────────────┘
                                                        │
                                               Wilson Score Ranking
                                                        │
                                              Improved Future Results
```

### Learning Components

| Component | Purpose |
|-----------|---------|
| `DiagnosisFeedback` | Records all user feedback with full context |
| `LearnedMapping` | Stores successful fault→solution patterns |
| `SourceRankingLearner` | Wilson score-based document prioritization |
| `ContrastiveLearningManager` | Collects data for embedding fine-tuning |

### Wilson Score Ranking

The system uses Wilson score confidence interval to rank sources reliably:

```
Wilson Score = (p + z²/2n - z√(p(1-p)/n + z²/4n²)) / (1 + z²/n)

Where:
- p = positive feedback ratio
- n = total feedback count
- z = 1.96 (95% confidence)
```

This prevents sources with few ratings from outranking well-tested sources.

### How Self-Learning Works

1. **Feedback Collection** - User submits 👍/👎 with optional reason and comment
2. **Signal Processing** - System extracts query patterns, source IDs, and context
3. **Score Update** - Wilson score recalculated for affected sources
4. **Pattern Storage** - Successful fault→solution mappings saved to `learned_mappings`
5. **Future Boost** - Matching patterns get relevance boost (up to 1.5x) in retrieval
6. **Negative Avoidance** - Low-scoring sources deprioritized in results

### Learning Metrics (Admin Dashboard)

| Metric | Description |
|--------|-------------|
| Total Feedback | All-time feedback submissions |
| Positive Rate | Percentage of helpful ratings |
| Learning Events | Patterns learned this week |
| Top Sources | Highest-rated documents |
| Problem Areas | Topics needing improvement |

---

## Local Development

### Environment Setup

```bash
# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt
pip install -r requirements-phase2.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### Environment Variables

```bash
# Copy example configuration
cp .env.example .env
# Edit .env with your settings
```

Key variables:

```bash
# MongoDB
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DATABASE=desoutter

# Ollama LLM
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_TEMPERATURE=0.1

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cuda  # or 'cpu'

# API
API_HOST=0.0.0.0
API_PORT=8000
JWT_SECRET=your-secret-key-change-in-production

# RAG Settings
RAG_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.7
USE_HYBRID_SEARCH=true
HYBRID_SEMANTIC_WEIGHT=0.6
HYBRID_BM25_WEIGHT=0.4
```

### Running Services Locally

```bash
# Terminal 1: Start backend
python scripts/run_api.py

# Terminal 2: Start frontend
cd frontend && npm run dev
```

---

## Docker Services

The `docker-compose.desoutter.yml` orchestrates 4 services:

| Service | Port | Description | Resources |
|---------|------|-------------|-----------|
| **mongodb** | 27017 | Database | 1 core, 2GB RAM |
| **ollama** | 11434 | LLM server with GPU | 2 cores, 8GB RAM, GPU |
| **desoutter-api** | 8000 | FastAPI backend | 3 cores, 12GB RAM |
| **desoutter-frontend** | 3001 | React frontend | 1 core, 1GB RAM |

### Docker Commands

```bash
# Start all services
docker-compose -f docker-compose.desoutter.yml up -d

# View logs
docker-compose -f docker-compose.desoutter.yml logs -f

# Stop all services
docker-compose -f docker-compose.desoutter.yml down

# Rebuild after code changes
docker-compose -f docker-compose.desoutter.yml up -d --build
```

---

## Testing

### Running Tests

```bash
# Run full test suite
./scripts/run_baseline_test.sh

# Run specific test module
pytest tests/test_rag_comprehensive.py -v

# Test hybrid search
python scripts/test_hybrid_search.py

# Test product filtering
python scripts/test_product_filtering.py
```

### Test Categories

| Category | Scripts | Purpose |
|----------|---------|---------|
| RAG Pipeline | `test_rag.py`, `test_rag_comprehensive.py` | End-to-end retrieval testing |
| Hybrid Search | `test_hybrid_search.py` | BM25 + Semantic fusion |
| Response Validation | `test_response_validator.py` | Hallucination detection |
| Product Filtering | `test_product_filtering.py` | Family-specific retrieval |
| Cache Performance | `test_cache.py` | Hit rate and speedup |
| Context Grounding | `test_context_grounding.py` | Uncertainty handling |

---

## Key Files Reference

| File | Description |
|------|-------------|
| [src/llm/rag_engine.py](src/llm/rag_engine.py) | 14-stage RAG pipeline orchestrator (81KB, main logic) |
| [src/llm/hybrid_search.py](src/llm/hybrid_search.py) | BM25 + Semantic search with RRF fusion |
| [src/llm/self_learning.py](src/llm/self_learning.py) | Feedback learning engine with Wilson scores |
| [src/llm/intent_detector.py](src/llm/intent_detector.py) | 8-category query classification |
| [src/llm/response_validator.py](src/llm/response_validator.py) | Hallucination and validation checks |
| [src/llm/confidence_scorer.py](src/llm/confidence_scorer.py) | Multi-factor confidence calculation |
| [src/llm/context_optimizer.py](src/llm/context_optimizer.py) | Token budget and deduplication |
| [src/llm/response_cache.py](src/llm/response_cache.py) | LRU + TTL caching layer |
| [src/api/main.py](src/api/main.py) | FastAPI routes and middleware |
| [src/documents/semantic_chunker.py](src/documents/semantic_chunker.py) | Intelligent document segmentation |
| [src/database/mongo_client.py](src/database/mongo_client.py) | MongoDB operations and models |
| [config/ai_settings.py](config/ai_settings.py) | RAG parameters and thresholds |
| [frontend/src/App.jsx](frontend/src/App.jsx) | Main React component |

---

## Security Considerations

### Production Checklist

1. **Change JWT_SECRET** - Never use default secret in production
2. **Restrict CORS origins** - Replace `*` with specific domains
3. **Enable rate limiting** - Protect against DoS attacks
4. **Enable MongoDB authentication** - Require username/password
5. **Use HTTPS** - TLS termination via reverse proxy
6. **Audit input validation** - Sanitize all user queries

### Current Limitations

| Risk | Status | Mitigation |
|------|--------|------------|
| Default credentials | Medium | Change on first deployment |
| Open CORS policy | Medium | Configure allowed origins |
| No rate limiting | Medium | Add nginx/API gateway limits |
| Single GPU dependency | Low | CPU fallback available |

---

## Roadmap

### Completed (January 2026)
- [x] 14-stage RAG Pipeline with 96% test pass rate
- [x] Hybrid Search (BM25 + Semantic + RRF)
- [x] Self-Learning Feedback Loop
- [x] Multi-turn Conversation
- [x] GPU Acceleration with RTX A2000
- [x] Intelligent Product Filtering
- [x] 28,414 chunks re-ingested with product metadata

### In Progress (Q1 2026)
- [ ] Freshdesk Ticket Integration
- [ ] Controller Units Scraping

### Planned (Q2 2026)
- [ ] Field Installation Support (extend beyond repair to on-site tool setup and commissioning)
- [ ] Qdrant Migration (10x scalability, 100M+ vectors)
- [ ] Prompt Caching (40% latency reduction)
- [ ] Async Ingestion Queue (Celery + Redis)
- [ ] Fine-tuned Embeddings (15-20% accuracy improvement)
- [ ] Advanced KPI Dashboard
- [ ] Service Management System Integration

---

## Additional Documentation

| Document | Description |
|----------|-------------|
| [TECHNICAL_ANALYSIS.md](TECHNICAL_ANALYSIS.md) | Deep-dive architecture and implementation details |
| [QUICKSTART.md](QUICKSTART.md) | Rapid deployment guide |
| [ROADMAP.md](ROADMAP.md) | Detailed development roadmap |
| [CHANGELOG.md](CHANGELOG.md) | Version history and changes |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Python: Follow PEP 8, use type hints
- JavaScript: ESLint + Prettier configuration
- Commits: Conventional commits format

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Fatih Bayram** - [@fatihhbayram](https://github.com/fatihhbayram)

---

## Acknowledgments

- **Ollama** - Local LLM serving infrastructure
- **ChromaDB** - High-performance vector database
- **HuggingFace** - Sentence transformers and model hub
- **FastAPI** - Modern Python web framework
- **LangChain** - RAG orchestration framework

---

<p align="center">
  <strong>Powered by</strong> Ollama | ChromaDB | FastAPI | React | BM25
  <br><br>
  <em>Production-Ready Enterprise RAG System v1.8.0</em>
</p>
