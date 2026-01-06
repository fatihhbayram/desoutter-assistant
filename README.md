# 🔧 Desoutter Assistant

> **AI-Powered Technical Support System for Industrial Tools**

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Tests](https://img.shields.io/badge/Tests-96%25%20Passing-success)

An enterprise-grade **RAG (Retrieval-Augmented Generation)** system that provides intelligent, context-aware repair and troubleshooting assistance for Desoutter industrial tools. Built with a self-learning feedback loop that continuously improves response quality.

**Repository**: [github.com/fatihhbayram/desoutter-assistant](https://github.com/fatihhbayram/desoutter-assistant)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Hybrid Search** | BM25 keyword + Semantic vector search with RRF (Reciprocal Rank Fusion) |
| 🎯 **Intelligent Product Filtering** | Auto-detects product family from queries, filters retrieval to relevant docs only |
| 📈 **Pattern-Based Boosting** | Regex error code detection + phrase matching for bulletin prioritization |
| 🧠 **Self-Learning RAG** | Learns from user feedback to improve future suggestions |
| 🎯 **96% Test Pass Rate** | Comprehensive automated test suite with 25 scenarios |
| 🔄 **Multi-turn Conversation** | Follow-up questions with context preservation |
| ⚡ **Response Caching** | LRU + TTL cache with ~100,000x speedup for repeated queries |
| 🚫 **Hallucination Prevention** | Context grounding + response validation + "I don't know" logic |
| 📊 **Intent Detection** | 8 query types with specialized prompts |
| 💾 **GPU Acceleration** | NVIDIA GPU inference for fast LLM responses |
| 🌐 **Multi-Language** | Turkish and English interface support |
| 🔐 **JWT Authentication** | Role-based access control (Admin / Technician) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                      (React Frontend)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         FASTAPI                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Routes    │  │  Services   │  │     RAG Engine          │ │
│  │  /api/chat  │──│  diagnosis  │──│  • Hybrid Search        │ │
│  │  /api/learn │  │  feedback   │  │  • Query Expansion      │ │
│  │  /api/docs  │  │  document   │  │  • Product Filtering    │ │
│  └─────────────┘  └─────────────┘  │  • Intent Detection     │ │
│                                     └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │                    │                      │
         ▼                    ▼                      ▼
┌─────────────┐      ┌─────────────┐        ┌─────────────┐
│   MongoDB   │      │   Ollama    │        │  ChromaDB   │
│  Feedback   │      │ Qwen2.5:7b  │        │  Vectors    │
│  Mappings   │      │   (GPU)     │        │  Documents  │
└─────────────┘      └─────────────┘        └─────────────┘
```

### 🔥 14-Stage RAG Pipeline

The core of our system - a production-grade retrieval pipeline that achieves **96% test pass rate**:

```
┌──────────────────────────────────────────────────────────────┐
│                       USER QUERY                             │
└──────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────────┐
    │                         ▼                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ 1️⃣  OFF-TOPIC DETECTION                         │  │
    │  │     Rejects non-relevant queries               │  │
    │  └────────────────────────────────────────────────┘  │
    │                         │                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ 2️⃣  LANGUAGE DETECTION (TR/EN)                  │  │
    │  │     Auto-detects query language                │  │
    │  └────────────────────────────────────────────────┘  │
    │                         │                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ 3️⃣  RESPONSE CACHE CHECK                        │  │
    │  │     ~100,000x speedup on cache hit             │  │
    │  └────────────────────────────────────────────────┘  │
    │                         │                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ 4️⃣  SELF-LEARNING CONTEXT                       │  │
    │  │     Applies learned mappings & boosts          │  │
    │  └────────────────────────────────────────────────┘  │
    │                         │                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ 5️⃣  HYBRID RETRIEVAL                            │  │
    │  │     • Semantic Search (0.7 weight)             │  │
    │  │     • BM25 Keyword Search (0.3 weight)         │  │
    │  │     • RRF Fusion (k=60)                        │  │
    │  └────────────────────────────────────────────────┘  │
    │                         │                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ 6️⃣  STRICT PRODUCT FILTERING                    │  │
    │  │     Prevents cross-product contamination       │  │
    │  └────────────────────────────────────────────────┘  │
    │                         │                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ 7️⃣  CAPABILITY FILTERING                        │  │
    │  │     WiFi/Battery content filtering             │  │
    │  └────────────────────────────────────────────────┘  │
    │                         │                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ 8️⃣  CONTEXT GROUNDING                           │  │
    │  │     Returns "I don't know" if uncertain        │  │
    │  └────────────────────────────────────────────────┘  │
    │                         │                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ 9️⃣  CONTEXT OPTIMIZATION                        │  │
    │  │     8K token budget, deduplication             │  │
    │  └────────────────────────────────────────────────┘  │
    │                         │                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ 🔟 INTENT DETECTION                             │  │
    │  │     8 intent types with custom prompts         │  │
    │  └────────────────────────────────────────────────┘  │
    │                         │                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ 1️⃣1️⃣ LLM GENERATION                              │  │
    │  │      Qwen2.5:7b with GPU acceleration          │  │
    │  └────────────────────────────────────────────────┘  │
    │                         │                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ 1️⃣2️⃣ RESPONSE VALIDATION                         │  │
    │  │      Hallucination & forbidden content check   │  │
    │  └────────────────────────────────────────────────┘  │
    │                         │                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ 1️⃣3️⃣ CONFIDENCE SCORING                          │  │
    │  │      Multi-factor scoring algorithm            │  │
    │  └────────────────────────────────────────────────┘  │
    │                         │                             │
    │  ┌────────────────────────────────────────────────┐  │
    │  │ 1️⃣4️⃣ SAVE & CACHE                                │  │
    │  │      MongoDB persistence + response cache      │  │
    │  └────────────────────────────────────────────────┘  │
    │                         │                             │
    └─────────────────────────┼─────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                       AI RESPONSE                            │
│              (With confidence score & sources)               │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA (optional, for faster inference)
- Ollama with `qwen2.5:7b-instruct` model

### Run with Docker

```bash
# Clone the repository
git clone https://github.com/fatihhbayram/desoutter-assistant.git
cd desoutter-assistant

# Start API (connect to existing ai-net network with MongoDB & Ollama)
sudo docker run -d --name desoutter-api \
  --network ai-net \
  -p 8000:8000 \
  -e MONGO_HOST=mongodb \
  -e OLLAMA_BASE_URL=http://ollama:11434 \
  -e OLLAMA_MODEL=qwen2.5:7b-instruct \
  -v desoutter_data:/app/data \
  -v /path/to/documents:/app/documents \
  -v huggingface_cache:/root/.cache/huggingface \
  --gpus all \
  desoutter-api

# Start Frontend
cd frontend && docker build -t desoutter-frontend .
docker run -d --name desoutter-frontend -p 3001:3001 desoutter-frontend
```

### Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:3001 | Main user interface |
| **API Docs** | http://localhost:8000/docs | Swagger API documentation |
| **Simple UI** | http://localhost:8000/ui | Lightweight web interface |

### Default Users

| Username | Password | Role |
|----------|----------|------|
| admin | admin123 | Admin |
| tech | tech123 | Technician |

> 📖 For detailed setup instructions, see [QUICKSTART.md](QUICKSTART.md)

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Pass Rate** | 96% (24/25 scenarios) |
| **Total Products** | 451 (71 wireless, 380 cable) |
| **ChromaDB Chunks** | ~28,414 semantic chunks |
| **Documents Indexed** | 541 (121 PDF + 420 Word) |
| **Freshdesk Tickets** | 2,249 scraped & ingested |
| **Domain Terms** | 351 Desoutter-specific |
| **BM25 Index Terms** | 19,032 unique terms |
| **Intent Types** | 8 specialized categories |
| **Fault Categories** | 15 relevance filters |
| **LLM Model** | Qwen2.5:7b-instruct |
| **Embedding Model** | all-MiniLM-L6-v2 (384-dim) |
| **GPU** | NVIDIA RTX A2000 (6GB) |
| **Cache Speedup** | ~100,000x for repeated queries |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **LLM** | Ollama + Qwen2.5:7b | Natural language understanding & generation |
| **Vector DB** | ChromaDB | Semantic document storage & retrieval |
| **Keyword Search** | BM25 | Fast keyword-based retrieval |
| **Backend** | FastAPI (Python) | REST API & business logic |
| **Frontend** | React + Vite | User interface |
| **Database** | MongoDB | Data persistence, feedback storage |
| **Embeddings** | Sentence Transformers | Document vectorization |
| **Deployment** | Docker + Docker Compose | Containerization |
| **Infrastructure** | Proxmox VM | Virtualization platform |
| **GPU** | NVIDIA RTX A2000 | Model acceleration |

---

## 📚 API Endpoints

### Authentication
- `POST /auth/login` - Login and get JWT token
- `GET /auth/me` - Validate token and get user info

### Diagnosis
- `POST /diagnose` - Get AI-powered repair suggestion
- `POST /diagnose/feedback` - Submit feedback (👍/👎) for learning
- `GET /diagnose/history` - Get user's diagnosis history

### Conversation
- `POST /conversation/start` - Start or continue multi-turn conversation
- `GET /conversation/{id}` - Get conversation history
- `DELETE /conversation/{id}` - End conversation

### Admin
- `GET /admin/dashboard` - Comprehensive dashboard statistics
- `GET /admin/metrics/health` - System health status
- `GET /admin/metrics/stats` - Performance statistics
- `GET /admin/learning/stats` - Self-learning statistics
- `POST /admin/documents/upload` - Upload document (PDF, DOCX, PPTX)
- `POST /admin/documents/ingest` - Process documents into RAG

---

## 🧠 Self-Learning System

The system learns from user feedback to continuously improve:

```
User Query → RAG Retrieval → LLM Response → User Feedback
                                                  ↓
                                         ┌───────────────┐
                                         │ 👍 Positive   │───→ Reinforce mapping
                                         │ 👎 Negative   │───→ Record pattern to avoid
                                         └───────────────┘
                                                  ↓
                                         Wilson Score Ranking
                                                  ↓
                                         Improved Future Results
```

**Learning Components:**
- **DiagnosisFeedback**: Records all user feedback
- **LearnedMapping**: Stores successful fault-solution patterns
- **SourceRankingLearner**: Wilson score-based source prioritization
- **ContrastiveLearningManager**: Collects training data for embedding fine-tuning

---

## 📁 Project Structure

```
desoutter-assistant/
├── src/
│   ├── api/              # FastAPI application
│   ├── database/         # MongoDB client, feedback models
│   ├── documents/        # PDF processor, semantic chunker
│   ├── llm/              # RAG engine, hybrid search, self-learning
│   ├── scraper/          # Product & ticket scraper
│   └── vectordb/         # ChromaDB client
├── frontend/             # React Vite application
├── config/               # Configuration files
├── documents/            # Technical manuals & service bulletins
├── scripts/              # Utility & test scripts
├── test_results/         # Automated test outputs
├── Dockerfile
└── docker-compose.yml
```

---

## 🔧 Configuration

### Environment Variables

```bash
# MongoDB
MONGO_HOST=mongodb
MONGO_PORT=27017
MONGO_DATABASE=desoutter

# Ollama LLM
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=qwen2.5:7b-instruct

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cuda  # or cpu

# Hybrid Search
USE_HYBRID_SEARCH=true
HYBRID_SEMANTIC_WEIGHT=0.7
HYBRID_BM25_WEIGHT=0.3

# JWT
JWT_SECRET=your-secret-key
```

---

## 🧪 Testing

```bash
# Health check
curl http://localhost:8000/health

# Login and get token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123"}' | jq -r '.access_token')

# Diagnose with token
curl -X POST http://localhost:8000/diagnose \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"part_number":"6151659770","fault_description":"motor not starting","language":"en"}'
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Rapid deployment guide |
| [PROXMOX_DEPLOYMENT.md](PROXMOX_DEPLOYMENT.md) | Infrastructure setup for Proxmox |
| [RAG_QUALITY_IMPROVEMENT.md](RAG_QUALITY_IMPROVEMENT.md) | Technical deep-dive into RAG system |
| [ROADMAP.md](ROADMAP.md) | Development roadmap and future plans |
| [CHANGELOG.md](CHANGELOG.md) | Version history and changes |

---

## 🗺️ Roadmap

**Current Status:** Production-Ready RAG System with Self-Learning

### Completed (Jan 2026)
- ✅ Intelligent Product Filtering (ChromaDB where clause)
- ✅ Pattern-based Product Extraction (no manual mappings)
- ✅ 26,528 chunks re-ingested with product metadata

### Completed (Dec 2025)
- ✅ Hybrid Search (BM25 + Semantic + RRF)
- ✅ Self-Learning Feedback Loop
- ✅ Multi-turn Conversation
- ✅ Intent Detection (8 types)
- ✅ Response Validation & Hallucination Prevention
- ✅ GPU Acceleration

### In Progress
- 🔄 Freshdesk Ticket Integration
- 🔄 Controller Units Scraping

### Planned
- 📋 Service Management System
- 📋 KPI Dashboard
- 📋 Embedding Fine-tuning

> 📖 See [ROADMAP.md](ROADMAP.md) for detailed planning

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Fatih Bayram**

- GitHub: [@fatihhbayram](https://github.com/fatihhbayram)

---

<p align="center">
  <strong>Powered by</strong> Ollama • ChromaDB • FastAPI • React • BM25
  <br>
  🏗️ Running on Proxmox AI Infrastructure
</p>
