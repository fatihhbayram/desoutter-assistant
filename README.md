# Desoutter Repair Assistant

AI-powered repair assistant for Desoutter industrial tools. Uses RAG (Retrieval Augmented Generation) with **self-learning capabilities** to provide intelligent repair suggestions based on technical manuals and service bulletins.

**Repository**: https://github.com/fatihhbayramm/desoutter-assistant

## 🎯 Key Features

- **🧠 Self-Learning RAG**: System learns from user feedback to improve future suggestions
- **🎯 RAG Relevance Filtering**: 15 fault categories filter irrelevant documents (Phase 0.1 - NEW!)
- **🔌 Connection Architecture**: 6 product family mappings for intelligent troubleshooting
- **📚 Domain Embeddings**: 351 Desoutter-specific terms with query enhancement (Phase 3.1)
- **🔄 Multi-turn Conversation**: Follow-up questions with context preservation (Phase 3.5)
- **📊 Performance Monitoring**: Query latency, cache hit rates, health status (Phase 5)
- **🔍 Hybrid Search**: BM25 keyword + Semantic vector search with RRF fusion
- **⚡ Response Caching**: LRU + TTL cache with ~100,000x speedup for repeated queries
- **📊 Admin Dashboard**: Comprehensive analytics with trends, top products, and feedback stats
- **🎯 Context Optimization**: Token budget management, deduplication, warning prioritization
- **💾 GPU Acceleration**: NVIDIA GPU inference for fast LLM responses
- **Smart Product Scraping**: Handles Next.js rendered pages with advanced image extraction from DatoCMS assets
- **MongoDB Integration**: Stores 451 products (71 wireless, 380 non-wireless)
- **ChromaDB Vector Store**: 6,798 semantic chunks from 541 documents
- **Multi-Format Documents**: Support for PDF, Word (DOCX), PowerPoint (PPTX), Excel (XLSX)
- **Document Viewer**: Open source documents directly from diagnosis results
- **Source Relevance Feedback**: Rate each retrieved document as relevant/irrelevant
- **Multi-Language UI**: Turkish and English interface support
- **Responsive Design**: Works on desktop, tablet, and mobile
- **JWT Authentication**: Role-based access control (Admin / Technician)
- **Modern Stack**: FastAPI + React + Docker Compose for easy deployment

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with CUDA (optional, for faster inference)
- Ollama with `qwen2.5:7b-instruct` or `llama3:latest` model

### Run with Docker

```bash
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
cd frontend
docker build -t desoutter-frontend .
docker run -d --name desoutter-frontend -p 3001:3001 desoutter-frontend
```

### Access
- **Frontend**: http://localhost:3001
- **API Docs**: http://localhost:8000/docs
- **Simple UI**: http://localhost:8000/ui

### Default Users
| Username | Password | Role |
|----------|----------|------|
| admin | admin123 | Admin |
| tech | tech123 | Technician |

## 📚 API Endpoints

### Authentication
- `POST /auth/login` - Login and get JWT token

### Diagnosis
- `POST /diagnose` - Get AI-powered repair suggestion
- `POST /diagnose/feedback` - Submit feedback (👍/👎) for learning
- `GET /diagnose/history` - Get user's diagnosis history
- `GET /products` - List all products
- `GET /stats` - System statistics

### Admin (requires admin role)
- `GET /admin/dashboard` - Get comprehensive dashboard statistics
- `GET /admin/users` - List users
- `POST /admin/users` - Create user
- `DELETE /admin/users/{username}` - Delete user
- `GET /admin/documents` - List uploaded documents
- `POST /admin/documents/upload` - Upload document (PDF, DOCX, PPTX)
- `DELETE /admin/documents/{type}/{filename}` - Delete document
- `POST /admin/documents/ingest` - Process documents into RAG

### Documents
- `GET /documents/download/{filename}` - Download/view source document

## 🖥️ Frontend Features

### Admin Panel
- System statistics dashboard
- User management (add/delete users)
- **RAG Document Management**:
  - Upload PDF, Word (DOCX), PowerPoint (PPTX) documents
  - View uploaded documents list with format icons
  - Delete documents
  - Re-index all documents into vector database
- Maintenance actions (scraper, refresh data)

### Technician Panel
- Product browser with grid/list view (237+ products)
- Search and filters (series, wireless)
- Product selection with details
- Fault description input (Turkish/English)
- AI-powered diagnosis with confidence level
- **Feedback System**: 👍/👎 buttons for self-learning
- **Source Document Viewer**: Open related documents directly from results

## 🧠 Self-Learning RAG System

The system learns from user feedback to continuously improve:

### How It Works
1. **User submits feedback** after receiving a diagnosis
2. **Positive feedback (👍)**: Reinforces the fault-solution mapping
3. **Negative feedback (👎)**: Records the pattern to avoid, offers alternative

### Learning Components
- `DiagnosisFeedback`: Records all user feedback
- `LearnedMapping`: Stores successful fault-solution patterns
- `DiagnosisHistory`: Tracks all diagnoses for analytics

### Confidence Score
- Based on positive/negative feedback ratio
- Boosted with more samples (max at 10+ feedbacks)
- High confidence patterns prioritized in future suggestions

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
EMBEDDING_DEVICE=cpu  # or cuda

# JWT
JWT_SECRET=your-secret-key
```

## 📁 Project Structure

```
desoutter-assistant/
├── src/
│   ├── api/           # FastAPI application
│   ├── database/      # MongoDB client, feedback models
│   ├── documents/     # PDF processor, chunker, embeddings
│   ├── llm/           # Ollama client, RAG engine, feedback engine
│   ├── scraper/       # Product scraper
│   ├── vectordb/      # ChromaDB client
│   └── utils/         # Logger, helpers
├── frontend/
│   └── src/           # React Vite application
├── config/
│   ├── settings.py    # Base configuration
│   └── ai_settings.py # AI/RAG configuration
├── documents/
│   ├── manuals/       # PDF manuals
│   └── bulletins/     # Service bulletins
├── scripts/           # Utility scripts
├── Dockerfile
└── docker-compose.yml
```

## 📝 Recent Updates

### 2025-12-22: ALL RAG PHASES COMPLETE ✨ **NEW**
- ✅ **Phase 3.1 Domain Embeddings**
  - 351 Desoutter-specific terms (27 product series, 29 error codes)
  - Query enhancement with synonym expansion
  - Domain-aware similarity boosting
  - Entity extraction (products, error codes, components)
  - New admin endpoints: `/admin/domain/*`
- ✅ **Phase 6 Self-Learning Feedback Loop**
  - Wilson score interval for source ranking
  - Keyword-to-source mapping
  - Training data collection for future fine-tuning
  - New admin endpoints: `/admin/learning/*`
- ✅ **Phase 5.1 Performance Metrics**
  - Query latency tracking (retrieval, LLM, total time)
  - Cache hit/miss rate monitoring
  - P95/P99 latency percentiles
  - Health status monitoring with issue detection
  - New admin endpoints: `/admin/metrics/*`
- ✅ **Phase 3.5 Multi-turn Conversation**
  - Conversation session management (30 min timeout)
  - Context preservation across follow-up questions
  - Reference resolution ("it", "this tool" → actual product)
  - History-aware prompts for better context
  - New endpoints: `/conversation/*`
- ✅ **Document Update**: 487 documents (484 bulletins + 3 manuals)

### 2025-12-17: Async Concurrency & Domain Knowledge
- ✅ **Async Concurrency Fix**: Wrapped blocking LLM calls with `asyncio.to_thread()` 
  - Multiple users can now access the system simultaneously
  - Health/products endpoints respond in ~40ms even during diagnose operations
  - Previously blocked for 30+ seconds during LLM inference
- ✅ **Desoutter Connection Architecture**: Dynamic tool connection guidance
  - WiFi-enabled tools: Connect via WiFi to Connect Unit or AP
  - Non-WiFi tools: Connect via tool cable to CVI3
  - No more incorrect "check ethernet" suggestions
- ✅ **Self-Learning Verified**: Feedback system actively learning
  - 4 learned mappings with confidence scores up to 1.0
  - Boost sources automatically applied for similar queries
  - Negative feedback excludes problematic sources

### 2025-12-17: Phase 4.1 Metadata Boosting
- ✅ **Metadata-Based Filtering & Boosting**
  - Service bulletins (ESD/ESB) get 1.5x boost
  - Procedure sections get 1.3x boost
  - Warning content gets 1.2x boost
- ✅ **Re-ingested Documents**: 1514 semantic chunks with full metadata
- ✅ **BM25 Index**: 17,050 unique terms for keyword search

### 2025-12-17: Phase 3.3 & 3.4 Complete
- ✅ **Source Relevance Feedback**: Users can rate each source as relevant/irrelevant
- ✅ **Context Window Optimization**: Token budget, deduplication, warning prioritization
- ✅ **Ollama GPU Activation**: NVIDIA RTX A2000 GPU inference enabled
- ✅ **Test Suites**: 5/5 context optimizer tests passing

### 2025-12-16: Phase 2.2 & 2.3 Complete
- ✅ **Hybrid Search**: BM25 keyword search + Semantic search combined
- ✅ **RRF Fusion**: Reciprocal Rank Fusion for score combination
- ✅ **Query Expansion**: Domain-specific synonym expansion (9 categories)
- ✅ **Response Caching**: LRU + TTL cache with ~100,000x speedup
- ✅ **Document Re-ingestion**: 276 docs → 2318 vectors with semantic chunks
- ✅ **Test Suites**: 5/5 hybrid search + 4/4 cache tests passing

**Details**: See [CHANGELOG.md](CHANGELOG.md) and [RAG_ENHANCEMENT_ROADMAP.md](RAG_ENHANCEMENT_ROADMAP.md)

### 2025-12-15: RAG Retrieval Quality Optimization
- ✅ **Similarity Threshold Optimization**: Dynamic filtering based on RAG_SIMILARITY_THRESHOLD config
- ✅ **L2 Distance Conversion**: Proper similarity score calculation from distance metrics
- ✅ **Testing & Tuning**: Thresholds tested from 0.85 to 0.30, optimal value confirmed
- ✅ **Multi-Format Support**: Excel (XLSX/XLS) document parsing added
- ✅ **Quality Validation**: Different fault types return relevant document combinations

**Details**: See [CHANGELOG.md](CHANGELOG.md) for technical implementation

### 2025-12-14: Tech Page Wizard & Infrastructure Fix
- ✅ **TechWizard Component**: 4-step wizard-style UI for technicians
- ✅ **MongoDB Configuration**: Fixed localhost connection
- ✅ **Backend Feedback Fix**: HTTP 422 error resolved
- ✅ **Infrastructure**: All 7 Docker services running and healthy

### 2025-12-09: Self-Learning RAG Feedback System
- ✅ **Feedback system**: 👍/👎 buttons for user feedback
- ✅ **Self-learning engine**: System learns from feedback
- ✅ **Diagnosis history**: All diagnoses saved to MongoDB
- ✅ **Learned mappings**: Fault-solution patterns stored

### 2025-12-08: Auto GPU Preload & Responsive Design
- ✅ **Ollama preload**: Model auto-loads to GPU on server restart
- ✅ **Responsive design**: Desktop, tablet, mobile support

### 2025-12-02: Security & UI Polish
- ✅ Session persistence across page refresh
- ✅ Auto-logout on token expiry

### 2025-12-01: Admin Document Management
- ✅ RAG document management panel
- ✅ Multi-format upload support (PDF, DOCX, PPTX, XLSX, XLS)
- ✅ Document ingestion to vector DB

## 🧪 Testing

```bash
# Health check
curl http://localhost:8000/health

# Login
curl -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123"}'

# Diagnose (with token)
curl -X POST http://localhost:8000/diagnose \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_TOKEN' \
  -d '{"part_number":"6151659770","fault_description":"motor not starting","language":"en"}'
```

## 📊 System Metrics (16 Aralık 2025)

| Metrik | Değer |
|--------|-------|
| Toplam Ürün | 237 |
| VectorDB Chunks | **2309** (1080 original + 1229 semantic) |
| Yüklü Dokuman | 276 (bulletins + manuals) |
| BM25 Index | **13026 unique terms** |
| RAG Similarity Threshold | 0.30 (dynamic, configurable) |
| Sources Per Diagnosis | 3-5 relevant documents |
| LLM Model | qwen2.5:7b-instruct |
| Embedding Model | all-MiniLM-L6-v2 (384-dim) |
| **Semantic Chunking** | **✅ Phase 1 COMPLETE** |
| **Hybrid Search** | **✅ Phase 2.2 COMPLETE** |
| Chunking Strategy | Recursive with structure preservation |
| Chunk Size | 400 characters with 100 char overlap |
| Metadata Fields | 14 per chunk (importance, keywords, type, etc) |
| Document Type Detection | 5 types (Manual, Bulletin, Guide, Catalog, Safety) |
| Fault Keywords | 9 categories (motor, noise, mechanical, electrical, etc) |
| Query Expansion | 9 synonym categories (domain-specific) |
| Hybrid Weights | Semantic: 0.7, BM25: 0.3 |
| GPU | NVIDIA RTX A2000 (6GB) |

## 📖 Additional Documentation

- `QUICKSTART.md` — Quick setup steps
- `PROXMOX_DEPLOYMENT.md` — Proxmox deployment notes
- `PHASE2_STRUCTURE.md` — Phase 2 architecture
- `CHANGELOG.md` — Detailed changelog (See [16 Aralık 2025 update](CHANGELOG.md#-16-aralık-2025-pazartesi))
- `ROADMAP.md` — Development roadmap

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details.

---

**Powered by**: Ollama + ChromaDB + FastAPI + React + BM25

🏗️ Running on Proxmox AI Infrastructure
