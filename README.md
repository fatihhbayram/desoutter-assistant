# Desoutter Repair Assistant

AI-powered repair assistant for Desoutter industrial tools. Uses RAG (Retrieval Augmented Generation) with **self-learning capabilities** to provide intelligent repair suggestions based on technical manuals and service bulletins.

**Repository**: https://github.com/fatihhbayram/desoutter-assistant

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
- Product browser with grid/list view (451 products)
- Search and filters (series, wireless)
- Product selection with details
- Fault description input (Turkish/English)
- AI-powered diagnosis with confidence level
- **Feedback System**: 👍/👎 buttons for self-learning
- **Source Document Viewer**: Open related documents directly from results

## 📊 System Metrics (31 Aralık 2025)

| Metrik | Değer |
|--------|-------|
| Toplam Ürün | 451 (71 wireless, 380 non-wireless) |
| ChromaDB Chunks | ~22,889 semantic chunks (Improved Granularity) |
| Dökümanlar | 541 (121 PDF + 420 Word) |
| Fault Categories | 15 (relevance filtering) |
| Domain Terms | 351 Desoutter-specific |
| LLM Model | qwen2.5:7b-instruct |
| Embedding Model | all-MiniLM-L6-v2 (384-dim) |
| GPU | NVIDIA RTX A2000 (6GB) |
| Hybrid Search | BM25 + Semantic + RRF |
| Response Cache | LRU + TTL (~100,000x speedup) |

## 📝 Recent Updates

### 2025-12-31: Source Citation Enhancement (Critical Fix) 🎯 **NEW**
- **2025-12-31:** Smart Product Recognition (Metadata Enrichment for precision filtering). 🎯 **NEW**
- ✅ **Fixed Page Number Extraction**
  - Refactored `clean_text` to preserve paragraph structure
  - Improved regex for robust page marker detection
- ✅ **Full Re-Ingestion Complete**
  - Database purged and rebuilt
  - 100% of documents (541) re-processed
  - **Chunk count increased to ~22,889** due to better granularity (paragraph-level splitting)
- ✅ **Verified Metadata**
  - All chunks now possess valid `page_number` and `section` tags
  - RAG responses now provide accurate citations (e.g., "Page 12")

### 2025-12-30: RAG Quality Improvements Phase 2 🚀 **NEW**
- ✅ **Intent Detection Integration (Priority 3)**
  - 8 intent types with dynamic prompt selection
  - Intent metadata in API responses
  - Specialized prompts for each query type
- ✅ **Content Deduplication (Priority 4)**
  - SHA-256 hash-based duplicate detection
  - Prevents duplicate chunks in vector DB
  - Configurable via `ENABLE_DEDUPLICATION`
- ✅ **Adaptive Chunk Sizing (Priority 5)**
  - Document type-based chunk sizing
  - Troubleshooting: 200 tokens (precision)
  - Manuals: 400 tokens (context)
- ✅ **Wireless Detection Bug Fix**
  - Fixed migration script logic
  - 80 battery tools corrected (EABS, EIBS, EPB, EAB series)
  - System now provides accurate WiFi troubleshooting

### 2025-12-29: RAG Quality Improvements Phase 1 🌟
- ✅ **Context Grounding ("I Don't Know" Logic)**
  - Multi-factor context sufficiency scoring
  - Automatic "I don't know" responses for insufficient context
  - Significant reduction in hallucinations
- ✅ **Response Validation System**
  - Post-processing layer to catch hallucinations
  - Validates numerical values against source context
  - Detects forbidden content (e.g. WiFi suggestions for non-WiFi tools)
  - Auto-flags low-quality responses

### 2025-12-27: RAG Relevance Filtering & Query Processing
- ✅ **Phase 0.1 RAG Relevance Filtering**
  - 15 fault categories with negative keyword filtering
  - Word boundary matching (prevents false positives)
  - 70-80% reduction in irrelevant results
  - Test: 10/10 intent detection passed
- ✅ **Phase 2.2 Query Processor**
  - Centralized query processing (normalize, keywords, intent)
  - Turkish + English support
  - Language detection
- ✅ **Document Expansion**: 541 docs → 6,798 chunks (+116%)

### 2025-12-22: ALL RAG PHASES COMPLETE
- ✅ **Phase 3.1 Domain Embeddings**: 351 Desoutter-specific terms
- ✅ **Phase 6 Self-Learning Feedback Loop**: Wilson score ranking
- ✅ **Phase 5.1 Performance Metrics**: Query latency, cache hit rates
- ✅ **Phase 3.5 Multi-turn Conversation**: Context preservation

**Details**: See [CHANGELOG.md](CHANGELOG.md) and [RAG_QUALITY_IMPROVEMENT.md](RAG_QUALITY_IMPROVEMENT.md)

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

## 📖 Additional Documentation

- `QUICKSTART.md` — Quick setup steps
- `PROXMOX_DEPLOYMENT.md` — Proxmox deployment notes
- `PHASE2_STRUCTURE.md` — Phase 2 architecture
- `CHANGELOG.md` — Detailed changelog
- `ROADMAP.md` — Development roadmap
- `RAG_ENHANCEMENT_ROADMAP.md` — RAG enhancement phases

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
