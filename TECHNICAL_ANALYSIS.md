# Desoutter Assistant - Technical Deep-Dive Analysis

**Document Version:** 1.0.0
**Analysis Date:** January 15, 2026
**Author:** Senior AI Software Architect
**Project Version:** v1.8.0

---

## Executive Summary

The **Desoutter Assistant** is an enterprise-grade Retrieval-Augmented Generation (RAG) system engineered for intelligent technical support in industrial tool maintenance. The system combines advanced AI/ML techniques including hybrid search (semantic + BM25), self-learning feedback loops, and multi-stage quality validation to deliver accurate, context-aware repair suggestions for Desoutter industrial tools.

**Key Metrics:**
- **Test Pass Rate:** 96% (24/25 scenarios)
- **Document Corpus:** 541 documents (28,414 semantic chunks)
- **Product Coverage:** 451 industrial tools
- **Freshdesk Integration:** 2,249 support tickets ingested
- **Response Cache Speedup:** ~100,000x for repeated queries

---

## 1. Tech Stack Identification

### 1.1 Core Programming Languages
- **Python 3.11+** - Primary backend language
- **JavaScript (ES6+)** - Frontend development
- **SQL/NoSQL** - Data persistence layer

### 1.2 AI/ML Framework & Libraries

#### LLM & Embeddings
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **LLM Inference** | Ollama | 0.1.6 | Local GPU-accelerated LLM serving |
| **Model** | Qwen2.5:7b-instruct | 7B params | Technical content generation |
| **Embeddings** | Sentence Transformers | 2.2.2 | Document vectorization |
| **Embedding Model** | all-MiniLM-L6-v2 | 384-dim | Semantic similarity search |
| **Deep Learning** | PyTorch | 2.1.2 | Neural network backend |
| **Transformers** | HuggingFace Transformers | 4.36.2 | Model loading & inference |

#### RAG Components
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Vector Database** | ChromaDB | 0.4.22 | Semantic document storage & retrieval |
| **Orchestration** | LangChain | 0.1.0 | RAG workflow management |
| **Keyword Search** | BM25 (Custom) | N/A | Sparse retrieval for exact matches |
| **Fusion Algorithm** | RRF (k=60) | N/A | Score combination for hybrid search |

### 1.3 Backend Framework & API
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Web Framework** | FastAPI | 0.109.0 | REST API server |
| **ASGI Server** | Uvicorn | 0.25.0 | Async request handling |
| **Authentication** | PyJWT | 2.8.0 | JWT token-based auth |
| **Password Hashing** | Passlib + Bcrypt | 1.7.4 | Secure credential storage |
| **HTTP Client** | HTTPX | 0.25.2 | Async HTTP requests |

### 1.4 Database Layer
| Component | Technology | Purpose |
|-----------|------------|---------|
| **Primary Database** | MongoDB | Document storage, feedback, user data |
| **Vector Storage** | ChromaDB | Semantic chunk embeddings |
| **Cache Layer** | LRU + TTL Cache | Response caching (3600s TTL) |

### 1.5 Document Processing
| Component | Technology | Purpose |
|-----------|------------|---------|
| **PDF Extraction** | PyPDF2 + pdfplumber | Text extraction from manuals |
| **Word Documents** | python-docx | DOCX processing |
| **Presentations** | python-pptx | PPTX slide extraction |
| **Spreadsheets** | openpyxl | Excel file parsing |
| **Tokenization** | tiktoken | Token counting for LLM context |

### 1.6 Web Scraping & Data Collection
| Component | Technology | Purpose |
|-----------|------------|---------|
| **HTTP Client** | aiohttp | Async web scraping |
| **HTML Parser** | BeautifulSoup4 + lxml | DOM parsing & extraction |
| **Freshdesk Integration** | Custom async scraper | Support ticket ingestion |

### 1.7 Frontend Stack
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **UI Framework** | React | 18.2.0 | Component-based UI |
| **Build Tool** | Vite | 5.0.0 | Fast development & bundling |
| **HTTP Client** | Axios | 1.6.0 | API communication |
| **Plugin** | @vitejs/plugin-react | 4.2.0 | React Fast Refresh |

### 1.8 DevOps & Infrastructure
| Component | Technology | Purpose |
|-----------|------------|---------|
| **Containerization** | Docker | Application packaging |
| **Orchestration** | Docker Compose | Multi-container management |
| **Virtualization** | Proxmox VM | Infrastructure platform |
| **GPU** | NVIDIA RTX A2000 (6GB) | LLM acceleration |
| **Networking** | Docker ai-net bridge | Inter-service communication |
| **Logging** | coloredlogs | Structured application logging |

### 1.9 Testing & Quality
| Component | Technology | Purpose |
|-----------|------------|---------|
| **Test Framework** | pytest | Unit & integration testing |
| **Async Testing** | pytest-asyncio | Async test support |
| **Validation** | Pydantic | Data validation & serialization |

---

## 2. Architecture Breakdown

### 2.1 High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER LAYER                               │
│  ┌──────────────────┐              ┌────────────────────────┐   │
│  │  React Frontend  │◄─────────────│   FastAPI REST API     │   │
│  │  (Port 3001)     │   HTTP/REST  │   (Port 8000)          │   │
│  │  - Auth UI       │              │   - JWT Auth           │   │
│  │  - Chat UI       │              │   - CORS Middleware    │   │
│  │  - Admin Panel   │              │   - Rate Limiting      │   │
│  └──────────────────┘              └────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    RAG Engine Core                       │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │   Query     │  │   Hybrid     │  │   Context    │   │   │
│  │  │  Processor  │─▶│   Search     │─▶│  Optimizer   │   │   │
│  │  └─────────────┘  └──────────────┘  └──────────────┘   │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │   Intent    │  │   Response   │  │  Confidence  │   │   │
│  │  │  Detector   │  │  Validator   │  │   Scorer     │   │   │
│  │  └─────────────┘  └──────────────┘  └──────────────┘   │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │  Product    │  │  Citation    │  │    Cache     │   │   │
│  │  │  Filter     │  │  Formatter   │  │  (LRU+TTL)   │   │   │
│  │  └─────────────┘  └──────────────┘  └──────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Self-Learning Engine (Phase 6)             │   │
│  │  - Feedback Collection  - Wilson Score Ranking          │   │
│  │  - Learned Mappings     - Query Boosting                │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │                    │                      │
         ▼                    ▼                      ▼
┌──────────────┐     ┌──────────────┐      ┌──────────────┐
│   MongoDB    │     │    Ollama    │      │   ChromaDB   │
│              │     │              │      │              │
│ - Users      │     │ - Qwen2.5:7b │      │ - Embeddings │
│ - Feedback   │     │ - GPU Accel. │      │ - BM25 Index │
│ - Mappings   │     │ - Temp: 0.1  │      │ - 28.4K docs │
│ - Products   │     │              │      │              │
└──────────────┘     └──────────────┘      └──────────────┘
```

### 2.2 Data Flow Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                   INPUT STAGE                                 │
└───────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
        ┌───────────────────┐   ┌────────────────────┐
        │   Web Scraping    │   │  Document Upload   │
        │   - Products      │   │  - PDF Manuals     │
        │   - Tickets       │   │  - DOCX Files      │
        │   - Categories    │   │  - PPTX Slides     │
        └─────────┬─────────┘   └────────┬───────────┘
                  │                      │
                  ▼                      ▼
        ┌───────────────────────────────────────────┐
        │          DATA PROCESSING                  │
        │  ┌────────────────────────────────────┐   │
        │  │    1. Text Extraction              │   │
        │  │       (PyPDF2/pdfplumber/docx)     │   │
        │  └────────────────────────────────────┘   │
        │  ┌────────────────────────────────────┐   │
        │  │    2. Semantic Chunking            │   │
        │  │       (Context-aware splitting)    │   │
        │  └────────────────────────────────────┘   │
        │  ┌────────────────────────────────────┐   │
        │  │    3. Product Extraction           │   │
        │  │       (Pattern-based detection)    │   │
        │  └────────────────────────────────────┘   │
        │  ┌────────────────────────────────────┐   │
        │  │    4. Embedding Generation         │   │
        │  │       (all-MiniLM-L6-v2, 384-dim)  │   │
        │  └────────────────────────────────────┘   │
        │  ┌────────────────────────────────────┐   │
        │  │    5. Deduplication                │   │
        │  │       (SHA256 content hashing)     │   │
        │  └────────────────────────────────────┘   │
        └───────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                     STORAGE STAGE                             │
│  ┌─────────────────┐         ┌──────────────────────────┐    │
│  │    MongoDB      │         │      ChromaDB            │    │
│  │  - Metadata     │         │  - Vector Embeddings     │    │
│  │  - Raw Content  │         │  - BM25 Keyword Index    │    │
│  │  - Products     │         │  - Metadata Filters      │    │
│  └─────────────────┘         └──────────────────────────┘    │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                  RETRIEVAL STAGE (14-Stage RAG)               │
│                                                               │
│  1️⃣  Off-topic Detection (reject non-tool queries)            │
│  2️⃣  Language Detection (TR/EN)                               │
│  3️⃣  Response Cache Check (~100,000x speedup on hit)          │
│  4️⃣  Self-Learning Context (apply learned mappings)           │
│  5️⃣  Hybrid Retrieval (Semantic 60% + BM25 40% + RRF)         │
│  6️⃣  Strict Product Filtering (prevent cross-contamination)   │
│  7️⃣  Capability Filtering (WiFi/Battery content)              │
│  8️⃣  Context Grounding (return "I don't know" if uncertain)   │
│  9️⃣  Context Optimization (8K token budget, deduplication)    │
│  🔟 Intent Detection (8 specialized categories)               │
│  1️⃣1️⃣ LLM Generation (Qwen2.5:7b with GPU)                     │
│  1️⃣2️⃣ Response Validation (hallucination detection)            │
│  1️⃣3️⃣ Confidence Scoring (multi-factor algorithm)             │
│  1️⃣4️⃣ Save & Cache (MongoDB persistence + LRU cache)          │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                     OUTPUT STAGE                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  AI Response Delivered to User:                         │ │
│  │  - Repair Suggestion                                    │ │
│  │  - Confidence Score (0.0-1.0)                           │ │
│  │  - Source Citations (with page numbers)                 │ │
│  │  - Follow-up Questions                                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                  FEEDBACK LOOP (Self-Learning)                │
│  User Feedback → MongoDB → Wilson Score → Future Boosting    │
└───────────────────────────────────────────────────────────────┘
```

### 2.3 RAG Pipeline Workflow

The system implements a sophisticated **14-stage RAG pipeline** that achieves 96% test pass rate:

#### Stage 1-4: Pre-Processing
1. **Off-topic Detection:** Filters non-relevant queries using intent classification
2. **Language Detection:** Auto-detects Turkish/English for localized responses
3. **Cache Check:** Returns cached response if available (~100,000x speedup)
4. **Self-Learning Context:** Applies learned query→solution mappings from user feedback

#### Stage 5-7: Retrieval
5. **Hybrid Search:**
   - Semantic Search (60% weight): Meaning-based retrieval via embeddings
   - BM25 Keyword Search (40% weight): Exact term matching
   - RRF Fusion (k=60): Combines both scores using Reciprocal Rank Fusion
6. **Product Filtering:** ChromaDB where clause filters to relevant product family
7. **Capability Filtering:** Excludes WiFi docs for cable tools, battery docs for non-battery tools

#### Stage 8-10: Context Processing
8. **Context Grounding:** Returns "I don't know" if top similarity < 0.35 or insufficient docs
9. **Context Optimization:**
   - Deduplication of semantically similar chunks
   - Token budget management (8K tokens)
   - Service bulletin prioritization (4.0x boost)
10. **Intent Detection:** Classifies into 8 categories (troubleshooting, calibration, specs, etc.)

#### Stage 11-13: Generation & Validation
11. **LLM Generation:** Qwen2.5:7b generates response with GPU acceleration
12. **Response Validation:**
    - Hallucination detection via numerical value verification
    - Forbidden content filtering
    - Uncertainty phrase counting (max 2)
13. **Confidence Scoring:** Multi-factor score based on similarity, doc count, validation

#### Stage 14: Persistence
14. **Save & Cache:** Stores to MongoDB, updates response cache, records metrics

---

## 3. Directory Structure

```
desoutter-assistant/
│
├── config/                         # Configuration files
│   ├── __init__.py                 # Base config (MongoDB, paths)
│   ├── settings.py                 # Phase 1 settings (scraping, DB)
│   └── ai_settings.py              # Phase 2 settings (RAG, LLM, embeddings)
│
├── src/                            # Source code
│   ├── __init__.py
│   │
│   ├── api/                        # FastAPI REST API
│   │   ├── __init__.py
│   │   └── main.py                 # Main API server (auth, diagnosis, admin)
│   │
│   ├── database/                   # MongoDB layer
│   │   ├── __init__.py
│   │   ├── mongo_client.py         # MongoDB connection & operations
│   │   ├── models.py               # Product, User data models (Pydantic)
│   │   └── feedback_models.py      # Feedback, LearnedMapping models
│   │
│   ├── documents/                  # Document processing
│   │   ├── __init__.py
│   │   ├── document_processor.py   # PDF/DOCX/PPTX text extraction
│   │   ├── semantic_chunker.py     # Context-aware text chunking
│   │   ├── embeddings.py           # Sentence transformer wrapper
│   │   ├── product_extractor.py    # Pattern-based product detection
│   │   └── pdf_processor.py        # PDF-specific processing
│   │
│   ├── llm/                        # LLM & RAG components
│   │   ├── __init__.py
│   │   ├── rag_engine.py           # Main RAG orchestrator (14-stage)
│   │   ├── hybrid_search.py        # Semantic + BM25 + RRF fusion
│   │   ├── ollama_client.py        # Ollama API client
│   │   ├── prompts.py              # System & RAG prompt templates
│   │   ├── intent_detector.py      # Query intent classification (8 types)
│   │   ├── query_processor.py      # Query preprocessing & expansion
│   │   ├── context_optimizer.py    # Token budget & deduplication
│   │   ├── context_grounding.py    # "I don't know" logic
│   │   ├── response_validator.py   # Hallucination detection
│   │   ├── confidence_scorer.py    # Multi-factor confidence scoring
│   │   ├── response_cache.py       # LRU + TTL response caching
│   │   ├── citation_formatter.py   # Source citation formatting
│   │   ├── conversation.py         # Multi-turn conversation manager
│   │   ├── self_learning.py        # Feedback-based learning engine
│   │   ├── feedback_engine.py      # Wilson score ranking
│   │   ├── domain_vocabulary.py    # 351 Desoutter-specific terms
│   │   ├── domain_embeddings.py    # Domain-specific embedding loader
│   │   ├── relevance_filter.py     # Fault category filtering
│   │   └── performance_metrics.py  # Query timing & metrics tracking
│   │
│   ├── scraper/                    # Web scraping modules
│   │   ├── __init__.py
│   │   ├── desoutter_scraper.py    # Main Desoutter website scraper
│   │   ├── parsers.py              # HTML parsing utilities
│   │   ├── product_categorizer.py  # Product classification
│   │   ├── ticket_scraper.py       # Freshdesk ticket scraper (async)
│   │   ├── ticket_scraper_sync.py  # Freshdesk ticket scraper (sync)
│   │   ├── ticket_preprocessor.py  # Ticket text cleaning
│   │   └── ingest_tickets.py       # Ticket ingestion to MongoDB
│   │
│   ├── utils/                      # Utility modules
│   │   ├── __init__.py
│   │   ├── logger.py               # Colored logging setup
│   │   └── http_client.py          # Async HTTP client wrapper
│   │
│   └── vectordb/                   # Vector database layer
│       ├── __init__.py
│       └── chroma_client.py        # ChromaDB client (CRUD operations)
│
├── frontend/                       # React frontend
│   ├── src/
│   │   ├── App.jsx                 # Main application component
│   │   ├── TechWizard.jsx          # Technician chat interface
│   │   ├── main.jsx                # React entry point
│   │   ├── App.css                 # Styling
│   │   └── components/
│   │       ├── MetricsDashboard.jsx       # Admin metrics dashboard
│   │       ├── DomainManagement.jsx       # Domain term management
│   │       └── LearningInsights.jsx       # Self-learning analytics
│   ├── public/                     # Static assets
│   ├── package.json                # NPM dependencies
│   ├── vite.config.js              # Vite build configuration
│   └── Dockerfile                  # Frontend container image
│
├── scripts/                        # Utility scripts
│   ├── run_api.py                  # Start FastAPI server
│   ├── scrape_all.py               # Scrape all Desoutter products
│   ├── ingest_documents.py         # Process PDFs into ChromaDB
│   ├── reingest_documents.py       # Re-process with updated metadata
│   ├── reset_vectordb.py           # Clear ChromaDB for fresh start
│   ├── ingest_tickets.py           # Import Freshdesk tickets
│   ├── export_data.py              # Export MongoDB data
│   ├── query_vector_db.py          # Test ChromaDB queries
│   ├── run_baseline_test.sh        # Automated test suite runner
│   └── clean_low_quality_chunks.py # Remove poor-quality chunks
│
├── documents/                      # Document storage
│   ├── manuals/                    # Product technical manuals (PDF)
│   └── bulletins/                  # Service bulletins (PDF)
│
├── data/                           # Runtime data directory
│   ├── vectordb/chroma/            # ChromaDB persistence
│   ├── logs/                       # Application logs
│   ├── exports/                    # Data exports
│   ├── tickets/                    # Scraped ticket JSON
│   └── cache/                      # Response cache persistence
│
├── test_results/                   # Automated test outputs
│
├── tests/                          # Test suite
│   └── fixtures/                   # Test data fixtures
│
├── Dockerfile                      # Backend container image
├── docker-compose.desoutter.yml    # Multi-container orchestration
├── requirements.txt                # Phase 1 dependencies (scraping)
├── requirements-phase2.txt         # Phase 2 dependencies (RAG/AI)
├── README.md                       # User-facing documentation
├── QUICKSTART.md                   # Quick deployment guide
├── CHANGELOG.md                    # Version history
├── ROADMAP.md                      # Future development plan
└── .env.example                    # Environment variable template
```

---

## 4. Component Details

### 4.1 RAG Engine (`src/llm/rag_engine.py`)
**Purpose:** Core orchestrator for the 14-stage RAG pipeline

**Key Features:**
- Hybrid search coordination (semantic + BM25)
- Self-learning feedback integration
- Product-aware filtering
- Bulletin score boosting (4.0x multiplier)
- Context grounding & hallucination prevention
- Multi-language support (TR/EN)

**Key Methods:**
- `diagnose()`: Main entry point for repair suggestions
- `_hybrid_retrieval()`: Executes semantic + BM25 fusion
- `_apply_metadata_boost()`: Service bulletin prioritization
- `_filter_by_product_strict()`: Product family filtering
- `_calculate_confidence()`: Multi-factor confidence scoring

### 4.2 Hybrid Search (`src/llm/hybrid_search.py`)
**Purpose:** Combines semantic and keyword-based retrieval

**Components:**
- `BM25Index`: Custom BM25 implementation with TF-IDF
- `HybridSearcher`: RRF fusion coordinator
- `SearchResult`: Unified result data structure

**Algorithm:**
```
RRF_score(doc) = Σ(weight_i / (k + rank_i))
Final_score = 0.6 * RRF_semantic + 0.4 * RRF_bm25
```

### 4.3 Self-Learning Engine (`src/llm/self_learning.py`)
**Purpose:** Continuous improvement from user feedback

**Learning Mechanisms:**
- Positive feedback → Reinforce query-solution mappings
- Negative feedback → Record patterns to avoid
- Wilson score confidence interval ranking
- Query boosting based on historical success

**Data Model:**
```python
LearnedMapping {
    query_pattern: str
    solution_keywords: List[str]
    positive_count: int
    negative_count: int
    wilson_score: float
}
```

### 4.4 Document Processor (`src/documents/document_processor.py`)
**Purpose:** Multi-format document text extraction

**Supported Formats:**
- PDF: PyPDF2 + pdfplumber
- Word: python-docx (DOCX)
- PowerPoint: python-pptx (PPTX)
- Excel: openpyxl (XLSX)

**Processing Pipeline:**
1. Text extraction
2. Semantic chunking (context-aware splitting)
3. Product family extraction (pattern-based)
4. Metadata enrichment (doc_type, section, importance)
5. Embedding generation (384-dim vectors)
6. Deduplication (SHA256 hashing)

### 4.5 Semantic Chunker (`src/documents/semantic_chunker.py`)
**Purpose:** Intelligent document segmentation

**Features:**
- Context-aware splitting (preserves semantic units)
- Document type detection (manual, bulletin, guide)
- Section preservation (warnings, procedures, specs)
- Product metadata extraction
- Adaptive chunk sizing (300-800 tokens)

**Metadata Schema:**
```python
ChunkMetadata {
    doc_type: str              # "service_bulletin" | "technical_manual"
    section_type: str          # "procedure" | "warning" | "specs"
    product_family: str        # "CVI3" | "ERS" | "EPB"
    product_models: List[str]  # ["CVI3", "CVI3-1500"]
    is_generic: bool           # Cross-product content flag
    importance_score: float    # 0.0-1.0 relevance score
}
```

### 4.6 Product Extractor (`src/documents/product_extractor.py`)
**Purpose:** Pattern-based product family detection

**Detection Strategies:**
- Filename parsing (40+ regex patterns)
- Content keyword matching
- Part number pattern recognition (e.g., "6159 12 3456")
- Family alias resolution (ESRS → ERS, CVI III → CVI3)

**Example Patterns:**
```regex
CVI3:     r'\bCVI[\s\-_]?3\b'
EPB:      r'\bEPB\b'
ERS:      r'\bE[RS]{2}\b'
ESLT:     r'\bESLT[\s\-_]?\d+'
CONNECT:  r'\bCONNECT[\s\-_]?[WG]?\b'
```

### 4.7 Feedback Engine (`src/llm/feedback_engine.py`)
**Purpose:** Statistical ranking of learned patterns

**Wilson Score Algorithm:**
```
wilson_score = (p + z²/2n - z√(p(1-p)/n + z²/4n²)) / (1 + z²/n)

Where:
  p = positive_count / total
  n = total feedback count
  z = 1.96 (95% confidence)
```

**Use Case:** Prioritizes query-solution mappings with statistical confidence

---

## 5. System Architecture Diagram (Mermaid.js)

```mermaid
graph TB
    subgraph "User Interface Layer"
        A[React Frontend<br/>Port 3001]
        B[FastAPI<br/>Port 8000]
    end

    subgraph "Authentication"
        C[JWT Auth<br/>PyJWT + Bcrypt]
    end

    subgraph "RAG Engine Core"
        D[Query Processor<br/>Language Detection<br/>Intent Classification]
        E[Hybrid Search<br/>Semantic 60%<br/>BM25 40%<br/>RRF Fusion]
        F[Context Optimizer<br/>8K Token Budget<br/>Deduplication<br/>Bulletin Boost 4.0x]
        G[Ollama LLM<br/>Qwen2.5:7b<br/>GPU Accelerated]
        H[Response Validator<br/>Hallucination Check<br/>Confidence Scoring]
        I[Response Cache<br/>LRU + TTL<br/>100,000x Speedup]
    end

    subgraph "Self-Learning System"
        J[Feedback Engine<br/>Wilson Score Ranking]
        K[Learned Mappings<br/>MongoDB Storage]
    end

    subgraph "Document Processing"
        L[Document Processor<br/>PDF/DOCX/PPTX]
        M[Semantic Chunker<br/>Context-Aware Splitting]
        N[Product Extractor<br/>Pattern Recognition]
        O[Embeddings Generator<br/>all-MiniLM-L6-v2]
    end

    subgraph "Data Layer"
        P[MongoDB<br/>Users, Feedback, Mappings]
        Q[ChromaDB<br/>Vector Embeddings<br/>BM25 Index]
        R[Ollama Service<br/>Model Storage]
    end

    subgraph "Data Ingestion"
        S[Web Scraper<br/>Products & Tickets]
        T[Document Upload<br/>Manuals & Bulletins]
    end

    A -->|HTTP/REST| B
    B --> C
    C --> D
    D --> I
    I -->|Cache Miss| E
    E --> F
    F --> G
    G --> H
    H -->|Store Response| I
    H -->|User Feedback| J
    J --> K
    K -->|Boost Queries| E

    S --> L
    T --> L
    L --> M
    M --> N
    N --> O
    O --> Q

    E <--> Q
    G <--> R
    B <--> P
    K <--> P

    style D fill:#e1f5ff
    style E fill:#fff9e1
    style G fill:#ffe1f5
    style H fill:#e1ffe1
    style J fill:#f5e1ff
```

---

## 6. Future Roadmap & Technical Improvements

### 6.1 Scalability Improvements

#### 1. **Distributed Vector Database (Qdrant Migration)**
**Problem:** ChromaDB is single-node and limited to ~1M vectors
**Solution:** Migrate to Qdrant with horizontal scaling
- **Expected Impact:** 10x query throughput, 100M+ vector capacity
- **Implementation:** Replace ChromaDBClient with QdrantClient
- **Effort:** 2-3 days (client swap + data migration script)

**Technical Details:**
```python
# Current: ChromaDB (single-node)
vectordb = ChromaDBClient()

# Future: Qdrant (distributed)
from qdrant_client import QdrantClient
vectordb = QdrantClient(url="http://qdrant-cluster:6333")
vectordb.create_collection(
    collection_name="desoutter_docs",
    vectors_config={
        "size": 384,
        "distance": "Cosine"
    },
    shard_number=4,  # Horizontal scaling
    replication_factor=2  # High availability
)
```

#### 2. **Prompt Caching for Repeated System Prompts**
**Problem:** System prompts (500-1000 tokens) recomputed on every LLM call
**Solution:** Implement prompt caching at Ollama level
- **Expected Impact:** 30-40% reduction in LLM latency
- **Implementation:** Use Ollama's `/api/generate` with `keep_alive` parameter
- **Effort:** 1 day (API client update)

**Technical Details:**
```python
# Current: No caching
response = ollama.generate(
    model="qwen2.5:7b-instruct",
    prompt=system_prompt + context + query
)

# Future: Cached system prompt
response = ollama.generate(
    model="qwen2.5:7b-instruct",
    prompt=query,
    system=system_prompt,  # Cached separately
    context=context_embedding,  # Reuse across queries
    keep_alive="10m"  # Keep model loaded
)
```

#### 3. **Asynchronous Document Ingestion Queue**
**Problem:** Document upload blocks API response during ingestion
**Solution:** Celery + Redis task queue for background processing
- **Expected Impact:** Instant upload response, 5x ingestion throughput
- **Implementation:** Celery worker pool + Redis broker
- **Effort:** 3-4 days (Celery setup + task migration)

**Architecture:**
```
User Upload → FastAPI → Redis Queue → Celery Worker → ChromaDB
                ↓
          Instant 202 Accepted

Worker Pool: 4 parallel workers
Tasks: extract_text → chunk → embed → store
```

### 6.2 Performance Improvements

#### 1. **GPU Batch Inference for Embeddings**
**Problem:** Embeddings generated one-by-one (serial processing)
**Solution:** Batch embedding generation on GPU
- **Expected Impact:** 10x faster document ingestion
- **Implementation:** Use sentence-transformers batch encoding
- **Effort:** 1 day (refactor embeddings.py)

**Code:**
```python
# Current: Serial (slow)
for chunk in chunks:
    embedding = model.encode(chunk.content)

# Future: Batched (10x faster)
batch_texts = [c.content for c in chunks]
embeddings = model.encode(
    batch_texts,
    batch_size=64,
    device='cuda',
    show_progress_bar=True
)
```

#### 2. **Pre-computed Query Embeddings for Common Queries**
**Problem:** Top 20 queries represent 60% of traffic, recomputed each time
**Solution:** Pre-compute & cache embeddings for frequent query patterns
- **Expected Impact:** 80% latency reduction for common queries
- **Implementation:** Warm cache on startup with top queries
- **Effort:** 1 day (cache pre-warming script)

#### 3. **Quantized LLM Models (GGUF Format)**
**Problem:** Qwen2.5:7b requires 14GB VRAM, limits GPU utilization
**Solution:** Use 4-bit quantized GGUF model (3.5GB VRAM)
- **Expected Impact:** 2x model capacity, 30% faster inference
- **Trade-off:** Minimal quality loss (<2% accuracy drop)
- **Implementation:** Convert to GGUF with llama.cpp
- **Effort:** 2 days (model conversion + testing)

### 6.3 Quality Improvements

#### 1. **Fine-tuned Embedding Model on Domain Data**
**Problem:** Generic all-MiniLM-L6-v2 lacks Desoutter-specific terminology
**Solution:** Fine-tune on 2,249 Freshdesk tickets + feedback pairs
- **Expected Impact:** 15-20% retrieval accuracy improvement
- **Training Data:** Positive feedback pairs (query → solution)
- **Method:** Siamese network with contrastive loss
- **Effort:** 1 week (data preparation + training + evaluation)

**Training Pipeline:**
```python
from sentence_transformers import SentenceTransformer, InputExample, losses

# Load positive feedback pairs
train_examples = [
    InputExample(texts=["motor not starting", "Check E06 error bulletin"]),
    InputExample(texts=["WiFi connection lost", "ESDE23028 Connect-W fix"])
]

# Fine-tune
model = SentenceTransformer('all-MiniLM-L6-v2')
train_dataloader = DataLoader(train_examples, batch_size=8)
loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, loss)], epochs=5)
```

#### 2. **Multi-turn Conversation Context Window**
**Problem:** Current system loses context after 3-4 turns
**Solution:** Implement sliding window context with conversation history
- **Expected Impact:** Better follow-up question handling
- **Implementation:** Store last N turns in conversation.py
- **Effort:** 2 days (context accumulation + LLM prompt adjustment)

#### 3. **Confidence Calibration via Post-hoc Training**
**Problem:** Confidence scores not well-calibrated (over-confident on wrong answers)
**Solution:** Train logistic regression on (features, correct/incorrect) pairs
- **Expected Impact:** Calibrated confidence scores for better UX
- **Features:** Similarity score, doc count, validation flags, token overlap
- **Training Data:** 500+ manually labeled test cases
- **Effort:** 3 days (labeling + training + integration)

---

## 7. Key Strengths

1. **Production-Grade Architecture:** Modular, testable, well-documented codebase
2. **Advanced RAG Pipeline:** 14-stage pipeline with 96% test pass rate
3. **Self-Learning Capability:** Continuous improvement from user feedback
4. **Hallucination Prevention:** Multi-layer validation (grounding, validation, confidence)
5. **Performance Optimization:** Response caching, GPU acceleration, hybrid search
6. **Enterprise Features:** JWT auth, RBAC, admin dashboard, metrics tracking

---

## 8. Technical Debt & Risks

### 8.1 Identified Debt
- **Hard-coded configuration:** Many settings in ai_settings.py should be runtime-configurable
- **Monolithic RAG engine:** rag_engine.py is 1000+ lines, needs refactoring into sub-modules
- **Limited test coverage:** Only 25 scenarios, needs expansion to 100+ edge cases
- **No CI/CD pipeline:** Manual testing and deployment process

### 8.2 Security Considerations
- **JWT secret:** Using default secret in development (must change in production)
- **CORS policy:** Currently allows all origins (`*`), should restrict to frontend domain
- **Input validation:** Limited sanitization on user queries (potential injection risk)
- **Rate limiting:** No rate limiting implemented (DoS vulnerability)

### 8.3 Operational Risks
- **Single GPU dependency:** RTX A2000 failure halts all LLM inference
- **No database backup:** MongoDB has no automated backup strategy
- **Vendor lock-in:** Tightly coupled to Ollama (should abstract LLM interface)

---

## 9. Deployment Architecture

### 9.1 Container Orchestration

```
Proxmox VM (Ubuntu 22.04 LTS)
│
├── Docker Network: ai-net (bridge)
│   │
│   ├── Container: mongodb
│   │   ├── Image: mongo:7.0
│   │   ├── Port: 27017
│   │   └── Volume: mongodb_data
│   │
│   ├── Container: ollama
│   │   ├── Image: ollama/ollama:latest
│   │   ├── Port: 11434
│   │   ├── GPU: NVIDIA RTX A2000
│   │   └── Volume: ollama_models
│   │
│   ├── Container: desoutter-api
│   │   ├── Image: desoutter-api:latest
│   │   ├── Port: 8000
│   │   ├── GPU: Shared with Ollama
│   │   └── Volumes:
│   │       ├── desoutter_data:/app/data
│   │       ├── documents:/app/documents
│   │       └── huggingface_cache:/root/.cache/huggingface
│   │
│   └── Container: desoutter-frontend
│       ├── Image: desoutter-frontend:latest
│       ├── Port: 3001
│       └── Env: VITE_API_URL=http://desoutter-api:8000
│
└── Host Resources
    ├── CPU: 8 cores (Intel Xeon)
    ├── RAM: 32 GB
    ├── GPU: NVIDIA RTX A2000 (6GB VRAM)
    └── Storage: 500GB SSD
```

### 9.2 Resource Allocation

| Component | CPU | RAM | Disk | GPU |
|-----------|-----|-----|------|-----|
| MongoDB | 1 core | 2 GB | 20 GB | - |
| Ollama | 2 cores | 8 GB | 50 GB | 6 GB VRAM |
| FastAPI | 3 cores | 12 GB | 100 GB | Shared |
| Frontend | 1 core | 1 GB | 1 GB | - |
| ChromaDB | 1 core | 8 GB | 100 GB | - |

---

## 10. Conclusion

The Desoutter Assistant represents a state-of-the-art implementation of production-grade RAG architecture. The system successfully combines modern AI/ML techniques (hybrid search, self-learning, hallucination prevention) with robust software engineering practices (modular design, comprehensive testing, performance optimization).

**Recommendation:** Focus near-term efforts on the three scalability improvements (Qdrant migration, prompt caching, async ingestion) to prepare for production workloads at scale.

---

**Document Prepared By:** Senior AI Software Architect
**Review Status:** Draft v1.0
**Next Review:** Q2 2026
