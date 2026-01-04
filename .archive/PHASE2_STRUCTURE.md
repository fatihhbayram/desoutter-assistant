# Desoutter Intelligent Repair Assistant - Enhanced Project Structure

```
desoutter-scraper/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── config/
│   ├── __init__.py
│   ├── settings.py                   # Tüm configuration
│   └── logging_config.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── scraper/                      # ✅ EXISTING - Web scraping
│   │   ├── __init__.py
│   │   ├── desoutter_scraper.py
│   │   └── parsers.py
│   │
│   ├── database/                     # ✅ EXISTING - MongoDB
│   │   ├── __init__.py
│   │   ├── mongo_client.py
│   │   └── models.py
│   │
│   ├── documents/                    # 🆕 NEW - Document processing
│   │   ├── __init__.py
│   │   ├── pdf_processor.py         # PDF extraction
│   │   ├── chunker.py               # Text chunking
│   │   └── embeddings.py            # Generate embeddings
│   │
│   ├── vectordb/                     # 🆕 NEW - Vector database
│   │   ├── __init__.py
│   │   ├── chroma_client.py         # ChromaDB operations
│   │   └── retriever.py             # Semantic search
│   │
│   ├── llm/                          # 🆕 NEW - LLM integration
│   │   ├── __init__.py
│   │   ├── ollama_client.py         # Ollama API client
│   │   ├── prompts.py               # Prompt templates
│   │   └── rag_engine.py            # RAG pipeline
│   │
│   ├── api/                          # 🆕 NEW - Web API
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI app
│   │   ├── routes.py                # API endpoints
│   │   └── schemas.py               # Request/response models
│   │
│   └── utils/                        # ✅ EXISTING + Enhanced
│       ├── __init__.py
│       ├── logger.py
│       └── http_client.py
│
├── scripts/
│   ├── scrape_all.py                # ✅ EXISTING
│   ├── scrape_single.py             # ✅ EXISTING
│   ├── export_data.py               # ✅ EXISTING
│   ├── ingest_documents.py          # 🆕 NEW - Ingest PDFs to vectorDB
│   ├── test_rag.py                  # 🆕 NEW - Test RAG system
│   └── run_api.py                   # 🆕 NEW - Start web API
│
├── data/
│   ├── logs/
│   ├── exports/
│   ├── cache/
│   ├── documents/                   # 🆕 NEW - PDF manuals & bulletins
│   │   ├── manuals/                 # Repair manuals
│   │   └── bulletins/               # Technical bulletins
│   └── vectordb/                    # 🆕 NEW - ChromaDB storage
│
├── frontend/                        # 🆕 NEW - Web interface (optional)
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
└── tests/
    ├── test_scraper.py
    ├── test_rag.py                  # 🆕 NEW
    └── test_api.py                  # 🆕 NEW
```


## Phase 2 - New Components (short)

1) Document processing (`src/documents/`)
- PDF text extraction, chunking, and embedding generation.

2) Vector DB (`src/vectordb/`)
- Store chunks + embeddings (Chroma). Support filtering by product/model on the client side.

3) LLM integration (`src/llm/`)
- Ollama client + prompt templates, RAG glue code.

4) Web API (`src/api/`)
- FastAPI endpoints (POST `/diagnose`, GET `/products`, health endpoints). Keep handlers small and testable.

5) Frontend (`frontend/`)
- Minimal React app for product selection and sending `/diagnose` requests to the API.

## Data Flow:

```
[Technician Input]
     ↓
[Web UI] → [FastAPI]
     ↓
[RAG Engine]
     ↓
[Vector Search] → Find relevant manual sections
     ↓
[LLM (Ollama)] → Generate repair suggestion
     ↓
[Response] → Display to technician
```

## Next Steps:

1. ✅ Scraper → MongoDB (DONE)
2. 🔄 Ingest PDFs → Vector DB
3. 🔄 Build RAG pipeline
4. 🔄 Create Web API
5. 🔄 Build frontend

Would you like me to:
A) Start with PDF processing & embeddings?
B) Create the RAG engine first?
C) Build the complete system step by step?
