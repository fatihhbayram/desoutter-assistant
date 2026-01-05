# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### In Progress
- Controller units scraping (10 units)

---

## [1.6.0] - 2026-01-05

### Added
- **Intelligent Product Filtering** - Query-time ChromaDB filtering for product-specific retrieval
  - Pattern-based `IntelligentProductExtractor` with 40+ regex patterns
  - Automatic product family detection from filenames and content
  - `_build_product_filter()` method builds ChromaDB `where` clause
  - `extract_product_from_query()` for retrieval-time product detection
- **New utility scripts**
  - `scripts/reset_vectordb.py` - Clear ChromaDB for fresh re-ingestion
  - `scripts/reingest_documents.py` - Re-process documents with new metadata
  - `scripts/test_product_filtering.py` - Validate product filtering

### Changed
- **Semantic Chunker** - Added `product_family`, `product_models`, `is_generic` metadata fields
- **Document Processor** - Uses new `get_product_metadata()` API
- **RAG Engine** - Passes `where_filter` to hybrid search for ChromaDB filtering
- **Full re-ingestion**: 26,528 chunks with product metadata
- **Test pass rate**: 91.7% for product filtering tests

### Fixed
- Cross-product contamination in retrieval (e.g., CVI3 query no longer returns EPB/ERS docs)

---

## [1.5.0] - 2026-01-04

### Added
- **Freshdesk Ticket Scraper Integration** - Real customer Q&A data from support portal
  - Async ticket scraper with PDF attachment processing
  - Resume capability for interrupted scrapes
  - RAG ingestion pipeline for ticket content
- **PDF Content Encoding Fix** - CID decoder for proper PDF character extraction

### Changed
- Updated documentation for ticket scraper usage

---

## [1.4.0] - 2025-12-31

### Added
- **Source Citation Enhancement** - Page numbers now correctly cited in RAG responses
  - Refactored `clean_text` to preserve paragraph structure
  - Improved regex for robust page marker detection (`--- Page X ---`)
- **Smart Product Recognition** - Automatic product metadata enrichment
  - `ProductExtractor` module for product family identification
  - Automatic tagging: `EPB_...pdf` → `[BATTERY_TOOL, EPB]`

### Changed
- Full database re-ingestion: 541 documents → ~22,889 chunks
- Increased chunk granularity for better precision

### Fixed
- Page number extraction from source documents
- Product category metadata propagation during ingestion

---

## [1.3.0] - 2025-12-30

### Added
- **Intent Detection Integration** - 8 intent types with dynamic prompt selection
  - Troubleshooting, specifications, installation, calibration
  - Maintenance, connection, error codes, general
  - Intent metadata included in API responses
- **Content Deduplication** - SHA-256 hash-based duplicate detection
  - Configurable via `ENABLE_DEDUPLICATION` environment variable
  - Prevents duplicate chunks in vector database
- **Adaptive Chunk Sizing** - Document type-based chunk sizes
  - Troubleshooting guides: 200 tokens (precision)
  - Technical manuals: 400 tokens (context)

### Fixed
- Wireless detection bug for battery tools (EABS, EIBS, EPB, EAB series)
- 80 battery tools corrected for accurate WiFi troubleshooting

---

## [1.2.0] - 2025-12-29

### Added
- **Context Grounding ("I Don't Know" Logic)** - Multi-factor context sufficiency scoring
  - Automatic "I don't know" responses for insufficient context
  - Significant reduction in hallucinations
- **Response Validation System** - Post-processing layer to catch hallucinations
  - Validates numerical values against source context
  - Detects forbidden content (WiFi suggestions for non-WiFi tools)
  - Auto-flags low-quality responses for review

### Changed
- Test coverage improved: 7/7 grounding tests passing (100%)
- Response validation: 7/8 tests passing (87.5%)

---

## [1.1.0] - 2025-12-27

### Added
- **RAG Relevance Filtering** - 15 fault categories with negative keyword filtering
  - Categories: WiFi, motor, torque, battery, display, touchscreen, etc.
  - Word boundary regex matching prevents false positives
  - 70-80% reduction in irrelevant search results
- **Connection Architecture Mapping** - 6 product family categories
  - CVI3 Range (corded tools)
  - Battery WiFi tools (EPBC, EABC, EABS, BLRTC, ELC)
  - Standalone battery tools (EPB, EPBA, EAB)
  - Connect family units
- **Query Processor** - Centralized query processing
  - Normalization, keyword extraction, intent detection
  - Turkish + English language support

### Changed
- Document ingestion: 541 documents → 6,798 chunks (116% increase)
- Wireless field updated: 71 wireless, 380 non-wireless, 0 null

---

## [1.0.0] - 2025-12-22

### Added
- **Phase 3.1: Domain Embeddings** - 351 Desoutter-specific terms
  - 27 product series, 29 error codes
  - Query enhancement with domain synonyms
  - Term weight learning from feedback
- **Phase 5.1: Performance Metrics** - Comprehensive monitoring
  - Query latency tracking (retrieval, LLM, total)
  - Cache hit/miss rate monitoring
  - P95/P99 latency percentiles
  - Health status endpoints: `/admin/metrics/*`
- **Phase 3.5: Multi-turn Conversation** - Session management
  - Context preservation across conversation turns
  - Reference resolution ("it", "this" → actual product)
  - 30-minute session timeout
  - Endpoints: `/conversation/*`
- **Phase 6: Self-Learning Feedback Loop** - Wilson score ranking
  - Feedback signal processing (explicit + implicit)
  - Source ranking learner with boost/demote factors
  - Keyword-to-source mapping
  - Training data collection for embedding retraining
  - Endpoints: `/admin/learning/*`

### Changed
- RAG engine fully integrated with self-learning components
- Hybrid search applies learned source boosts

---

## [0.9.0] - 2025-12-18

### Added
- **ProductModel Schema v2** - Comprehensive product categorization
  - `tool_category`: battery_tightening, cable_tightening, electric_drilling
  - `wireless_info`: WiFi capability detection with 3-tier logic
  - `platform_connection`: Compatible controller platforms
  - `modular_system`: XPB modular tool support
- **Smart Upsert Logic** - Preserves existing data during updates
  - Rejects placeholder values
  - Merges new data with existing records

### Fixed
- Frontend placeholder image filter
- WiFi detection logic (3 iterations refined)

---

## [0.8.0] - 2025-12-17

### Added
- **Phase 3.3: Source Relevance Feedback** - Per-document relevance UI
  - ✓/✗ buttons on each source document card
  - Visual feedback with green/red borders
  - Relevance summary before feedback submission
- **Phase 3.4: Context Window Optimization** - 8K token budget management
  - Jaccard similarity deduplication (85% threshold)
  - Warning prioritization (safety content first)
  - Procedure prioritization (actionable steps)
- **Phase 4.1: Metadata-Based Filtering** - Document type boosting
  - Service bulletins: 1.5x boost
  - Procedures: 1.3x boost
  - Warnings: 1.2x boost
- **Ollama GPU Activation** - NVIDIA RTX A2000 GPU inference
  - GPU memory: 4MiB → 4832MiB (model loaded)
  - LLM inference GPU-accelerated

### Fixed
- Async concurrency issue - blocking calls moved to thread pool
  - Health check: 30+ seconds → 40ms
  - Products list: 30+ seconds → 45ms

---

## [0.7.0] - 2025-12-16

### Added
- **Phase 2.1: Document Re-ingestion** - 276 documents with semantic chunking
  - Output: 1229 semantic chunks with rich metadata
  - Total in ChromaDB: 2309 vectors
- **Phase 2.2: Hybrid Search** - BM25 + Semantic + RRF Fusion
  - `HybridSearcher` class (700+ lines)
  - `BM25Index`: 13,026 unique terms indexed
  - `QueryExpander`: 9 domain synonym categories
  - Configurable weights: semantic=0.7, BM25=0.3
- **Phase 2.3: Response Caching** - LRU + TTL cache
  - ~100,000x speedup for repeated queries

---

## [0.6.0] - 2025-12-15

### Added
- **Phase 1: Semantic Chunking** - Structure-aware document processing
  - `SemanticChunker` module (420+ lines)
  - 5 document type classifications
  - 9 fault keyword categories
  - 14-field metadata per chunk
  - Recursive character-level chunking
- **RAG Retrieval Quality Optimization** - Dynamic similarity threshold
  - Optimal threshold: 0.30
  - Returns 3-5 relevant documents per query

### Changed
- Chunk size: 400 characters with 100 character overlap
- Document processor integrated with semantic chunking

---

## [0.5.0] - 2025-12-14

### Added
- **TechWizard Component** - 4-step wizard-style UI
  - Step 1: Product Search & Filter
  - Step 2: Product Selection
  - Step 3: Fault Description
  - Step 4: Diagnosis Results & Feedback
- Responsive CSS styling for mobile devices
- Progress bar with step indicators

### Fixed
- MongoDB configuration (localhost instead of Docker IP)
- Feedback API request body validation (HTTP 422)

---

## [0.4.0] - 2025-12-09

### Added
- **Admin Dashboard** - Comprehensive analytics
  - Overview cards: total diagnoses, today, this week, active users
  - Daily trend chart (last 7 days)
  - Confidence breakdown (high/medium/low)
  - Feedback statistics
  - Top diagnosed products
  - Common fault keywords
- **Self-Learning RAG Feedback System**
  - User feedback collection (👍/👎)
  - Negative feedback reason selection
  - `DiagnosisFeedback` model
  - `LearnedMapping` for fault-solution patterns
  - `FeedbackLearningEngine` class

---

## [0.3.0] - 2025-12-04

### Added
- **Document Viewer** - Open source documents from diagnosis results
  - `/documents/download/{filename}` endpoint
  - Support for PDF, DOCX, PPTX formats
- **Multi-Format Document Support**
  - PDF: PyPDF2 + pdfplumber
  - Word: python-docx
  - PowerPoint: python-pptx

### Changed
- Product catalog expanded: 237 → 451 products
  - Battery Tightening Tools: 151 products
  - Cable Tightening Tools: 272 products
  - Electric Drilling Tools: 28 products

### Fixed
- RAG distance threshold (L2 distance < 2.0)
- Model name product search
- Sources empty response issue

---

## [0.2.0] - 2025-12-02

### Added
- **Session Persistence** - Token validation on page refresh
- **Auto-Logout** - Axios interceptor for 401 responses
- Professional header design with gradient background
- Footer with author branding and social links
- Role-based UI (API docs link only for admins)

### Fixed
- `/auth/me` endpoint Header() dependency

---

## [0.1.0] - 2025-11-22 - 2025-12-01

### Added
- **Initial Release**
- FastAPI backend with product CRUD operations
- MongoDB database integration
- Web scraper for Desoutter product catalog
- React frontend with grid/list view
- Ollama LLM integration (Qwen2.5:7b-instruct)
- ChromaDB vector database
- RAG engine with PDF processing
- JWT authentication (admin/technician roles)
- User management (CRUD)
- RAG document management (upload, delete, ingest)

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.6.0 | 2026-01-05 | Intelligent product filtering, ChromaDB where clause |
| 1.5.0 | 2026-01-04 | Freshdesk ticket integration |
| 1.4.0 | 2025-12-31 | Source citation enhancement |
| 1.3.0 | 2025-12-30 | Intent detection, deduplication |
| 1.2.0 | 2025-12-29 | Context grounding, validation |
| 1.1.0 | 2025-12-27 | Relevance filtering, connection mapping |
| 1.0.0 | 2025-12-22 | Domain embeddings, self-learning, metrics |
| 0.9.0 | 2025-12-18 | ProductModel Schema v2 |
| 0.8.0 | 2025-12-17 | Source feedback, context optimization, GPU |
| 0.7.0 | 2025-12-16 | Hybrid search, response caching |
| 0.6.0 | 2025-12-15 | Semantic chunking |
| 0.5.0 | 2025-12-14 | TechWizard UI |
| 0.4.0 | 2025-12-09 | Admin dashboard, feedback system |
| 0.3.0 | 2025-12-04 | Document viewer, multi-format |
| 0.2.0 | 2025-12-02 | Session management, UI polish |
| 0.1.0 | 2025-11-22 | Initial release |

---

[Unreleased]: https://github.com/fatihhbayram/desoutter-assistant/compare/v1.6.0...HEAD
[1.6.0]: https://github.com/fatihhbayram/desoutter-assistant/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/fatihhbayram/desoutter-assistant/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/fatihhbayram/desoutter-assistant/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/fatihhbayram/desoutter-assistant/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/fatihhbayram/desoutter-assistant/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/fatihhbayram/desoutter-assistant/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/fatihhbayram/desoutter-assistant/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/fatihhbayram/desoutter-assistant/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/fatihhbayram/desoutter-assistant/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/fatihhbayram/desoutter-assistant/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/fatihhbayram/desoutter-assistant/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/fatihhbayram/desoutter-assistant/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/fatihhbayram/desoutter-assistant/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/fatihhbayram/desoutter-assistant/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/fatihhbayram/desoutter-assistant/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/fatihhbayram/desoutter-assistant/releases/tag/v0.1.0
