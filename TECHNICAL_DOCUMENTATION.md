# Desoutter Assistant - Technical Documentation

> **Comprehensive technical guide for developers, architects, and AI engineers**

**Target Audience**: Junior Developers, Senior Developers, System Architects, AI/ML Engineers
**Last Updated**: March 23, 2026
**Version**: 1.8.0

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture Deep Dive](#architecture-deep-dive)
4. [Core Components](#core-components)
5. [Data Models](#data-models)
6. [API Reference](#api-reference)
7. [RAG Pipeline (14 Stages)](#rag-pipeline-14-stages)
8. [Database Schema](#database-schema)
9. [Deployment Architecture](#deployment-architecture)
10. [Development Guide](#development-guide)
11. [Testing Strategy](#testing-strategy)
12. [Performance Optimization](#performance-optimization)
13. [Troubleshooting Guide](#troubleshooting-guide)
14. [Future Roadmap](#future-roadmap)

---

## Executive Summary

### What is Desoutter Assistant?

Desoutter Assistant is an **enterprise-grade AI-powered technical support system** designed to assist technicians in diagnosing and repairing industrial tools. It combines:

- **Retrieval-Augmented Generation (RAG)** for context-aware responses
- **Hybrid Search** (Semantic + BM25) for comprehensive document retrieval
- **Self-Learning** from user feedback using Wilson score ranking
- **Multi-turn Conversations** with context preservation
- **GPU-accelerated Inference** using NVIDIA RTX A2000

### Key Metrics

| Metric | Value | Note |
|--------|-------|------|
| Test Pass Rate | 85% (34/40) | Production-ready |
| Vector DB Size | 4,082 chunks | 384-dim, language-filtered (multilingual PDF denoised) |
| Documents Indexed | 541 + 2,249 tickets | PDF, Word, Freshdesk |
| Products Supported | 451 | 71 wireless, 380 cable |
| Intent Types | 15 categories | Expanded from 8 |
| Avg Response Time | 23.6s (non-cached) | 2.4ms cached |
| Cache Speedup | ~100,000x | LRU + TTL |
| Hallucination Rate | <2% | Context grounding enabled |

---

## System Overview

### Technology Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                      DESOUTTER ASSISTANT                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │   Frontend   │  │   Backend    │  │    AI/ML Layer     │   │
│  │              │  │              │  │                    │   │
│  │  React 18.2  │──│  FastAPI     │──│  Ollama + Qwen2.5  │   │
│  │  Vite 5.0    │  │  Python 3.11 │  │  PyTorch 2.1.2     │   │
│  │  Axios       │  │  JWT Auth    │  │  HuggingFace       │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │   Database   │  │   Vector DB  │  │    Orchestration   │   │
│  │              │  │              │  │                    │   │
│  │  MongoDB 7.0 │  │  Qdrant      │  │  LangChain 0.1     │   │
│  │  27017       │  │  v1.7.4      │  │  Docker Compose    │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### High-Level Data Flow

```
User Query
    │
    ├─> [1] Authentication (JWT)
    │
    ├─> [2] RAG Engine (14-stage pipeline)
    │       │
    │       ├─> Off-topic Detection
    │       ├─> Language Detection (TR/EN)
    │       ├─> Cache Check (LRU + TTL)
    │       ├─> Self-Learning Boost
    │       ├─> Hybrid Retrieval (Semantic + BM25)
    │       ├─> Product Filtering (Qdrant)
    │       ├─> Context Grounding (>0.35 threshold)
    │       ├─> Context Optimization (8K tokens)
    │       ├─> Intent Detection (15 types)
    │       ├─> LLM Generation (Qwen2.5:7b)
    │       ├─> Response Validation
    │       ├─> Confidence Scoring
    │       └─> Save & Cache
    │
    └─> [3] Response + Sources + Confidence
            │
            └─> [4] User Feedback → Self-Learning Update
```

---

## Architecture Deep Dive

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐         ┌──────────────────┐                 │
│  │  Web Browser     │         │  Mobile PWA      │  (Future)       │
│  │  (React SPA)     │         │  (React Native)  │                 │
│  └────────┬─────────┘         └────────┬─────────┘                 │
│           │                            │                            │
│           └────────────────┬───────────┘                            │
│                            │                                        │
└────────────────────────────┼────────────────────────────────────────┘
                             │ HTTPS / WSS
┌────────────────────────────┼────────────────────────────────────────┐
│                      API GATEWAY LAYER                              │
├────────────────────────────┼────────────────────────────────────────┤
│                            ▼                                        │
│                   ┌─────────────────┐                               │
│                   │  FastAPI Server │                               │
│                   │   (Port 8000)   │                               │
│                   │                 │                               │
│                   │  • CORS Middleware                              │
│                   │  • JWT Auth      │                              │
│                   │  • Rate Limiting │ (Planned)                    │
│                   │  • Request Validation                           │
│                   └────────┬────────┘                               │
│                            │                                        │
└────────────────────────────┼────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                             │
├────────────────────────────┼────────────────────────────────────────┤
│                            │                                        │
│  ┌─────────────────────────┴─────────────────────────┐             │
│  │            RAG Engine (rag_engine.py)             │             │
│  │                                                   │             │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐      │
│  │  │ Intent       │  │ Hybrid       │  │ Self-Learning   │      │
│  │  │ Detector     │  │ Searcher     │  │ Engine          │      │
│  │  └──────────────┘  └──────────────┘  └─────────────────┘      │
│  │                                                   │             │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐      │
│  │  │ Context      │  │ Response     │  │ Confidence      │      │
│  │  │ Optimizer    │  │ Validator    │  │ Scorer          │      │
│  │  └──────────────┘  └──────────────┘  └─────────────────┘      │
│  │                                                   │             │
│  └───────────────────────────────────────────────────┘             │
│                            │                                        │
│  ┌─────────────────────────┴─────────────────────────┐             │
│  │       El-Harezmi Pipeline (In Progress)           │             │
│  │  5-stage intelligent processing (stage1-5)        │             │
│  └───────────────────────────────────────────────────┘             │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────┐
│                      DATA LAYER                                     │
├────────────────────────────┼────────────────────────────────────────┤
│                            │                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │  MongoDB    │  │   Qdrant    │  │   Ollama    │                │
│  │  (27017)    │  │   (6333)    │  │   (11434)   │                │
│  │             │  │             │  │             │                │
│  │ • users     │  │ • 4,082     │  │ • Qwen2.5   │                │
│  │ • products  │  │   chunks    │  │   :7b       │                │
│  │ • feedback  │  │ • 384-dim   │  │ • GPU       │                │
│  │ • mappings  │  │   vectors   │  │   accel     │                │
│  │ • sessions  │  │ • Metadata  │  │             │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Request Flow Example: "My EPB tool shows error E018"

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Authentication                                          │
├─────────────────────────────────────────────────────────────────┤
│ POST /diagnose                                                  │
│ Header: Authorization: Bearer eyJhbG...                         │
│ Body: {                                                         │
│   "part_number": "6151659000",                                  │
│   "fault_description": "error E018",                            │
│   "language": "en"                                              │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: RAG Engine Processing (14 stages)                      │
├─────────────────────────────────────────────────────────────────┤
│ [Stage 1] Off-topic Detection → ✅ Relevant                     │
│ [Stage 2] Language Detection → English                          │
│ [Stage 3] Cache Check → ❌ Not cached                           │
│ [Stage 4] Self-Learning → Apply learned boosts                  │
│ [Stage 5] Hybrid Retrieval:                                     │
│           • Semantic Search (60%): "error transducer fault"     │
│           • BM25 Search (40%): "E018" exact match               │
│           • RRF Fusion → Top 7 documents                        │
│ [Stage 6] Product Filter → Filter by "EPB" family              │
│ [Stage 7] Capability Filter → N/A                               │
│ [Stage 8] Context Grounding → Score: 0.89 (✅ > 0.35)          │
│ [Stage 9] Context Optimization → 8K token budget               │
│ [Stage 10] Intent Detection → "error_code"                      │
│ [Stage 11] LLM Generation → Qwen2.5:7b (GPU)                    │
│ [Stage 12] Response Validation → ✅ No hallucination           │
│ [Stage 13] Confidence Scoring → 0.89                            │
│ [Stage 14] Save & Cache → MongoDB + LRU cache                   │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Response                                                │
├─────────────────────────────────────────────────────────────────┤
│ {                                                               │
│   "diagnosis_id": "diag_abc123",                                │
│   "suggestion": "Error E018: Transducer Fault...",             │
│   "confidence": 0.89,                                           │
│   "sources": [                                                  │
│     {                                                           │
│       "document": "ESDE25004_EPB8_Transducer_Issue.pdf",       │
│       "page": 3,                                                │
│       "similarity": 0.91                                        │
│     }                                                           │
│   ],                                                            │
│   "intent": "error_code"                                        │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. RAG Engine (`src/llm/rag_engine.py`)

**Purpose**: Orchestrates the 14-stage RAG pipeline for generating AI-powered repair suggestions.

**Key Features**:
- Singleton pattern for efficient resource usage
- Lazy initialization of components
- Multi-stage pipeline with graceful fallbacks
- Self-learning integration

**Main Class**: `RAGEngine`

```python
class RAGEngine:
    """
    RAG Engine for repair suggestions with self-learning capabilities.

    Attributes:
        embeddings: EmbeddingsGenerator for text vectorization
        vectordb: QdrantDBClient for semantic search
        llm: OllamaClient for LLM generation
        hybrid_searcher: HybridSearcher for BM25 + Semantic fusion
        self_learning_engine: SelfLearningEngine for feedback-based learning
        response_cache: ResponseCache for query caching
    """
```

**Key Methods**:

```python
def generate_suggestion(
    self,
    part_number: str,
    fault_description: str,
    language: str = "en",
    conversation_id: Optional[str] = None
) -> Dict:
    """
    Generate repair suggestion using 14-stage RAG pipeline.

    Args:
        part_number: Product part number (e.g., "6151659000")
        fault_description: User's description of the fault
        language: Response language ("en" or "tr")
        conversation_id: Optional conversation ID for multi-turn

    Returns:
        Dict with keys:
            - suggestion: AI-generated repair suggestion
            - confidence: Confidence score (0.0-1.0)
            - sources: List of source documents with citations
            - intent: Detected query intent
            - diagnosis_id: Unique identifier for this diagnosis
    """
```

**Pipeline Stages**:

1. **Off-topic Detection**: Filters queries unrelated to Desoutter tools
2. **Language Detection**: Auto-detects Turkish or English
3. **Cache Check**: Returns cached response if available
4. **Self-Learning**: Applies learned query→source boosts
5. **Hybrid Retrieval**: Combines Semantic (60%) + BM25 (40%)
6. **Product Filtering**: Filters by product family (EPB, EAD, etc.)
7. **Capability Filtering**: Filters WiFi/Battery-specific content
8. **Context Grounding**: Refuses to answer if confidence < 0.35
9. **Context Optimization**: Deduplicates and fits 8K token budget
10. **Intent Detection**: Classifies into 15 intent types
11. **LLM Generation**: Generates response using Qwen2.5:7b
12. **Response Validation**: Detects hallucinations and uncertain phrases
13. **Confidence Scoring**: Multi-factor confidence calculation
14. **Save & Cache**: Persists to MongoDB and updates cache

---

### 2. Hybrid Search (`src/llm/hybrid_search.py`)

**Purpose**: Combines semantic (dense) and keyword (sparse) retrieval for comprehensive document matching.

**Algorithm**: Reciprocal Rank Fusion (RRF)

```python
class HybridSearcher:
    """
    Hybrid search combining Semantic (BERT) and BM25 (keyword) retrieval.

    Features:
    - Reciprocal Rank Fusion (RRF) for score fusion
    - Dynamic weight adjustment based on query type
    - Query expansion with domain synonyms

    Default Weights:
    - Semantic: 60% (meaning-based matching)
    - BM25: 40% (exact keyword matching)
    """

    def search(
        self,
        query: str,
        top_k: int = 7,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword matching.

        Process:
        1. Generate semantic embedding (384-dim)
        2. Perform Qdrant vector search → semantic_results
        3. Perform BM25 search on text index → bm25_results
        4. Apply RRF fusion: score = 1 / (k + rank)
        5. Combine and re-rank results
        6. Return top_k documents
        """
```

**RRF Formula**:

```
RRF_score(doc) = Σ [ 1 / (k + rank_i(doc)) ]

Where:
- k = 60 (RRF constant, reduces impact of high ranks)
- rank_i(doc) = rank of document in retrieval method i
- Final score = semantic_weight * RRF_semantic + bm25_weight * RRF_bm25
```

**Dynamic Weight Adjustment**:

```python
def get_dynamic_weights(query: str) -> tuple:
    """
    Adjust weights based on query characteristics:

    - Error codes (E018, I004) → BM25: 60%, Semantic: 40%
      Reason: Exact code matching is critical

    - Troubleshooting keywords → BM25: 45%, Semantic: 55%
      Reason: Balanced approach for symptom matching

    - General questions → BM25: 40%, Semantic: 60%
      Reason: Meaning more important than exact words
    """
```

---

### 3. Self-Learning Engine (`src/llm/self_learning.py`)

**Purpose**: Continuously improves response quality using user feedback and Wilson score ranking.

**Key Features**:
- Wilson score confidence interval for reliable ranking
- Source-level feedback tracking
- Query→Source pattern learning
- Contrastive learning data collection

**Wilson Score Formula**:

```
Wilson Score = (p̂ + z²/2n - z√(p̂(1-p̂)/n + z²/4n²)) / (1 + z²/n)

Where:
- p̂ = proportion of positive feedback
- n = total feedback count
- z = 1.96 (95% confidence interval)

This formula prevents sources with few ratings (e.g., 1/1 = 100%)
from outranking well-tested sources (e.g., 80/100 = 80%).
```

**Learning Cycle**:

```
1. User submits feedback (👍 helpful / 👎 not helpful)
   ↓
2. FeedbackEngine records:
   - diagnosis_id (links to query + sources)
   - rating ("helpful" | "not_helpful" | "partially_helpful")
   - reason (optional: "solved_problem", "incorrect_info", etc.)
   - comment (free text)
   ↓
3. SelfLearningEngine processes feedback:
   - Updates Wilson score for each source
   - Creates learned_mapping (query → source boost)
   - Collects contrastive pairs for embedding fine-tuning
   ↓
4. Next time similar query arrives:
   - Learned boosts applied to source scores
   - Higher-ranked sources retrieved first
   - Better response generated
```

**Data Structures**:

```python
# MongoDB Collection: diagnosis_feedback
{
    "_id": ObjectId("..."),
    "diagnosis_id": "diag_abc123",
    "rating": "helpful",  # helpful | not_helpful | partially_helpful
    "reason": "solved_problem",
    "comment": "The cable connector solution fixed the issue",
    "query": "error E018 transducer fault",
    "sources_used": [
        {
            "source_id": "src_xyz789",
            "document": "ESDE25004_EPB8_Transducer_Issue.pdf",
            "relevance_score": 0.91
        }
    ],
    "timestamp": ISODate("2026-03-23T10:30:00Z")
}

# MongoDB Collection: learned_mappings
{
    "_id": ObjectId("..."),
    "query_pattern": "E018.*transducer",  # Regex pattern
    "source_boost": {
        "ESDE25004_EPB8_Transducer_Issue.pdf": 1.5,  # 1.5x boost
        "EPB_Technical_Manual.pdf": 1.2
    },
    "confidence": 0.85,  # Wilson score
    "positive_count": 17,
    "total_count": 20,
    "last_updated": ISODate("2026-03-23T10:30:00Z")
}
```

---

### 4. Intent Detector (`src/llm/intent_detector.py`)

**Purpose**: Classifies user queries into 15 intent categories for targeted response generation.

**Intent Types** (Expanded from 8 to 15):

| Intent | Description | Example Queries |
|--------|-------------|-----------------|
| `error_code` | Error code lookup | "E018 nedir?", "What is error I004?" |
| `troubleshooting` | Problem diagnosis | "motor not starting", "tool won't calibrate" |
| `specification` | Technical specs | "max torque?", "EPB-1800 specifications" |
| `calibration` | Calibration procedures | "how to calibrate", "zero setting" |
| `maintenance` | Maintenance tasks | "lubrication interval", "replace battery" |
| `connection` | Network/WiFi setup | "connect to CVI3", "WiFi pairing" |
| `installation` | Setup procedures | "first time setup", "installation guide" |
| `configuration` | Parameter setup | "pset configuration", "torque settings" |
| `compatibility` | Tool-controller compat | "EPB compatible with CVI3?", "which platform?" |
| `procedure` | Step-by-step instructions | "disassembly procedure", "replacement steps" |
| `firmware` | Firmware update/downgrade | "firmware update", "version compatibility" |
| `comparison` | Model comparison | "EPB vs EAD", "which tool is better?" |
| `capability_query` | Feature inquiry | "WiFi var mı?", "does it have bluetooth?" |
| `accessory_query` | Accessory questions | "battery type", "charger compatibility" |
| `general` | Unclear/mixed intent | Default fallback |

**Detection Method**: Keyword-based pattern matching (reliable and fast)

```python
class IntentDetector:
    """
    Keyword-based intent classification for query routing.

    Priority Order (highest to lowest):
    1. error_code - Regex patterns for error codes
    2. troubleshooting - Problem keywords
    3. specifications - Spec keywords
    4. [remaining intents...]
    5. general - Default fallback
    """

    def detect_intent(self, query: str) -> IntentResult:
        """
        Detect query intent using pattern matching.

        Returns:
            IntentResult(
                intent: QueryIntent,
                confidence: float (0.0-1.0),
                matched_patterns: List[str],
                secondary_intent: Optional[QueryIntent]
            )
        """
```

---

### 5. Context Optimizer (`src/llm/context_optimizer.py`)

**Purpose**: Optimizes retrieved context to fit within LLM token budget (8K tokens) while maximizing relevance.

**Techniques**:
1. **Semantic Deduplication**: Removes near-duplicate chunks (>90% similarity)
2. **Relevance Sorting**: Prioritizes high-confidence sources
3. **Token Budget Management**: Truncates to fit 8K limit
4. **Metadata Preservation**: Keeps source citations intact

```python
class ContextOptimizer:
    """
    Optimizes retrieved context for LLM input.

    Token Budget: 8000 tokens

    Optimization Steps:
    1. Remove duplicate chunks (cosine similarity > 0.9)
    2. Sort by relevance score (descending)
    3. Fit within token budget (truncate if needed)
    4. Format with source citations
    """

    def optimize_context(
        self,
        retrieved_docs: List[Dict],
        token_budget: int = 8000
    ) -> str:
        """
        Optimize context to fit token budget.

        Returns:
            Formatted context string with citations:

            [Source: doc1.pdf, Page 5]
            Content from page 5...

            [Source: doc2.pdf, Page 12]
            Content from page 12...
        """
```

---

### 6. Response Validator (`src/llm/response_validator.py`)

**Purpose**: Detects hallucinations and validates response quality before returning to user.

**Validation Checks**:

1. **Uncertainty Phrase Detection**:
   - Flags responses with >2 uncertain phrases
   - Examples: "might", "probably", "I think", "not sure"

2. **Numerical Value Verification**:
   - Checks if numbers in response exist in source context
   - Prevents fabricated specifications

3. **Forbidden Content Detection**:
   - Blocks responses with dangerous advice
   - Examples: "ignore safety", "bypass protection"

4. **Minimum Length Check**:
   - Ensures response is at least 30 characters
   - Prevents empty or trivial responses

```python
class ResponseValidator:
    """
    Validates LLM responses to prevent hallucinations.

    Validation Flags:
    - uncertain: >2 uncertain phrases detected
    - unverified_numbers: Numbers not found in context
    - forbidden_content: Dangerous advice detected
    - too_short: Response < 30 characters
    """

    def validate(
        self,
        response: str,
        context: str
    ) -> ValidationResult:
        """
        Validate response against context.

        Returns:
            ValidationResult(
                is_valid: bool,
                flags: List[str],
                confidence_penalty: float
            )
        """
```

---

## Data Models

### Product Model (`src/database/models.py`)

**Schema Version**: v2 (Enhanced)

```python
class ProductModel(BaseModel):
    """
    Enhanced Product Schema v2

    Schema Evolution:
    - v1: Basic product info (legacy)
    - v2: Added categorization, platform relationships, wireless detection
    """

    # Core Fields (v1 - Backward Compatible)
    product_id: str                # Unique ID (part number)
    model_name: str                # e.g., "EPBC8-1800-4Q"
    part_number: str               # e.g., "6151659000"
    series_name: str               # e.g., "EPB"
    category: str                  # Legacy category
    product_url: str               # Desoutter product page
    image_url: str                 # Product image
    description: str               # Product description

    # Technical Specifications
    min_torque: str                # e.g., "0.5 Nm"
    max_torque: str                # e.g., "18 Nm"
    speed: str                     # e.g., "1800 RPM"
    output_drive: str              # e.g., "1/4\" hex"
    wireless_communication: str    # "Yes" or "No" (legacy)
    weight: str                    # e.g., "1.2 kg"

    # Enhanced Fields (v2)
    tool_category: str             # battery_tightening | cable_tightening |
                                   # electric_drilling | platform

    tool_type: Optional[str]       # pistol | angle_head | inline |
                                   # screwdriver | drill | fixtured

    product_family: str            # EPB | EAD | XPB | CVI | CVIR | etc.

    # Conditional Fields (based on tool_category)
    wireless: Optional[WirelessInfo]              # For battery_tightening
    platform_connection: Optional[PlatformConnection]  # For cable_tightening
    modular_system: Optional[ModularSystem]       # For electric_drilling

    # Metadata
    schema_version: int = 2
    scraped_date: str
    updated_at: str
    status: str                    # active | retired | discontinued


class WirelessInfo(BaseModel):
    """Wireless capability for battery tools"""
    capable: bool
    detection_method: str          # model_name_C | existing_field |
                                   # standalone_text_found | not_applicable
    compatible_platforms: List[str]  # ["CVI3", "Connect"]
    compatible_platform_ids: List[str]  # MongoDB ObjectIds


class PlatformConnection(BaseModel):
    """Platform connection for cable tools"""
    required: bool = True
    compatible_platforms: List[str]  # ["CVI3", "CVIR II", "ESP-C"]
    compatible_platform_ids: List[str]


class ModularSystem(BaseModel):
    """Modular system for drilling tools"""
    is_base_tool: bool             # XPB-Modular, XPB-One
    is_attachment: bool            # Tightening Head, Drilling Head
    attachment_type: Optional[str]  # tightening | drilling
    compatible_base_tools: List[str]
```

### Feedback Model (`src/database/feedback_models.py`)

```python
class DiagnosisFeedback(BaseModel):
    """User feedback on AI diagnosis"""

    diagnosis_id: str              # Links to diagnosis
    rating: str                    # helpful | not_helpful | partially_helpful
    reason: Optional[str]          # solved_problem | incorrect_info |
                                   # missing_details | wrong_product | etc.
    comment: Optional[str]         # Free text feedback

    # Context
    query: str                     # Original query
    part_number: str              # Product queried
    sources_used: List[SourceInfo]  # Sources in the response

    # Metadata
    user_id: str
    timestamp: datetime
    session_id: Optional[str]


class LearnedMapping(BaseModel):
    """Learned query→source patterns"""

    query_pattern: str             # Regex pattern (e.g., "E018.*transducer")
    source_boost: Dict[str, float]  # Document → boost multiplier

    # Wilson Score Stats
    confidence: float              # Wilson score (0.0-1.0)
    positive_count: int            # Number of helpful ratings
    total_count: int               # Total feedback count

    # Metadata
    created_at: datetime
    last_updated: datetime
    last_used: datetime
```

---

## API Reference

### Authentication Endpoints

#### POST `/auth/login`

**Description**: Authenticate user and get JWT token

**Request**:
```json
{
  "username": "admin",
  "password": "admin123"
}
```

**Response** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "username": "admin",
    "role": "admin"
  }
}
```

**Error** (401 Unauthorized):
```json
{
  "detail": "Invalid credentials"
}
```

---

#### GET `/auth/me`

**Description**: Get current user info

**Headers**:
```
Authorization: Bearer <token>
```

**Response** (200 OK):
```json
{
  "username": "admin",
  "role": "admin",
  "created_at": "2026-01-15T10:30:00Z"
}
```

---

### Diagnosis Endpoints

#### POST `/diagnose`

**Description**: Get AI-powered repair suggestion

**Headers**:
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Request**:
```json
{
  "part_number": "6151659000",
  "fault_description": "motor not starting, error code E018",
  "language": "en",
  "conversation_id": null  // Optional for multi-turn
}
```

**Response** (200 OK):
```json
{
  "diagnosis_id": "diag_20260323_abc123",
  "suggestion": "Error Code E018: Torque Out of Range!\n\nCause: The error code E018 indicates a transducer fault due to an incorrect cable assembly with the wrong connector.\n\nSolution:\n1. Check the cable assembly and connector compatibility\n2. Verify transducer connection is secure\n3. Replace cable if connector type is incorrect\n4. Perform calibration after cable replacement\n\nSafety Note: Disconnect tool from power before inspecting connections.",
  "confidence": 0.89,
  "sources": [
    {
      "document": "ESDE25004_ERS_range_EPB8_Transducer_Issue.pdf",
      "page": 3,
      "snippet": "E018 indicates transducer fault - check cable assembly connector type...",
      "similarity": 0.91,
      "source_type": "service_bulletin"
    },
    {
      "document": "EPB_Technical_Manual.pdf",
      "page": 47,
      "snippet": "Transducer calibration procedure after cable replacement...",
      "similarity": 0.85,
      "source_type": "technical_manual"
    }
  ],
  "intent": "error_code",
  "language_detected": "en",
  "processing_time": 23.6
}
```

---

#### POST `/diagnose/feedback`

**Description**: Submit feedback on diagnosis

**Headers**:
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Request**:
```json
{
  "diagnosis_id": "diag_20260323_abc123",
  "rating": "helpful",  // helpful | not_helpful | partially_helpful
  "reason": "solved_problem",  // Optional
  "comment": "The cable connector solution fixed the issue immediately"  // Optional
}
```

**Response** (200 OK):
```json
{
  "status": "success",
  "message": "Feedback recorded. Thank you for helping improve the system!"
}
```

---

### Conversation Endpoints

#### POST `/conversation/start`

**Description**: Start or continue a multi-turn conversation

**Request**:
```json
{
  "message": "My EPB tool is not starting",
  "part_number": "6151659030",
  "session_id": null  // null for new, or existing session_id to continue
}
```

**Response** (200 OK):
```json
{
  "session_id": "sess_xyz789",
  "message": "I'll help you troubleshoot your EPB tool. Can you tell me if any error codes are displayed?",
  "context_summary": "User reported EPB tool not starting. Awaiting error code information.",
  "turn_number": 1
}
```

---

### Admin Endpoints

#### GET `/admin/dashboard`

**Description**: Get system metrics and statistics

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Response** (200 OK):
```json
{
  "products": {
    "total": 451,
    "battery_tools": 71,
    "cable_tools": 380,
    "wireless_capable": 71
  },
  "documents": {
    "total": 541,
    "pdf": 121,
    "docx": 420,
    "freshdesk_tickets": 2249
  },
  "vector_db": {
    "total_chunks": 4082,
    "dimension": 384,
    "collection": "desoutter_docs_v2"
  },
  "queries": {
    "total_queries": 1523,
    "cache_hit_rate": 0.67,
    "avg_response_time": 23.6
  },
  "feedback": {
    "total_feedback": 428,
    "positive_rate": 0.82,
    "learned_mappings": 89
  }
}
```

---

#### POST `/admin/documents/upload`

**Description**: Upload document for RAG ingestion

**Headers**:
```
Authorization: Bearer <admin_token>
Content-Type: multipart/form-data
```

**Request**:
```
file: <PDF/DOCX/PPTX file>
document_type: service_bulletin  // Optional
```

**Response** (200 OK):
```json
{
  "status": "success",
  "filename": "ESDE25004_EPB8_Transducer_Issue.pdf",
  "file_size": 245760,
  "message": "Document uploaded. Run /admin/documents/ingest to process."
}
```

---

#### POST `/admin/documents/ingest`

**Description**: Process uploaded documents into vector database

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Response** (200 OK):
```json
{
  "status": "success",
  "documents_processed": 3,
  "chunks_created": 147,
  "duplicates_removed": 12,
  "processing_time": 45.2
}
```

---

## RAG Pipeline (14 Stages)

Detailed breakdown of each stage in the RAG pipeline.

### Stage 1: Off-topic Detection

**Purpose**: Filter queries unrelated to Desoutter tools

**Method**: Keyword matching against Desoutter-specific terms

**Keywords**:
- Product families: EPB, EAD, XPB, CVI, CVIR, ESP, etc.
- Tool types: screwdriver, tightening, drilling, torque, etc.
- Technical terms: calibration, pset, controller, transducer, etc.

**Logic**:
```python
if no_desoutter_keywords_found(query):
    return {
        "suggestion": "I'm specifically designed to help with Desoutter industrial tools. "
                      "Please ask questions about Desoutter products, repairs, or specifications.",
        "confidence": 0.0,
        "sources": [],
        "intent": "off_topic"
    }
```

---

### Stage 2: Language Detection

**Purpose**: Auto-detect query language (Turkish or English)

**Detection Criteria**:
1. Turkish-specific characters: ç, ğ, ı, ş, ö, ü
2. Common Turkish words: nedir, nasıl, hata, arıza, etc.

**Logic**:
```python
def detect_language(query: str) -> str:
    # Check for Turkish characters
    if any(char in query for char in 'çğışöüÇĞİŞÖÜ'):
        return "tr"

    # Check for Turkish words
    turkish_words = ['nedir', 'nasıl', 'hata', 'arıza', ...]
    if any(word in query.lower() for word in turkish_words):
        return "tr"

    return "en"  # Default to English
```

---

### Stage 3: Cache Check

**Purpose**: Return cached response if query was asked recently

**Cache Key**: MD5 hash of normalized query + part_number

**TTL**: 3600 seconds (1 hour)

**Speedup**: ~100,000x faster than full pipeline

**Logic**:
```python
cache_key = hashlib.md5(
    f"{normalize(query)}:{part_number}".encode()
).hexdigest()

cached_response = cache.get(cache_key)
if cached_response:
    logger.info(f"✅ Cache hit for query: {query[:50]}...")
    return cached_response  # Return immediately
```

---

### Stage 4: Self-Learning Context

**Purpose**: Apply learned query→source boosts from user feedback

**Process**:
1. Query self-learning engine for matching patterns
2. Get source boost multipliers (e.g., 1.5x for doc A)
3. Apply boosts during retrieval

**Example**:
```python
# Learned mapping: "E018.*transducer" → boost ESDE25004.pdf by 1.5x
learned_boosts = self_learning_engine.get_boosts(query)
# learned_boosts = {"ESDE25004_EPB8_Transducer_Issue.pdf": 1.5}

# Apply during retrieval
for doc in retrieved_docs:
    if doc['filename'] in learned_boosts:
        doc['score'] *= learned_boosts[doc['filename']]
```

---

### Stage 5: Hybrid Retrieval

**Purpose**: Combine semantic (meaning) and BM25 (keyword) search

**Sub-stages**:

1. **Semantic Search** (60% weight):
   - Generate 384-dim embedding with `all-MiniLM-L6-v2`
   - Query Qdrant vector database
   - Top 7 results by cosine similarity

2. **BM25 Search** (40% weight):
   - Tokenize query into terms
   - Query BM25 index (19,032 terms)
   - Top 7 results by BM25 score

3. **RRF Fusion**:
   - Apply Reciprocal Rank Fusion
   - Formula: `score = 1 / (k + rank)` where k=60
   - Combine scores: `0.6 * RRF_semantic + 0.4 * RRF_bm25`

4. **Re-rank and Deduplicate**:
   - Sort by combined score
   - Remove duplicates (same doc_id)
   - Return top 7 unique documents

**Code Flow**:
```python
# 1. Semantic search
query_vector = embeddings.encode(query)  # 384-dim
semantic_results = qdrant.search(
    collection="desoutter_docs_v2",
    query_vector=query_vector,
    limit=7
)

# 2. BM25 search
bm25_results = bm25_index.search(query, top_k=7)

# 3. RRF Fusion
fused_results = rrf_fusion(
    semantic_results,
    bm25_results,
    semantic_weight=0.6,
    bm25_weight=0.4,
    rrf_k=60
)

# 4. Return top 7
return fused_results[:7]
```

---

### Stage 6: Product Filtering

**Purpose**: Filter documents to only those relevant to queried product family

**Extraction Method**: `IntelligentProductExtractor` using 40+ regex patterns

**Product Families**: EPB, EAD, XPB, CVI, CVIR, ESP, ETP, etc.

**Qdrant Filtering**:
```python
# Extract product family from query
product_family = extractor.extract_product_family(query)  # e.g., "EPB"

# Apply Qdrant filter
filter_condition = Filter(
    must=[
        FieldCondition(
            key="product_family",
            match=MatchAny(any=[product_family])
        )
    ]
)

# Re-query with filter
filtered_results = qdrant.search(
    collection="desoutter_docs_v2",
    query_vector=query_vector,
    query_filter=filter_condition,
    limit=7
)
```

**Impact**: Prevents cross-product contamination (e.g., EPB query gets EPB docs only)

---

### Stage 7: Capability Filtering

**Purpose**: Filter WiFi/Battery-specific content based on query

**Detection**:
- WiFi queries: "wifi", "wireless", "connect", "pairing"
- Battery queries: "battery", "charge", "dock", "power"

**Logic**:
```python
if "wifi" in query.lower() or "wireless" in query.lower():
    # Filter to only wireless-capable products
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="wireless.capable",
                match=MatchValue(value=True)
            )
        ]
    )
```

---

### Stage 8: Context Grounding

**Purpose**: Refuse to answer if retrieved context is insufficient

**Metrics**:
1. **Top Document Similarity**: Must be > 0.35
2. **Average Similarity**: Should be > 0.30
3. **Relevant Document Count**: At least 2 documents

**Logic**:
```python
top_similarity = retrieved_docs[0]['score']
avg_similarity = mean([doc['score'] for doc in retrieved_docs])
relevant_count = sum(1 for doc in retrieved_docs if doc['score'] > 0.3)

if top_similarity < 0.35 or relevant_count < 2:
    return {
        "suggestion": "I don't have enough information to confidently answer this question. "
                      "Could you provide more details or check the product manual?",
        "confidence": 0.0,
        "sources": [],
        "intent": "insufficient_context"
    }
```

**Benefits**:
- Prevents hallucinations
- Maintains user trust
- Acceptable "I don't know" rate: 10-15%

---

### Stage 9: Context Optimization

**Purpose**: Optimize retrieved context to fit 8K token budget

**Steps**:

1. **Semantic Deduplication**:
   ```python
   # Remove near-duplicate chunks (>90% similarity)
   deduplicated = []
   for chunk in chunks:
       if not any(cosine_sim(chunk, existing) > 0.9 for existing in deduplicated):
           deduplicated.append(chunk)
   ```

2. **Relevance Sorting**:
   ```python
   # Sort by score (descending)
   sorted_chunks = sorted(deduplicated, key=lambda x: x['score'], reverse=True)
   ```

3. **Token Budget Fitting**:
   ```python
   # Fit within 8000 tokens
   optimized_context = ""
   token_count = 0

   for chunk in sorted_chunks:
       chunk_tokens = count_tokens(chunk['text'])
       if token_count + chunk_tokens <= 8000:
           optimized_context += f"\n[Source: {chunk['document']}, Page {chunk['page']}]\n"
           optimized_context += chunk['text'] + "\n"
           token_count += chunk_tokens
       else:
           break
   ```

---

### Stage 10: Intent Detection

**Purpose**: Classify query into 15 intent categories

**Intents**: (See [Intent Detector](#4-intent-detector-srcllmintent_detectorpy) section)

**Usage**: Selects appropriate system prompt and response format

**Example**:
```python
intent = intent_detector.detect_intent(query)
# intent = "error_code"

# Select system prompt
if intent == "error_code":
    system_prompt = get_error_code_prompt()
elif intent == "troubleshooting":
    system_prompt = get_troubleshooting_prompt()
else:
    system_prompt = get_general_prompt()
```

---

### Stage 11: LLM Generation

**Purpose**: Generate natural language response using Qwen2.5:7b

**LLM**: Ollama + Qwen2.5:7b-instruct

**Parameters**:
- Temperature: 0.1 (deterministic)
- Max Tokens: 512
- Timeout: 10 seconds
- GPU: NVIDIA RTX A2000 (6GB VRAM)

**Prompt Structure**:
```
SYSTEM PROMPT:
You are an expert technician assistant for Desoutter industrial tools.
Your role is to provide accurate, safe, and practical repair suggestions.
Always cite sources and prioritize safety.

CONTEXT:
[Source: ESDE25004_EPB8_Transducer_Issue.pdf, Page 3]
E018 indicates transducer fault due to incorrect cable connector...

[Source: EPB_Technical_Manual.pdf, Page 47]
Transducer calibration procedure after cable replacement...

USER QUERY:
My EPB tool shows error E018. Motor not starting.

ASSISTANT RESPONSE:
```

**Generation**:
```python
response = ollama_client.generate(
    model="qwen2.5:7b-instruct",
    system_prompt=system_prompt,
    context=optimized_context,
    query=query,
    temperature=0.1,
    max_tokens=512,
    timeout=10
)
```

---

### Stage 12: Response Validation

**Purpose**: Detect hallucinations and quality issues

**Checks**: (See [Response Validator](#6-response-validator-srcllmresponse_validatorpy) section)

1. Uncertainty phrase count
2. Numerical value verification
3. Forbidden content detection
4. Minimum length check

**Action**:
```python
validation = validator.validate(response, context)

if not validation.is_valid:
    logger.warning(f"Response validation failed: {validation.flags}")
    # Apply confidence penalty
    confidence_score *= (1.0 - validation.confidence_penalty)
```

---

### Stage 13: Confidence Scoring

**Purpose**: Calculate multi-factor confidence score (0.0-1.0)

**Factors**:

1. **Top Document Similarity** (40% weight):
   ```python
   similarity_score = retrieved_docs[0]['score']  # 0.0-1.0
   ```

2. **Context Coverage** (30% weight):
   ```python
   relevant_count = sum(1 for doc in retrieved_docs if doc['score'] > 0.3)
   coverage_score = min(relevant_count / 5.0, 1.0)
   ```

3. **Response Quality** (20% weight):
   ```python
   quality_score = 1.0
   if validation.has_uncertainty_phrases:
       quality_score -= 0.2
   if validation.has_unverified_numbers:
       quality_score -= 0.3
   ```

4. **Intent Confidence** (10% weight):
   ```python
   intent_confidence = intent_detector.confidence  # 0.0-1.0
   ```

**Final Score**:
```python
confidence = (
    0.4 * similarity_score +
    0.3 * coverage_score +
    0.2 * quality_score +
    0.1 * intent_confidence
)
```

---

### Stage 14: Save & Cache

**Purpose**: Persist diagnosis to MongoDB and update cache

**MongoDB Save**:
```python
diagnosis_record = {
    "_id": f"diag_{timestamp}_{random_id}",
    "query": query,
    "part_number": part_number,
    "suggestion": response,
    "confidence": confidence_score,
    "sources": [
        {
            "document": doc['document'],
            "page": doc['page'],
            "similarity": doc['score'],
            "snippet": doc['text'][:200]
        }
        for doc in retrieved_docs
    ],
    "intent": intent,
    "language": language,
    "processing_time": elapsed_time,
    "timestamp": datetime.utcnow()
}

mongodb.diagnoses.insert_one(diagnosis_record)
```

**Cache Update**:
```python
cache_key = generate_cache_key(query, part_number)
cache.set(
    key=cache_key,
    value=diagnosis_record,
    ttl=3600  # 1 hour
)
```

---

## Database Schema

### MongoDB Collections

#### 1. `users`

```javascript
{
  "_id": ObjectId("..."),
  "username": "admin",
  "password_hash": "$2b$12$...",  // Bcrypt hash
  "role": "admin",  // admin | technician
  "created_at": ISODate("2026-01-15T10:30:00Z"),
  "last_login": ISODate("2026-03-23T14:20:00Z"),
  "status": "active"  // active | disabled
}
```

**Indexes**:
- `username` (unique)
- `role`

---

#### 2. `products`

```javascript
{
  "_id": ObjectId("..."),
  "product_id": "6151659000",
  "model_name": "EPBC8-1800-4Q",
  "part_number": "6151659000",
  "series_name": "EPB",
  "product_family": "EPB",

  // Technical Specs
  "min_torque": "0.5 Nm",
  "max_torque": "18 Nm",
  "speed": "1800 RPM",
  "output_drive": "1/4\" hex",
  "weight": "1.2 kg",

  // Enhanced Categorization (v2)
  "tool_category": "battery_tightening",
  "tool_type": "pistol",
  "wireless": {
    "capable": true,
    "detection_method": "model_name_C",
    "compatible_platforms": ["CVI3", "Connect"],
    "compatible_platform_ids": ["<CVI3_ObjectId>", "<Connect_ObjectId>"]
  },

  // Metadata
  "schema_version": 2,
  "scraped_date": "2025-11-15",
  "updated_at": "2026-01-10T12:00:00Z",
  "status": "active"
}
```

**Indexes**:
- `product_id` (unique)
- `part_number`
- `product_family`
- `tool_category`
- `wireless.capable`

---

#### 3. `diagnoses`

```javascript
{
  "_id": "diag_20260323_abc123",
  "query": "error E018 transducer fault",
  "part_number": "6151659000",
  "suggestion": "Error Code E018: Torque Out of Range!...",
  "confidence": 0.89,
  "sources": [
    {
      "document": "ESDE25004_EPB8_Transducer_Issue.pdf",
      "page": 3,
      "similarity": 0.91,
      "snippet": "E018 indicates transducer fault..."
    }
  ],
  "intent": "error_code",
  "language": "en",
  "processing_time": 23.6,
  "user_id": "admin",
  "timestamp": ISODate("2026-03-23T10:30:00Z")
}
```

**Indexes**:
- `_id` (unique)
- `user_id`
- `part_number`
- `timestamp` (descending)

---

#### 4. `diagnosis_feedback`

```javascript
{
  "_id": ObjectId("..."),
  "diagnosis_id": "diag_20260323_abc123",
  "rating": "helpful",  // helpful | not_helpful | partially_helpful
  "reason": "solved_problem",
  "comment": "The cable connector solution fixed the issue",

  // Context (for learning)
  "query": "error E018 transducer fault",
  "part_number": "6151659000",
  "sources_used": [
    {
      "source_id": "src_xyz789",
      "document": "ESDE25004_EPB8_Transducer_Issue.pdf",
      "relevance_score": 0.91
    }
  ],

  // Metadata
  "user_id": "tech_user",
  "timestamp": ISODate("2026-03-23T10:35:00Z")
}
```

**Indexes**:
- `diagnosis_id`
- `rating`
- `timestamp` (descending)

---

#### 5. `learned_mappings`

```javascript
{
  "_id": ObjectId("..."),
  "query_pattern": "E018.*transducer",  // Regex pattern
  "source_boost": {
    "ESDE25004_EPB8_Transducer_Issue.pdf": 1.5,
    "EPB_Technical_Manual.pdf": 1.2
  },

  // Wilson Score Stats
  "confidence": 0.85,
  "positive_count": 17,
  "total_count": 20,

  // Metadata
  "created_at": ISODate("2026-02-01T08:00:00Z"),
  "last_updated": ISODate("2026-03-23T10:35:00Z"),
  "last_used": ISODate("2026-03-23T10:30:00Z")
}
```

**Indexes**:
- `query_pattern` (unique)
- `confidence` (descending)
- `last_used` (descending)

---

#### 6. `conversation_sessions`

```javascript
{
  "_id": "sess_xyz789",
  "user_id": "tech_user",
  "part_number": "6151659030",
  "turns": [
    {
      "turn_number": 1,
      "user_message": "My EPB tool is not starting",
      "assistant_message": "Can you tell me if any error codes are displayed?",
      "timestamp": ISODate("2026-03-23T10:30:00Z")
    },
    {
      "turn_number": 2,
      "user_message": "Yes, it shows E018",
      "assistant_message": "Error E018 indicates a transducer fault...",
      "timestamp": ISODate("2026-03-23T10:32:00Z")
    }
  ],
  "context_summary": "User reported EPB tool not starting with error E018. Diagnosed transducer fault.",
  "status": "active",  // active | completed
  "created_at": ISODate("2026-03-23T10:30:00Z"),
  "updated_at": ISODate("2026-03-23T10:32:00Z")
}
```

**Indexes**:
- `_id` (unique)
- `user_id`
- `status`
- `updated_at` (descending)

---

### Qdrant Collections

#### `desoutter_docs_v2`

**Purpose**: Stores document embeddings for semantic search

**Schema**:
```python
{
    "id": "chunk_abc123",  # Unique chunk ID
    "vector": [0.123, -0.456, ...],  # 384-dimensional embedding
    "payload": {
        # Document Info
        "document": "ESDE25004_EPB8_Transducer_Issue.pdf",
        "document_type": "service_bulletin",  # service_bulletin | technical_manual | user_manual
        "page": 3,
        "chunk_index": 5,

        # Product Info
        "product_family": "EPB",
        "product_models": ["EPBC8-1800-4Q", "EPBC8-2000-4Q"],

        # Content
        "text": "E018 indicates transducer fault due to incorrect cable connector...",
        "heading": "Error Code E018 - Transducer Fault",

        # Metadata
        "importance_score": 0.95,  # 0.0-1.0
        "is_procedure": True,
        "has_error_code": True,
        "error_codes": ["E018"],

        # Deduplication
        "content_hash": "sha256_hash...",

        # Timestamps
        "ingested_at": "2026-01-15T10:00:00Z",
        "updated_at": "2026-01-15T10:00:00Z"
    }
}
```

**Indexes**:
- `vector` (HNSW index for fast ANN search)
- `payload.product_family` (for filtering)
- `payload.document_type` (for boosting)
- `payload.error_codes` (for exact matching)

**Vector Index Config**:
```python
{
    "vectors": {
        "size": 384,
        "distance": "Cosine"
    },
    "hnsw_config": {
        "m": 16,
        "ef_construct": 100,
        "full_scan_threshold": 10000
    }
}
```

---

## Deployment Architecture

### Infrastructure Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    PROXMOX VIRTUALIZATION                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │            Ubuntu 22.04 LTS VM                           │ │
│  │            CPU: 8 cores (Xeon)                           │ │
│  │            RAM: 32 GB                                    │ │
│  │            Storage: 500 GB SSD                           │ │
│  │            GPU: NVIDIA RTX A2000 (6GB VRAM, PCIe)       │ │
│  │                                                          │ │
│  │  ┌────────────────────────────────────────────────────┐ │ │
│  │  │         Docker Compose Stack                       │ │ │
│  │  │                                                    │ │ │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │ │ │
│  │  │  │ MongoDB  │  │  Qdrant  │  │    Ollama    │   │ │ │
│  │  │  │  (27017) │  │  (6333)  │  │   (11434)    │   │ │ │
│  │  │  │  2GB RAM │  │  4GB RAM │  │   8GB RAM    │   │ │ │
│  │  │  │          │  │          │  │   GPU access │   │ │ │
│  │  │  └──────────┘  └──────────┘  └──────────────┘   │ │ │
│  │  │                                                    │ │ │
│  │  │  ┌──────────────────┐  ┌────────────────────┐   │ │ │
│  │  │  │  desoutter-api   │  │ desoutter-frontend │   │ │ │
│  │  │  │    (8000)        │  │      (3001)        │   │ │ │
│  │  │  │  12GB RAM        │  │    1GB RAM         │   │ │ │
│  │  │  │  GPU access      │  │                    │   │ │ │
│  │  │  └──────────────────┘  └────────────────────┘   │ │ │
│  │  │                                                    │ │ │
│  │  └────────────────────────────────────────────────────┘ │ │
│  │                                                          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                           │
                           │ Ethernet (1 Gbps)
                           ▼
┌────────────────────────────────────────────────────────────────┐
│                    NETWORK LAYER                                │
│                                                                │
│  ┌──────────────┐         ┌──────────────┐                    │
│  │ Load Balancer│         │   Firewall   │                    │
│  │   (Future)   │         │   (iptables) │                    │
│  └──────────────┘         └──────────────┘                    │
└────────────────────────────────────────────────────────────────┘
```

### Resource Allocation

| Service | CPU Cores | RAM | Disk | GPU | Priority |
|---------|-----------|-----|------|-----|----------|
| **MongoDB** | 1 | 2 GB | 50 GB | - | High |
| **Qdrant** | 2 | 4 GB | 100 GB | - | High |
| **Ollama** | 2 | 8 GB | 100 GB | RTX A2000 | Critical |
| **desoutter-api** | 3 | 12 GB | 50 GB | RTX A2000 | Critical |
| **desoutter-frontend** | 1 | 1 GB | 10 GB | - | Medium |
| **Total** | **9** | **27 GB** | **310 GB** | **1 GPU** | - |

### Network Configuration

```bash
# Docker network: ai-net
docker network create ai-net

# Service DNS names (internal):
mongodb        → mongodb:27017
qdrant         → qdrant:6333
ollama         → ollama:11434
desoutter-api  → desoutter-api:8000

# External access:
Frontend: http://<VM_IP>:3001
API: http://<VM_IP>:8000
API Docs: http://<VM_IP>:8000/docs
```

### Docker Compose Configuration

```yaml
version: '3.8'

services:
  mongodb:
    image: mongo:7.0
    container_name: mongodb
    restart: unless-stopped
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    networks:
      - ai-net
    healthcheck:
      test: echo 'db.runCommand("ping").ok' | mongosh localhost:27017/test --quiet
      interval: 30s
      timeout: 10s
      retries: 3

  qdrant:
    image: qdrant/qdrant:v1.7.4
    container_name: qdrant
    restart: unless-stopped
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - ai-net

  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    restart: unless-stopped
    runtime: nvidia
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    networks:
      - ai-net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  desoutter-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: desoutter-api
    restart: unless-stopped
    runtime: nvidia
    ports:
      - "8000:8000"
    volumes:
      - desoutter_data:/app/data
      - ./documents:/app/documents
    environment:
      - MONGO_HOST=mongodb
      - QDRANT_HOST=qdrant
      - OLLAMA_BASE_URL=http://ollama:11434
      - OLLAMA_MODEL=qwen2.5:7b-instruct
      - EMBEDDING_DEVICE=cuda
      - JWT_SECRET=${JWT_SECRET}
    depends_on:
      mongodb:
        condition: service_healthy
      qdrant:
        condition: service_started
      ollama:
        condition: service_started
    networks:
      - ai-net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  desoutter-frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: desoutter-frontend
    restart: unless-stopped
    ports:
      - "3001:3001"
    environment:
      - VITE_API_URL=http://<VM_IP>:8000
    networks:
      - ai-net

networks:
  ai-net:
    driver: bridge

volumes:
  mongodb_data:
  qdrant_data:
  ollama_models:
  desoutter_data:
```

---

## Development Guide

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- NVIDIA GPU (optional but recommended)
- 16GB RAM minimum

### Local Development Setup

#### 1. Clone Repository

```bash
git clone https://github.com/fatihhbayram/desoutter-assistant.git
cd desoutter-assistant
```

#### 2. Python Environment

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 3. Environment Variables

```bash
cp .env.example .env
# Edit .env with your settings
```

#### 4. Start Services

```bash
# Start MongoDB, Qdrant, Ollama
docker-compose up -d mongodb qdrant ollama

# Wait for services to initialize
sleep 30

# Pull Qwen2.5 model (if not already pulled)
docker exec -it ollama ollama pull qwen2.5:7b-instruct
```

#### 5. Run API Server

```bash
# Option A: Direct Python (for debugging)
python src/api/main.py

# Option B: Uvicorn with reload (for development)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 6. Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Access:
- Frontend: http://localhost:5173 (Vite dev server)
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Project Structure for Developers

```
desoutter-assistant/
│
├── src/                           # Backend source code
│   ├── api/                       # FastAPI routes
│   │   ├── main.py               # Main API app (authentication, endpoints)
│   │   └── el_harezmi_router.py  # El-Harezmi pipeline routes
│   │
│   ├── llm/                       # RAG and AI components
│   │   ├── rag_engine.py         # Main RAG orchestrator (14-stage pipeline)
│   │   ├── hybrid_search.py      # Hybrid search (Semantic + BM25)
│   │   ├── self_learning.py      # Feedback learning engine
│   │   ├── intent_detector.py    # Query intent classification (15 types)
│   │   ├── response_validator.py # Hallucination detection
│   │   ├── confidence_scorer.py  # Confidence calculation
│   │   ├── context_optimizer.py  # Token budget management
│   │   ├── response_cache.py     # LRU + TTL caching
│   │   ├── prompts.py            # System prompts and templates
│   │   ├── ollama_client.py      # Ollama LLM client
│   │   ├── conversation.py       # Multi-turn conversation manager
│   │   ├── domain_embeddings.py  # Domain-specific embeddings (351 terms)
│   │   ├── domain_vocabulary.py  # Desoutter terminology
│   │   └── feedback_engine.py    # Feedback processing
│   │
│   ├── database/                  # MongoDB operations
│   │   ├── mongo_client.py       # MongoDB connection and CRUD
│   │   ├── models.py             # Product, User models (Pydantic)
│   │   └── feedback_models.py    # Feedback, LearnedMapping models
│   │
│   ├── vectordb/                  # Qdrant vector database
│   │   └── qdrant_client.py      # Qdrant operations (search, insert, filter)
│   │
│   ├── documents/                 # Document processing
│   │   ├── pdf_processor.py      # PDF extraction (PyPDF2, pdfplumber)
│   │   ├── document_classifier.py # Document type classification
│   │   ├── embeddings.py         # Embedding generation (all-MiniLM-L6-v2)
│   │   ├── product_extractor.py  # Product family extraction (40+ patterns)
│   │   └── chunkers/             # Chunking strategies
│   │       ├── semantic_chunker.py       # Sentence-based chunking
│   │       ├── table_aware_chunker.py    # Table-preserving chunking
│   │       ├── step_preserving_chunker.py # Procedure step chunking
│   │       └── hybrid_chunker.py         # Combined strategy
│   │
│   ├── el_harezmi/                # El-Harezmi 5-stage pipeline (in progress)
│   │   ├── pipeline.py           # Main pipeline orchestrator
│   │   ├── stage1_intent_classifier.py   # Intent detection
│   │   ├── stage2_retrieval_strategy.py  # Adaptive retrieval
│   │   ├── stage3_info_extraction.py     # Information extraction
│   │   ├── stage4_kg_validation.py       # Knowledge graph validation
│   │   └── stage5_response_formatter.py  # Response formatting
│   │
│   ├── scraper/                   # Web scraping (Freshdesk, product pages)
│   │   ├── desoutter_scraper.py  # Product scraper
│   │   ├── ticket_scraper_sync.py # Freshdesk ticket scraper
│   │   ├── ticket_preprocessor.py # Ticket text preprocessing
│   │   └── ingest_tickets.py     # Ticket ingestion to Qdrant
│   │
│   └── utils/                     # Utilities
│       ├── logger.py             # Logging setup
│       └── helpers.py            # Helper functions
│
├── frontend/                      # React frontend
│   ├── src/
│   │   ├── App.jsx               # Main app (routing, auth)
│   │   ├── TechWizard.jsx        # Technician chat interface
│   │   ├── MetricsDashboard.jsx  # Admin dashboard (planned)
│   │   └── components/           # Reusable components
│   ├── package.json
│   └── vite.config.js
│
├── config/                        # Configuration
│   ├── ai_settings.py            # RAG parameters, LLM settings, thresholds
│   └── settings.py               # Base settings (MongoDB, paths)
│
├── scripts/                       # Utility scripts
│   ├── run_api.py                # Start API server
│   ├── ingest_documents.py       # Document ingestion script
│   ├── test_hybrid_search.py     # Test hybrid search
│   ├── test_product_filtering.py # Test product filtering
│   └── run_baseline_test.sh      # Test suite runner
│
├── tests/                         # Test files
│   ├── test_rag_comprehensive.py # RAG pipeline tests
│   ├── test_hybrid_search.py     # Hybrid search tests
│   ├── test_intent_detector.py   # Intent detection tests
│   └── fixtures/                 # Test data
│
├── documents/                     # Document storage
│   ├── manuals/                  # Technical manuals (PDF)
│   └── bulletins/                # Service bulletins (PDF)
│
├── data/                          # Runtime data
│   ├── logs/                     # Application logs
│   ├── cache/                    # Response cache
│   └── documents/                # Processed documents
│
├── docker-compose.yml             # Multi-container orchestration
├── Dockerfile                     # Backend container image
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── README.md                      # Project overview
├── QUICKSTART.md                  # Quick setup guide
├── ROADMAP.md                     # Development roadmap
└── TECHNICAL_DOCUMENTATION.md     # This file
```

### Adding a New Feature

#### Example: Add a new intent type "safety_warning"

1. **Update Intent Detector** (`src/llm/intent_detector.py`):

```python
class QueryIntent(str, Enum):
    # Existing intents...
    SAFETY_WARNING = "safety_warning"  # New intent

class IntentDetector:
    def __init__(self):
        # Add safety warning keywords
        self.safety_patterns = [
            'safety', 'warning', 'caution', 'danger', 'hazard',
            'güvenlik', 'uyarı', 'tehlike'  # Turkish
        ]

    def detect_intent(self, query: str) -> IntentResult:
        # Add safety check (high priority)
        if self._has_safety_keywords(query):
            return IntentResult(
                intent=QueryIntent.SAFETY_WARNING,
                confidence=0.95,
                matched_patterns=self.safety_patterns
            )
        # Existing logic...
```

2. **Add System Prompt** (`src/llm/prompts.py`):

```python
def get_safety_warning_prompt() -> str:
    """System prompt for safety warnings"""
    return """You are a safety-focused assistant for Desoutter industrial tools.
When providing safety warnings:
1. **Highlight dangers clearly**
2. List all safety precautions
3. Reference official safety documents
4. Use urgent but professional tone

Always include: "⚠️ SAFETY WARNING" at the top of your response."""
```

3. **Update RAG Engine** (`src/llm/rag_engine.py`):

```python
def generate_suggestion(self, ...):
    # Detect intent
    intent_result = self.intent_detector.detect_intent(query)

    # Select system prompt based on intent
    if intent_result.intent == QueryIntent.SAFETY_WARNING:
        system_prompt = get_safety_warning_prompt()
    elif intent_result.intent == QueryIntent.ERROR_CODE:
        system_prompt = get_error_code_prompt()
    # ... other intents
```

4. **Add Tests** (`tests/test_intent_detector.py`):

```python
def test_safety_warning_detection():
    detector = IntentDetector()

    # Test English
    result = detector.detect_intent("safety warning for EPB tool")
    assert result.intent == QueryIntent.SAFETY_WARNING
    assert result.confidence > 0.9

    # Test Turkish
    result = detector.detect_intent("EPB aleti için güvenlik uyarısı")
    assert result.intent == QueryIntent.SAFETY_WARNING
```

5. **Run Tests**:

```bash
pytest tests/test_intent_detector.py::test_safety_warning_detection -v
```

---

## Testing Strategy

### Test Pyramid

```
           ┌─────────────────┐
           │   End-to-End    │  (10%)  - Full RAG pipeline tests
           │   Tests (E2E)   │          40 scenarios
           └─────────────────┘
                  │
         ┌────────────────────┐
         │  Integration Tests │  (30%)  - Multi-component tests
         │   (Hybrid Search,  │          API + DB, Vector search
         │    Product Filter) │
         └────────────────────┘
                  │
      ┌────────────────────────┐
      │     Unit Tests         │  (60%)  - Single component tests
      │  (Intent Detector,     │          Each function isolated
      │   Validator, Scorer)   │
      └────────────────────────┘
```

### Test Files

| File | Coverage | Description |
|------|----------|-------------|
| `test_rag_comprehensive.py` | E2E | 40 test scenarios covering all intent types |
| `test_hybrid_search.py` | Integration | BM25 + Semantic fusion, RRF algorithm |
| `test_product_filtering.py` | Integration | Product family extraction and Qdrant filtering |
| `test_intent_detector.py` | Unit | All 15 intent types, Turkish/English |
| `test_response_validator.py` | Unit | Hallucination detection, uncertain phrases |
| `test_confidence_scorer.py` | Unit | Multi-factor confidence calculation |
| `test_context_grounding.py` | Unit | Insufficient context detection |
| `test_cache.py` | Unit | LRU + TTL cache operations |

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_rag_comprehensive.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only E2E tests
pytest tests/test_rag_comprehensive.py -v -m e2e

# Run baseline test suite (40 scenarios)
./scripts/run_baseline_test.sh
```

### Test Results (Baseline)

```
========== Desoutter Assistant - Baseline Test Results ==========
Test Pass Rate: 85% (34/40 scenarios)
Timeout Rate: 0% (0/40 timeouts)

By Intent:
- error_code:        5/5  (100%)  ✅
- troubleshooting:   8/9  (89%)   ✅
- specification:     6/6  (100%)  ✅
- calibration:       4/4  (100%)  ✅
- maintenance:       3/4  (75%)   ⚠️
- connection:        3/3  (100%)  ✅
- installation:      2/3  (67%)   ⚠️
- configuration:     3/3  (100%)  ✅
- compatibility:     0/1  (0%)    ❌
- capability_query:  0/2  (0%)    ❌

By Product Family:
- EPB:   18/20  (90%)   ✅
- EAD:   8/10   (80%)   ✅
- CVI:   5/6    (83%)   ✅
- XPB:   3/4    (75%)   ⚠️

Avg Response Time: 23.6s (non-cached)
Cache Hit Rate: 67%
```

---

## Performance Optimization

### Performance Bottlenecks & Solutions

| Bottleneck | Impact | Solution | Result |
|------------|--------|----------|--------|
| **LLM Generation** | 20-30s | GPU acceleration (RTX A2000) | 5-10s |
| **Repeated Queries** | Full pipeline | LRU + TTL caching | 2.4ms (100,000x) |
| **Large Context** | Token limit exceeded | Context optimization (8K budget) | Fits all cases |
| **Vector Search** | 500-1000ms | HNSW index | <100ms |
| **BM25 Search** | 200-300ms | Inverted index | <50ms |
| **Embedding Generation** | 100-200ms | Batch processing | <50ms |

### Optimization Techniques

#### 1. Response Caching

```python
from src.llm.response_cache import get_response_cache

cache = get_response_cache(
    max_size=1000,
    default_ttl=3600,  # 1 hour
    enable_similarity=True  # Semantic similarity matching
)

# Cache hit example
cache_key = generate_cache_key(query, part_number)
cached = cache.get(cache_key)
if cached:
    return cached  # 2.4ms response time
```

**Speedup**: ~100,000x for exact matches, ~1,000x for similar queries

---

#### 2. GPU Acceleration

```python
# Ollama with GPU
OLLAMA_BASE_URL = "http://ollama:11434"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
EMBEDDING_DEVICE = "cuda"  # Use GPU for embeddings

# Docker GPU passthrough
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

**Speedup**:
- LLM Generation: 30s → 10s (3x)
- Embedding Generation: 200ms → 50ms (4x)

---

#### 3. Context Optimization

```python
# Token budget management
optimizer = ContextOptimizer(token_budget=8000)

optimized_context = optimizer.optimize_context(
    retrieved_docs,
    token_budget=8000
)

# Techniques:
# 1. Semantic deduplication (>90% similarity)
# 2. Relevance sorting (high scores first)
# 3. Token truncation (fit 8K limit)
```

**Result**: 100% of queries fit within 8K token limit

---

#### 4. Hybrid Search RRF

```python
# Parallel search
semantic_results = qdrant.search(query_vector, top_k=7)  # 100ms
bm25_results = bm25_index.search(query, top_k=7)         # 50ms

# RRF fusion (vectorized)
fused = rrf_fusion(semantic_results, bm25_results)  # 10ms

# Total: 160ms (vs 250ms sequential)
```

**Speedup**: 1.5x through parallelization

---

#### 5. Database Indexing

```python
# MongoDB indexes
db.products.create_index([("part_number", 1)])
db.products.create_index([("product_family", 1)])
db.diagnoses.create_index([("timestamp", -1)])
db.learned_mappings.create_index([("confidence", -1)])

# Qdrant HNSW index
qdrant.create_collection(
    collection_name="desoutter_docs_v2",
    vectors_config={
        "size": 384,
        "distance": "Cosine"
    },
    hnsw_config={
        "m": 16,
        "ef_construct": 100
    }
)
```

**Result**:
- MongoDB queries: <10ms
- Qdrant ANN search: <100ms

---

### Performance Monitoring

```python
from src.llm.performance_metrics import get_performance_monitor

monitor = get_performance_monitor()

# Record query metrics
with monitor.track_query("diagnose"):
    result = rag_engine.generate_suggestion(...)

# Get metrics
metrics = monitor.get_metrics()
print(f"Avg response time: {metrics['avg_response_time']}s")
print(f"Cache hit rate: {metrics['cache_hit_rate']}")
print(f"Queries/hour: {metrics['queries_per_hour']}")
```

---

## Troubleshooting Guide

### Common Issues

#### 1. "Connection refused" to MongoDB/Qdrant/Ollama

**Symptoms**:
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solution**:
```bash
# Check if services are running
docker-compose ps

# Check service health
docker-compose logs mongodb
docker-compose logs qdrant
docker-compose logs ollama

# Restart services
docker-compose restart mongodb qdrant ollama

# Wait for services to initialize
sleep 30
```

---

#### 2. GPU not detected / CUDA errors

**Symptoms**:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution**:
```bash
# Verify NVIDIA driver
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Check Ollama GPU usage
docker exec -it ollama nvidia-smi

# Fallback to CPU
# In .env:
EMBEDDING_DEVICE=cpu
```

---

#### 3. Slow first request (30-60 seconds)

**Cause**: Model loading into GPU memory

**Solution**:
```bash
# Pre-load model
docker exec -it ollama ollama pull qwen2.5:7b-instruct

# Keep Ollama running
docker-compose restart ollama

# Subsequent requests should be <10s
```

---

#### 4. "I don't know" responses for valid queries

**Cause**: Context grounding threshold too strict

**Solution**:
```python
# In config/ai_settings.py:
RAG_SIMILARITY_THRESHOLD = 0.30  # Lower = more permissive (was 0.35)
MIN_SIMILARITY_FOR_ANSWER = 0.30  # Lower threshold

# Or in .env:
RAG_SIMILARITY_THRESHOLD=0.25
MIN_SIMILARITY_FOR_ANSWER=0.25
```

---

#### 5. Empty vector database (no search results)

**Symptoms**:
```
No documents found in Qdrant collection
```

**Solution**:
```bash
# Check Qdrant collection
curl http://localhost:6333/collections/desoutter_docs_v2

# Re-ingest documents
curl -X POST -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/admin/documents/ingest

# Check chunk count
curl http://localhost:6333/collections/desoutter_docs_v2
# Expected: ~4,082 chunks (language-filtered)
```

---

#### 6. JWT "Invalid token" errors

**Symptoms**:
```json
{"detail": "Invalid token"}
```

**Solution**:
```bash
# Check JWT_SECRET matches in .env
echo $JWT_SECRET

# Get fresh token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123"}' | jq -r '.access_token')

# Test token
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/auth/me
```

---

#### 7. Out of memory errors

**Symptoms**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution**:
```python
# Reduce batch size
# In config/ai_settings.py:
EMBEDDING_BATCH_SIZE = 32  # Reduce from 64

# Or reduce context
RAG_TOP_K = 5  # Reduce from 7

# Or use CPU
EMBEDDING_DEVICE = "cpu"
```

---

## Future Roadmap

### Q2 2026 - El-Harezmi Pipeline

**Status**: In Progress

**Goal**: Replace 14-stage Legacy RAG with 5-stage El-Harezmi pipeline

**Stages**:
1. **Intent Classification**: Adaptive intent detection
2. **Retrieval Strategy**: Dynamic retrieval based on intent
3. **Information Extraction**: Structured extraction from documents
4. **Knowledge Graph Validation**: Cross-reference with KG
5. **Response Formatting**: Intent-specific formatting

**Expected Improvements**:
- Latency: 23.6s → <15s (37% reduction)
- Accuracy: 85% → 90% (5% improvement)
- "I don't know" rate: 12% → 8% (33% reduction)

---

### Q3 2026 - Service Management System

**Components**:
- Device registry (serial numbers, warranty tracking)
- Service request management (repairs, calibrations)
- Customer management
- KPI dashboard (supervisor/manager analytics)

**API Endpoints** (Planned):
- `GET /api/devices`
- `POST /api/services`
- `PUT /api/services/{id}/status`
- `GET /api/kpi/overview`

---

### Q4 2026 - Advanced Features

- **Embedding Fine-tuning**: Domain-specific embeddings (15-20% accuracy improvement)
- **Vision AI**: Photo-based fault detection
- **Predictive Maintenance**: Failure prediction system
- **Mobile PWA**: Progressive Web App for field use

---

## Conclusion

This technical documentation provides a comprehensive overview of the Desoutter Assistant system. For questions or contributions, please refer to:

- [README.md](README.md) - Project overview
- [QUICKSTART.md](QUICKSTART.md) - Quick setup guide
- [ROADMAP.md](ROADMAP.md) - Development roadmap
- [GitHub Issues](https://github.com/fatihhbayram/desoutter-assistant/issues) - Bug reports and feature requests

---

**Last Updated**: March 23, 2026
**Version**: 1.8.0
**Maintained by**: Fatih Bayram ([@fatihhbayram](https://github.com/fatihhbayram))
