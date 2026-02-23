"""
=============================================================================
Phase 2 Configuration - AI & RAG Components
=============================================================================
This module contains all AI-related configuration settings for the
Desoutter Repair Assistant, including:
- Document storage paths (manuals, bulletins)
- Vector database settings (Qdrant)
- Embedding model configuration (HuggingFace sentence-transformers)
- LLM settings (Ollama with qwen2.5:7b-instruct)
- RAG (Retrieval Augmented Generation) parameters
- API server configuration

Configuration Priority:
1. Environment variables (highest priority)
2. .env file
3. Default values in this file (lowest priority)

Usage:
    from config.ai_settings import OLLAMA_MODEL, RAG_TOP_K
=============================================================================
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Import base settings (DATA_DIR, DATABASE settings, etc.)
from .settings import *

# =============================================================================
# DOCUMENT STORAGE CONFIGURATION
# =============================================================================
# These directories store PDF documents used for RAG knowledge base
# - manuals/: Product technical manuals and repair guides
# - bulletins/: Service bulletins and technical updates

# Documents are mounted at /app/documents in Docker
DOCUMENTS_DIR = BASE_DIR / "documents"
MANUALS_DIR = DOCUMENTS_DIR / "manuals"
BULLETINS_DIR = DOCUMENTS_DIR / "bulletins"

# Auto-create document directories on module import
for directory in [DOCUMENTS_DIR, MANUALS_DIR, BULLETINS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# VECTOR DATABASE CONFIGURATION (Qdrant)
# =============================================================================
# Qdrant stores document embeddings for semantic search (migrated from ChromaDB)
# Qdrant runs as a Docker service — see ai-stack.yml

VECTORDB_DIR = DATA_DIR / "vectordb"
VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

# Qdrant connection settings (set via environment in ai-stack.yml)
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "desoutter_docs_v2")

# =============================================================================
# EMBEDDING MODEL CONFIGURATION (HuggingFace)
# =============================================================================
# Embeddings convert text into numerical vectors for similarity search
# Models are downloaded from HuggingFace and cached locally

# Default model: all-MiniLM-L6-v2 (fast, good quality, 384 dimensions)
# Alternatives:
#   - "sentence-transformers/all-mpnet-base-v2": Better quality, slower, 768 dim
#   - "intfloat/multilingual-e5-small": Multilingual support (Turkish/English)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Device for embedding computation: 'cpu' (default) or 'cuda' (GPU acceleration)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# Embedding output dimension (do not change - must match model)
EMBEDDING_DIMENSION = 384

# Pooling method for token-level embeddings
EMBEDDING_POOLING = os.getenv("EMBEDDING_POOLING", "mean")  # 'mean', 'cls', 'max'

# Batch size for processing multiple texts at once (higher = faster, more memory)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# === Phase 2: Domain Embeddings (Future Enhancement) ===
# Path to fine-tuned domain-specific embedding model
# When available, this overrides EMBEDDING_MODEL for better domain accuracy
DOMAIN_EMBEDDING_MODEL_PATH = os.getenv(
    "DOMAIN_EMBEDDING_MODEL_PATH",
    None  # Will be set after fine-tuning on Phase 2
)

# Enable domain embeddings when fine-tuned model is available
USE_DOMAIN_EMBEDDINGS = os.getenv("USE_DOMAIN_EMBEDDINGS", "false").lower() == "true"

# Domain embeddings fine-tuning parameters (reference only)
# These are used by the domain_embedding_trainer.py script:
# - Training data: Positive feedback pairs (fault -> solution)
# - Training method: Siamese network with contrastive loss
# - Epochs: 3-5 with early stopping
# - Batch size: 8 (depends on VRAM)
# - Learning rate: 2e-5
# - Warmup steps: 100
DOMAIN_EMBEDDING_TRAINING_ENABLED = os.getenv("DOMAIN_EMBEDDING_TRAINING_ENABLED", "false").lower() == "true"

# =============================================================================
# TEXT CHUNKING CONFIGURATION
# =============================================================================
# Documents are split into smaller chunks for better retrieval accuracy
# Smaller chunks = more precise retrieval, but may lose context
# Larger chunks = more context, but may include irrelevant information

# Maximum tokens per chunk (recommended: 300-500 for technical documents)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))

# Overlap between consecutive chunks (prevents cutting off mid-sentence)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# =============================================================================
# LLM CONFIGURATION (Ollama)
# =============================================================================
# Ollama provides local LLM inference with GPU acceleration
# Models are stored in the ollama Docker volume

# Ollama API endpoint - uses Docker network hostname
# For local development: http://localhost:11434
# For Docker deployment: http://ollama:11434
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# LLM model to use for generating repair suggestions
# Available models on this Proxmox system:
#   - "llama3:latest": Meta Llama 3 8B (general purpose)
#   - "qwen2.5:7b-instruct": Qwen 2.5 7B (better for technical/structured content)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# Temperature controls randomness in responses
# 0.0 = deterministic, always same answer
# 0.1 = very focused (recommended for technical content)
# 0.7+ = more creative/varied responses
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

# Request timeout in seconds (increase for slower hardware)
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# Maximum tokens in LLM response (num_predict parameter)
# Higher = longer responses, but may exceed context window
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "512"))

# =============================================================================
# RAG (Retrieval Augmented Generation) SETTINGS
# =============================================================================
# RAG combines retrieved context with LLM generation for accurate answers
# The system retrieves relevant document chunks and provides them to the LLM

# Number of most similar chunks to retrieve for context
# Higher = more context but may include less relevant information
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# Minimum similarity score (0.0-1.0) to consider a chunk relevant
# Higher = stricter matching, may miss relevant info
# Lower = more results, may include noise
# Set to 0.30: Filters off absolute worst matches while keeping top results
# Best combined with user feedback learning for quality improvement
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.30"))

# =============================================================================
# HYBRID SEARCH CONFIGURATION (Phase 2.2)
# =============================================================================
# Hybrid search combines semantic (dense) and BM25 (sparse) retrieval
# for improved coverage of both semantic meaning and exact keyword matches

# Enable hybrid search (combines semantic + BM25)
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"

# Weight for semantic search in fusion (0.0-1.0)
# Reduced from 0.7 to 0.6 to give more weight to exact keyword matches
HYBRID_SEMANTIC_WEIGHT = float(os.getenv("HYBRID_SEMANTIC_WEIGHT", "0.6"))

# Weight for BM25 keyword search in fusion (0.0-1.0)
# Increased from 0.3 to 0.4 for better error code and technical term matching
HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.4"))

# RRF (Reciprocal Rank Fusion) constant - higher = more uniform weighting
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))

# Enable query expansion with domain synonyms
ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"

# Maximum query expansions to generate
MAX_QUERY_EXPANSIONS = int(os.getenv("MAX_QUERY_EXPANSIONS", "3"))


def get_dynamic_weights(query: str) -> tuple:
    """
    Adjust search weights based on query characteristics.
    
    - Error codes → Higher BM25 (exact match important)
    - Troubleshooting → Balanced with slight BM25 emphasis
    - General questions → Higher Semantic (meaning important)
    
    Returns:
        (bm25_weight, semantic_weight) tuple
    """
    import re
    
    query_upper = query.upper()
    
    # Error code present → BM25 dominant
    if re.search(r'\b[EI]\d{2,4}\b', query_upper) or 'TRD-' in query_upper:
        return (0.6, 0.4)
    
    # Troubleshooting keywords → Balanced with BM25 emphasis
    troubleshooting_keywords = [
        'error', 'issue', 'problem', 'fault', 'failure', 'not working',
        'failed', 'broken', 'stuck', 'wont', "won't", 'unable'
    ]
    if any(kw in query.lower() for kw in troubleshooting_keywords):
        return (0.45, 0.55)
    
    # Default → Semantic emphasis
    return (HYBRID_BM25_WEIGHT, HYBRID_SEMANTIC_WEIGHT)

# =============================================================================
# METADATA-BASED FILTERING AND BOOSTING (Phase 4.1)
# =============================================================================
# Boost certain document types and metadata fields for better relevance

# Enable metadata-based score boosting
ENABLE_METADATA_BOOST = os.getenv("ENABLE_METADATA_BOOST", "true").lower() == "true"

# Boost factor for service bulletins (ESD/ESB documents)
# Applied: score *= SERVICE_BULLETIN_BOOST
# Increased from 2.5 to 4.0 to ensure bulletins rank above generic manual content
# especially for symptom-based queries where semantic similarity may be lower
SERVICE_BULLETIN_BOOST = float(os.getenv("SERVICE_BULLETIN_BOOST", "4.0"))

# Boost factor for procedure sections (step-by-step instructions)
PROCEDURE_BOOST = float(os.getenv("PROCEDURE_BOOST", "1.3"))

# Boost factor for warning/caution sections
WARNING_BOOST = float(os.getenv("WARNING_BOOST", "1.2"))

# Boost factor based on importance_score metadata (0-1)
# Applied: score *= (1 + importance_score * IMPORTANCE_BOOST_FACTOR)
IMPORTANCE_BOOST_FACTOR = float(os.getenv("IMPORTANCE_BOOST_FACTOR", "0.3"))

# =============================================================================
# API SERVER CONFIGURATION
# =============================================================================
# FastAPI server settings for the repair assistant API

# Bind address: 0.0.0.0 listens on all network interfaces
API_HOST = os.getenv("API_HOST", "0.0.0.0")

# Port for the API server (default: 8000)
API_PORT = int(os.getenv("API_PORT", "8000"))

# Auto-reload on code changes (disable in production)
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# =============================================================================
# LANGUAGE SETTINGS
# =============================================================================
# The assistant supports multiple languages for responses

# Default response language: 'en' (English) or 'tr' (Turkish)
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")

# System prompts define the AI assistant's role and behavior
# English system prompt - used when language='en'
SYSTEM_PROMPT_EN = """You are an expert technician assistant for Desoutter industrial tools. 
Your role is to provide accurate, safe, and practical repair suggestions based on technical manuals and bulletins.
Always prioritize safety and follow manufacturer guidelines."""

# Turkish system prompt - used when language='tr'
SYSTEM_PROMPT_TR = """Desoutter endüstriyel aletleri için uzman teknisyen asistanısınız. 
Göreviniz teknik kılavuzlar ve bültenlere dayanarak doğru, güvenli ve pratik onarım önerileri sunmaktır.
Her zaman güvenliği önceliklendirin ve üretici talimatlarını takip edin."""

# =============================================================================
# DOCUMENT PROCESSING SETTINGS
# =============================================================================
# Configuration for PDF document parsing and extraction

# Extract images from PDFs (disabled by default - increases processing time)
PDF_EXTRACT_IMAGES = os.getenv("PDF_EXTRACT_IMAGES", "false").lower() == "true"

# Extract tables from PDFs (enabled - useful for specifications)
PDF_EXTRACT_TABLES = os.getenv("PDF_EXTRACT_TABLES", "true").lower() == "true"

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================
# Caching and optimization options

# Enable response caching (recommended for production)
USE_CACHE = os.getenv("USE_CACHE", "true").lower() == "true"

# Cache time-to-live in seconds (1 hour default)
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

# =============================================================================
# RESPONSE GROUNDING CONFIGURATION (Priority 1)
# =============================================================================
# Context sufficiency scoring to prevent hallucinations
# System will refuse to answer when retrieved context is inadequate

# Enable context grounding (recommended for production)
ENABLE_CONTEXT_GROUNDING = os.getenv("ENABLE_CONTEXT_GROUNDING", "true").lower() == "true"

# Overall sufficiency threshold (0.0-1.0)
# 0.5 = balanced (recommended), 0.3 = permissive, 0.7 = strict
CONTEXT_SUFFICIENCY_THRESHOLD = float(os.getenv("CONTEXT_SUFFICIENCY_THRESHOLD", "0.5"))

# Minimum similarity for top document to consider answering
# Top retrieved document must exceed this threshold
MIN_SIMILARITY_FOR_ANSWER = float(os.getenv("MIN_SIMILARITY_FOR_ANSWER", "0.35"))

# Minimum number of relevant documents for confident answer
# Having multiple docs increases confidence
MIN_DOCS_FOR_CONFIDENCE = int(os.getenv("MIN_DOCS_FOR_CONFIDENCE", "2"))

# =============================================================================
# RESPONSE VALIDATION CONFIGURATION (Priority 2)
# =============================================================================
# Post-processing validation to detect hallucinations and quality issues

# Enable response validation (recommended for production)
ENABLE_RESPONSE_VALIDATION = os.getenv("ENABLE_RESPONSE_VALIDATION", "true").lower() == "true"

# Flag responses with uncertainty phrases ("might", "probably", etc.)
FLAG_UNCERTAINTY_PHRASES = os.getenv("FLAG_UNCERTAINTY_PHRASES", "true").lower() == "true"

# Verify numerical values exist in context
VERIFY_NUMERICAL_VALUES = os.getenv("VERIFY_NUMERICAL_VALUES", "true").lower() == "true"

# Minimum response length in characters
MIN_RESPONSE_LENGTH = int(os.getenv("MIN_RESPONSE_LENGTH", "30"))

# Maximum uncertain phrases before flagging (2 = balanced)
MAX_UNCERTAINTY_COUNT = int(os.getenv("MAX_UNCERTAINTY_COUNT", "2"))

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Control what gets logged for debugging and analytics

# Log user queries (for debugging and improving the system)
LOG_QUERIES = os.getenv("LOG_QUERIES", "true").lower() == "true"

# Log LLM responses (useful for quality analysis)
LOG_RESPONSES = os.getenv("LOG_RESPONSES", "true").lower() == "true"

# =============================================================================
# DEDUPLICATION CONFIGURATION (Priority 4)
# =============================================================================
# Prevent duplicate content (identical chunks) from bloating the vector DB
# Uses SHA256 hash of normalized content

# Enable content deduplication during ingestion
ENABLE_DEDUPLICATION = os.getenv("ENABLE_DEDUPLICATION", "true").lower() == "true"

# Log duplicate ratio to database (for analytics)
LOG_DUPLICATE_RATIO = os.getenv("LOG_DUPLICATE_RATIO", "true").lower() == "true"

# =============================================================================
# PERFORMANCE OPTIMIZATION FLAGS (NEW)
# =============================================================================
# Disable slow features for faster response times
# Use these to test baseline performance and identify bottlenecks

# Domain embeddings: 351 Desoutter terms for query expansion
# Impact: ~50-100ms per query, may or may not improve retrieval
# Set to False to disable and test baseline performance
ENABLE_DOMAIN_EMBEDDINGS = os.getenv("ENABLE_DOMAIN_EMBEDDINGS", "false").lower() == "true"

# Fault filtering: 15 category rules to exclude irrelevant docs
# Impact: ~30ms per query, may over-filter in some cases
# Set to False to disable filtering
ENABLE_FAULT_FILTERING = os.getenv("ENABLE_FAULT_FILTERING", "false").lower() == "true"

# Adaptive chunking: Varies chunk size based on document type
# Impact: Affects ingestion, not query time directly
# Set to False to use fixed chunk size (FIXED_CHUNK_SIZE)
ENABLE_ADAPTIVE_CHUNKING = os.getenv("ENABLE_ADAPTIVE_CHUNKING", "false").lower() == "true"

# Fixed chunk size when adaptive chunking is disabled (characters)
FIXED_CHUNK_SIZE = int(os.getenv("FIXED_CHUNK_SIZE", "600"))

# =============================================================================
# TIMEOUT CONFIGURATION (NEW)
# =============================================================================
# Prevent slow operations from blocking responses

# LLM generation timeout in seconds
# If LLM takes longer, return error instead of waiting
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "10"))
