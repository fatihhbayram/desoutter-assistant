"""
=============================================================================
Phase 2 Configuration - AI & RAG Components
=============================================================================
This module contains all AI-related configuration settings for the
Desoutter Repair Assistant, including:
- Document storage paths (manuals, bulletins)
- Vector database settings (ChromaDB)
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

DOCUMENTS_DIR = DATA_DIR / "documents"
MANUALS_DIR = DOCUMENTS_DIR / "manuals"
BULLETINS_DIR = DOCUMENTS_DIR / "bulletins"

# Auto-create document directories on module import
for directory in [DOCUMENTS_DIR, MANUALS_DIR, BULLETINS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# VECTOR DATABASE CONFIGURATION (ChromaDB)
# =============================================================================
# ChromaDB stores document embeddings for semantic search
# The vector database enables fast similarity search for RAG retrieval

VECTORDB_DIR = DATA_DIR / "vectordb"
VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

# Database type: 'chroma' (default) or 'qdrant' (alternative)
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chroma")

# ChromaDB persistence directory - stores the vector index on disk
CHROMA_PERSIST_DIR = str(VECTORDB_DIR / "chroma")

# Collection name within ChromaDB - logical grouping of documents
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "desoutter_docs")

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

# Batch size for processing multiple texts at once (higher = faster, more memory)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

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
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.7"))

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
# LOGGING CONFIGURATION
# =============================================================================
# Control what gets logged for debugging and analytics

# Log user queries (for debugging and improving the system)
LOG_QUERIES = os.getenv("LOG_QUERIES", "true").lower() == "true"

# Log LLM responses (useful for quality analysis)
LOG_RESPONSES = os.getenv("LOG_RESPONSES", "true").lower() == "true"
