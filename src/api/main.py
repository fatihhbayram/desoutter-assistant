"""
=============================================================================
FastAPI Application for Desoutter Repair Assistant
=============================================================================
Main API server providing:
- Authentication: JWT-based login for admin and technician roles
- Product Management: List and retrieve product information from MongoDB
- AI Diagnosis: RAG-powered repair suggestions using Ollama LLM
- Document Management: Upload, delete, and ingest PDF documents for RAG
- User Management: Admin-only user CRUD operations

Endpoints Overview:
- GET /health          - Health check
- GET /products        - List all products
- POST /diagnose       - Get AI repair suggestion
- POST /auth/login     - User authentication
- GET /admin/users     - List users (admin only)
- GET /admin/documents - List uploaded documents (admin only)
- POST /admin/documents/upload  - Upload PDF document (admin only)
- POST /admin/documents/ingest  - Process documents into RAG (admin only)

API Documentation: /docs (Swagger UI) or /redoc
=============================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Authentication libraries
import jwt
from passlib.context import CryptContext

# Async support for blocking operations
import asyncio
from functools import partial

# Internal modules
from src.llm.rag_engine import RAGEngine  # RAG engine for AI responses
from src.database import MongoDBClient     # MongoDB client wrapper
from src.utils.logger import setup_logger  # Logging utility
from src.documents.document_processor import DocumentProcessor, SUPPORTED_EXTENSIONS  # Document text extraction
# from src.documents.chunker import TextChunker  # REMOVED: Now using SemanticChunker via DocumentProcessor
from src.documents.embeddings import EmbeddingsGenerator  # Embedding generation
from src.vectordb import ChromaDBClient  # Vector database client

# El-Harezmi v2 router
from src.api.el_harezmi_router import router as el_harezmi_router

# Initialize logger for this module
logger = setup_logger(__name__)

# -----------------------------------------------------------------------------
# FASTAPI APPLICATION INITIALIZATION
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Desoutter Repair Assistant API",
    description="AI-powered repair suggestions for Desoutter industrial tools",
    version="2.0.0"  # Updated for El-Harezmi integration
)

# Include El-Harezmi v2 router
app.include_router(el_harezmi_router)

# -----------------------------------------------------------------------------
# CORS MIDDLEWARE CONFIGURATION
# -----------------------------------------------------------------------------
# Enable Cross-Origin Resource Sharing for frontend access
# NOTE: In production, replace "*" with specific frontend domain(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict to frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers (including Authorization)
)

# -----------------------------------------------------------------------------
# GLOBAL STATE
# -----------------------------------------------------------------------------
# RAG engine singleton - initialized on first request or at startup
rag_engine = None

# -----------------------------------------------------------------------------
# AUTHENTICATION CONFIGURATION
# -----------------------------------------------------------------------------
# JWT-based authentication for API security

# Secret key for signing JWT tokens - CHANGE THIS IN PRODUCTION!
JWT_SECRET = os.environ.get("JWT_SECRET", "change-this-in-prod")

# JWT algorithm (HS256 = HMAC with SHA-256)
JWT_ALG = "HS256"

# Token expiration time (12 hours by default)
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 12

# Password hashing context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Optional secret for /auth/seed endpoint (first-run user creation)
SEED_SECRET = os.environ.get("SEED_SECRET")


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def get_rag_engine():
    """
    Get or create RAG engine singleton instance.
    
    The RAG engine is expensive to initialize (loads embedding model),
    so we use a singleton pattern to reuse it across requests.
    
    Returns:
        RAGEngine: The initialized RAG engine instance
    """
    global rag_engine
    if rag_engine is None:
        logger.info("Initializing RAG Engine...")
        rag_engine = RAGEngine()
    return rag_engine


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create a JWT access token for authenticated users.
    
    Args:
        data: Payload to encode (typically {"sub": username, "role": role})
        expires_delta: Optional custom expiration time
        
    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a bcrypt hash.
    
    Args:
        plain_password: User-provided password
        hashed_password: Stored bcrypt hash from database
        
    Returns:
        bool: True if password matches, False otherwise
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password to hash
        
    Returns:
        str: Bcrypt hash suitable for storage
    """
    return pwd_context.hash(password)


# -----------------------------------------------------------------------------
# PYDANTIC MODELS (Request/Response Schemas)
# -----------------------------------------------------------------------------
# These models define the structure of API requests and responses
# FastAPI uses them for validation and automatic documentation

class DiagnoseRequest(BaseModel):
    """Request body for AI diagnosis endpoint."""
    part_number: str              # Product part number (e.g., "6151659030")
    fault_description: str        # User's description of the problem
    language: Optional[str] = "en"  # Response language: 'en' or 'tr'
    # Retry parameters (for feedback-based retry)
    is_retry: Optional[bool] = False
    retry_of: Optional[str] = None  # Original diagnosis_id to retry
    excluded_sources: Optional[List[str]] = None  # Sources to exclude
    # Multi-turn conversation (Phase 3.5)
    session_id: Optional[str] = None  # Conversation session ID for follow-ups


class ConversationRequest(BaseModel):
    """Request for multi-turn conversation."""
    session_id: Optional[str] = None  # Existing session ID or None for new
    message: str                       # User message
    part_number: Optional[str] = None  # Product context
    language: Optional[str] = "en"


class SourceInfo(BaseModel):
    """Source document information with citation details."""
    source: str  # Filename
    page: Optional[int] = None  # Page number (if available)
    section: Optional[str] = None  # Section title
    similarity: str  # Similarity score
    excerpt: str  # Text excerpt


class DiagnoseResponse(BaseModel):
    """Response body for AI diagnosis endpoint."""
    suggestion: str       # AI-generated repair suggestion
    confidence: str       # Confidence level: 'high', 'medium', or 'low'
    product_model: str    # Product model name
    part_number: str      # Product part number
    sources: List[SourceInfo]  # Source documents with citations
    language: str         # Language of the response
    diagnosis_id: Optional[str] = None  # Unique ID for feedback
    response_time_ms: Optional[int] = None  # Response time in milliseconds
    # Priority 1: Context Grounding Metadata
    sufficiency_score: Optional[float] = None
    sufficiency_reason: Optional[str] = None
    sufficiency_factors: Optional[dict] = None
    sufficiency_recommendation: Optional[str] = None
    # Priority 2: Response Validation Metadata
    validation: Optional[dict] = None
    # Priority 3: Intent Detection Metadata
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None


class SourceRelevanceFeedback(BaseModel):
    """Feedback for a single source document."""
    source: str          # Document name/path
    relevant: bool       # True if relevant, False if not


class FeedbackRequest(BaseModel):
    """Request body for submitting feedback on a diagnosis."""
    diagnosis_id: str              # ID of the diagnosis to give feedback on
    feedback_type: str             # 'positive' or 'negative'
    negative_reason: Optional[str] = None  # Reason if negative
    user_comment: Optional[str] = None     # Additional comment
    correct_solution: Optional[str] = None # User's correct solution (if known)
    source_relevance: Optional[List[SourceRelevanceFeedback]] = None  # Per-source relevance feedback


class LoginRequest(BaseModel):
    """Request body for user login."""
    username: str  # User's username (case-insensitive)
    password: str  # User's password


class TokenResponse(BaseModel):
    """Response body for successful login."""
    access_token: str           # JWT access token
    token_type: str = "bearer"  # Token type (always "bearer")
    role: str                   # User's role: 'admin' or 'technician'


class SeedRequest(BaseModel):
    """Request body for seeding default users."""
    secret: str  # Secret key to authorize seeding


class CreateUserRequest(BaseModel):
    """Request body for creating a new user (admin only)."""
    username: str                    # New user's username
    password: str                    # New user's password
    role: str = "technician"         # Role: 'admin' or 'technician'


# -----------------------------------------------------------------------------
# AUTHENTICATION HELPERS
# -----------------------------------------------------------------------------

def verify_admin_token(authorization: str) -> dict:
    """Verify token and check admin role"""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Desoutter Repair Assistant API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_engine": "initialized" if rag_engine else "not initialized"
    }


@app.get("/products")
async def list_products():
    """List all products (Tools + CVI3 Units)"""
    try:
        # Get regular tools
        with MongoDBClient(collection_name="products") as db:
            tools = db.get_products(limit=0)
        
        # Get CVI3 units
        with MongoDBClient(collection_name="tool_units") as db:
            units = db.get_products(limit=0)
        
        # Add type field to distinguish
        for tool in tools:
            if '_id' in tool:
                tool['_id'] = str(tool['_id'])
            tool['product_type'] = 'tool'
        
        for unit in units:
            if '_id' in unit:
                unit['_id'] = str(unit['_id'])
            unit['product_type'] = 'cvi3_controller'
        
        # Merge lists
        all_products = tools + units
        
        return {
            "total": len(all_products),
            "tools_count": len(tools),
            "cvi3_units_count": len(units),
            "products": all_products
        }
    except Exception as e:
        logger.error(f"Error listing products: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/{part_number}")
async def get_product(part_number: str):
    """Get specific product details"""
    try:
        with MongoDBClient() as db:
            products = db.get_products({"part_number": part_number}, limit=1)
            
            if not products:
                raise HTTPException(status_code=404, detail="Product not found")
            
            product = products[0]
            if '_id' in product:
                product['_id'] = str(product['_id'])
            
            return product
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting product: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(
    request: DiagnoseRequest,
    authorization: str = Header(default=None)
):
    """
    Generate repair suggestion based on fault description.
    Now with self-learning capabilities from user feedback.
    
    Uses asyncio.to_thread() to run blocking LLM calls in a thread pool,
    preventing the event loop from blocking during long-running operations.
    """
    try:
        # Get username from token if available
        username = "anonymous"
        if authorization and authorization.startswith("Bearer "):
            try:
                token = authorization.split(" ")[1]
                payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
                username = payload.get("sub", "anonymous")
            except:
                pass
        
        rag = get_rag_engine()
        
        # Run blocking LLM call in thread pool to avoid blocking event loop
        # This allows other requests to be processed while waiting for LLM response
        result = await asyncio.to_thread(
            rag.generate_repair_suggestion,
            part_number=request.part_number,
            fault_description=request.fault_description,
            language=request.language,
            username=username,
            excluded_sources=request.excluded_sources,
            is_retry=request.is_retry or False,
            retry_of=request.retry_of
        )
        
        return result
    except Exception as e:
        logger.error(f"Error in diagnose: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diagnose/stream")
async def diagnose_stream(request: DiagnoseRequest):
    """
    Stream repair suggestion in real-time.
    
    Uses run_in_executor to run blocking generator in thread pool,
    preventing event loop blocking during streaming.
    """
    try:
        rag = get_rag_engine()
        
        async def generate():
            # Run sync generator in thread pool
            loop = asyncio.get_event_loop()
            
            # Create sync iterator
            def get_chunks():
                return list(rag.stream_repair_suggestion(
                    part_number=request.part_number,
                    fault_description=request.fault_description,
                    language=request.language
                ))
            
            # Get all chunks in thread pool to avoid blocking
            chunks = await loop.run_in_executor(None, get_chunks)
            
            for chunk in chunks:
                yield f"data: {json.dumps(chunk)}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Error in stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# FEEDBACK ENDPOINTS - Self-Learning RAG System
# =============================================================================

@app.post("/diagnose/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    authorization: str = Header(default=None)
):
    """
    Submit feedback for a diagnosis.
    Positive feedback reinforces the solution, negative feedback triggers learning.
    
    Uses asyncio.to_thread() for database operations to avoid blocking.
    """
    try:
        # Get username from token
        username = "anonymous"
        if authorization and authorization.startswith("Bearer "):
            try:
                token = authorization.split(" ")[1]
                payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
                username = payload.get("sub", "anonymous")
            except:
                pass
        
        from src.llm.feedback_engine import FeedbackLearningEngine
        from src.database.feedback_models import FeedbackType, NegativeFeedbackReason
        
        feedback_engine = FeedbackLearningEngine()
        
        # Convert string to enum
        feedback_type = FeedbackType(request.feedback_type)
        negative_reason = None
        if request.negative_reason:
            try:
                negative_reason = NegativeFeedbackReason(request.negative_reason)
            except:
                negative_reason = NegativeFeedbackReason.OTHER
        
        # Process source relevance feedback if provided
        source_relevance_data = None
        if request.source_relevance:
            source_relevance_data = [
                {"source": sr.source, "relevant": sr.relevant}
                for sr in request.source_relevance
            ]
        
        # Run feedback processing in thread pool to avoid blocking
        result = await asyncio.to_thread(
            feedback_engine.submit_feedback,
            diagnosis_id=request.diagnosis_id,
            feedback_type=feedback_type,
            username=username,
            negative_reason=negative_reason,
            user_comment=request.user_comment,
            correct_solution=request.correct_solution,
            source_relevance=source_relevance_data
        )
        
        return result
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/diagnose/history")
async def get_diagnosis_history(
    authorization: str = Header(...),
    limit: int = 20,
    skip: int = 0
):
    """
    Get diagnosis history for the current user.
    """
    try:
        # Get username from token
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization")
        
        token = authorization.split(" ")[1]
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        username = payload.get("sub")
        
        from src.llm.feedback_engine import FeedbackLearningEngine
        feedback_engine = FeedbackLearningEngine()
        
        history = feedback_engine.get_user_history(username, limit, skip)
        
        # Convert ObjectId to string
        for item in history:
            if '_id' in item:
                item['_id'] = str(item['_id'])
        
        return {"history": history, "count": len(history)}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/diagnose/feedback-stats")
async def get_feedback_stats(authorization: str = Header(...)):
    """
    Get feedback statistics (admin only).
    """
    try:
        # Verify admin
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization")
        
        token = authorization.split(" ")[1]
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        role = payload.get("role")
        
        if role != "admin":
            raise HTTPException(status_code=403, detail="Admin only")
        
        from src.llm.feedback_engine import FeedbackLearningEngine
        feedback_engine = FeedbackLearningEngine()
        
        stats = feedback_engine.get_feedback_stats()
        return stats
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        with MongoDBClient() as db:
            product_count = db.count_products()
        
        rag = get_rag_engine()
        doc_count = rag.vectordb.get_count()
        
        # Add feedback stats
        feedback_stats = {}
        try:
            from src.llm.feedback_engine import FeedbackLearningEngine
            feedback_engine = FeedbackLearningEngine()
            feedback_stats = feedback_engine.get_feedback_stats()
        except:
            pass
        
        return {
            "products_in_db": product_count,
            "documents_in_vectordb": doc_count,
            "rag_engine_status": "ready",
            "feedback_stats": feedback_stats
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DASHBOARD ENDPOINT
# =============================================================================

@app.get("/admin/dashboard")
async def get_dashboard(authorization: str = Header(...)):
    """
    Get comprehensive dashboard statistics for admin panel.
    
    Returns:
    - Overview: Total diagnoses, today/week/month counts, active users
    - Top Products: Most diagnosed products
    - Top Faults: Most common fault keywords
    - Daily Trend: Last 7 days diagnosis count
    - Confidence Breakdown: High/medium/low confidence distribution
    - Feedback Stats: Satisfaction rate, feedback breakdown
    """
    # Verify admin token
    verify_admin_token(authorization)
    
    try:
        from src.llm.feedback_engine import FeedbackLearningEngine
        feedback_engine = FeedbackLearningEngine()
        dashboard_data = feedback_engine.get_dashboard_stats()
        
        # Add system info
        with MongoDBClient() as db:
            product_count = db.count_products()
        
        rag = get_rag_engine()
        doc_count = rag.vectordb.get_count()
        
        dashboard_data["system"] = {
            "products_in_db": product_count,
            "documents_in_vectordb": doc_count,
            "rag_engine_status": "ready"
        }
        
        return dashboard_data
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Auth endpoints
@app.post("/auth/login", response_model=TokenResponse)
async def auth_login(req: LoginRequest):
    try:
        with MongoDBClient() as db:
            users = db.get_collection("users")
            username = (req.username or "").strip().lower()
            user = users.find_one({"username": username})
            if not user:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            password = (req.password or "").strip()
            if not verify_password(password, user.get("password_hash", "")):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            role = user.get("role", "technician")
            token = create_access_token({"sub": username, "role": role})
            return {"access_token": token, "role": role, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@app.get("/auth/me")
async def auth_me(authorization: Optional[str] = Header(None)):
    try:
        token = None
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization.split(" ", 1)[1]
        if not token:
            raise HTTPException(status_code=401, detail="Missing token")
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return {"username": payload.get("sub"), "role": payload.get("role")}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


# Simple HTML UI
@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """Simple web interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Desoutter Repair Assistant</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            .form-group { margin: 20px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, textarea, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
            textarea { min-height: 100px; }
            button { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
            button:hover { background: #0056b3; }
            .result { margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 4px; }
            .loading { color: #666; font-style: italic; }
        </style>
    </head>
    <body>
        <h1>🔧 Desoutter Repair Assistant</h1>
        <p>Enter product details and fault description to get AI-powered repair suggestions.</p>
        
        <div class="form-group">
            <label>Part Number:</label>
            <input type="text" id="partNumber" placeholder="e.g., 6151659030">
        </div>
        
        <div class="form-group">
            <label>Fault Description:</label>
            <textarea id="faultDescription" placeholder="Describe the problem..."></textarea>
        </div>
        
        <div class="form-group">
            <label>Language:</label>
            <select id="language">
                <option value="en">English</option>
                <option value="tr">Türkçe</option>
            </select>
        </div>
        
        <button onclick="getSuggestion()">Get Repair Suggestion</button>
        
        <div id="result" class="result" style="display:none;">
            <h3>Repair Suggestion:</h3>
            <div id="suggestionText"></div>
        </div>
        
        <script>
            async function getSuggestion() {
                const partNumber = document.getElementById('partNumber').value;
                const faultDescription = document.getElementById('faultDescription').value;
                const language = document.getElementById('language').value;
                
                if (!partNumber || !faultDescription) {
                    alert('Please fill in all fields');
                    return;
                }
                
                const resultDiv = document.getElementById('result');
                const suggestionText = document.getElementById('suggestionText');
                
                resultDiv.style.display = 'block';
                suggestionText.innerHTML = '<p class="loading">Generating suggestion...</p>';
                
                try {
                    const response = await fetch('/diagnose', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            part_number: partNumber,
                            fault_description: faultDescription,
                            language: language
                        })
                    });

                    // Handle non-OK HTTP responses
                    if (!response.ok) {
                        let errText = `Request failed (${response.status})`;
                        try {
                            const errData = await response.json();
                            errText = errData.detail || errData.error || JSON.stringify(errData);
                        } catch (e) {}
                        suggestionText.innerHTML = `<p style="color: red;">Error: ${errText}</p>`;
                        return;
                    }

                    const data = await response.json();

                    // Defensive defaults
                    const suggestion = (data && data.suggestion) ? data.suggestion : 'No suggestion available';
                    const productModel = (data && data.product_model) ? data.product_model : (partNumber || 'Unknown');
                    const confidence = (data && data.confidence) ? data.confidence : 'low';
                    const sources = (data && Array.isArray(data.sources)) ? data.sources : [];

                    let html = `
                        <p><strong>Product:</strong> ${productModel}</p>
                        <p><strong>Confidence:</strong> ${confidence}</p>
                        <hr>
                        <div style="white-space: pre-wrap;">${suggestion}</div>
                    `;

                    if (sources.length > 0) {
                        html += '<hr><p><strong>Sources:</strong></p><ul>';
                        sources.forEach(src => {
                            const s = src && src.source ? src.source : 'Unknown';
                            const sim = src && src.similarity ? src.similarity : '?';
                            html += `<li>${s} (similarity: ${sim})</li>`;
                        });
                        html += '</ul>';
                    }

                    suggestionText.innerHTML = html;
                } catch (error) {
                    suggestionText.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.on_event("startup")
async def startup_event():
    """Startup event - initialize RAG engine in background thread"""
    logger.info("🚀 Starting Desoutter Repair Assistant API")
    
    # Initialize RAG engine in thread pool to avoid blocking startup
    # This pre-loads embedding model and BM25 index
    await asyncio.to_thread(get_rag_engine)
    
    # Seed default users if missing
    try:
        with MongoDBClient() as db:
            users = db.get_collection("users")
            # normalize usernames to lowercase
            if not users.find_one({"username": "admin"}):
                users.insert_one({
                    "username": "admin",
                    "password_hash": get_password_hash("admin123"),
                    "role": "admin"
                })
            if not users.find_one({"username": "tech"}):
                users.insert_one({
                    "username": "tech",
                    "password_hash": get_password_hash("tech123"),
                    "role": "technician"
                })
    except Exception as e:
        logger.warning(f"User seed failed: {e}")
    logger.info("✅ API ready")


# Admin-only user seeding endpoint
@app.post("/auth/seed")
async def auth_seed(req: SeedRequest):
    try:
        # If SEED_SECRET is set, require it; if not set, allow seeding (first-run convenience)
        if SEED_SECRET and req.secret != SEED_SECRET:
            raise HTTPException(status_code=403, detail="Forbidden")
        with MongoDBClient() as db:
            users = db.get_collection("users")
            users.update_one({"username": "admin"}, {"$set": {
                "username": "admin",
                "password_hash": get_password_hash("admin123"),
                "role": "admin"
            }}, upsert=True)
            users.update_one({"username": "tech"}, {"$set": {
                "username": "tech",
                "password_hash": get_password_hash("tech123"),
                "role": "technician"
            }}, upsert=True)
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Seed error: {e}")
        raise HTTPException(status_code=500, detail="Seed failed")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    logger.info("👋 Shutting down API")


# Admin endpoints
@app.get("/admin/users")
async def admin_list_users(authorization: Optional[str] = Header(None)):
    """List all users (admin only)"""
    verify_admin_token(authorization or "")
    try:
        with MongoDBClient() as db:
            users_col = db.get_collection("users")
            users = list(users_col.find({}, {"_id": 0, "username": 1, "role": 1}))
            return {"users": users}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List users error: {e}")
        raise HTTPException(status_code=500, detail="Failed to list users")


@app.post("/admin/users")
async def admin_create_user(req: CreateUserRequest, authorization: Optional[str] = Header(None)):
    """Create a new user (admin only)"""
    verify_admin_token(authorization or "")
    username = (req.username or "").strip().lower()
    if not username or not req.password:
        raise HTTPException(status_code=400, detail="Username and password required")
    if req.role not in ["admin", "technician"]:
        raise HTTPException(status_code=400, detail="Role must be admin or technician")
    try:
        with MongoDBClient() as db:
            users_col = db.get_collection("users")
            if users_col.find_one({"username": username}):
                raise HTTPException(status_code=400, detail="User already exists")
            users_col.insert_one({
                "username": username,
                "password_hash": get_password_hash(req.password),
                "role": req.role
            })
            return {"status": "ok", "username": username}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create user error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user")


@app.delete("/admin/users/{username}")
async def admin_delete_user(username: str, authorization: Optional[str] = Header(None)):
    """Delete a user (admin only)"""
    verify_admin_token(authorization or "")
    username = username.strip().lower()
    try:
        with MongoDBClient() as db:
            users_col = db.get_collection("users")
            result = users_col.delete_one({"username": username})
            if result.deleted_count == 0:
                raise HTTPException(status_code=404, detail="User not found")
            return {"status": "ok", "deleted": username}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete user error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete user")


@app.post("/admin/scrape")
async def admin_trigger_scrape(authorization: Optional[str] = Header(None)):
    """Trigger scraping (admin only) - placeholder"""
    verify_admin_token(authorization or "")
    # This is a placeholder - actual scraping would be async/background
    logger.info("Scrape triggered by admin")
    return {"status": "ok", "message": "Scrape job queued (placeholder)"}


# =============================================================================
# DOCUMENT MANAGEMENT ENDPOINTS
# =============================================================================
# These endpoints allow admins to upload, list, delete, and ingest documents
# Supported formats: PDF, Word (DOCX), PowerPoint (PPTX)
# Documents are stored in the filesystem and processed into the RAG vector database

# Base directory for document uploads (mapped to host via Docker volume)
UPLOAD_DIR = Path("/app/documents")

# Subdirectories for different document types
MANUALS_DIR = UPLOAD_DIR / "manuals"      # Technical manuals and repair guides
BULLETINS_DIR = UPLOAD_DIR / "bulletins"  # Service bulletins and updates

# Auto-create directories if they don't exist
MANUALS_DIR.mkdir(parents=True, exist_ok=True)
BULLETINS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/admin/documents")
async def admin_list_documents(authorization: Optional[str] = Header(None)):
    """List all uploaded documents (admin only)"""
    verify_admin_token(authorization or "")
    try:
        documents = []
        
        # Supported extensions for document listing
        doc_extensions = ['*.pdf', '*.docx', '*.doc', '*.pptx', '*.ppt', '*.xlsx', '*.xls']
        
        # List manuals
        if MANUALS_DIR.exists():
            for ext in doc_extensions:
                for f in MANUALS_DIR.glob(ext):
                    stat = f.stat()
                    documents.append({
                        "filename": f.name,
                        "type": "manual",
                        "format": f.suffix.lower().replace('.', ''),
                        "size": stat.st_size,
                        "uploaded": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        # List bulletins
        if BULLETINS_DIR.exists():
            for ext in doc_extensions:
                for f in BULLETINS_DIR.glob(ext):
                    stat = f.stat()
                    documents.append({
                        "filename": f.name,
                        "type": "bulletin",
                        "format": f.suffix.lower().replace('.', ''),
                        "size": stat.st_size,
                        "uploaded": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        # Sort by upload date descending
        documents.sort(key=lambda x: x["uploaded"], reverse=True)
        
        return {"documents": documents, "total": len(documents)}
    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail="Failed to list documents")


@app.get("/documents/download/{filename}")
async def download_document(filename: str):
    """
    Download/view a document by filename.
    This endpoint is public so technicians can view source documents from diagnosis.
    """
    # Security: sanitize filename to prevent directory traversal
    safe_filename = Path(filename).name
    
    # Search in both directories
    for directory in [MANUALS_DIR, BULLETINS_DIR]:
        file_path = directory / safe_filename
        if file_path.exists() and file_path.is_file():
            # Determine media type
            ext = file_path.suffix.lower()
            media_types = {
                '.pdf': 'application/pdf',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.doc': 'application/msword',
                '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                '.ppt': 'application/vnd.ms-powerpoint',
                '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                '.xls': 'application/vnd.ms-excel'
            }
            media_type = media_types.get(ext, 'application/octet-stream')
            
            logger.info(f"Document download: {safe_filename}")
            return FileResponse(
                path=file_path,
                filename=safe_filename,
                media_type=media_type
            )
    
    raise HTTPException(status_code=404, detail="Document not found")


@app.post("/admin/documents/upload")
async def admin_upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form(...),
    authorization: Optional[str] = Header(None)
):
    """Upload a document for RAG (admin only). Supports PDF, Word (DOCX), PowerPoint (PPTX)"""
    verify_admin_token(authorization or "")
    
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    allowed_extensions = {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls'}
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: PDF, DOCX, PPTX, XLSX, XLS"
        )
    
    # Validate doc type
    if doc_type not in ["manual", "bulletin"]:
        raise HTTPException(status_code=400, detail="doc_type must be 'manual' or 'bulletin'")
    
    # Determine target directory
    target_dir = MANUALS_DIR if doc_type == "manual" else BULLETINS_DIR
    target_path = target_dir / file.filename
    
    try:
        # Save file
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Document uploaded: {file.filename} ({doc_type})")
        
        return {
            "status": "ok",
            "filename": file.filename,
            "type": doc_type,
            "message": "Document uploaded. Use /admin/documents/ingest to process into RAG."
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload: {str(e)}")


@app.delete("/admin/documents/{doc_type}/{filename}")
async def admin_delete_document(
    doc_type: str,
    filename: str,
    authorization: Optional[str] = Header(None)
):
    """Delete a document (admin only)"""
    verify_admin_token(authorization or "")
    
    if doc_type not in ["manual", "bulletin"]:
        raise HTTPException(status_code=400, detail="Invalid doc_type")
    
    target_dir = MANUALS_DIR if doc_type == "manual" else BULLETINS_DIR
    target_path = target_dir / filename
    
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Delete file
        target_path.unlink()
        
        # Also remove from vector DB
        try:
            chroma = ChromaDBClient()
            chroma.delete_by_source(filename)
            logger.info(f"Removed {filename} from vector DB")
        except Exception as e:
            logger.warning(f"Could not remove from vector DB: {e}")
        
        logger.info(f"Document deleted: {filename}")
        return {"status": "ok", "deleted": filename}
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")


@app.post("/admin/documents/ingest")
async def admin_ingest_documents(authorization: Optional[str] = Header(None)):
    """Process and ingest all documents into RAG vector database (admin only)"""
    verify_admin_token(authorization or "")
    
    try:
        logger.info("Starting document ingestion...")
        
        # Initialize processors
        doc_processor = DocumentProcessor()
        # Note: DocumentProcessor uses SemanticChunker internally (Phase 1-2 enhancement)
        embeddings_gen = EmbeddingsGenerator()
        chroma = ChromaDBClient()
        
        all_documents = []
        
        # Process manuals (PDF, DOCX, PPTX) with semantic chunking
        if MANUALS_DIR.exists():
            manuals = doc_processor.process_directory(
                MANUALS_DIR, 
                enable_semantic_chunking=True  # Use semantic chunker (Phase 1)
            )
            all_documents.extend(manuals)
            logger.info(f"Processed {len(manuals)} manuals")
        
        # Process bulletins (PDF, DOCX, PPTX) with semantic chunking
        if BULLETINS_DIR.exists():
            bulletins = doc_processor.process_directory(
                BULLETINS_DIR,
                enable_semantic_chunking=True  # Use semantic chunker (Phase 1)
            )
            all_documents.extend(bulletins)
            logger.info(f"Processed {len(bulletins)} bulletins")
        
        if not all_documents:
            return {"status": "ok", "message": "No documents to process", "chunks_added": 0}
        
        # Extract semantic chunks from processed documents
        # DocumentProcessor already generated chunks with rich metadata
        all_chunks = []
        for doc in all_documents:
            if doc.get('chunks'):
                all_chunks.extend(doc['chunks'])
        logger.info(f"Extracted {len(all_chunks)} semantic chunks")
        
        # Generate embeddings
        texts = [c["text"] for c in all_chunks]
        embeddings = embeddings_gen.generate_embeddings(texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Clear existing and add new
        chroma.clear_collection()
        added = chroma.add_documents(all_chunks, embeddings)
        
        logger.info(f"✅ Ingestion complete: {added} chunks added to vector DB")
        
        return {
            "status": "ok",
            "documents_processed": len(all_documents),
            "chunks_added": added,
            "message": f"Successfully ingested {len(all_documents)} documents ({added} chunks)"
        }
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# =============================================================================
# PHASE 2.3: RESPONSE CACHE MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/admin/cache/stats")
async def get_cache_stats(authorization: str = Header(...)):
    """
    Get response cache statistics.
    
    Returns:
    - size: Current number of cached entries
    - max_size: Maximum cache capacity
    - hit_rate: Cache hit rate percentage
    - hits/misses: Total hit and miss counts
    - evictions: Number of LRU evictions
    - ttl_expirations: Number of TTL-based expirations
    """
    verify_admin_token(authorization)
    
    try:
        rag = get_rag_engine()
        
        if not rag.response_cache:
            return {
                "status": "disabled",
                "message": "Response cache is not enabled"
            }
        
        stats = rag.response_cache.get_stats()
        return {
            "status": "enabled",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/cache/clear")
async def clear_cache(authorization: str = Header(...)):
    """
    Clear all cached responses.
    
    Useful after:
    - Document updates/re-ingestion
    - Model changes
    - Configuration updates
    """
    verify_admin_token(authorization)
    
    try:
        rag = get_rag_engine()
        
        if not rag.response_cache:
            return {
                "status": "disabled",
                "message": "Response cache is not enabled"
            }
        
        # Get stats before clearing
        stats_before = rag.response_cache.get_stats()
        entries_before = stats_before.get("size", 0)
        
        # Clear cache
        rag.response_cache.clear()
        
        logger.info(f"✅ Cache cleared: {entries_before} entries removed")
        
        return {
            "status": "ok",
            "message": f"Cache cleared successfully",
            "entries_removed": entries_before
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/cache/entry")
async def delete_cache_entry(
    part_number: str,
    fault_description: str,
    language: str = "en",
    authorization: str = Header(...)
):
    """
    Delete a specific cache entry.
    
    Args:
    - part_number: Product part number
    - fault_description: Fault description used in the query
    - language: Language code (en/tr)
    """
    verify_admin_token(authorization)
    
    try:
        rag = get_rag_engine()
        
        if not rag.response_cache:
            return {
                "status": "disabled",
                "message": "Response cache is not enabled"
            }
        
        # Build cache key (same format as in generate_repair_suggestion)
        cache_key = f"{part_number}:{fault_description}:{language}"
        
        # Try to delete
        deleted = rag.response_cache.delete(cache_key)
        
        if deleted:
            logger.info(f"✅ Cache entry deleted: {cache_key[:50]}...")
            return {
                "status": "ok",
                "message": "Cache entry deleted successfully"
            }
        else:
            return {
                "status": "not_found",
                "message": "Cache entry not found"
            }
    except Exception as e:
        logger.error(f"Error deleting cache entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PHASE 5: PERFORMANCE METRICS ENDPOINTS
# =============================================================================

@app.get("/admin/metrics/health")
async def get_system_health(authorization: str = Header(...)):
    """
    Get overall system health status and metrics.
    
    Returns:
    - status: healthy, warning, or degraded
    - issues: list of detected issues
    - last_hour: metrics for the last hour
    - last_24h: metrics for the last 24 hours
    """
    verify_admin_token(authorization)
    
    try:
        from src.llm.performance_metrics import get_performance_monitor
        monitor = get_performance_monitor()
        
        return monitor.get_health_status()
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/metrics/stats")
async def get_performance_stats(
    hours: int = 1,
    authorization: str = Header(...)
):
    """
    Get aggregated performance statistics for a time period.
    
    Args:
    - hours: Number of hours to include (default: 1)
    
    Returns:
    - Query counts (total, cache hits/misses)
    - Latency stats (avg, p95, p99)
    - Retrieval quality metrics
    - Confidence distribution
    - Feedback accuracy
    """
    verify_admin_token(authorization)
    
    try:
        from src.llm.performance_metrics import get_performance_monitor
        monitor = get_performance_monitor()
        
        stats = monitor.get_stats(hours=hours)
        return stats.to_dict()
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/metrics/queries")
async def get_recent_queries(
    limit: int = 20,
    authorization: str = Header(...)
):
    """
    Get recent query details for debugging.
    
    Args:
    - limit: Maximum number of queries to return (default: 20)
    """
    verify_admin_token(authorization)
    
    try:
        from src.llm.performance_metrics import get_performance_monitor
        monitor = get_performance_monitor()
        
        return {
            "queries": monitor.get_recent_queries(limit=limit),
            "count": limit
        }
    except Exception as e:
        logger.error(f"Error getting recent queries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/metrics/slow")
async def get_slow_queries(
    threshold_ms: float = 10000,
    limit: int = 10,
    authorization: str = Header(...)
):
    """
    Get slow queries exceeding a threshold.
    
    Args:
    - threshold_ms: Minimum response time to include (default: 10000ms = 10s)
    - limit: Maximum number of queries to return (default: 10)
    """
    verify_admin_token(authorization)
    
    try:
        from src.llm.performance_metrics import get_performance_monitor
        monitor = get_performance_monitor()
        
        return {
            "slow_queries": monitor.get_slow_queries(threshold_ms=threshold_ms, limit=limit),
            "threshold_ms": threshold_ms
        }
    except Exception as e:
        logger.error(f"Error getting slow queries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/metrics/reset")
async def reset_metrics(authorization: str = Header(...)):
    """
    Reset all performance metrics (for testing).
    """
    verify_admin_token(authorization)
    
    try:
        from src.llm.performance_metrics import get_performance_monitor
        monitor = get_performance_monitor()
        monitor.reset()
        
        return {
            "status": "ok",
            "message": "Performance metrics reset successfully"
        }
    except Exception as e:
        logger.error(f"Error resetting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PHASE 3.5: MULTI-TURN CONVERSATION ENDPOINTS
# =============================================================================

@app.post("/conversation/start")
async def start_conversation(
    request: ConversationRequest,
    authorization: str = Header(default=None)
):
    """
    Start a new conversation session or continue existing one.
    
    Returns:
    - session_id: Unique session identifier
    - response: AI response to the message
    """
    try:
        # Get username from token
        username = "anonymous"
        if authorization and authorization.startswith("Bearer "):
            try:
                token = authorization.split(" ")[1]
                payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
                username = payload.get("sub", "anonymous")
            except:
                pass
        
        from src.llm.conversation import get_conversation_manager
        manager = get_conversation_manager()
        
        # Get or create session
        session = manager.get_or_create_session(
            session_id=request.session_id,
            user_id=username,
            part_number=request.part_number,
            language=request.language
        )
        
        # Resolve references in query
        resolved_query = manager.resolve_references(request.message, session)
        
        # Add user message to session
        session.add_turn("user", request.message)
        
        # Generate response using RAG
        rag = get_rag_engine()
        
        # Build context-aware prompt
        if session.turns and len(session.turns) > 1:
            # Multi-turn: include conversation context
            context_prompt = manager.build_conversation_prompt(
                session=session,
                current_query=resolved_query
            )
            fault_description = context_prompt
        else:
            fault_description = resolved_query
        
        # Get diagnosis
        result = await asyncio.to_thread(
            rag.generate_repair_suggestion,
            part_number=request.part_number or "general",
            fault_description=fault_description,
            language=request.language,
            username=username
        )
        
        # Add assistant response to session
        session.add_turn("assistant", result.get("suggestion", ""), {
            "confidence": result.get("confidence"),
            "sources": [s.get("source") for s in result.get("sources", [])]
        })
        
        return {
            "session_id": session.session_id,
            "response": result.get("suggestion"),
            "confidence": result.get("confidence"),
            "sources": result.get("sources", []),
            "turn_count": len(session.turns),
            "context_preserved": len(session.turns) > 2
        }
    except Exception as e:
        logger.error(f"Error in conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{session_id}")
async def get_conversation(
    session_id: str,
    authorization: str = Header(default=None)
):
    """
    Get conversation history for a session.
    """
    try:
        from src.llm.conversation import get_conversation_manager
        manager = get_conversation_manager()
        
        session = manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        return {
            "session_id": session.session_id,
            "product_context": session.product_context,
            "part_number": session.part_number,
            "turn_count": len(session.turns),
            "history": session.get_history(max_turns=20),
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversation/{session_id}")
async def end_conversation(
    session_id: str,
    authorization: str = Header(default=None)
):
    """
    End and delete a conversation session.
    """
    try:
        from src.llm.conversation import get_conversation_manager
        manager = get_conversation_manager()
        
        if manager.delete_session(session_id):
            return {"status": "ok", "message": "Conversation ended"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/conversations/stats")
async def get_conversation_stats(authorization: str = Header(...)):
    """
    Get conversation manager statistics (admin only).
    """
    verify_admin_token(authorization)
    
    try:
        from src.llm.conversation import get_conversation_manager
        manager = get_conversation_manager()
        
        return manager.get_stats()
    except Exception as e:
        logger.error(f"Error getting conversation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PHASE 6: SELF-LEARNING FEEDBACK LOOP API
# =============================================================================

@app.get("/admin/learning/stats")
async def get_learning_stats(authorization: str = Header(...)):
    """
    Get self-learning system statistics (admin only).
    
    Returns:
        - Source learning scores (count, confidence)
        - Keyword mappings (count, success rate)
        - Activity metrics (events per day)
        - Embedding training readiness
    """
    verify_admin_token(authorization)
    
    try:
        from src.llm.self_learning import get_self_learning_engine
        engine = get_self_learning_engine()
        
        return engine.get_learning_stats()
    except Exception as e:
        logger.error(f"Error getting learning stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/learning/top-sources")
async def get_top_learned_sources(
    limit: int = 10,
    authorization: str = Header(...)
):
    """
    Get top-performing document sources based on learned feedback (admin only).
    
    Args:
        limit: Maximum number of sources to return (default: 10)
        
    Returns:
        List of sources with scores and confidence levels
    """
    verify_admin_token(authorization)
    
    try:
        from src.llm.self_learning import get_self_learning_engine
        engine = get_self_learning_engine()
        
        return {
            "top_sources": engine.get_top_learned_sources(limit=limit),
            "total_sources": engine.get_learning_stats()["source_learning"]["total_sources"]
        }
    except Exception as e:
        logger.error(f"Error getting top sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class LearningQueryRequest(BaseModel):
    """Request model for learning recommendations query"""
    keywords: List[str]


@app.post("/admin/learning/recommendations")
async def get_learning_recommendations(
    request: LearningQueryRequest,
    authorization: str = Header(...)
):
    """
    Get source recommendations based on learned patterns (admin only).
    
    Args:
        keywords: List of fault keywords to query
        
    Returns:
        - boost_sources: Sources to prioritize
        - avoid_sources: Sources to demote
        - confidence: Learning confidence score
    """
    verify_admin_token(authorization)
    
    try:
        from src.llm.self_learning import get_self_learning_engine
        engine = get_self_learning_engine()
        
        return engine.get_recommendations_for_query(request.keywords)
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/learning/training-status")
async def get_training_status(authorization: str = Header(...)):
    """
    Get embedding retraining status and statistics (admin only).
    
    Returns:
        - Training data availability
        - Sample counts
        - Retraining readiness
    """
    verify_admin_token(authorization)
    
    try:
        from src.llm.self_learning import get_self_learning_engine
        engine = get_self_learning_engine()
        
        stats = engine.embedding_retrainer.get_training_data_stats()
        history = engine.embedding_retrainer.get_retraining_history(limit=5)
        
        return {
            "training_data": stats,
            "recent_jobs": history
        }
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/learning/schedule-retraining")
async def schedule_embedding_retraining(authorization: str = Header(...)):
    """
    Schedule an embedding retraining job (admin only).
    
    Note: Actual training runs as a separate process.
    This endpoint marks data and creates a job record.
    
    Returns:
        - Job status and ID
        - Sample count
    """
    verify_admin_token(authorization)
    
    try:
        from src.llm.self_learning import get_self_learning_engine
        engine = get_self_learning_engine()
        
        result = engine.embedding_retrainer.schedule_retraining()
        return result
    except Exception as e:
        logger.error(f"Error scheduling retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ResetLearningRequest(BaseModel):
    """Request model for learning reset"""
    confirm: bool = False


@app.post("/admin/learning/reset")
async def reset_learning_data(
    request: ResetLearningRequest,
    authorization: str = Header(...)
):
    """
    Reset all learned data (admin only).
    
    WARNING: This deletes all accumulated learning data!
    Requires confirm=true in request body.
    
    Returns:
        - Reset status
        - Deleted record counts
    """
    verify_admin_token(authorization)
    
    try:
        from src.llm.self_learning import get_self_learning_engine
        engine = get_self_learning_engine()
        
        result = engine.reset_learning(confirm=request.confirm)
        
        if result.get("status") == "reset_complete":
            logger.warning(f"Learning data reset by admin: {result['deleted']}")
        
        return result
    except Exception as e:
        logger.error(f"Error resetting learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PHASE 3.1: DOMAIN EMBEDDINGS API
# =============================================================================

@app.get("/admin/domain/stats")
async def get_domain_stats(authorization: str = Header(...)):
    """
    Get domain embeddings statistics (admin only).
    
    Returns:
        - Vocabulary size
        - Term weights count
        - Product series, error codes, etc.
        - Contrastive learning stats
    """
    verify_admin_token(authorization)
    
    try:
        from src.llm.domain_embeddings import get_domain_embeddings_engine
        engine = get_domain_embeddings_engine()
        
        return engine.get_stats()
    except Exception as e:
        logger.error(f"Error getting domain stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/domain/vocabulary")
async def get_domain_vocabulary(authorization: str = Header(...)):
    """
    Get domain vocabulary information (admin only).
    
    Returns:
        - Tool types
        - Product series
        - Error codes
        - Components
        - Symptoms
        - Procedures
    """
    verify_admin_token(authorization)
    
    try:
        from src.llm.domain_embeddings import get_domain_embeddings_engine
        engine = get_domain_embeddings_engine()
        
        return engine.get_vocabulary_info()
    except Exception as e:
        logger.error(f"Error getting vocabulary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class QueryEnhancementRequest(BaseModel):
    """Request model for query enhancement"""
    query: str


@app.post("/admin/domain/enhance-query")
async def enhance_query(
    request: QueryEnhancementRequest,
    authorization: str = Header(...)
):
    """
    Enhance a query with domain knowledge (admin only).
    
    Useful for testing domain enhancement.
    
    Args:
        query: Original query text
        
    Returns:
        - Original query
        - Enhanced query
        - Extracted entities
        - Context keywords
        - Term weights
    """
    verify_admin_token(authorization)
    
    try:
        from src.llm.domain_embeddings import get_domain_embeddings_engine
        engine = get_domain_embeddings_engine()
        
        return engine.enhance_query(request.query)
    except Exception as e:
        logger.error(f"Error enhancing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/domain/error-codes")
async def get_error_codes(authorization: str = Header(...)):
    """
    Get all Desoutter error codes with descriptions (admin only).
    
    Returns:
        Dict of error codes and their meanings
    """
    verify_admin_token(authorization)
    
    try:
        from src.llm.domain_embeddings import DomainVocabulary
        
        return {
            "error_codes": DomainVocabulary.ERROR_CODES,
            "total": len(DomainVocabulary.ERROR_CODES)
        }
    except Exception as e:
        logger.error(f"Error getting error codes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/domain/product-series")
async def get_product_series(authorization: str = Header(...)):
    """
    Get all Desoutter product series with descriptions (admin only).
    
    Returns:
        Dict of product series codes and their meanings
    """
    verify_admin_token(authorization)
    
    try:
        from src.llm.domain_embeddings import DomainVocabulary
        
        return {
            "product_series": DomainVocabulary.PRODUCT_SERIES,
            "total": len(DomainVocabulary.PRODUCT_SERIES)
        }
    except Exception as e:
        logger.error(f"Error getting product series: {e}")
        raise HTTPException(status_code=500, detail=str(e))
