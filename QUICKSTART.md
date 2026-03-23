# Quick Start Guide

Get Desoutter Assistant running in under 10 minutes.

---

## Prerequisites

Before you begin, ensure you have the following installed:

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Docker | 20.10+ | `docker --version` |
| Docker Compose | 2.0+ | `docker compose version` |
| Git | 2.30+ | `git --version` |
| NVIDIA Driver | 525+ (optional) | `nvidia-smi` |
| NVIDIA Container Toolkit | Latest (optional) | `nvidia-ctk --version` |

> **Note**: GPU is optional but recommended for faster LLM inference.

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/fatihhbayram/desoutter-assistant.git
cd desoutter-assistant
```

### Step 2: Environment Setup

Create your environment configuration:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# MongoDB Configuration
MONGO_HOST=mongodb
MONGO_PORT=27017
MONGO_DATABASE=desoutter

# Ollama LLM Configuration
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_TEMPERATURE=0.1

# Qdrant Vector Database
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=desoutter_docs_v2

# Embeddings Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cuda  # Use 'cpu' if no GPU

# RAG Configuration
RAG_TOP_K=7
RAG_SIMILARITY_THRESHOLD=0.30
USE_HYBRID_SEARCH=true
HYBRID_SEMANTIC_WEIGHT=0.6
HYBRID_BM25_WEIGHT=0.4

# JWT Secret (change in production!)
JWT_SECRET=your-secure-secret-key-here

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### Step 3: Launch Services

```bash
# Start all services
docker-compose up -d

# Verify all containers are running
docker-compose ps

# Check service health
docker-compose logs -f desoutter-api
```

### Step 4: Verify Installation

Run these health checks to ensure everything is working:

```bash
# Check API health
curl http://localhost:8000/health
# Expected: {"status":"healthy","timestamp":"..."}

# Check Ollama connection
curl http://localhost:11434/api/tags
# Expected: {"models":[{"name":"qwen2.5:7b-instruct",...}]}

# Check if documents are indexed (requires auth)
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123"}' | jq -r '.access_token')

curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/admin/dashboard
# Expected: Dashboard statistics with product count, document count, etc.
```

---

## First Query

### Login

```bash
# Get authentication token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123"}' | jq -r '.access_token')

echo "Token: $TOKEN"
```

### Make a Diagnosis Request

```bash
curl -X POST http://localhost:8000/diagnose \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "part_number": "6151659770",
    "fault_description": "motor not starting",
    "language": "en"
  }'
```

**Expected Response:**

```json
{
  "diagnosis_id": "diag_abc123",
  "suggestion": "Based on the documentation, here are the steps to troubleshoot...",
  "confidence": 0.89,
  "sources": [
    {
      "document": "EPB_Troubleshooting_Guide.pdf",
      "page": 12,
      "similarity": 0.85
    }
  ],
  "intent": "troubleshooting"
}
```

### Try Multi-turn Conversation

```bash
# Start a conversation
curl -X POST http://localhost:8000/conversation/start \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "message": "My EPB tool is not starting",
    "part_number": "6151659030"
  }'

# Continue with follow-up (use session_id from response)
curl -X POST http://localhost:8000/conversation/start \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "session_id": "sess_xyz789",
    "message": "What about the battery?"
  }'
```

---

## Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:3001 | Main user interface |
| **API Docs** | http://localhost:8000/docs | Swagger documentation |
| **API Health** | http://localhost:8000/health | Health check endpoint |

---

## Default Users

| Username | Password | Role | Permissions |
|----------|----------|------|-------------|
| `admin` | `admin123` | Admin | Full access, user management, document upload |
| `tech` | `tech123` | Technician | Diagnosis, feedback, conversation |

> **Security Warning**: Change these passwords before deploying to production!

---

## Troubleshooting

### Container Not Starting

```bash
# Check container logs
docker-compose logs desoutter-api

# Common issues:
# - MongoDB not reachable: Check MONGO_HOST setting
# - Ollama not running: Verify Ollama container is up
# - Port conflicts: Check if ports 8000, 3001 are available
```

### GPU Not Detected

```bash
# Verify NVIDIA driver
nvidia-smi

# Verify container can see GPU
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# If using Docker Compose, ensure deploy section includes GPU resources
```

### Slow First Request

The first request may take 30-60 seconds as:
1. Ollama loads the model into GPU memory
2. Embeddings are initialized
3. Qdrant indexes are loaded

Subsequent requests should be much faster (2-5 seconds for non-cached queries).

### Empty Search Results

```bash
# Check if documents are ingested
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/admin/dashboard

# If vector DB is empty, ingest documents
curl -X POST -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/admin/documents/ingest
```

### Connection Refused Errors

```bash
# Check all services are running
docker-compose ps

# Restart services if needed
docker-compose restart

# Check network connectivity
docker network inspect ai-net
```

---

## Next Steps

1. **Upload Documents**: Add your technical manuals and service bulletins via the Admin panel
2. **Configure Users**: Create technician accounts with appropriate permissions
3. **Explore API**: Review the [API documentation](http://localhost:8000/docs) for all endpoints
4. **Production Deployment**: See [PROXMOX_DEPLOYMENT.md](PROXMOX_DEPLOYMENT.md) for infrastructure setup
5. **Review Roadmap**: Check [ROADMAP.md](ROADMAP.md) for upcoming features

---

## Additional Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Project overview and features |
| [ROADMAP.md](ROADMAP.md) | Development roadmap |
| [PROXMOX_DEPLOYMENT.md](PROXMOX_DEPLOYMENT.md) | Production deployment guide |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/fatihhbayram/desoutter-assistant/issues)
- **API Documentation**: http://localhost:8000/docs (when running)
- **Configuration**: Review [config/ai_settings.py](config/ai_settings.py) for all available settings
