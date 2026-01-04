# ⚡ Quick Start Guide

Get Desoutter Assistant running in under 10 minutes.

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Docker | 20.10+ | `docker --version` |
| Docker Compose | 2.0+ | `docker compose version` |
| Git | 2.30+ | `git --version` |
| NVIDIA Driver | 525+ | `nvidia-smi` |
| NVIDIA Container Toolkit | Latest | `nvidia-ctk --version` |

> **Note**: GPU is optional but recommended for faster LLM inference.

---

## 🚀 Installation

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

# Embeddings Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cuda  # Use 'cpu' if no GPU

# RAG Configuration
RAG_SIMILARITY_THRESHOLD=0.30
USE_HYBRID_SEARCH=true

# JWT Secret (change in production!)
JWT_SECRET=your-secure-secret-key-here
```

### Step 3: Launch Services

#### Option A: Docker Compose (Recommended)

```bash
# Start all services
docker compose up -d

# Verify all containers are running
docker compose ps
```

#### Option B: Individual Containers

```bash
# Start API container
docker run -d --name desoutter-api \
  --network ai-net \
  -p 8000:8000 \
  -e MONGO_HOST=mongodb \
  -e OLLAMA_BASE_URL=http://ollama:11434 \
  -e OLLAMA_MODEL=qwen2.5:7b-instruct \
  -v desoutter_data:/app/data \
  -v $(pwd)/documents:/app/documents \
  -v huggingface_cache:/root/.cache/huggingface \
  --gpus all \
  desoutter-api

# Start Frontend container
cd frontend
docker build -t desoutter-frontend .
docker run -d --name desoutter-frontend -p 3001:3001 desoutter-frontend
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

# Check MongoDB (from inside container)
docker exec -it desoutter-api python3 -c \
  "from src.database import MongoDBClient; db=MongoDBClient(); db.connect(); print('Products:', db.count_products())"
# Expected: Products: 451
```

---

## 🎯 First Query

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
  "confidence": "high",
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

## 🌐 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:3001 | Main user interface |
| **API Docs** | http://localhost:8000/docs | Swagger documentation |
| **Simple UI** | http://localhost:8000/ui | Lightweight web interface |
| **Mongo Express** | http://localhost:8081 | Database admin (if enabled) |

---

## 👤 Default Users

| Username | Password | Role | Permissions |
|----------|----------|------|-------------|
| `admin` | `admin123` | Admin | Full access, user management, document upload |
| `tech` | `tech123` | Technician | Diagnosis, feedback, conversation |

> ⚠️ **Security Warning**: Change these passwords before deploying to production!

---

## 🔧 Troubleshooting

### Container Not Starting

```bash
# Check container logs
docker logs desoutter-api

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

# If using Docker Compose, ensure deploy section is correct:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: all
#           capabilities: [gpu]
```

### Slow First Request

The first request may take 30-60 seconds as:
1. Ollama loads the model into GPU memory
2. Embeddings are initialized
3. ChromaDB indexes are loaded

Subsequent requests should be much faster (2-5 seconds).

### Empty Search Results

```bash
# Check if documents are ingested
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/admin/documents

# If empty, run document ingestion
curl -X POST -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/admin/documents/ingest
```

---

## 📚 Next Steps

1. **Upload Documents**: Add your technical manuals and service bulletins via the Admin panel
2. **Configure Users**: Create technician accounts with appropriate permissions
3. **Explore API**: Review the [API documentation](http://localhost:8000/docs) for all endpoints
4. **Production Deployment**: See [PROXMOX_DEPLOYMENT.md](PROXMOX_DEPLOYMENT.md) for infrastructure setup

---

## 📖 Additional Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Project overview and features |
| [PROXMOX_DEPLOYMENT.md](PROXMOX_DEPLOYMENT.md) | Production deployment guide |
| [RAG_QUALITY_IMPROVEMENT.md](RAG_QUALITY_IMPROVEMENT.md) | Technical RAG documentation |
| [ROADMAP.md](ROADMAP.md) | Development roadmap |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## 🆘 Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/fatihhbayram/desoutter-assistant/issues)
- **API Documentation**: http://localhost:8000/docs (when running)
