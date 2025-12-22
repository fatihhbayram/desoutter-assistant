# 🚀 Quick Start Guide

> **Last Updated:** 22 December 2025

## ⚡ 5-Minute Setup (Proxmox ai.server)

### 1. Transfer Files

```bash
# On your Windows PC
# Use WinSCP or upload to GitHub, then clone on ai.server

# On ai.server VM
cd ~/
git clone <your-repo>  # or upload via SCP
cd desoutter-assistant
```

### 2. One-Command Setup

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create virtual environment
- Install dependencies
- Create .env file
- Create data directories

### 3. Configure

```bash
# Edit .env
nano .env

# Update if needed (defaults should work):
# OLLAMA_BASE_URL=http://localhost:11434  # ✓ Same VM
# MONGO_HOST=172.18.0.5                  # ✓ Already correct
```

---

## 🐳 Docker Quick Start (Recommended)

```bash
# Start API (connect to existing ai-net network)
cd /home/adentechio/desoutter-assistant
sudo docker cp config/settings.py desoutter-api:/app/config/
sudo docker cp src/llm/. desoutter-api:/app/src/llm/
sudo docker cp src/api/main.py desoutter-api:/app/src/api/
sudo docker restart desoutter-api
```

### Access Points
- **Frontend**: http://localhost:3001
- **API Docs**: http://localhost:8000/docs
- **Simple UI**: http://localhost:8000/ui

### Default Users
| Username | Password | Role |
|----------|----------|------|
| admin | admin123 | Admin |
| tech | tech123 | Technician |

---

## 📊 New Features (22 Dec 2025)

### Performance Metrics API
```bash
# Check system health
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/admin/metrics/health

# Get statistics (last hour)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/admin/metrics/stats?hours=1

# View slow queries
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/admin/metrics/slow
```

### Multi-turn Conversation API
```bash
# Start a conversation
curl -X POST http://localhost:8000/conversation/start \
  -H "Content-Type: application/json" \
  -d '{"message": "My EPB tool is not starting", "part_number": "6151659030"}'

# Continue conversation (use session_id from response)
curl -X POST http://localhost:8000/conversation/start \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc12345", "message": "What about the battery?"}'

# Get conversation history
curl http://localhost:8000/conversation/abc12345
```

---

## 🔧 Manual Setup (Development)

```bash
# Create venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run API
python3 scripts/run_api.py
```

## 🧪 Test Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Diagnose
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{"part_number":"6151659770","fault_description":"does not start","language":"en"}'
```

---

## 📚 Documentation

- [README.md](README.md) - Full documentation
- [CHANGELOG.md](CHANGELOG.md) - Development log
- [RAG_ENHANCEMENT_ROADMAP.md](RAG_ENHANCEMENT_ROADMAP.md) - Technical roadmap
- [PROXMOX_DEPLOYMENT.md](PROXMOX_DEPLOYMENT.md) - Deployment guide
- Ollama: `curl http://localhost:11434/api/tags`
- Mongo: `mongosh --host <mongo_host> --port 27017 --eval "db.version()"`

More details: see `PROXMOX_DEPLOYMENT.md` and `PHASE2_STRUCTURE.md`.

````
curl http://localhost:8000/health         # API

Troubleshooting — Chroma `$contains` error

```bash
# Symptom: API returns 500 and logs show:
# Expected where operator ... got $contains

# Fix: ensure API uses client-side filtering (no $contains in code)
sudo docker cp desoutter-assistant/desoutter-scraper/src/llm/rag_engine.py desoutter-api:/app/src/llm/rag_engine.py
sudo docker cp desoutter-assistant/desoutter-scraper/src/vectordb/chroma_client.py desoutter-api:/app/src/vectordb/chroma_client.py
sudo docker restart desoutter-api

# Verify:
sudo docker exec desoutter-api sh -lc "grep -Rn '\$contains' /app/src || true"

# Re-test:
curl -s -X POST http://127.0.0.1:8000/diagnose \
	-H 'Content-Type: application/json' \
	-d '{"part_number":"6151659770","fault_description":"does not start","language":"en"}' | jq
```
