# 🚀 Quick Start Guide

## ⚡ 5-Minute Setup (Proxmox ai.server)

### 1. Transfer Files

```bash
# On your Windows PC
# Use WinSCP or upload to GitHub, then clone on ai.server

# On ai.server VM
cd ~/
git clone <your-repo>  # or upload via SCP
cd desoutter-scraper
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
````markdown
# Quickstart — minimal, tested

These steps assume you have cloned the repository on the target VM (ai.server) and will run the service locally. See `PROXMOX_DEPLOYMENT.md` for Proxmox-specific notes.

1) Create Python venv and install

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-phase2.txt || true  # phase2 deps optional
```

2) Create .env (copy example) and data dirs

```bash
cp .env.proxmox .env  # or cp .env.example .env
mkdir -p data/logs data/exports data/documents/manuals data/documents/bulletins
```

3) Run initial scrape (non-destructive)

```bash
python3 scripts/scrape_single.py   # sample
# or: python3 scripts/scrape_all.py
```

4) (Optional) Ingest PDFs to vector DB

```bash
python3 scripts/ingest_documents.py
```

5) Start the API (development)

```bash
python3 scripts/run_api.py
# or: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

6) Open UI in browser (if frontend served by API)

```
http://<ai.server.ip>:8000/ui
```

7) Test Diagnose endpoint

```bash
curl -s -X POST http://127.0.0.1:8000/diagnose \
	-H 'Content-Type: application/json' \
	-d '{"part_number":"6151659770","fault_description":"does not start","language":"en"}' | jq
```

Notes
- Defaults in `.env.proxmox` assume Ollama runs locally on ai.server at port 11434 and MongoDB is reachable at the address listed there.
- This guide avoids destructive steps (no automatic drops). If you want to wipe and re-scrape, tell me and I'll prepare a helper with explicit confirmation.

Troubleshooting quick checks
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
