# Desoutter Repair Assistant - Proxmox Deployment Guide

## 🏗️ Your Infrastructure

Based on your GitHub repo, you have:

```
Proxmox VE Host
├── VM: ai.server (GPU Passthrough - RTX A2000 6G)
│   ├── Ollama (port 11434)
│   ├── Open WebUI (port 3000)
│   └── Ubuntu 24.04
│
├── Docker Containers
│   ├── MongoDB (172.18.0.5:27017)
│   ├── n8n
│   └── Cloudflare Tunnel
│
└── Network: 192.168.1.x
```

# Desoutter Repair Assistant — Proxmox deployment notes

This page contains concise, tested steps to run the project on your ai.server VM (Proxmox). It assumes Ollama runs locally on the same VM and MongoDB is reachable from that VM.

Quick checklist

- Clone the repository on ai.server
- Create Python 3.11 venv and install dependencies
- Copy `.env.proxmox` → `.env` and adapt if needed
- Create data directories and run a non-destructive initial scrape
- (Optional) ingest PDFs and start the API

1) Prepare the VM

```bash
# SSH to ai.server (Proxmox console or ssh)
ssh user@ai.server
cd ~
git clone https://github.com/fatihhbayramm/desoutter-assistant.git
cd desoutter-assistant
```

2) Create venv and install

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3) Configure environment and directories

```bash
cp .env.proxmox .env   # inspect and update values as needed
mkdir -p data/logs data/exports data/documents/manuals data/documents/bulletins
```

4) Verify services

```bash
# Ollama (local)
curl http://localhost:11434/api/tags || echo "ollama unreachable"

# Mongo (remote or container)
# mongosh --host <mongo_host> --port 27017 --eval "db.adminCommand('ping')"
```

5) Run a non-destructive scrape and check DB

```bash
python3 scripts/scrape_single.py
python3 -c "from src.database import MongoDBClient; db=MongoDBClient(); db.connect(); print('product count:', db.count_products())"
```

6) Ingest PDFs (Phase 2)

```bash
# Place PDFs under data/documents/{manuals,bulletins}/
python3 scripts/ingest_documents.py
```

7) Start API (dev)

```bash
python3 scripts/run_api.py
# or: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Systemd (production)

Create `/etc/systemd/system/desoutter-api.service` with the working directory and venv path for your user, then enable and start the unit. See `PROXMOX_DEPLOYMENT.md` in repo for an example unit file.

Notes

- The instructions intentionally avoid destructive commands. If you want a helper to wipe `products` and re-scrape, I can add an explicit script that requires your confirmation.
- If Mongo is a Docker container, ensure the container network is accessible from ai.server.

If you'd like, I can now commit these doc updates and push them to GitHub.
mongosh --host 172.18.0.5 --port 27017 --eval "db.adminCommand('ping')"
