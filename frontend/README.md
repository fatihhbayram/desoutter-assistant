# Desoutter Repair Assistant - Frontend

Modern React frontend with Vite.

## Setup

```bash
cd frontend
npm install
```

## Development

```bash
# Start dev server (with hot reload)
npm run dev

# Access at: http://localhost:3001

# Or run on 3001 for parity with Docker
npm run dev -- --host 0.0.0.0 --port 3001
# Access at: http://localhost:3001
```

## Production Build

```bash
# Build for production
npm run build

# Serve the API to serve static files
# Files will be in: frontend/dist/
```

## Features

- 🎨 Modern, responsive UI
- ⚡ Fast with Vite
- 🔄 Real-time product selection
- 🌍 Bilingual (EN/TR)
- 📱 Mobile-friendly
- ✨ Smooth animations

## Architecture

```
Frontend (React + Vite)
    ↓ HTTP
FastAPI Backend
    ↓
RAG Engine → Ollama LLM
    ↓
ChromaDB + MongoDB
```

## Environment Variables

Create `.env` in frontend directory:

```env
VITE_API_URL=http://192.168.1.125:8000
```

## Quick Start (Docker)

```bash
sudo docker build -t desoutter-frontend:dev desoutter-scraper/frontend
sudo docker run -d --name desoutter-frontend \
    -p 3001:3001 \
    -e VITE_API_URL=http://192.168.1.125:8000 \
    desoutter-frontend:dev

# Open the app
xdg-open http://localhost:3001 || echo "Open http://localhost:3001"
```

## Troubleshooting

- If the page is blank or calls fail, ensure `VITE_API_URL` is reachable (try `/health` and `/stats`).
- In Docker on Linux, use your host LAN IP instead of `localhost` for `VITE_API_URL`.
- If port 3001 is busy, change the mapping (e.g., `-p 8080:3001`) and open `http://localhost:8080`.

## Recent UI Improvements

- Contained product details content to prevent overflow outside its card.
- Wider layout on large screens (1280–1400px breakpoints).
