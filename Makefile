.PHONY: help venv install setup run-single run-all run-api ingest export clean test

help:
	@echo "Desoutter Assistant - Available Commands:"
	@echo ""
	@echo "  make venv          - Create python virtualenv (venv)"
	@echo "  make install       - Install dependencies into active venv"
	@echo "  make setup         - Setup project (create .env, directories)"
	@echo "  make run-single    - Scrape single series"
	@echo "  make run-all       - Scrape all categories"
	@echo "  make run-api       - Start the API (dev)"
	@echo "  make ingest        - Ingest PDFs into vector DB"
	@echo "  make export-json   - Export data to JSON"
	@echo "  make export-csv    - Export data to CSV"
	@echo "  make clean         - Clean logs and cache"
	@echo "  make test          - Run tests"
	@echo ""

venv:
	@echo "🛠 Creating venv (if missing)..."
	@if [ ! -d venv ]; then python3 -m venv venv && echo "✅ venv created"; fi
	@echo "Activate with: source venv/bin/activate"

install:
	@echo "📦 Installing dependencies into venv (activate first)"
	@pip install -r requirements.txt

setup:
	@echo "⚙️  Setting up project..."
	@if [ ! -f .env ]; then cp .env.proxmox .env && echo "✅ Created .env from .env.proxmox"; fi
	@mkdir -p data/logs data/exports data/cache data/documents/manuals data/documents/bulletins
	@echo "✅ Setup complete!"

run-single:
	@echo "🚀 Running single series scraper..."
	python3 scripts/scrape_single.py

run-all:
	@echo "🚀 Running all categories scraper..."
	python3 scripts/scrape_all.py

run-api:
	@echo "🚀 Starting API (development)..."
	python3 scripts/run_api.py

ingest:
	@echo "📥 Ingesting PDFs to vector DB..."
	python3 scripts/ingest_documents.py

export-json:
	@echo "📤 Exporting to JSON..."
	python3 scripts/export_data.py --format json

export-csv:
	@echo "📤 Exporting to CSV..."
	python3 scripts/export_data.py --format csv

clean:
	@echo "🧹 Cleaning logs and cache..."
	rm -f data/logs/*.log || true
	rm -f data/exports/* || true
	rm -rf data/cache/* || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete || true
	@echo "✅ Clean complete!"

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v
