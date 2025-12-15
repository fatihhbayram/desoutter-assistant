#!/bin/bash
# Quick setup and run script for Desoutter Scraper

set -e

echo "=================================================="
echo "Desoutter Scraper - Quick Setup"
echo "=================================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python $(python3 --version) found"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created"
    echo "⚠️  Please edit .env file with your settings before running"
else
    echo "✅ .env file already exists"
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/logs data/exports data/cache
echo "✅ Directories created"
echo ""

echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env file if needed: nano .env"
echo "  2. Run single series scraper: python3 scripts/scrape_single.py"
echo "  3. Or run all categories: python3 scripts/scrape_all.py"
echo ""
echo "Quick commands with Makefile:"
echo "  make run-single   - Scrape single series"
echo "  make run-all      - Scrape all categories"
echo "  make export-json  - Export to JSON"
echo "  make help         - Show all commands"
echo ""
