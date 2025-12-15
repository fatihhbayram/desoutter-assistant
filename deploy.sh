#!/bin/bash
# Complete Deployment Script for ai.server (192.168.1.125)

set -e

echo "=================================================================="
echo "🚀 Desoutter Repair Assistant - Complete Deployment"
echo "=================================================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required"
    exit 1
fi

echo "✅ Python $(python3 --version) found"

# Create venv
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "📥 Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -r requirements-phase2.txt

# Create .env
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.proxmox .env
    echo "✅ .env created (edit if needed)"
else
    echo "✅ .env already exists"
fi

# Create directories
echo "📁 Creating data directories..."
mkdir -p data/{logs,exports,documents/manuals,documents/bulletins,vectordb}

echo ""
echo "=================================================================="
echo "✅ Backend Setup Complete!"
echo "=================================================================="
echo ""

# Frontend setup
if command -v npm &> /dev/null; then
    echo "📦 Setting up frontend..."
    cd frontend
    
    if [ ! -d "node_modules" ]; then
        echo "📥 Installing Node dependencies..."
        npm install
    else
        echo "✅ Node modules already installed"
    fi
    
    # Create .env for frontend
    if [ ! -f ".env" ]; then
        echo "VITE_API_URL=http://192.168.1.125:8000" > .env
    fi
    
    cd ..
    
    echo ""
    echo "=================================================================="
    echo "✅ Frontend Setup Complete!"
    echo "=================================================================="
else
    echo "⚠️  npm not found - frontend setup skipped"
    echo "   Install Node.js to use the React frontend"
fi

echo ""
echo "=================================================================="
echo "📋 Next Steps:"
echo "=================================================================="
echo ""
echo "1. Verify connections:"
echo "   curl http://localhost:11434/api/tags  # Ollama"
echo "   mongosh --host 172.18.0.5 --eval 'db.version()'  # MongoDB"
echo ""
echo "2. Scrape products:"
echo "   python3 scripts/scrape_single.py"
echo ""
echo "3. Add PDF manuals to:"
echo "   data/documents/manuals/"
echo "   data/documents/bulletins/"
echo ""
echo "4. Ingest documents:"
echo "   python3 scripts/ingest_documents.py"
echo ""
echo "5. Test RAG:"
echo "   python3 scripts/test_rag.py"
echo ""
echo "6. Start API:"
echo "   python3 scripts/run_api.py"
echo "   Access: http://192.168.1.125:8000"
echo ""
echo "7. Start Frontend (optional):"
echo "   cd frontend && npm run dev"
echo "   Access: http://192.168.1.125:3000"
echo ""
echo "=================================================================="
