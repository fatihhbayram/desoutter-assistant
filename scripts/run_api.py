#!/usr/bin/env python3
"""
Run FastAPI server
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from config.ai_settings import API_HOST, API_PORT, API_RELOAD

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Starting Desoutter Repair Assistant API")
    print("=" * 80)
    print(f"Server: http://{API_HOST}:{API_PORT}")
    print(f"Docs: http://{API_HOST}:{API_PORT}/docs")
    print(f"UI: http://{API_HOST}:{API_PORT}/ui")
    print("=" * 80)
    print("")
    
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD
    )
