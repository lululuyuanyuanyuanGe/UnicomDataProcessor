#!/usr/bin/env python3
"""
Startup script for the Asian Information Data Processor API
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

import uvicorn
from api.main import app


def main():
    """Start the API server"""
    print("🚀 Starting Asian Information Data Processor API...")
    print(f"📁 Project root: {project_root}")
    print("🌐 Server will be available at:")
    print("   • API: http://localhost:8000")
    print("   • Docs: http://localhost:8000/docs")
    print("   • Health: http://localhost:8000/api/health")
    print("\n⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()