#!/usr/bin/env python3
"""
Simple startup script for the Asian Information Data Processor API
"""

import sys
import os
from pathlib import Path

# Set environment variable to handle Unicode
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Asian Information Data Processor API...")
    print("Server will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Start the server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )