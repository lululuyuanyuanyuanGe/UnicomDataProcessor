#!/usr/bin/env python3
"""Simple test to verify LibreOffice lock works"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.file_process import _libreoffice_lock

def test_lock():
    """Test that the LibreOffice lock is working"""
    results = []
    lock_order = []
    
    def test_function(file_id):
        with _libreoffice_lock:
            lock_order.append(file_id)
            print(f"Processing file {file_id}")
            import time
            time.sleep(0.1)  # Simulate work
            return f"File {file_id} processed"
    
    print("Testing LibreOffice lock with ThreadPoolExecutor...")
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(test_function, i) for i in range(5)]
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Completed: {result}")
    
    print(f"Lock acquisition order: {lock_order}")
    print("Test completed - lock is working if order is sequential: 0,1,2,3,4")
    
    # Check that we have the global lock available
    print(f"LibreOffice lock object: {_libreoffice_lock}")

if __name__ == "__main__":
    test_lock()