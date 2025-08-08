#!/usr/bin/env python3
"""Test script for parallel file processing with LibreOffice lock fix"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from fileProcessAgent import FileProcessAgent

def test_parallel_processing():
    """Test the fixed parallel processing capability"""
    
    # Test files - use existing files from your data folder  
    test_files = [
        r"d:\asianInfo\数据\七田村\七田村2025年度党员名册2025.xls",
        r"d:\asianInfo\数据\七田村\城保名册.xls"
    ]
    
    # Check if test files exist
    existing_files = []
    for file_path in test_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
            print(f"Found test file: {Path(file_path).name}")
        else:
            print(f"Missing test file: {file_path}")
    
    if not existing_files:
        print("No test files found. Please provide valid file paths.")
        return
    
    print(f"\nTesting parallel processing with {len(existing_files)} files...")
    print("=" * 60)
    
    try:
        # Initialize agent
        agent = FileProcessAgent()
        
        # Run the agent with test files
        final_state = agent.run_file_process_agent(
            session_id="test_parallel",
            upload_files_path=existing_files,
            village_name="测试村"
        )
        
        print("\nParallel processing test completed successfully!")
        print("=" * 60)
        
        # Show results
        processed_results = final_state.get('processed_table_results', [])
        successful = [r for r in processed_results if r.get("success", False)]
        failed = [r for r in processed_results if not r.get("success", False)]
        
        print(f"Test Results:")
        print(f"  - Total files processed: {len(processed_results)}")
        print(f"  - Successful: {len(successful)}")
        print(f"  - Failed: {len(failed)}")
        
        if successful:
            print("\nSuccessful conversions:")
            for result in successful:
                name = result.get("chinese_table_name", "Unknown")
                headers = len(result.get("headers", []))
                print(f"  - {name}: {headers} headers")
        
        if failed:
            print("\nFailed conversions:")
            for result in failed:
                name = result.get("chinese_table_name", "Unknown")
                error = result.get("error", "Unknown error")
                print(f"  - {name}: {error}")
                
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parallel_processing()