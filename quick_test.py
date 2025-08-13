#!/usr/bin/env python3
"""
Quick test for processing your files
Edit the file_paths list below with your actual file paths
"""

import requests
import json


VILLAGE_NAME = "测试村"  # Change this to your village name

def quick_process_files():
    """Quick test to process your files"""
    print("🚀 Quick File Processing Test")
    print("=" * 40)
    
    
    # Test API connection
    try:
        health_response = requests.get("http://localhost:8000/api/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ API not responding. Is Docker container running?")
            return
        print("✅ API is running")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Check: docker-compose ps")
        return
    
    # Process files
    request_data = {
        "files_data": {"http://58.144.196.118:5019/ai-index/atts/images/QA_20250805152449915_党员信息.xlsx": "QA_20250805152449915_党员信息"},
        "village_name": VILLAGE_NAME
    }


    
    print("\n🔄 Processing files...")
    try:
        response = requests.post(
            "http://localhost:8000/api/process-files-with-ids",
            timeout=300,  # 5 minutes
            json=request_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS! Status: {data['status']['status']}")
            print(f"📋 Results:")
            print(f"   Session ID: {data['session_id']}")
            print(f"   Input files: {len(data['input_files'])}")
            print(f"   Successfully processed: {len(data['processed_files'])}")
            print(f"   Tables found: {len(data['table_files'])}")
            print(f"   Irrelevant files: {len(data['irrelevant_files'])}")
            
            # Show table details
            if data['table_files']:
                print("\n📊 Detected Tables:")
                for i, table in enumerate(data['table_files'], 1):
                    print(f"   {i}. {table['chinese_table_name']}")
                    print(f"      Headers: {table['header_count']} columns")
                    if table['headers']:
                        print(f"      Columns: {', '.join(table['headers'][:5])}{'...' if len(table['headers']) > 5 else ''}")
            
            # Show irrelevant files
            if data['irrelevant_files']:
                print(f"\n🗑️  Irrelevant Files: {data['irrelevant_files']}")
            
            print(f"\n📊 Summary: {data['summary']}")
            
        else:
            print(f"❌ FAILED! Status: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"   Error: {response.text}")
                
    except Exception as e:
        print(f"❌ Processing error: {e}")

if __name__ == "__main__":
    quick_process_files()