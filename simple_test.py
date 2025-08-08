#!/usr/bin/env python3
"""
Simple test script for the Docker-deployed API
Tests file processing with the volume mount setup
"""

import requests
import json


def test_health():
    """Test if API is running"""
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health Check: {data['status']}")
            print(f"   Services: {data['services']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False


def test_path_validation():
    """Test path validation to see what paths are accessible"""
    print("\n🔍 Testing path validation...")
    
    # Test some common paths that might exist in your /temp mount
    test_paths = [
        "/temp/test.txt",  # This maps to your HOST_FILES_PATH
        "/app/data/test.txt"  # Local test path
    ]
    
    try:
        params = [("file_paths", path) for path in test_paths]
        
        response = requests.get(
            "http://localhost:8000/api/validate-paths",
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"📋 Validation Results: {data['valid_files']}/{data['total_files']} files valid")
            print(f"📁 Allowed base paths: {data['allowed_base_paths']}")
            
            for result in data['validation_results']:
                status = "✅ Valid" if result['valid'] else "❌ Invalid"
                print(f"   {status}: {result['path']}")
                if result['error']:
                    print(f"      Error: {result['error']}")
        else:
            print(f"❌ Path validation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Path validation error: {e}")


def process_files_example():
    """Example of processing files - you'll need to adjust paths to actual files"""
    print("\n📊 Testing file processing...")
    
    # You need to replace these paths with actual files that exist in your HOST_FILES_PATH
    # These paths should be accessible inside the container at /temp/*
    request_data = {
        "file_paths": [
            # "/temp/your_actual_file.xlsx",  # Replace with real file path
            # "/temp/another_file.csv"        # Replace with real file path
        ],
        "village_name": "测试村"
    }
    
    if not request_data["file_paths"]:
        print("⚠️  No file paths provided. Please edit the script and add real file paths.")
        print("   Example: '/temp/your_file.xlsx' (maps to your HOST_FILES_PATH)")
        return
    
    try:
        response = requests.post(
            "http://localhost:8000/api/process-files",
            json=request_data,
            timeout=180  # 3 minutes
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Processing completed: {data['status']['status']}")
            print(f"   Session ID: {data['session_id']}")
            print(f"   Input files: {len(data['input_files'])}")
            print(f"   Processed files: {len(data['processed_files'])}")
            print(f"   Table files found: {len(data['table_files'])}")
            print(f"   Irrelevant files: {len(data['irrelevant_files'])}")
            
            if data['table_files']:
                print("   📋 Detected tables:")
                for table in data['table_files']:
                    print(f"      - {table['chinese_table_name']}: {table['header_count']} headers")
        else:
            print(f"❌ File processing failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ File processing error: {e}")


def test_database_vectorization():
    """Test database vectorization"""
    print("\n🗄️  Testing database vectorization...")
    
    request_data = {
        "force_refresh": False
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/revectorize-database",
            json=request_data,
            timeout=300  # 5 minutes
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Vectorization completed: {data['status']['status']}")
            print(f"   Total tables: {data['total_tables']}")
            print(f"   Message: {data['status']['message']}")
        else:
            print(f"❌ Vectorization failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Vectorization error: {e}")


def main():
    """Run simple API tests"""
    print("🚀 Simple API Test for Docker Deployment")
    print("=" * 50)
    
    # Test 1: Health Check
    if not test_health():
        print("\n❌ API is not running. Please check:")
        print("   1. Docker container is running: docker-compose ps")
        print("   2. Logs: docker-compose logs fileprocessor-api")
        return
    
    # Test 2: Path Validation
    test_path_validation()
    
    # Test 3: File Processing (you need to add real file paths)
    process_files_example()
    
    # Test 4: Database Vectorization
    print("\n❓ Would you like to test database vectorization?")
    print("   This may take several minutes and requires database connection")
    user_input = input("   Test vectorization? (y/n): ").lower().strip()
    if user_input in ['y', 'yes']:
        test_database_vectorization()
    else:
        print("   Skipping database vectorization test")
    
    print("\n" + "=" * 50)
    print("🎉 Simple API test completed!")
    print(f"📖 API Documentation: http://localhost:8000/docs")
    
    # Show next steps
    print("\n📝 Next Steps:")
    print("1. Add real file paths to the process_files_example() function")
    print("2. Make sure your HOST_FILES_PATH contains the files you want to process")
    print("3. File paths in API calls should start with /temp/ (mapped to your host path)")
    print("4. Check docker-compose logs if you encounter issues")


def show_volume_mount_info():
    """Show information about current volume mount setup"""
    print("\n📁 Volume Mount Information:")
    print("   Your docker-compose.yml maps:")
    print("   Host: ${HOST_FILES_PATH} → Container: /temp")
    print("   ")
    print("   To process files:")
    print("   1. Put files in your HOST_FILES_PATH directory")
    print("   2. Use paths like '/temp/your_file.xlsx' in API calls")
    print("   3. The container will access them via the /temp mount")


if __name__ == "__main__":
    show_volume_mount_info()
    main()