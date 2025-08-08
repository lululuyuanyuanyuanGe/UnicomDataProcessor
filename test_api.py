#!/usr/bin/env python3
"""
Test script for the Asian Information Data Processor API
"""

import asyncio
import aiohttp
import json
from pathlib import Path
import tempfile


async def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing health endpoint...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("http://localhost:8000/api/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"Health check passed: {data['status']}")
                    print(f"   Services: {data['services']}")
                    return True
                else:
                    print(f"Health check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"Failed to connect to API: {e}")
            return False


async def test_database_info():
    """Test database info endpoints"""
    print("\nTesting database info endpoints...")
    
    async with aiohttp.ClientSession() as session:
        # Test database info
        try:
            async with session.get("http://localhost:8000/api/database-info") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Database info retrieved: {data['total_tables']} tables")
                    print(f"   Last updated: {data.get('last_updated', 'Unknown')}")
                else:
                    print(f"⚠️ Database info warning: {response.status}")
        except Exception as e:
            print(f"❌ Database info failed: {e}")
        
        # Test table list
        try:
            async with session.get("http://localhost:8000/api/database-tables") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Database tables listed: {data['total_tables']} tables")
                    print(f"   Status: {data['status']}")
                    if data['tables'][:3]:  # Show first 3 tables
                        print(f"   Sample tables: {data['tables'][:3]}")
                else:
                    print(f"⚠️ Database tables warning: {response.status}")
        except Exception as e:
            print(f"❌ Database tables failed: {e}")


async def test_database_vectorization():
    """Test database vectorization endpoint"""
    print("\nTesting database vectorization...")
    
    async with aiohttp.ClientSession() as session:
        try:
            request_data = {
                "force_refresh": False,
                "specific_tables": None
            }
            
            async with session.post(
                "http://localhost:8000/api/revectorize-database",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Database vectorization completed: {data['status']['status']}")
                    print(f"   Total tables processed: {data['total_tables']}")
                    print(f"   Message: {data['status']['message']}")
                else:
                    error_data = await response.text()
                    print(f"❌ Database vectorization failed: {response.status}")
                    print(f"   Error: {error_data}")
        except asyncio.TimeoutError:
            print("⚠️ Database vectorization timed out (this is normal for large databases)")
        except Exception as e:
            print(f"❌ Database vectorization error: {e}")


async def test_file_processing():
    """Test file processing endpoint with a sample file"""
    print("\nTesting file processing...")
    
    # Create a sample CSV file for testing
    sample_csv_content = """姓名,年龄,职业
张三,25,工程师
李四,30,教师
王五,28,医生"""
    
    async with aiohttp.ClientSession() as session:
        try:
            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field('village_name', '测试村')
            
            # Add the sample CSV file
            data.add_field(
                'files',
                sample_csv_content.encode('utf-8'),
                filename="D:\asianInfo\dataProcessor\temp_files\燕云村残疾人名单.xlsx",
                content_type='text/csv'
            )
            
            async with session.post(
                "http://localhost:8000/api/process-files",
                data=data,
                timeout=aiohttp.ClientTimeout(total=180)  # 3 minute timeout
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ File processing completed: {data['status']['status']}")
                    print(f"   Session ID: {data['session_id']}")
                    print(f"   Processed files: {len(data['processed_files'])}")
                    print(f"   Table files: {len(data['table_files'])}")
                    print(f"   Summary: {data['summary']}")
                else:
                    error_data = await response.text()
                    print(f"❌ File processing failed: {response.status}")
                    print(f"   Error: {error_data}")
        except asyncio.TimeoutError:
            print("⚠️ File processing timed out")
        except Exception as e:
            print(f"❌ File processing error: {e}")


async def main():
    """Run all API tests"""
    print("Starting API tests for Asian Information Data Processor")
    print("=" * 60)
    
    # Test health endpoint first
    health_ok = await test_health_endpoint()
    
    if not health_ok:
        print("\nAPI is not running. Please start the API server first:")
        print("   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return
    
    # Test database endpoints
    await test_database_info()
    
    # Test file processing
    await test_file_processing()
    
    # Test database vectorization (optional, takes longer)
    print("\nWould you like to test database vectorization? (This may take several minutes)")
    print("   This will update database embeddings...")
    # For automated testing, we'll skip this for now
    print("   Skipping database vectorization test (uncomment to enable)")
    # await test_database_vectorization()
    
    print("\n" + "=" * 60)
    print("API tests completed!")
    print("\nAPI Documentation available at: http://localhost:8000/docs")
    print("Alternative docs at: http://localhost:8000/redoc")


if __name__ == "__main__":
    asyncio.run(main())