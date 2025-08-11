#!/usr/bin/env python3
import os
import re
import requests
from urllib.parse import quote, urlparse, unquote
import time


def extract_chinese_and_parentheses(filename):
    """Extract Chinese characters and parentheses from filename, preserve extension"""
    name_without_ext = os.path.splitext(filename)[0]
    extension = os.path.splitext(filename)[1]
    
    # Find Chinese characters and parentheses with content
    matches = re.findall(r'[\u4e00-\u9fff]+|\([^)]*\)', name_without_ext)
    chinese_name = ''.join(matches)
    
    # If no Chinese characters found, use original filename
    if not chinese_name:
        return filename
    
    return chinese_name + extension


def download_and_rename_from_dict(url_file_dict, download_folder="downloads_files", delay_seconds=1):
    """
    Download files from a dictionary of {url: file_id} and return {complete_file_path: file_id}
    
    Args:
        url_file_dict: Dictionary with download URLs as keys and file_ids as values
        download_folder: Folder to save downloaded files
        delay_seconds: Delay between downloads to be server-friendly
    
    Returns:
        Dictionary with complete file paths as keys and original file_ids as values
    """
    
    print(f"🚀 Starting downloads to folder: {download_folder}")
    print(f"📁 Total files to process: {len(url_file_dict)}\n")
    
    # Create download folder if it doesn't exist
    os.makedirs(download_folder, exist_ok=True)
    
    # Result dictionary: {complete_file_path: file_id}
    result_dict = {}
    
    # Track statistics
    successful_downloads = 0
    failed_downloads = 0
    
    for i, (url, file_id) in enumerate(url_file_dict.items(), 1):
        try:
            print(f"[{i}/{len(url_file_dict)}] Processing URL: {url}")
            print(f"                     File ID: {file_id}")
            
            # Extract original filename from URL
            parsed_url = urlparse(url)
            original_filename = unquote(os.path.basename(parsed_url.path))
            
            # Generate new filename with Chinese characters only
            new_filename = extract_chinese_and_parentheses(original_filename)
            
            # Handle potential duplicate filenames
            full_path = os.path.join(download_folder, new_filename)
            
            if os.path.exists(full_path):
                os.remove(full_path)
                print(f"🗑️  Removed existing: {new_filename}")

            print(f"📥 Original: {original_filename}")
            print(f"💾 Saving as: {new_filename}")
            
            # Set headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Download the file
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save the file
            with open(full_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Add to result dictionary with COMPLETE PATH as key
            result_dict[full_path] = file_id
            
            file_size_mb = os.path.getsize(full_path) / (1024*1024)
            print(f"✅ Success! Size: {file_size_mb:.2f} MB")
            print(f"🗂️  Mapped: {full_path} → {file_id}\n")
            
            successful_downloads += 1
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {e}")
            failed_downloads += 1
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            failed_downloads += 1
        
        # Add delay between downloads to be server-friendly
        if i < len(url_file_dict):
            time.sleep(delay_seconds)
    
    # Print summary
    print("=" * 60)
    print("📋 DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✅ Successful downloads: {successful_downloads}")
    print(f"❌ Failed downloads: {failed_downloads}")
    print(f"📦 Total files in result: {len(result_dict)}")
    
    if result_dict:
        print("\n🎉 Final mapping (Complete file path → File ID):")
        for file_path, file_id in result_dict.items():
            print(f"   • {file_path} → {file_id}")
    
    print(f"\n📁 All files saved to: {os.path.abspath(download_folder)}")
    
    return result_dict


# Example usage function
def example_usage():
    """Example of how to use the function"""
    
    # Example input dictionary - replace with your actual data
    url_file_dict = {
        "http://58.144.196.118:5019/ai-index/atts/images/QA_20250805124450569_党员信息.xlsx": "file_001",
    }
    
    # Process the downloads
    result = download_and_rename_from_dict(
        url_file_dict=url_file_dict,
        download_folder="downloads_files",
        delay_seconds=1
    )
    
    return result


if __name__ == "__main__":
    print("🔥 URL Dictionary File Downloader and Renamer")
    print("=" * 60)
    
    # Run example - replace this with your actual function call
    result_mapping = example_usage()
    
    print("🎯 Script completed!")
    print(f"🗂️  Final result dictionary: {result_mapping}")
