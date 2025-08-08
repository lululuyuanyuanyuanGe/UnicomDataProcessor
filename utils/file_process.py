from __future__ import annotations
from bs4 import BeautifulSoup
from pathlib import Path
import re
import os
import json
from pathlib import Path
import subprocess
import chardet
from typing import Union, List, Dict
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.modelRelated import invoke_model

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

def detect_and_process_file_paths(user_input: str) -> list:
    """检测用户输入中的文件路径并验证文件是否存在，返回结果为用户上传的文件路径组成的数列"""
    file_paths = []
    processed_paths = set()  # Track already processed paths to avoid duplicates
    
    # 改进的文件路径检测模式，支持中文字符
    # Windows路径模式 (C:\path\file.ext 或 D:\path\file.ext) - 支持中文字符
    windows_pattern = r'[A-Za-z]:[\\\\/](?:[^\\\\/\s\n\r]+[\\\\/])*[^\\\\/\s\n\r]+\.\w+'
    # 相对路径模式 (./path/file.ext 或 ../path/file.ext) - 支持中文字符
    relative_pattern = r'\.{1,2}[\\\\/](?:[^\\\\/\s\n\r]+[\\\\/])*[^\\\\/\s\n\r]+\.\w+'
    # 简单文件名模式 (filename.ext) - 支持中文字符
    filename_pattern = r'\b[a-zA-Z0-9_\u4e00-\u9fff\-\(\)（）]+\.[a-zA-Z0-9]+\b'
    
    patterns = [windows_pattern, relative_pattern, filename_pattern]
    
    # Run the absolute path pattern first
    for match in re.findall(patterns[0], user_input):
        if match in processed_paths:
            continue
        processed_paths.add(match)
        _log_existence(match, file_paths)

    # Run the relative path pattern
    for match in re.findall(patterns[1], user_input):
        if match in processed_paths:
            continue
        processed_paths.add(match)
        _log_existence(match, file_paths)
        
    # Run the filename pattern if we didn't find any files
    if not file_paths:
        for match in re.findall(patterns[2], user_input):
            if match in processed_paths:
                continue
            processed_paths.add(match)
            _log_existence(match, file_paths)

    return file_paths


# -- 小工具函数 ------------------------------------------------------------
def _log_existence(path: str, container: list):
    if os.path.exists(path):
        container.append(path)
        print(f"✅ 检测到文件: {path}")
    else:
        print(f"⚠️ 文件路径无效或文件不存在: {path}")


def process_file_to_text(file_path: str | Path) -> str | None:
    """
    Efficiently process a file to readable text content in memory.
    
    This function does: 1 read → process in memory → return text
    Instead of: read → write temp file → read temp file → write final file
    
    Returns:
        str: The processed text content, or None if processing failed
    """
    source_path = Path(file_path)
    file_extension = source_path.suffix.lower()
    
    # Define file type categories
    spreadsheet_extensions = {'.xlsx', '.xls', '.xlsm', '.ods', '.csv'}
    text_extensions = {'.txt', '.md', '.json', '.xml', '.html', '.htm', '.py', '.js', '.css', '.sql', '.log'}
    document_extensions = {'.docx', '.doc', '.pptx', '.ppt'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg'}
    
    try:
        # Handle spreadsheet files
        if file_extension in spreadsheet_extensions:
            return _process_spreadsheet(source_path)
        
        # Handle document files (DOCX, DOC, etc.)
        elif file_extension in document_extensions:
            return _process_doc_file(source_path)
        
        # Handle plain text files
        elif file_extension in text_extensions:
            return _read_text_auto(source_path)
        
        # Handle image files - return metadata since we can't convert to text
        elif file_extension in image_extensions:
            return f"Image file: {source_path.name}\nFile size: {source_path.stat().st_size} bytes\nFormat: {file_extension}"
        
        # Handle other file types
        else:
            # Try to detect if it's a text file by MIME type
            import mimetypes
            mime_type, _ = mimetypes.guess_type(str(source_path))
            
            if mime_type and mime_type.startswith('text/'):
                return _read_text_auto(source_path)
            else:
                # For binary files, return metadata
                return f"Binary file: {source_path.name}\nFile size: {source_path.stat().st_size} bytes\nType: {mime_type or 'unknown'}"
                
    except Exception as e:
        print(f"❌ Error processing file {file_path}: {e}")
        return None


# Global lock for LibreOffice operations to prevent concurrent access issues
import threading
_libreoffice_lock = threading.Lock()


# ──────────────────────── private helpers ─────────────────────── #
def _read_text_auto(path: Path) -> str:
    """Best-effort text loader with encoding detection."""
    data = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "big5"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    if chardet:
        enc = chardet.detect(data).get("encoding")
        if enc:
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                pass
    return data.decode("utf-8", errors="replace")

def _process_spreadsheet():
    # This function should use pandas that convertts the excel to html or csv format, then save them inside the txt file with the same
    # file name but txt suffix under the temp folder
    pass

def _process_doc_file():
    # We will use mamoth library to extratct the content of the word files, and save it in the txt format under the temp folder
    pass

def _process_pdf_file():
    # We will support the PDF files in later versions, for now just safely ignore the pdf files
    pass

def move_template_files_safely(processed_template_file: str, original_files_list: list[str], dest_dir_name: str = "template_files") -> dict[str, str]:
    """
    Safely move both processed and original template files to the template_files directory.
    
    This function handles moving both the processed template file (.txt) and its corresponding
    original file to the template_files folder, with proper error handling and logging.
    
    Args:
        processed_template_file: Path to the processed template file (.txt)
        original_files_list: List of original file paths to search for the corresponding original file
        dest_dir_name: Name of the destination directory under conversations/files/user_uploaded_files/
        
    Returns:
        dict: {
            "processed_template_path": str,  # Path to moved processed template file
            "original_template_path": str    # Path to moved original template file (or empty if not found)
        }
    """
    import shutil
    
    try:
        # Create destination directories
        dest_dir = Path(f"conversations/files/user_uploaded_files/{dest_dir_name}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Create original_file subdirectory within template_files
        original_dest_dir = dest_dir / "original_file"
        original_dest_dir.mkdir(parents=True, exist_ok=True)
        
        processed_template_path = Path(processed_template_file)
        result = {
            "processed_template_path": "",
            "original_template_path": ""
        }
        
        print(f"📁 正在移动模板文件: {processed_template_path.name}")
        
        # Move the processed template file
        processed_target_path = dest_dir / processed_template_path.name
        
        # Handle existing processed file
        if processed_target_path.exists():
            print(f"⚠️ 处理模板文件已存在: {processed_target_path.name}")
            try:
                processed_target_path.unlink()
                print(f"🗑️ 已删除旧的处理模板文件: {processed_target_path.name}")
            except Exception as delete_error:
                print(f"❌ 删除旧的处理模板文件失败: {delete_error}")
                result["processed_template_path"] = processed_template_file
                return result
        
        # Move processed template file
        try:
            shutil.move(str(processed_template_path), str(processed_target_path))
            result["processed_template_path"] = str(processed_target_path)
            print(f"✅ 处理模板文件已移动到: {processed_target_path}")
        except Exception as move_error:
            print(f"❌ 移动处理模板文件失败: {move_error}")
            result["processed_template_path"] = processed_template_file
            return result
        
        # Find and move the corresponding original file
        template_file_stem = processed_template_path.stem
        original_file_found = False
        
        print(f"🔍 正在寻找对应的原始模板文件: {template_file_stem}")
        
        for original_file in original_files_list:
            original_file_path = Path(original_file)
            if original_file_path.stem == template_file_stem:
                print(f"📋 找到对应的原始文件: {original_file_path.name}")
                
                # Move the original file to the original_file subdirectory
                original_target_path = original_dest_dir / original_file_path.name
                
                # Handle existing original file
                if original_target_path.exists():
                    print(f"⚠️ 原始模板文件已存在: {original_target_path.name}")
                    try:
                        original_target_path.unlink()
                        print(f"🗑️ 已删除旧的原始模板文件: {original_target_path.name}")
                    except Exception as delete_error:
                        print(f"❌ 删除旧的原始模板文件失败: {delete_error}")
                        # Continue with moving even if deletion failed
                
                # Move original file
                try:
                    shutil.move(str(original_file_path), str(original_target_path))
                    result["original_template_path"] = str(original_target_path)
                    print(f"✅ 原始模板文件已移动到: {original_target_path}")
                    original_file_found = True
                    break
                except Exception as move_error:
                    print(f"❌ 移动原始模板文件失败: {move_error}")
                    # Continue searching for other matching files
        
        if not original_file_found:
            print(f"⚠️ 未找到对应的原始模板文件: {template_file_stem}")
            result["original_template_path"] = ""
        
        return result
        
    except Exception as e:
        print(f"❌ 移动模板文件过程中出错: {e}")
        return {
            "processed_template_path": processed_template_file,
            "original_template_path": ""
        }

def convert_html_to_excel(html_file_path: str, output_dir: str = None) -> str:
    """
    Convert HTML file to Excel format using session-specific output directory
    
    Args:
        html_file_path: Path to the HTML file to convert
        output_dir: Output directory path (should be session-specific)
    
    Returns:
        str: Path to the converted Excel file
    """
    # Function implementation placeholder
    pass

def delete_files_from_staging_area(file_paths: list[str]) -> dict[str, list[str]]:
    """Delete irrelevant files from staging area.
    
    Args:
        file_paths: List of file paths to delete
        
    Returns:
        dict: {
            "deleted_files": list[str],  # Successfully deleted files
            "failed_deletes": list[str]  # Files that failed to delete
        }
    """
    from pathlib import Path
    
    deleted_files = []
    failed_deletes = []
    
    for file_path in file_paths:
        try:
            file_to_delete = Path(file_path)
            if file_to_delete.exists():
                file_to_delete.unlink()
                deleted_files.append(str(file_to_delete))
                print(f"🗑️ 已删除无关文件: {file_to_delete.name}")
            else:
                print(f"⚠️ 文件不存在，跳过删除: {file_path}")
        except Exception as e:
            failed_deletes.append(file_path)
            print(f"❌ 删除文件失败 {file_path}: {e}")
    
    print(f"📊 删除结果: 成功 {len(deleted_files)} 个，失败 {len(failed_deletes)} 个")
    
    return {
        "deleted_files": deleted_files,
        "failed_deletes": failed_deletes
    }

def analyze_single_file(file_path: str) -> tuple[str, str, str]:
    """Analyze a single file and return (file_path, classification, file_name)"""   
    try:
        source_path = Path(file_path)
        print(f"🔍 正在分析文件: {source_path.name}")
        
        if not source_path.exists():
            print(f"❌ 文件不存在: {file_path}")
            return file_path, "irrelevant", source_path.name
        
        # Read file content for analysis
        file_content = source_path.read_text(encoding='utf-8')
        # Truncate content for analysis (to avoid token limits)
        analysis_content = file_content[:5000] if len(file_content) > 2000 else file_content
        
        # Create individual analysis prompt for this file
        system_prompt = f"""你是一个表格生成智能体，需要分析用户上传的文件内容是不是一个包含有效数据集的表格文件，最直观的是用户上传了一个excel表格，并且并非模板表格，里面有具体的数据，此时将文件

        仔细检查不要把补充文件错误划分为模板文件反之亦然，补充文件里面是有数据的，模板文件里面是空的，或者只有一两个例子数据
        注意：所有文件已转换为txt格式，表格以HTML代码形式呈现，请根据内容而非文件名或后缀判断。

        当前分析文件:
        文件名: {source_path.name}
        文件路径: {file_path}
        文件内容:
        {analysis_content}

        请严格按照以下JSON格式回复，只返回这一个文件的分类结果（不要添加任何其他文字），不要将返回内容包裹在```json```中：
        {{
            "classification": "irrelevant" | "table"
        }}"""
        
        # Get LLM analysis for this file
        print("📤 正在调用LLM进行文件分类...")
        analysis_response = invoke_model(model_name="deepseek-ai/DeepSeek-V3", messages=[SystemMessage(content=system_prompt)])

        # Parse JSON response for this file
        try:
            # Extract JSON from response
            response_content = analysis_response.strip()
            print(f"📥 LLM分类响应: {response_content}")
            
            # Remove markdown code blocks if present
            if response_content.startswith('```'):
                response_content = response_content.split('\n', 1)[1]
                response_content = response_content.rsplit('\n', 1)[0]
            
            file_classification = json.loads(response_content)
            classification_type = file_classification.get("classification", "irrelevant")
            
            print(f"✅ 文件 {source_path.name} 分类为: {classification_type}")
            return file_path, classification_type, source_path.name
            
        except json.JSONDecodeError as e:
            print(f"❌ 文件 {source_path.name} JSON解析错误: {e}")
            print(f"LLM响应: {analysis_response}")
            # Fallback: mark as irrelevant for safety
            return file_path, "irrelevant", source_path.name
        
    except Exception as e:
        print(f"❌ 处理文件出错 {file_path}: {e}")
        # Return irrelevant on error
        return file_path, "irrelevant", Path(file_path).name