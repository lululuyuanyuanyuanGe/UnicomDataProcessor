from __future__ import annotations
from bs4 import BeautifulSoup
from pathlib import Path
import re
import os
import json
from pathlib import Path
import subprocess
import chardet
import shutil
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


def process_file_for_LLM_capability(file_path: str | Path) -> str | None:
    """
    Process uploaded files to formats suitable for LLM analysis.
    
    This function converts different file types to LLM-analyzable formats:
    - Spreadsheets (Excel, CSV) → CSV format for tabular data analysis
    - Documents (Word, DOCX) → Plain text for content analysis  
    - Text files → As-is (already LLM-ready)
    - Other formats → Metadata or error handling
    
    The processed files are saved in the temp/ folder with appropriate extensions.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        str: Path to the processed file in temp folder, or None if processing failed
    """
    source_path = Path(file_path)
    
    if not source_path.exists():
        print(f"❌ File does not exist: {file_path}")
        return None
        
    file_extension = source_path.suffix.lower()
    
    # Create temp directory if it doesn't exist
    temp_dir = Path("temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Define file type categories
    spreadsheet_extensions = {'.xlsx', '.xls', '.xlsm', '.ods', '.csv'}
    text_extensions = {'.txt', '.md', '.json', '.xml', '.html', '.htm', '.py', '.js', '.css', '.sql', '.log'}
    document_extensions = {'.docx', '.doc', '.pptx', '.ppt'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg'}
    
    try:
        print(f"📄 Processing file for LLM analysis: {source_path.name}")
        
        # Handle spreadsheet files - convert to CSV for LLM analysis
        if file_extension in spreadsheet_extensions:
            return _process_spreadsheet(source_path, temp_dir)
        
        # Handle document files - extract text content
        elif file_extension in document_extensions:
            return _process_doc_file(source_path, temp_dir)
        
        # Handle plain text files - copy as-is since they're already LLM-ready
        elif file_extension in text_extensions:
            return _copy_text_file(source_path, temp_dir)
        
        # Handle PDF files
        elif file_extension == '.pdf':
            return _process_pdf_file(source_path, temp_dir)
            
        # Handle image files - create metadata file
        elif file_extension in image_extensions:
            return _process_image_file(source_path, temp_dir)
            
        # Handle other file types
        else:
            print(f"⚠️ Unsupported file type: {file_extension}")
            return _create_unsupported_file_info(source_path, temp_dir)
                
    except Exception as e:
        print(f"❌ Error processing file for LLM capability {file_path}: {e}")
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

def _process_spreadsheet(source_path: Path, temp_dir: Path) -> str | None:
    """
    Convert Excel/spreadsheet files to CSV format for LLM analysis.
    Also saves a copy of the original Excel file in temp folder with original name.
    
    Args:
        source_path: Path to the source spreadsheet file
        temp_dir: Directory to save the processed file
        
    Returns:
        str: Path to the created CSV file in temp folder, or None if failed
    """
    try:
        print(f"📊 Converting spreadsheet to CSV: {source_path.name}")
        
        # Generate output filenames
        base_name = source_path.stem
        csv_output_file = temp_dir / f"{base_name}.csv"
        
        # Save copy of original file to temp folder with original name
        if source_path.suffix.lower() != '.csv':
            original_copy_path = temp_dir / source_path.name
            shutil.copy2(source_path, original_copy_path)
            print(f"📁 Original Excel file copied to temp: {original_copy_path.name}")
        
        # Read the spreadsheet file
        if source_path.suffix.lower() == '.csv':
            # If it's already CSV, just copy it
            df = pd.read_csv(source_path, encoding='utf-8')
        else:
            # For Excel files, read the first sheet
            df = pd.read_excel(source_path, sheet_name=0)
        
        # Save as CSV for LLM analysis
        df.to_csv(csv_output_file, index=False, encoding='utf-8')
        
        print(f"✅ Spreadsheet converted to CSV: {csv_output_file.name}")
        return str(csv_output_file)
        
    except Exception as e:
        print(f"❌ Failed to process spreadsheet {source_path.name}: {e}")
        return None

def _process_doc_file(source_path: Path, temp_dir: Path) -> str | None:
    """
    Extract text content from Word documents using mammoth library.
    
    Args:
        source_path: Path to the source document file
        temp_dir: Directory to save the processed file
        
    Returns:
        str: Path to the created text file in temp folder, or None if failed
    """
    try:
        print(f"📄 Extracting text from document: {source_path.name}")
        
        # Generate output filename
        base_name = source_path.stem
        output_file = temp_dir / f"{base_name}.txt"
        
        # Check file extension
        if source_path.suffix.lower() == '.docx':
            # Use mammoth for DOCX files
            import mammoth
            with open(source_path, "rb") as docx_file:
                result = mammoth.extract_raw_text(docx_file)
                text_content = result.value

        elif source_path.suffix.lower() == '.doc':
            # For .doc files, fall back to LibreOffice conversion
            print("⚠️ 请先将 .doc 文件转换成 docx文件")
            return None
        else:
            print(f"⚠️ Unsupported document format: {source_path.suffix}")
            return None
        
        # Save extracted text content
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        print(f"✅ Document text extracted: {output_file.name}")
        return str(output_file)
        
    except Exception as e:
        print(f"❌ Failed to process document {source_path.name}: {e}")
        return None

def _process_pdf_file(source_path: Path, temp_dir: Path) -> str | None:
    """
    Process PDF files - placeholder for future implementation.
    
    Args:
        source_path: Path to the source PDF file
        temp_dir: Directory to save the processed file
        
    Returns:
        None (not implemented yet)
    """
    print(f"⚠️ PDF processing not yet implemented: {source_path.name}")
    return None

def _copy_text_file(source_path: Path, temp_dir: Path) -> str | None:
    """
    Copy text files as-is since they're already LLM-ready.
    
    Args:
        source_path: Path to the source text file
        temp_dir: Directory to save the processed file
        
    Returns:
        str: Path to the copied file in temp folder, or None if failed
    """
    try:
        print(f"📝 Copying text file: {source_path.name}")
        
        # Generate output filename
        base_name = source_path.stem
        extension = source_path.suffix
        output_file = temp_dir / f"{base_name}{extension}"
        
        # Read and copy the text file
        content = _read_text_auto(source_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Text file copied: {output_file.name}")
        return str(output_file)
        
    except Exception as e:
        print(f"❌ Failed to copy text file {source_path.name}: {e}")
        return None

def _process_image_file(source_path: Path, temp_dir: Path) -> str | None:
    """
    Create metadata file for image files.
    
    Args:
        source_path: Path to the source image file
        temp_dir: Directory to save the processed file
        
    Returns:
        str: Path to the created metadata file in temp folder, or None if failed
    """
    try:
        print(f"🖼️ Creating metadata for image: {source_path.name}")
        
        # Generate output filename
        base_name = source_path.stem
        output_file = temp_dir / f"{base_name}_metadata.txt"
        
        # Create metadata content
        file_size = source_path.stat().st_size
        metadata_content = f"Image file: {source_path.name}\nFile size: {file_size} bytes\nFormat: {source_path.suffix}"
        
        # Save metadata
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(metadata_content)
        
        print(f"✅ Image metadata created: {output_file.name}")
        return str(output_file)
        
    except Exception as e:
        print(f"❌ Failed to create image metadata {source_path.name}: {e}")
        return None

def _create_unsupported_file_info(source_path: Path, temp_dir: Path) -> str | None:
    """
    Create info file for unsupported file types.
    
    Args:
        source_path: Path to the source file
        temp_dir: Directory to save the processed file
        
    Returns:
        str: Path to the created info file in temp folder, or None if failed
    """
    try:
        print(f"❓ Creating info for unsupported file: {source_path.name}")
        
        # Generate output filename
        base_name = source_path.stem
        output_file = temp_dir / f"{base_name}_info.txt"
        
        # Try to detect MIME type
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(source_path))
        
        # Create info content
        file_size = source_path.stat().st_size
        info_content = f"Unsupported file: {source_path.name}\nFile size: {file_size} bytes\nExtension: {source_path.suffix}\nMIME type: {mime_type or 'unknown'}"
        
        # Save info
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(info_content)
        
        print(f"✅ Unsupported file info created: {output_file.name}")
        return str(output_file)
        
    except Exception as e:
        print(f"❌ Failed to create unsupported file info {source_path.name}: {e}")
        return None
    

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
        system_prompt = f"""
你是一个“表格数据识别”智能体。你的任务是判断给定文件内容是否为“包含实际数据的表格文件”。注意：所有文件已被转换为txt；若是表格，其内容以HTML代码形式呈现。你必须仅依据文件内容进行判断，不能根据文件名、后缀或路径推断。

严格分类规则
- 输出仅允许两类：
  - "table": 文件中包含“非样例/非占位符”的、成规模的实际数据表格（通常是HTML表格，且有多行记录和多个字段，字段值多样且非占位符）。
  - "irrelevant": 其他任何情况，包括模板、空表、仅说明性文字、目录、无结构文本、仅少量示例行、仅字段定义/占位符。
- 模板的精确定义（必须判为 "irrelevant"）：
  - 只有表头/字段名、或字段说明、或占位符（如“示例”“样例”“模板”“必填”“填写示例”“N/A”“请输入”），无真实批量数据。
  - 只有极少数（如≤2行）示例数据或演示行；或大量单元格为空/0/NA/—。
  - 重复性的占位值或格式说明（如“YYYY-MM-DD”“示例：张三”）。
  - 数据行中字段值大面积重复、明显非实际（如全是“示例”“test”“sample”“模板”“N/A”）。
- 真实表格（判为 "table"）的最低要求（需同时满足）：
  - 有明确表头与多行数据记录（通常≥3行实际数据，不含表头）。
  - 多字段列且数据值分布有多样性，非占位符/非样例说明。
  - 若为HTML表格，<table> 内存在多个 <tr> 数据行，且 <td> 含具体值（非空、非说明性文本）。

判定要点与启发式
- 聚焦 HTML 结构：<table>、<thead>、<tbody>、<tr>、<th>/<td>。没有表格结构且内容非结构化时，通常为 "irrelevant"。
- 模板信号：
  - 含“模板/样例/示例/示范/范本/填写说明/请填写/必填/选填/示例值/请输入/格式：YYYY-MM-DD”等。
  - 数据区大量为空、统一占位、或仅1–2行示例。
  - 字段说明性文本多于数据内容。
- 真实数据信号：
  - 多行记录且单元格值差异明显（不同人名/日期/金额/编号等）。
  - 数值、日期、文本混合且看起来真实。
- 若同时出现表头与少量（≤2）示例行，优先判为模板（"irrelevant"）。
- 不要因为存在 <table> 结构就直接判为 "table"；必须确认包含成规模真实数据。

输入
当前分析文件:
文件名: {source_path.name}
文件路径: {file_path}
文件内容:
{analysis_content}

输出格式（仅返回此JSON，禁止添加其他文字或注释）：
{{
    "classification": "irrelevant" | "table"
}}

在选择 "table" 前请核对：
- 数据行数 ≥ 3（不含表头/汇总行/空行）
- 单元格值非占位符且具有明显多样性
- 非说明性或引导性文本占比不高

无法确定时，请保守选择 "irrelevant"。
"""

        
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