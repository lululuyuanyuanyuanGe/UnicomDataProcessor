import sys
from pathlib import Path
import os
import stat

# Add root project directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


from typing import Dict, TypedDict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import shutil
from utils.modelRelated import invoke_model4extract_excel_headers
from utils.file_process import (    delete_files_from_staging_area,
                                    detect_and_process_file_paths)
from utils.process_links import download_and_rename_from_dict
from src.similarity_calculation import TableSimilarityCalculator
from utils.table_processing_helpers import (
    extract_headers_from_response, 
    extract_headers_from_txt_content,
    parse_llm_table_response,
    generate_fallback_table_name,
    create_table_description
)

import json
import re

from langgraph.graph import StateGraph, END, START
from langgraph.types import Send


class FileProcessState(TypedDict):
    session_id: str
    upload_files_path: list[dict] # Store all uploaded files with timestamps: [{"path": "file_path", "timestamp": "iso_timestamp", "file_id": "unique_id"}]
    new_upload_files_path: list[dict] # Track the new uploaded files in this round with timestamps and file_id
    new_upload_files_processed_path: list[dict] # Store the processed new uploaded files with timestamps and file_id
    original_files_path: list[dict] # Store the original files in original_file subfolder with timestamps
    table_files_path: list[dict]  # Store table files with timestamps and file_id
    table_headers2embed: list[str]  # Change from str to list[str] to handle multiple tables
    table_header_embeddings: list[float]
    processed_table_results: list[dict]  # Store results from concurrent per-file processing
    replacement_info: dict  # Store info about files being replaced: {file_path: (clean_name, old_file_path)}
    village_name: str


class FileProcessAgent:

    def __init__(self):
        # Thread lock for safe JSON file updates
        self._json_lock = threading.Lock()
        self.graph = self._build_graph().compile()

    def _build_graph(self):
        graph = StateGraph(FileProcessState)

        graph.add_node("file_upload", self._file_upload)
        graph.add_node("process_table_and_similarity", self._process_table_and_similarity)
        graph.add_node("summary_file_upload", self._summary_file_upload)

        graph.add_edge(START, "file_upload")
        graph.add_edge("file_upload", "process_table_and_similarity")
        graph.add_edge("process_table_and_similarity", "summary_file_upload")
        graph.add_edge("summary_file_upload", END)

        return graph

    def _create_initial_state(self, session_id: str = "1", upload_files_data: dict[str, str] = {}, village_name: str = "") -> FileProcessState:
        # Convert input file paths to dictionary format with timestamps
        # upload_files_data format: {file_path: file_id}
        current_timestamp = datetime.now().isoformat()
        upload_files_with_timestamps = [
            {"path": file_path, "timestamp": current_timestamp, "file_id": file_id} 
            for file_path, file_id in upload_files_data.items()
        ]
        
        return {
            "session_id": session_id,
            "upload_files_path": upload_files_with_timestamps,
            "new_upload_files_path": [],
            "new_upload_files_processed_path": [],
            "original_files_path": [],
            "uploaded_template_files_path": [],
            "table_files_path": [],
            "table_headers2embed": [],  # Changed from str to list[str]
            "table_header_embeddings": [],
            "processed_table_results": [],  # New field for concurrent processing results
            "replacement_info": {},  # Store replacement information
            "template_complexity": "",
            "village_name": village_name
        }

    def _file_upload(self, state: FileProcessState) -> FileProcessState:
        """This node will upload user's file to our system with concurrent processing"""
        print("\n🔍 开始执行: _file_upload")
        print("=" * 50)
        
        print("📁 正在检测用户输入中的文件路径...")
        uploaded_files_path = state["upload_files_path"]
        print(f"📋 检测到 {len(uploaded_files_path)} 个文件")
        
        if not uploaded_files_path:
            print("⚠️ 没有文件需要上传")
            print("✅ _file_upload 执行完成")
            print("=" * 50)
            return {
                "new_upload_files_path": [],
                "new_upload_files_processed_path": [],
                "table_files_path": []
            }
        
        # Extract file paths from the dictionary structure
        file_paths = [file_entry["path"] for file_entry in uploaded_files_path]
        
        # Create staging area (temp folder)
        project_root = Path.cwd()
        temp_dir = project_root / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 开始并发处理 {len(file_paths)} 个文件...")
        
        # Process files concurrently for LLM analysis
        processed_files = []
        max_workers = min(len(file_paths), 5)  # Limit concurrent processing
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file processing tasks
            future_to_file = {
                executor.submit(self._process_single_file_for_llm, file_path): file_path
                for file_path in file_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    processed_file_path = future.result()
                    if processed_file_path:
                        processed_files.append(processed_file_path)
                        print(f"✅ 文件处理完成: {Path(file_path).name} -> {Path(processed_file_path).name}")
                    else:
                        print(f"❌ 文件处理失败: {Path(file_path).name}")
                except Exception as e:
                    print(f"❌ 并发处理文件时发生错误 {file_path}: {e}")
        
        # Create processed files with timestamps and preserve file_id
        current_timestamp = datetime.now().isoformat()
        processed_files_with_timestamps = []
        for file_path in processed_files:
            # Find the corresponding file_id from the original uploaded files
            file_id = None
            for uploaded_file_entry in uploaded_files_path:
                # Match by filename stem since processed files may have different extensions
                uploaded_file_stem = Path(uploaded_file_entry["path"]).stem
                processed_file_stem = Path(file_path).stem
                if uploaded_file_stem == processed_file_stem:
                    file_id = uploaded_file_entry.get("file_id")
                    break
            
            processed_files_with_timestamps.append({
                "path": file_path,
                "timestamp": current_timestamp,
                "file_id": file_id
            })
        
        print(f"🎉 文件上传处理完成:")
        print(f"  - 输入文件数: {len(file_paths)}")
        print(f"  - 成功处理: {len(processed_files)} 个")
        print(f"  - 失败处理: {len(file_paths) - len(processed_files)} 个")
        print("✅ _file_upload 执行完成")
        print("=" * 50)

        return {
            "new_upload_files_path": uploaded_files_path,  # Keep original input files 
            "new_upload_files_processed_path": processed_files_with_timestamps,  # LLM-ready processed files
            "table_files_path": processed_files_with_timestamps  # Treat all processed files as table files
        }

    def _process_single_file_for_llm(self, file_path: str) -> str | None:
        """
        Process a single file for LLM analysis capability.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            str: Path to the processed file in temp folder, or None if failed
        """
        try:
            from utils.file_process import process_file_for_LLM_capability
            return process_file_for_LLM_capability(file_path)
        except Exception as e:
            print(f"❌ 单文件LLM处理失败 {file_path}: {e}")
            return None


    
    def _process_table_and_similarity(self, state: FileProcessState) -> FileProcessState:
        """Process all table files concurrently, each going through full pipeline"""
        print("\n🔍 开始执行: _process_table_and_similarity (并发处理)")
        print("=" * 50)
        
        table_files_with_timestamps = state["table_files_path"]
        
        print(f"📊 需要并发处理的表格文件: {len(table_files_with_timestamps)} 个")
        
        if not table_files_with_timestamps:
            print("⚠️ 没有表格文件需要处理")
            print("✅ _process_table_and_similarity 执行完成")
            print("=" * 50)
            return {"processed_table_results": [], "table_headers2embed": []}
        
        # Use ThreadPoolExecutor for concurrent per-file processing
        max_workers = min(len(table_files_with_timestamps), 5)  # Limit to avoid overwhelming resources
        all_results = []
        
        print(f"🚀 开始并发处理，使用 {max_workers} 个工作线程")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit pipeline for each file
            future_to_file = {
                executor.submit(self.process_table_pipeline, file_entry, state): file_entry 
                for file_entry in table_files_with_timestamps
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_entry = future_to_file[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    print(f"✅ 完成处理: {result.get('chinese_table_name', 'Unknown')}")
                except Exception as e:
                    print(f"❌ 处理失败 {file_entry['path']}: {e}")
                    # Add failed entry to results
                    all_results.append({
                        "file_path": file_entry.get("path", ""),
                        "chinese_table_name": f"处理失败_{Path(file_entry['path']).stem}",
                        "success": False,
                        "error": str(e)
                    })
        
        # Generate summary data
        successful_results = [r for r in all_results if r.get("success", False)]
        table_descriptions = [r["table_description"] for r in successful_results if r.get("table_description")]
        
        print(f"\n📊 并发处理总结:")
        print(f"  - 总文件数: {len(table_files_with_timestamps)}")
        print(f"  - 成功处理: {len(successful_results)}")
        print(f"  - 失败处理: {len(all_results) - len(successful_results)}")
        print(f"  - 生成表格描述: {len(table_descriptions)}")
        
        # Show successful table names
        if successful_results:
            print("✅ 成功处理的表格:")
            for result in successful_results:
                chinese_name = result.get("chinese_table_name", "Unknown")
                header_count = len(result.get("headers", []))
                print(f"  - {chinese_name}: {header_count} 个表头")
        
        print("✅ _process_table_and_similarity 执行完成")
        print("=" * 50)
        
        return {
            "processed_table_results": all_results,
            "table_headers2embed": table_descriptions  # List of table descriptions
        }


    # DISABLED: Uploaded files should not interact with data.json
    # They should only be stored in src/uploaded_files.json
    # def append_table_data_to_json(self, file_name: str, headers: list[str], full_response: str, village_name: str):
    #     """
    #     [DISABLED] Append table data to data.json file with proper structure
    #     Uploaded files now only use src/uploaded_files.json
    #     """
    #     pass
    
    def load_uploaded_files_json(self) -> Dict:
        """Load existing uploaded_files.json file with thread safety"""
        # Use absolute path to ensure we find the file regardless of working directory
        project_root = Path(__file__).resolve().parent.parent
        uploaded_files_json_path = project_root / "src" / "uploaded_files.json"
        
        with self._json_lock:
            try:
                if uploaded_files_json_path.exists():
                    with open(uploaded_files_json_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    return []
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"⚠️ 读取uploaded_files.json失败: {e}，创建新的数据结构")
                return []

    def save_uploaded_files_json(self, data: Dict):
        """Save data to uploaded_files.json file with thread safety"""
        # Use absolute path to ensure we find the file regardless of working directory
        project_root = Path(__file__).resolve().parent.parent
        uploaded_files_json_path = project_root / "src" / "uploaded_files.json"
        
        with self._json_lock:
            try:
                with open(uploaded_files_json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print("✅ 成功更新uploaded_files.json")
            except Exception as e:
                print(f"❌ 保存uploaded_files.json失败: {e}")

    def save_original_file_to_uploads(self, file_path: str, chinese_name: str) -> str:
        """Move original file to uploaded_files/ directory with Chinese name, handling replacements"""
        try:
            source_path = Path(file_path)
            
            # Create uploaded_files directory if it doesn't exist
            uploads_dir = Path("uploaded_files")
            uploads_dir.mkdir(exist_ok=True)
            
            file_extension = source_path.suffix
            
            # ALWAYS use clean name without timestamp - no more timestamped files
            new_filename = f"{chinese_name}{file_extension}"
            dest_path = uploads_dir / new_filename
            
            # Check if file already exists and log replacement
            if dest_path.exists():
                print(f"🔄 替换现有文件: {dest_path.name}")
            else:
                print(f"📁 保存新文件: {dest_path.name}")
            
            # Copy/replace the file (shutil.copy will overwrite existing files)
            print(f"📋 Source path: {source_path}")
            os.chmod(source_path, stat.S_IWRITE)
            shutil.copy(source_path, dest_path)

            print(f"✅ 文件已保存到: {dest_path.name} (无时间戳)")
            return str(dest_path)
            
        except Exception as e:
            print(f"❌ 移动文件到uploaded_files失败: {e}")
            return file_path  # Return original path if move fails


    def update_uploaded_files_json(self, chinese_table_name: str, table_data: dict):
        """Update uploaded_files.json with table data (thread-safe)"""
        try:
            # Load existing data
            data = self.load_uploaded_files_json()
            
            # Find the index of the existing table entry
            # Priority: First try to find entry without file_id, then any matching entry
            index = None
            for i, item in enumerate(data):
                if item.get("excel_name") == chinese_table_name:
                    # If we find an entry without file_id, prefer to replace that one
                    if item.get("file_id") is None:
                        index = i
                        break
                    # If we haven't found a better match yet, use this one
                    elif index is None:
                        index = i
            
            if index is not None:
                # Update existing table data
                print(f"🔄 替换现有表格数据: {chinese_table_name} (索引: {index})")
                data[index] = table_data
            else:
                # Add new table data
                print(f"📊 添加新表格数据: {chinese_table_name}")
                data.append(table_data)
            
            # Save back to file
            self.save_uploaded_files_json(data)
            
            print(f"✅ 已更新uploaded_files.json - 表格: {chinese_table_name}")
            
        except Exception as e:
            print(f"❌ 更新uploaded_files.json失败: {e}")

    def process_single_table_file(self, file_entry: dict, state: FileProcessState) -> dict:
        """Process one table file (header extraction and file management)"""
        file_path = file_entry["path"]
        file_timestamp = file_entry["timestamp"]
        file_id = file_entry.get("file_id")
        
        try:
            source_path = Path(file_path)
            print(f"🔍 正在处理单个表格文件: {source_path.name}")
            
            # Find corresponding original Excel file in temp folder
            table_file_stem = source_path.stem
            original_excel_file = None
            
            # Look for original Excel file in temp folder by exact name match
            temp_dir = Path("temp")
            if temp_dir.exists():
                # Search for original Excel file with same stem but different extensions
                for potential_file in temp_dir.glob(f"{table_file_stem}.*"):
                    if potential_file.suffix.lower() in {'.xlsx', '.xls', '.xlsm'}:
                        original_excel_file = potential_file
                        print(f"🔍 在temp文件夹找到原始Excel文件: {original_excel_file.name}")
                        break
            
            # Fallback: check original_files_path from state (legacy)
            if not original_excel_file:
                original_files_with_timestamps = state.get("original_files_path", [])
                for original_file_entry in original_files_with_timestamps:
                    original_file_path = original_file_entry["path"]
                    if Path(original_file_path).stem == table_file_stem:
                        original_excel_file = Path(original_file_path)
                        print(f"🔍 在state中找到原始Excel文件: {original_excel_file.name}")
                        break
            
            headers = []
            analysis_response = ""
            
            # Extract headers and table name using LLM
            chinese_table_name = ""
            try:
                if original_excel_file and original_excel_file.exists():
                    print(f"🔍 找到原始Excel文件: {original_excel_file}")
                    # Use screenshot-based analysis to extract headers and table name
                    print("📤 正在调用LLM提取表格信息...")
                    analysis_response = invoke_model4extract_excel_headers(
                        model_name="deepseek-ai/DeepSeek-V3", 
                        file_path=str(original_excel_file)
                    )
                    print("📥 表格信息提取响应接收成功")
                    
                    # Parse the LLM response to get table name and headers
                    llm_result = parse_llm_table_response(analysis_response)
                    
                    if llm_result["success"]:
                        # Use LLM provided table name and headers
                        chinese_table_name = llm_result["table_name"] or generate_fallback_table_name(source_path.name)
                        headers = llm_result["headers"]
                        print(f"✅ LLM提取成功 - 表格名: {chinese_table_name}, 表头数: {len(headers)}")
                    else:
                        # Fallback to old method
                        headers = extract_headers_from_response(analysis_response)
                        chinese_table_name = generate_fallback_table_name(source_path.name)
                        print(f"⚠️ LLM结构化提取失败，使用后备方法 - 表格名: {chinese_table_name}")
                    
                else:
                    print(f"⚠️ 未找到对应的原始Excel文件: {table_file_stem}")
                    # Fallback: try to extract from txt content
                    file_content = source_path.read_text(encoding='utf-8')
                    headers = extract_headers_from_txt_content(file_content, source_path.name)
                    chinese_table_name = generate_fallback_table_name(source_path.name)
                    analysis_response = f"从文本内容提取 - 表格名: {chinese_table_name}, 表头: {headers}"
                    
            except Exception as llm_error:
                print(f"❌ 表格信息提取失败: {llm_error}")
                # Fallback: try to extract from txt content
                try:
                    file_content = source_path.read_text(encoding='utf-8')
                    headers = extract_headers_from_txt_content(file_content, source_path.name)
                    chinese_table_name = generate_fallback_table_name(source_path.name)
                    analysis_response = f"提取失败回退 - 表格名: {chinese_table_name}, 表头: {headers}"
                except Exception as e:
                    print(f"❌ 所有提取方法都失败: {e}")
                    headers = []
                    chinese_table_name = generate_fallback_table_name(source_path.name)
                    analysis_response = f"完全失败 - 表格名: {chinese_table_name}, 错误: {str(e)}"
            
            # Move original file to uploaded_files directory
            new_file_path = ""
            if original_excel_file:
                # Check if this is a replacement operation
                replacement_info = state.get("replacement_info", {})
                replacement_mode = False
                old_file_path = ""
                
                # Find if this file is being replaced
                for file_key, (clean_name, old_path) in replacement_info.items():
                    if Path(file_key).stem == table_file_stem:
                        original_file_key = file_key
                        replacement_mode = True
                        old_file_path = old_path
                        break
                
                new_file_path = self.save_original_file_to_uploads(
                    str(original_excel_file), 
                    chinese_table_name
                )
            
            # Generate table description for embedding
            table_description = create_table_description(chinese_table_name, headers)
            
            # Return processed file data
            return {
                "file_path": file_path,
                "timestamp": file_timestamp,
                "file_id": file_id,
                "chinese_table_name": chinese_table_name,
                "headers": headers,
                "original_file_path": new_file_path,
                "table_description": table_description,
                "analysis_response": analysis_response,
                "success": True
            }
            
        except Exception as e:
            print(f"❌ 处理单个表格文件失败 {file_path}: {e}")
            return {
                "file_path": file_path,
                "timestamp": file_entry.get("timestamp", ""),
                "file_id": file_entry.get("file_id"),
                "chinese_table_name": f"处理失败_{Path(file_path).stem}",
                "headers": [],
                "original_file_path": "",
                "table_description": "",
                "analysis_response": f"处理失败: {str(e)}",
                "success": False
            }

    def select_similarity_for_single_table(self, table_data: dict) -> dict:
        """Find similarity matches for a single processed table"""
        try:
            table_description = table_data["table_description"]
            chinese_table_name = table_data["chinese_table_name"]
            
            print(f"🔍 正在计算相似度: {chinese_table_name}")
            
            if not table_description:
                print("⚠️ 没有表格描述，跳过相似度计算")
                table_data["similarity_match"] = {
                    "top_matches": [],
                    "best_match": None,
                    "similarity_scores": []
                }
                return table_data
            
            # Initialize similarity calculator
            calculator = TableSimilarityCalculator()
            
            # Find best matching tables
            results = calculator.get_best_matches(table_description, top_n=5)
            
            if results['success']:
                print(f"✅ 相似度计算完成: {chinese_table_name}")
                table_data["similarity_match"] = {
                    "top_matches": results.get('matches', []),
                    "best_match": results.get('top_match'),
                }
            else:
                print(f"❌ 相似度计算失败: {results.get('error', '未知错误')}")
                table_data["similarity_match"] = {
                    "top_matches": [],
                    "best_match": None,
                    "error": results.get('error', '相似度计算失败')
                }
                
            return table_data
            
        except Exception as e:
            print(f"❌ 相似度计算异常: {e}")
            table_data["similarity_match"] = {
                "top_matches": [],
                "best_match": None,
                "error": f"相似度计算异常: {str(e)}"
            }
            return table_data

    def process_table_pipeline(self, file_entry: dict, state: FileProcessState) -> dict:
        """Complete pipeline for one table file: process → similarity → save"""
        try:
            print(f"\n🚀 开始处理表格管道: {Path(file_entry['path']).name}")
            # Step 1: Process table file (header extraction)
            table_data = self.process_single_table_file(file_entry, state)
            
            if not table_data["success"]:
                print(f"❌ 表格处理失败，跳过相似度计算")
                return table_data
            
            # Step 2: Calculate similarity
            complete_data = self.select_similarity_for_single_table(table_data)
            
            # Step 3: Save to uploaded_files.json
            chinese_table_name = complete_data["chinese_table_name"]
            
            # Prepare data structure for JSON
            json_entry = {
                "excel_name": chinese_table_name,
                "timestamp": complete_data["timestamp"],
                "file_id": complete_data.get("file_id"),
                "headers": complete_data["headers"],
                "original_file_path": complete_data["original_file_path"],
                "table_description": complete_data["table_description"],
                "similarity_match": complete_data["similarity_match"]
            }
            
            # Save to JSON
            self.update_uploaded_files_json(chinese_table_name, json_entry)
            
            print(f"✅ 完整管道处理完成: {chinese_table_name}")
            return complete_data
            
        except Exception as e:
            print(f"❌ 表格管道处理失败: {e}")
            return {
                "file_path": file_entry.get("path", ""),
                "timestamp": file_entry.get("timestamp", ""),
                "file_id": file_entry.get("file_id"),
                "chinese_table_name": f"管道失败_{datetime.now().strftime('%H%M%S')}",
                "headers": [],
                "original_file_path": "",
                "table_description": "",
                "analysis_response": f"管道处理失败: {str(e)}",
                "similarity_match": {"error": str(e)},
                "success": False
            }

    
    def _summary_file_upload(self, state: FileProcessState) -> FileProcessState:
        """Summary node for file upload process"""
        
        
        print("\n🔍 开始执行: _summary_file_upload")
        print("=" * 50)
        
        # Log the final state summary
        print("📊 文件处理总结:")
        print(f"  - 上传文件总数: {len(state.get('upload_files_path', []))}")
        print(f"  - 新上传文件数: {len(state.get('new_upload_files_path', []))}")
        print(f"  - 表格文件数: {len(state.get('table_files_path', []))}")
        
        # Show concurrent processing results
        processed_results = state.get('processed_table_results', [])
        if processed_results:
            successful_results = [r for r in processed_results if r.get("success", False)]
            print(f"  - 并发处理表格数: {len(processed_results)}")
            print(f"  - 成功处理: {len(successful_results)}")
            print(f"  - 失败处理: {len(processed_results) - len(successful_results)}")
            
            if successful_results:
                total_headers = sum(len(r.get("headers", [])) for r in successful_results)
                print(f"  - 总表头数: {total_headers}")
                
                print("📋 处理成功的表格:")
                for result in successful_results:
                    chinese_name = result.get("chinese_table_name", "Unknown")
                    header_count = len(result.get("headers", []))
                    has_similarity = bool(result.get("similarity_match", {}).get("best_match"))
                    similarity_status = "✅ 有相似匹配" if has_similarity else "⚠️ 无相似匹配"
                    print(f"    - {chinese_name}: {header_count} 个表头, {similarity_status}")
        
        # Show table descriptions for embedding
        table_descriptions = state.get('table_headers2embed', [])
        if table_descriptions:
            print(f"  - 生成嵌入描述数: {len(table_descriptions)}")
            
        print("\n📁 文件存储信息:")
        print("  - 原始文件位置: uploaded_files/ 目录")
        print("  - 处理结果存储: src/uploaded_files.json")
        
        # Cleanup temp folder
        print("\n🧹 正在清理临时文件...")
        temp_dir = Path("temp")
        if temp_dir.exists():
            try:
                # Remove all files and subdirectories in temp folder
                shutil.rmtree(str(temp_dir), ignore_errors=True)
                temp_dir.mkdir(parents=True, exist_ok=True)  # Recreate empty temp folder
                print("✅ 临时文件夹已清理完成")
            except Exception as e:
                print(f"⚠️ 清理临时文件夹失败: {e}")
        else:
            print("⚠️ 临时文件夹不存在，跳过清理")
        
        print("✅ _summary_file_upload 执行完成")
        print("=" * 50)
        
        return {**state}

    def run_file_process_agent(self, session_id: str = "1", upload_files_data: dict[str, str] = {}, village_name: str = "ChatBI") -> FileProcessState:
        """Driver to run the process file agent"""
        print("\n🚀 开始运行 FileProcessAgent")
        print("=" * 60)
        renamed_files = download_and_rename_from_dict(upload_files_data)
        initial_state = self._create_initial_state(session_id = session_id, upload_files_data = renamed_files, village_name = village_name)
        config = {"configurable": {"thread_id": session_id}}

        print(f"📋 会话ID: {session_id}")
        print(f"📝 初始状态已创建")
        print("🔄 正在执行文件处理工作流...")

        try:
            final_state = self.graph.invoke(initial_state, config=config)

            print("\n🎉 FileProcessAgent 执行完成！")
            print("=" * 60)
            print("📊 最终结果:")
            print(f"- 上传文件数量: {len(final_state.get('upload_files_path', []))}")
            print(f"- 新上传文件数量: {len(final_state.get('new_upload_files_path', []))}")
            print(f"- 新上传文件已处理数量: {len(final_state.get('new_upload_files_processed_path', []))}")
            print(f"- 原始文件数量: {len(final_state.get('original_files_path', []))}")
            print(f"- 表格文件数量: {len(final_state.get('table_files_path', []))}")

            return final_state
        
        except Exception as e:
            print(f"❌ 执行过程中发生错误: {e}")
            return initial_state
if __name__ == "__main__":
    # upload_files_path = input("请输入上传文件路径: ")
    # upload_files_path_list = detect_and_process_file_paths(upload_files_path)
    # # Convert list to dict format with dummy file_ids for CLI usage
    # upload_files_data = {path: f"file_{i}" for i, path in enumerate(upload_files_path_list)}
    agent = FileProcessAgent()
    agent.run_file_process_agent(upload_files_data = {"http://58.144.196.118:5019/ai-index/atts/images/QA_20250805124450569_党员信息.xlsx": "file_001"})
