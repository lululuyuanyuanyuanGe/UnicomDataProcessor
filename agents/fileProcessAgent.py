import sys
from pathlib import Path

# Add root project directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


from typing import Dict, List, Optional, Any, TypedDict, Annotated
from datetime import datetime
from utils.modelRelated import invoke_model, invoke_model_with_screenshot
from utils.file_process import (retrieve_file_content, save_original_file,
                                    extract_filename, 
                                    ensure_location_structure, check_file_exists_in_data,
                                    get_available_locations, move_template_files_to_final_destination,
                                    move_supplement_files_to_final_destination, delete_files_from_staging_area,
                                    reconstruct_csv_with_headers, detect_and_process_file_paths)

import json

from langgraph.graph import StateGraph, END, START
from langgraph.constants import Send
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool


class FileProcessState(TypedDict):
    session_id: str
    upload_files_path: list[str] # Store all uploaded files
    new_upload_files_path: list[str] # Track the new uploaded files in this round
    new_upload_files_processed_path: list[str] # Store the processed new uploaded files
    original_files_path: list[str] # Store the original files in original_file subfolder
    table_files_path: list[str]
    irrelevant_files_path: list[str]
    irrelevant_original_files_path: list[str] # Track original files to be deleted with irrelevant files
    all_files_irrelevant: bool  # Flag to indicate all files are irrelevant
    village_name: str


class FileProcessAgent:


    def __init__(self):
        self.memory = MemorySaver()
        self.graph = self._build_graph().compile(checkpointer=self.memory)

    def _build_graph(self):
        graph = StateGraph(FileProcessState)

        graph.add_node("file_upload", self._file_upload)
        graph.add_node("analyze_uploaded_files", self._analyze_uploaded_files)
        graph.add_node("route_after_analyze_uploaded_files", self._route_after_analyze_uploaded_files)
        graph.add_node("process_table", self._process_table)
        graph.add_node("process_irrelevant", self._process_irrelevant)
        graph.add_node("summary_file_upload", self._summary_file_upload)

        graph.add_edge(START, "file_upload")
        graph.add_edge("file_upload", "analyze_uploaded_files")
        graph.add_conditional_edges("analyze_uploaded_files", self._route_after_analyze_uploaded_files)
        graph.add_edge("process_table", "summary_file_upload")
        graph.add_edge("process_irrelevant", "summary_file_upload")
        graph.add_edge("summary_file_upload", END)

        return graph

    def _create_initial_state(self, session_id: str = "1", upload_files_path: list[str] = [], village_name: str = "") -> FileProcessState:
        return {
            "session_id": session_id,
            "upload_files_path": upload_files_path,
            "new_upload_files_path": [],
            "new_upload_files_processed_path": [],
            "original_files_path": [],
            "uploaded_template_files_path": [],
            "table_files_path": [],
            "irrelevant_files_path": [],
            "irrelevant_original_files_path": [],
            "all_files_irrelevant": False,
            "template_complexity": "",
            "village_name": village_name
        }


    def _file_upload(self, state: FileProcessState) -> FileProcessState:
            """This node will upload user's file to our system"""
            print("\n🔍 开始执行: _file_upload")
            print("=" * 50)
            
            print("📁 正在检测用户输入中的文件路径...")
            detected_files = state["upload_files_path"]
            print(f"📋 检测到 {len(detected_files)} 个文件")
            
            # Load data.json with error handling
            data_file = Path("agents/data.json")
            try:
                with open(data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"⚠️ data.json文件出错: {e}")
                # Initialize empty structure if file is missing or corrupted
                data = {}
            
            print("🔍 正在检查文件是否已存在...")
            files_to_remove = []
            for file in detected_files:
                file_name = Path(file).name
                if check_file_exists_in_data(data, file_name):
                    files_to_remove.append(file)
                    print(f"⚠️ 文件 {file} 已存在")
            
            # Remove existing files from detected_files
            for file in files_to_remove:
                detected_files.remove(file)
            
            if not detected_files:
                print("⚠️ 没有新文件需要上传")
                print("✅ _file_upload 执行完成")
                print("=" * 50)
                return {
                    "new_upload_files_path": [],
                    "new_upload_files_processed_path": []
                }
            
            print(f"🔄 正在处理 {len(detected_files)} 个新文件...")
            
            # Create staging area for original files
            project_root = Path.cwd()
            staging_dir = project_root / "conversations" / state["session_id"] / "user_uploaded_files"
            staging_dir.mkdir(parents=True, exist_ok=True)
            
            # Process the files to get .txt versions
            processed_files = retrieve_file_content(detected_files, state["session_id"])
            
            # Save original files separately
            original_files = []
            for file_path in detected_files:
                try:
                    source_path = Path(file_path)
                    original_file_saved_path = save_original_file(source_path, staging_dir)
                    if original_file_saved_path:
                        original_files.append(original_file_saved_path)
                        print(f"💾 原始文件已保存: {Path(original_file_saved_path).name}")
                    else:
                        print(f"⚠️ 原始文件保存失败: {source_path.name}")
                except Exception as e:
                    print(f"❌ 保存原始文件时出错 {file_path}: {e}")
            
            print(f"✅ 文件处理完成: {len(processed_files)} 个处理文件, {len(original_files)} 个原始文件")
            print("✅ _file_upload 执行完成")
            print("=" * 50)
            
            # Update state with new files
            # Safely handle the case where upload_files_path might not exist in state
            existing_files = state.get("upload_files_path", [])
            existing_original_files = state.get("original_files_path", [])
            print("detected_files 类型: ", type(detected_files))
            print("existing_files 类型: ", type(existing_files))
            print("existing_original_files 类型: ", type(existing_original_files))
            print("processed_files 类型: ", type(processed_files))
            print("original_files 类型: ", type(original_files))
            return {
                "new_upload_files_path": detected_files,
                "upload_files_path": existing_files + detected_files,
                "new_upload_files_processed_path": processed_files,
                "original_files_path": existing_original_files + original_files
            }
    


    def _analyze_uploaded_files(self, state: FileProcessState) -> FileProcessState:
        """This node will analyze the user's uploaded files, it need to classify the file into template
        supplement, or irrelevant. If all files are irrelevant, it will flag for text analysis instead."""
        
        print("\n🔍 开始执行: _analyze_uploaded_files")
        print("=" * 50)
        
        import json
        from pathlib import Path
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Initialize classification results
        classification_results = {
            "tables": [],
            "irrelevant": []
        }
        
        # Process files one by one for better accuracy
        processed_files = []
        # Safely handle the case where new_upload_files_processed_path might not exist in state
        new_files_to_process = state.get("new_upload_files_processed_path", [])
        
        print(f"📁 需要分析的文件数量: {len(new_files_to_process)}")
        
        if not new_files_to_process:
            print("⚠️ 没有找到可处理的文件")
            print("✅ _analyze_uploaded_files 执行完成")
            print("=" * 50)
            return {
                "table_files_path": [],
                "irrelevant_files_path": [],
                "all_files_irrelevant": True  # Flag for routing to text analysis
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
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(len(new_files_to_process), 5)  # Limit to 5 concurrent requests
        print(f"🚀 开始并行处理文件，使用 {max_workers} 个工作线程")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file analysis tasks
            future_to_file = {
                executor.submit(analyze_single_file, file_path): file_path 
                for file_path in new_files_to_process
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_path_result, classification_type, file_name = future.result()
                    
                    # Add to appropriate category
                    if classification_type == "table":
                        classification_results["tables"].append(file_path_result)
                    else:  # irrelevant or unknown
                        classification_results["irrelevant"].append(file_path_result)
                    
                    processed_files.append(file_name)
                    
                except Exception as e:
                    print(f"❌ 并行处理文件任务失败 {file_path}: {e}")
                    # Add to irrelevant on error
                    classification_results["irrelevant"].append(file_path)
        
        print(f"🎉 并行文件分析完成:")
        print(f"  - 表格文件: {len(classification_results['tables'])} 个")
        print(f"  - 无关文件: {len(classification_results['irrelevant'])} 个")
        print(f"  - 成功处理: {len(processed_files)} 个文件")
        
        if not processed_files and not classification_results["irrelevant"]:
            print("⚠️ 没有找到可处理的文件")
            print("✅ _analyze_uploaded_files 执行完成")
            print("=" * 50)
            return {
                "table_files_path": [],
                "irrelevant_files_path": [],
                "all_files_irrelevant": True  # Flag for routing to text analysis
            }
        
        # Update state with classification results
        table_files = classification_results.get("tables", [])
        irrelevant_files = classification_results.get("irrelevant", [])
        
        # Create mapping of processed files to original files to track irrelevant originals
        irrelevant_original_files = []
        if irrelevant_files:
            original_files = state.get("original_files_path", [])
            processed_files = state.get("new_upload_files_processed_path", [])
            
            print("🔍 正在映射无关文件对应的原始文件...")
            
            # Create mapping based on filename (stem)
            for irrelevant_file in irrelevant_files:
                irrelevant_file_stem = Path(irrelevant_file).stem
                # Find the corresponding original file
                for original_file in original_files:
                    original_file_stem = Path(original_file).stem
                    if irrelevant_file_stem == original_file_stem:
                        irrelevant_original_files.append(original_file)
                        print(f"📋 映射无关文件: {Path(irrelevant_file).name} -> {Path(original_file).name}")
                        break
        
        # Check if all files are irrelevant
        # Safely handle the case where new_upload_files_processed_path might not exist in state
        new_files_processed_count = len(state.get("new_upload_files_processed_path", []))
        all_files_irrelevant = len(irrelevant_files) == new_files_processed_count
        
        if all_files_irrelevant:
            print("⚠️ 所有文件都被分类为无关文件")
            print("✅ _analyze_uploaded_files 执行完成")
            print("=" * 50)
            return {
                "table_files_path": [],
                "irrelevant_files_path": irrelevant_files,
                "irrelevant_original_files_path": irrelevant_original_files,
                "all_files_irrelevant": True  # Flag for routing
            }
        else:
            # Some files are relevant, proceed with normal flow
            print("✅ 文件分析完成，存在有效文件")
            print(f"  - 表格文件: {len(table_files)} 个")
            print(f"  - 无关文件: {len(irrelevant_files)} 个")
            print("✅ _analyze_uploaded_files 执行完成")
            print("=" * 50)
            
            return {
                "table_files_path": table_files,
                "irrelevant_files_path": irrelevant_files,
                "irrelevant_original_files_path": irrelevant_original_files,
                "all_files_irrelevant": False  # Flag for routing
            }
                
    def _route_after_analyze_uploaded_files(self, state: FileProcessState):
        """Route after analyzing uploaded files. Uses Send objects for all routing."""
        print("Debug: route_after_analyze_uploaded_files")
        
        # Check if all files are irrelevant - route to cleanup
        if state.get("all_files_irrelevant", False):
            sends = []
            if state.get("irrelevant_files_path"):
                sends.append(Send("process_irrelevant", state))
            return sends if sends else [Send("summary_file_upload", state)]
        
        # Some files are relevant - process them in parallel
        sends = []
        if state.get("table_files_path"):
            sends.append(Send("process_table", state))
        if state.get("irrelevant_files_path"):
            sends.append(Send("process_irrelevant", state))

        # The parallel nodes will automatically converge, then continue to summary
        return sends if sends else [Send("summary_file_upload", state)]  # Fallback
    
    def _process_table(self, state: FileProcessState) -> FileProcessState:
        """This node will process the table files and extract headers from them"""
        print("\n🔍 开始执行: _process_table")
        print("=" * 50)
        
        table_files = state["table_files_path"]
        
        print(f"📊 需要处理的表格文件: {len(table_files)} 个")
        
        if not table_files:
            print("⚠️ 没有表格文件需要处理")
            print("✅ _process_table 执行完成")
            print("=" * 50)
            return {}
        
        # Store all extracted headers for summary
        all_extracted_headers = {}
        
        def extract_headers_from_response(response: str) -> list[str]:
            """Extract headers from LLM response"""
            try:
                # Try to parse JSON response first
                import re
                # Look for table structure in the response
                if '表格结构' in response:
                    # Extract field names from JSON structure
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        json_data = json.loads(json_match.group())
                        # Navigate through the JSON to find field names
                        table_structure = None
                        for key, value in json_data.items():
                            if isinstance(value, dict) and '表格结构' in value:
                                table_structure = value['表格结构']
                                break
                            elif key == '表格结构' and isinstance(value, dict):
                                table_structure = value
                                break
                        
                        if table_structure:
                            headers = list(table_structure.keys())
                            return headers
                
                # Fallback: extract from plain text
                lines = response.split('\n')
                headers = []
                for line in lines:
                    line = line.strip()
                    if ':' in line and not line.startswith('{') and not line.startswith('}'):
                        header = line.split(':')[0].strip('" ')
                        if header and len(header) < 50:  # Reasonable header length
                            headers.append(header)
                
                return headers[:20]  # Limit to first 20 headers
                
            except Exception as e:
                print(f"提取表头失败: {e}")
                return []
        
        def extract_headers_from_txt_content(content: str, filename: str) -> list[str]:
            """Extract headers from HTML content in txt file"""
            try:
                import re
                headers = []
                
                # Look for table headers in HTML
                th_matches = re.findall(r'<th[^>]*>(.*?)</th>', content, re.IGNORECASE | re.DOTALL)
                if th_matches:
                    for th in th_matches:
                        # Clean HTML tags
                        clean_header = re.sub(r'<[^>]+>', '', th).strip()
                        if clean_header and len(clean_header) < 50:
                            headers.append(clean_header)
                
                # If no th tags, look for first row cells
                if not headers:
                    td_matches = re.findall(r'<td[^>]*>(.*?)</td>', content, re.IGNORECASE | re.DOTALL)
                    if td_matches:
                        # Take first few cells as potential headers
                        for i, td in enumerate(td_matches[:10]):
                            clean_header = re.sub(r'<[^>]+>', '', td).strip()
                            if clean_header and len(clean_header) < 50:
                                headers.append(clean_header)
                
                return headers[:15]  # Limit to 15 headers
                
            except Exception as e:
                print(f"从文本内容提取表头失败: {e}")
                return []
        
        # Process each table file
        for table_file in table_files:
            try:
                source_path = Path(table_file)
                print(f"🔍 正在处理表格文件: {source_path.name}")
                
                # Find corresponding original Excel file
                table_file_stem = Path(table_file).stem
                original_files = state.get("original_files_path", [])
                original_excel_file = None
                
                for original_file in original_files:
                    if Path(original_file).stem == table_file_stem:
                        original_excel_file = Path(original_file)
                        break
                
                headers = []
                
                try:
                    if original_excel_file and original_excel_file.exists():
                        print(f"🔍 找到原始Excel文件: {original_excel_file}")
                        # Use screenshot-based analysis to extract headers
                        print("📤 正在调用LLM提取表格表头...")
                        analysis_response = invoke_model_with_screenshot(
                            model_name="Qwen/Qwen2.5-VL-72B-Instruct", 
                            file_path=str(original_excel_file)
                        )
                        print("📥 表头提取响应接收成功")
                        
                        # Extract headers from the response
                        headers = extract_headers_from_response(analysis_response)
                        
                    else:
                        print(f"⚠️ 未找到对应的原始Excel文件: {table_file_stem}")
                        # Fallback: try to extract from txt content
                        file_content = source_path.read_text(encoding='utf-8')
                        headers = extract_headers_from_txt_content(file_content, source_path.name)
                        
                except Exception as llm_error:
                    print(f"❌ 表头提取失败: {llm_error}")
                    # Fallback: try to extract from txt content
                    try:
                        file_content = source_path.read_text(encoding='utf-8')
                        headers = extract_headers_from_txt_content(file_content, source_path.name)
                    except Exception as e:
                        print(f"❌ 文本内容提取也失败: {e}")
                        headers = []
                
                # Store extracted headers
                all_extracted_headers[source_path.name] = headers
                print(f"✅ 表格文件已处理: {source_path.name} (提取到 {len(headers)} 个表头)")
                
                if headers:
                    print(f"📋 表头列表: {', '.join(headers[:5])}{'...' if len(headers) > 5 else ''}")
                
            except Exception as e:
                print(f"❌ 处理表格文件出错 {table_file}: {e}")
                all_extracted_headers[Path(table_file).name] = []
        
        # Print summary of extracted headers
        print(f"\n📊 表头提取总结:")
        total_headers = 0
        for filename, headers in all_extracted_headers.items():
            print(f"  - {filename}: {len(headers)} 个表头")
            total_headers += len(headers)
        print(f"  - 总计: {total_headers} 个表头从 {len(table_files)} 个文件中提取")
        
        print("✅ _process_table 执行完成")
        print("=" * 50)
        
        return {"extracted_headers": all_extracted_headers}
    
        
    def _process_irrelevant(self, state: FileProcessState) -> FileProcessState:
        """This node will process the irrelevant files, it will delete the irrelevant files (both processed and original) from the staging area"""
        
        print("\n🔍 开始执行: _process_irrelevant")
        print("=" * 50)
        
        irrelevant_files = state["irrelevant_files_path"]
        irrelevant_original_files = state.get("irrelevant_original_files_path", [])
        
        print(f"🗑️ 需要删除的无关处理文件数量: {len(irrelevant_files)}")
        print(f"🗑️ 需要删除的无关原始文件数量: {len(irrelevant_original_files)}")
        
        # Combine all files to delete
        all_files_to_delete = irrelevant_files + irrelevant_original_files
        
        if all_files_to_delete:
            delete_result = delete_files_from_staging_area(all_files_to_delete)
            
            deleted_count = len(delete_result["deleted_files"])
            failed_count = len(delete_result["failed_deletes"])
            
            print(f"📊 删除结果: 成功 {deleted_count} 个，失败 {failed_count} 个 (总计 {len(all_files_to_delete)} 个文件)")
            
            if delete_result["failed_deletes"]:
                print("❌ 删除失败的文件:")
                for failed_file in delete_result["failed_deletes"]:
                    print(f"  - {failed_file}")
        else:
            print("⚠️ 没有无关文件需要删除")
        
        print("✅ _process_irrelevant 执行完成")
        print("=" * 50)
        
        return {}  # Return empty dict since this node doesn't need to update any state keys

    
    def _summary_file_upload(self, state: FileProcessState) -> FileProcessState:
        """Summary node for file upload process"""
        
        print("\n🔍 开始执行: _summary_file_upload")
        print("=" * 50)
        
        # Log the final state summary
        print("📊 文件处理总结:")
        print(f"  - 上传文件总数: {len(state.get('upload_files_path', []))}")
        print(f"  - 新上传文件数: {len(state.get('new_upload_files_path', []))}")
        print(f"  - 表格文件数: {len(state.get('table_files_path', []))}")
        print(f"  - 无关文件数: {len(state.get('irrelevant_files_path', []))}")
        
        # Show extracted headers summary if available
        extracted_headers = state.get('extracted_headers', {})
        if extracted_headers:
            print(f"  - 提取表头文件数: {len(extracted_headers)}")
            total_headers = sum(len(headers) for headers in extracted_headers.values())
            print(f"  - 总表头数: {total_headers}")
        
        print("✅ _summary_file_upload 执行完成")
        print("=" * 50)
        
        return {}

    def run_file_process_agent(self, session_id: str = "1", upload_files_path: list[str] = [], village_name: str = "ChatBI") -> FileProcessState:
        """Driver to run the process file agent"""
        print("\n🚀 开始运行 FileProcessAgent")
        print("=" * 60)

        initial_state = self._create_initial_state(session_id = session_id, upload_files_path = upload_files_path, village_name = village_name)
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
            print(f"- 无关文件数量: {len(final_state.get('irrelevant_files_path', []))}")

            return final_state
        
        except Exception as e:
            print(f"❌ 执行过程中发生错误: {e}")
            return initial_state
if __name__ == "__main__":
    upload_files_path = input("请输入上传文件路径: ")
    upload_files_path = detect_and_process_file_paths(upload_files_path)
    agent = FileProcessAgent()
    agent.run_file_process_agent(upload_files_path = upload_files_path)
