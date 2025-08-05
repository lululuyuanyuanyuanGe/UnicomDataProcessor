import sys
from pathlib import Path

# Add root project directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))



from typing import Dict, List, Optional, Any, TypedDict, Annotated
from datetime import datetime
from utils.modelRelated import invoke_model, invoke_model_with_screenshot
from utils.file_process import (retrieve_file_content, save_original_file,
                                    extract_filename, 
                                    ensure_location_structure, check_file_exists_in_data,
                                    get_available_locations, move_template_files_to_final_destination,
                                    move_supplement_files_to_final_destination, delete_files_from_staging_area,
                                    reconstruct_csv_with_headers)

import json

from langgraph.graph import StateGraph, END, START
from langgraph.constants import Send
from langgraph.graph.message import add_messages
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


class FileProcessState(TypedDict):
    session_id: str
    upload_files_path: list[str] # Store all uploaded files
    new_upload_files_path: list[str] # Track the new uploaded files in this round
    new_upload_files_processed_path: list[str] # Store the processed new uploaded files
    original_files_path: list[str] # Store the original files in original_file subfolder
    uploaded_template_files_path: list[str]
    supplement_files_path: dict[str, list[str]]
    irrelevant_files_path: list[str]
    irrelevant_original_files_path: list[str] # Track original files to be deleted with irrelevant files
    all_files_irrelevant: bool  # Flag to indicate all files are irrelevant
    template_complexity: str
    village_name: str


class FileProcessAgent:

    @tool
    def request_user_clarification(question: str, context: str = "") -> str:
        """
        询问用户澄清，和用户确认，或者询问用户补充信息，当你不确定的时候请询问用户

        参数：
            question: 问题
            context: 可选补充内容，解释为甚恶魔你需要一下信息
        """
        print("\n" + "="*60)
        print("🤔 需要您的确认")
        print("="*60)
        print(f"📋 {question}")
        if context:
            print(f"💡 {context}")
        print("="*60)
        
        user_response = input("👤 请输入您的选择: ").strip()
        
        print(f"✅ 您的选择: {user_response}")
        print("="*60 + "\n")
        
        return user_response
    
    tools = [request_user_clarification]


    def __init__(self):
        self.memory = MemorySaver()
        self.graph = self._build_graph().compile(checkpointer=self.memory)

    def _build_graph(self):
        graph = StateGraph(FileProcessState)

        graph.add_node("file_upload", self._file_upload)
        graph.add_node("analyze_uploaded_files", self._analyze_uploaded_files)
        graph.add_node("route_after_analyze_uploaded_files", self._route_after_analyze_uploaded_files)
        graph.add_node("process_supplement", self._process_supplement)
        graph.add_node("process_irrelevant", self._process_irrelevant)
        graph.add_node("process_template", self._process_template)
        graph.add_node("summary_file_upload", self._summary_file_upload)

        graph.add_edge(START, "file_upload")
        graph.add_edge("file_upload", "analyze_uploaded_files")
        graph.add_conditional_edges("analyze_uploaded_files", self._route_after_analyze_uploaded_files)
        graph.add_edge("process_template", "summary_file_upload")
        graph.add_edge("process_supplement", "summary_file_upload")
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
            "supplement_files_path": {"表格": [], "文档": []},
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
            "template": [],
            "supplement": {"表格": [], "文档": []},
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
                "uploaded_template_files_path": [],
                "supplement_files_path": {"表格": [], "文档": []},
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
                system_prompt = f"""你是一个表格生成智能体，需要分析用户上传的文件内容并进行分类。共有四种类型：

                1. **模板类型 (template)**: 空白表格模板，只有表头没有具体数据
                2. **补充表格 (supplement-表格)**: 已填写的完整表格，用于补充数据库
                3. **补充文档 (supplement-文档)**: 包含重要信息的文本文件，如法律条文、政策信息等
                4. **无关文件 (irrelevant)**: 与表格填写无关的文件

                仔细检查不要把补充文件错误划分为模板文件反之亦然，补充文件里面是有数据的，模板文件里面是空的，或者只有一两个例子数据
                注意：所有文件已转换为txt格式，表格以HTML代码形式呈现，请根据内容而非文件名或后缀判断。

                当前分析文件:
                文件名: {source_path.name}
                文件路径: {file_path}
                文件内容:
                {analysis_content}

                请严格按照以下JSON格式回复，只返回这一个文件的分类结果（不要添加任何其他文字），不要将返回内容包裹在```json```中：
                {{
                    "classification": "template" | "supplement-表格" | "supplement-文档" | "irrelevant"
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
                    if classification_type == "template":
                        classification_results["template"].append(file_path_result)
                    elif classification_type == "supplement-表格":
                        classification_results["supplement"]["表格"].append(file_path_result)
                    elif classification_type == "supplement-文档":
                        classification_results["supplement"]["文档"].append(file_path_result)
                    else:  # irrelevant or unknown
                        classification_results["irrelevant"].append(file_path_result)
                    
                    processed_files.append(file_name)
                    
                except Exception as e:
                    print(f"❌ 并行处理文件任务失败 {file_path}: {e}")
                    # Add to irrelevant on error
                    classification_results["irrelevant"].append(file_path)
        
        print(f"🎉 并行文件分析完成:")
        print(f"  - 模板文件: {len(classification_results['template'])} 个")
        print(f"  - 补充表格: {len(classification_results['supplement']['表格'])} 个")
        print(f"  - 补充文档: {len(classification_results['supplement']['文档'])} 个")
        print(f"  - 无关文件: {len(classification_results['irrelevant'])} 个")
        print(f"  - 成功处理: {len(processed_files)} 个文件")
        
        if not processed_files and not classification_results["irrelevant"]:
            print("⚠️ 没有找到可处理的文件")
            print("✅ _analyze_uploaded_files 执行完成")
            print("=" * 50)
            return {
                "uploaded_template_files_path": [],
                "supplement_files_path": {"表格": [], "文档": []},
                "irrelevant_files_path": [],
                "all_files_irrelevant": True  # Flag for routing to text analysis
            }
        
        # Update state with classification results
        uploaded_template_files = classification_results.get("template", [])
        supplement_files = classification_results.get("supplement", {"表格": [], "文档": []})
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
        all_files_irrelevant = (
            len(uploaded_template_files) == 0 and 
            len(supplement_files.get("表格", [])) == 0 and 
            len(supplement_files.get("文档", [])) == 0 and
            len(irrelevant_files) == new_files_processed_count
        )
        
        if all_files_irrelevant:
            print("⚠️ 所有文件都被分类为无关文件")
            print("✅ _analyze_uploaded_files 执行完成")
            print("=" * 50)
            return {
                "uploaded_template_files_path": [],
                "supplement_files_path": {"表格": [], "文档": []},
                "irrelevant_files_path": irrelevant_files,
                "irrelevant_original_files_path": irrelevant_original_files,
                "all_files_irrelevant": True  # Flag for routing
            }
        else:
            # Some files are relevant, proceed with normal flow
            analysis_summary = f"""文件分析完成:
            模板文件: {len(uploaded_template_files)} 个
            补充表格: {len(supplement_files.get("表格", []))} 个  
            补充文档: {len(supplement_files.get("文档", []))} 个
            无关文件: {len(irrelevant_files)} 个"""
            
            print("✅ 文件分析完成，存在有效文件")
            print("✅ _analyze_uploaded_files 执行完成")
            print("=" * 50)
            
            return {
                "uploaded_template_files_path": uploaded_template_files,
                "supplement_files_path": supplement_files,
                "irrelevant_files_path": irrelevant_files,
                "irrelevant_original_files_path": irrelevant_original_files,
                "all_files_irrelevant": False  # Flag for routing
            }
                
    def _route_after_analyze_uploaded_files(self, state: FileProcessState):
        """Route after analyzing uploaded files. Uses Send objects for all routing."""
        print("Debug: route_after_analyze_uploaded_files")
        
        # Check if all files are irrelevant - route to text analysis
        if state.get("all_files_irrelevant", False):
            # First clean up irrelevant files, then analyze text
            sends = []
            if state.get("irrelevant_files_path"):
                sends.append(Send("process_irrelevant", state))
            return sends if sends else [Send("summary_file_upload", state)]
        
        # Some files are relevant - process them in parallel
        sends = []
        if state.get("uploaded_template_files_path"):
            print("Debug: process_template")
            sends.append(Send("process_template", state))
        if state.get("supplement_files_path", {}).get("表格") or state.get("supplement_files_path", {}).get("文档"):
            print("Debug: process_supplement")
            sends.append(Send("process_supplement", state))
        if state.get("irrelevant_files_path"):
            print("Debug: process_irrelevant")
            sends.append(Send("process_irrelevant", state))

        # The parallel nodes will automatically converge, then continue to summary
        return sends if sends else [Send("summary_file_upload", state)]  # Fallback
    
    def _process_supplement(self, state: FileProcessState) -> FileProcessState:
        """This node will process the supplement files, it will analyze the supplement files and summarize the content of the files as well as stored the summary in data.json"""
        print("\n🔍 开始执行: _process_supplement")
        print("=" * 50)
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Load existing data.json with better error handling
        data_json_path = Path("agents/data.json")
        try:
            with open(data_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print("📝 data.json不存在，创建空的数据结构")
            data = {}
        except json.JSONDecodeError as e:
            print(f"⚠️ data.json格式错误: {e}")
            print("📝 备份原文件并创建新的数据结构")
            # Backup the corrupted file
            backup_path = data_json_path.with_suffix('.json.backup')
            if data_json_path.exists():
                data_json_path.rename(backup_path)
                print(f"📦 原文件已备份到: {backup_path}")
            data = {}
        
        # Use village_name from state as the location for all supplement files
        location = state["village_name"]
        
        table_files = state["supplement_files_path"]["表格"]
        document_files = state["supplement_files_path"]["文档"]
        
        print(f"📊 需要处理的表格文件: {len(table_files)} 个")
        print(f"📄 需要处理的文档文件: {len(document_files)} 个")
        
        # Collect new messages instead of directly modifying state
        new_messages = []
        
        def process_table_file(table_file: str) -> tuple[str, str, dict]:
            """Process a single table file and return (file_path, file_type, result_data)"""
            try:
                source_path = Path(table_file)
                print(f"🔍 表格文件路径: {source_path}")
                print(f"🔍 正在处理表格文件: {source_path.name}")
                
                
                # Use village_name as the location for this table file
                file_location = location
                

                print("📤 正在调用LLM进行表格分析...")
                
                try:
                    file_name = source_path.name
                    print(f"🔍 表格文件名: {file_name}")
                    
                    # Find the corresponding original Excel file from the uploaded files
                    table_file_stem = Path(table_file).stem
                    original_files = state.get("original_files_path", [])
                    original_excel_file = None
                    
                    for original_file in original_files:
                        if Path(original_file).stem == table_file_stem:
                            original_excel_file = Path(original_file)
                            break
                    
                    if original_excel_file and original_excel_file.exists():
                        print(f"🔍 找到原始Excel文件: {original_excel_file}")
                        analysis_response = invoke_model_with_screenshot(model_name="Qwen/Qwen2.5-VL-72B-Instruct", file_path=original_excel_file)
                        print("📥 表格分析响应接收成功")
                    else:
                        print(f"⚠️ 未找到对应的原始Excel文件: {table_file_stem}")
                        raise FileNotFoundError(f"Original Excel file not found for {table_file_stem}")
                        
                except Exception as llm_error:
                    print(f"❌ LLM调用失败: {llm_error}")
                    # Create fallback response  
                    analysis_response = f"表格文件分析失败: {str(llm_error)}，文件名: {source_path.name}"
                
                # Create result data with location information
                # Note: file_path will be updated after moving to final destination
                result_data = {
                    "file_key": source_path.name.split(".")[0],
                    "location": file_location,
                    "new_entry": {
                        "summary": analysis_response,
                        "file_path": str(table_file),  # This will be updated after moving
                        "original_file_path": str(original_excel_file) if original_excel_file else "",  # This will be updated after moving
                        "timestamp": datetime.now().isoformat(),
                        "file_size": source_path.stat().st_size
                    },
                    "analysis_response": analysis_response
                }
                
                print(f"✅ 表格文件已分析: {source_path.name} (位置: {file_location})")
                
                # Reconstruct CSV with headers using the analyzed structure
                try:
                    reconstructed_csv_path = reconstruct_csv_with_headers(
                        analysis_response, source_path.name, original_excel_file, village_name=state["village_name"]
                    )
                    if reconstructed_csv_path:
                        result_data["reconstructed_csv_path"] = reconstructed_csv_path
                        print(f"📊 CSV重构完成: {reconstructed_csv_path}")
                except Exception as csv_error:
                    print(f"❌ CSV重构失败: {csv_error}")
                    result_data["reconstructed_csv_path"] = ""
                
                return table_file, "table", result_data
                
            except Exception as e:
                print(f"❌ 处理表格文件出错 {table_file}: {e}")
                return table_file, "table", {
                    "file_key": Path(table_file).name,
                    "location": location,  # Use village_name as location on error
                    "new_entry": {
                        "summary": f"表格文件处理失败: {str(e)}",
                        "file_path": str(table_file),
                        "timestamp": datetime.now().isoformat(),
                        "file_size": 0
                    },
                    "analysis_response": f"表格文件处理失败: {str(e)}"
                }

        def process_document_file(document_file: str) -> tuple[str, str, dict]:
            """Process a single document file and return (file_path, file_type, result_data)"""
            try:
                source_path = Path(document_file)
                print(f"🔍 正在处理文档文件: {source_path.name}")
                
                file_content = source_path.read_text(encoding='utf-8')
                # file_content = file_content[:2000] if len(file_content) > 2000 else file_content
                file_name = extract_filename(document_file)
                print(f"🔍 文档文件名: {file_name}")
                
                # Use village_name as the location for this document file
                file_location = location
                print(f"📍 文档文件使用位置: {file_location}")
                
                system_prompt = """你是一位专业的文档分析专家，具备法律与政策解读能力。你的任务是阅读用户提供的 HTML 格式文件，并从中提取出最重要的 1-2 条关键信息进行总结，无需提取全部内容。

请遵循以下要求：

1. 忽略所有 HTML 标签（如 <p>、<div>、<table> 等），只关注文本内容；

2. 从文件中提取重要的项核心政策信息（例如补贴金额、适用对象、审批流程等），或者其他你觉得重要的信息；

3. 对提取的信息进行结构化总结，语言正式、逻辑清晰、简洁明了；

4. 输出格式为严格的 JSON，但不要包裹在```json中，直接返回json格式即可：
   {{
     "文件名": "内容总结"
   }}

5. 若提供多个文件，需分别处理并合并输出为一个 JSON 对象；

6. 输出语言应与输入文档保持一致（若文档为中文，则输出中文）；

7. 输出文件名和提供的文件名一致，不许有任何更改

请根据上述要求，对提供的 HTML 文件内容进行分析并返回结果。

文件内容:
{file_content}
""".format(file_name=file_name, file_content=file_content)

                print("📤 正在调用LLM进行文档分析...")
                print("确认文档分析提示词：\n", system_prompt)
                
                try:
                    analysis_response = invoke_model(model_name="Pro/deepseek-ai/DeepSeek-V3", messages=[SystemMessage(content=system_prompt)])
                    print("📥 文档分析响应接收成功")
                    analysis_response_dict = json.loads(analysis_response)
                    keys = list(analysis_response_dict.keys())
                    old_key = keys[0]
                    new_key = file_name
                    analysis_response_dict[new_key] = analysis_response_dict.pop(old_key)
                    analysis_response = json.dumps(analysis_response_dict, ensure_ascii=False)
                    print("📥 文档分析响应转换成功:", analysis_response)
                except Exception as llm_error:
                    print(f"❌ LLM调用失败: {llm_error}")
                    # Create fallback response
                    analysis_response = f"文档文件分析失败: {str(llm_error)}，文件名: {source_path.name}"

                # Create result data with location information
                # Note: file_path will be updated after moving to final destination
                result_data = {
                    "file_key": source_path.name,
                    "location": file_location,  # Single location
                    "new_entry": {
                        "summary": analysis_response,
                        "file_path": str(document_file),  # This will be updated after moving
                        "original_file_path": str(source_path),  # This will be updated after moving
                        "timestamp": datetime.now().isoformat(),
                        "file_size": source_path.stat().st_size
                    },
                    "analysis_response": analysis_response
                }
                
                print(f"✅ 文档文件已分析: {source_path.name} (位置: {file_location})")
                return document_file, "document", result_data
                
            except Exception as e:
                print(f"❌ 处理文档文件出错 {document_file}: {e}")
                return document_file, "document", {
                    "file_key": Path(document_file).name,
                    "location": location,  # Use village_name as location on error
                    "new_entry": {
                        "summary": f"文档文件处理失败: {str(e)}",
                        "file_path": str(document_file),
                        "timestamp": datetime.now().isoformat(),
                        "file_size": 0
                    },
                    "analysis_response": f"文档文件处理失败: {str(e)}"
                }

        # Use ThreadPoolExecutor for parallel processing
        all_files = [(file, "table") for file in table_files] + [(file, "document") for file in document_files]
        total_files = len(all_files)
        
        if total_files == 0:
            print("⚠️ 没有文件需要处理")
            print("✅ _process_supplement 执行完成")
            print("=" * 50)
            return {}
        
        max_workers = min(total_files, 5)  # Limit to 4 concurrent requests for supplement processing
        print(f"🚀 开始并行处理补充文件，使用 {max_workers} 个工作线程")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file processing tasks
            future_to_file = {}
            for file_path, file_type in all_files:
                if file_type == "table":
                    future = executor.submit(process_table_file, file_path)
                    # Implement the logic for 
                else:  # document
                    future = executor.submit(process_document_file, file_path)
                future_to_file[future] = (file_path, file_type)
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_file):
                file_path, file_type = future_to_file[future]
                try:
                    file_path, processed_file_type, result_data = future.result()
                    
                    # Add to new_messages
                    new_messages.append(AIMessage(content=result_data["analysis_response"]))
                    
                    # Update data.json structure with location-based storage
                    file_key = result_data["file_key"]
                    new_entry = result_data["new_entry"]
                    
                    # Both table and document files now use single location
                    file_location = result_data["location"]
                    # Ensure location structure exists in data
                    data = ensure_location_structure(data, file_location)
                    
                    if processed_file_type == "table":
                        if file_key in data[file_location]["表格"]:
                            print(f"⚠️ 表格文件 {file_key} 已存在于 {file_location}，将更新其内容")
                            # Preserve any additional fields that might exist
                            existing_entry = data[file_location]["表格"][file_key]
                            for key, value in existing_entry.items():
                                if key not in new_entry:
                                    new_entry[key] = value
                        else:
                            print(f"📝 添加新的表格文件: {file_key} 到 {file_location}")
                        data[file_location]["表格"][file_key] = new_entry
                    else:  # document - now also uses single location
                        if file_key in data[file_location]["文档"]:
                            print(f"⚠️ 文档文件 {file_key} 已存在于 {file_location}，将更新其内容")
                            # Preserve any additional fields that might exist
                            existing_entry = data[file_location]["文档"][file_key]
                            for key, value in existing_entry.items():
                                if key not in new_entry:
                                    new_entry[key] = value
                        else:
                            print(f"📝 添加新的文档文件: {file_key} 到 {file_location}")
                        data[file_location]["文档"][file_key] = new_entry
                    
                except Exception as e:
                    print(f"❌ 并行处理文件任务失败 {file_path}: {e}")
                    # Create fallback entry
                    fallback_response = f"文件处理失败: {str(e)}"
                    new_messages.append(AIMessage(content=fallback_response))
        
        print(f"🎉 并行文件处理完成，共处理 {total_files} 个文件")
        
        # Move supplement files to their final destinations and update data.json with new paths
        original_files = state.get("original_files_path", [])
        
        # Track moved files to update data.json paths
        moved_files_info = {}
        
        # Move table files to their final destination
        for table_file in table_files:
            # Find corresponding original file
            table_file_stem = Path(table_file).stem
            corresponding_original_file = ""
            
            for original_file in original_files:
                if Path(original_file).stem == table_file_stem:
                    corresponding_original_file = original_file
                    break
            
            try:
                move_result = move_supplement_files_to_final_destination(
                    table_file, corresponding_original_file, "table", village_name=state["village_name"]
                )
                print(f"✅ 表格文件已移动到最终位置: {Path(table_file).name}")
                
                # Store moved file info for later data.json update
                moved_files_info[Path(table_file).name] = {
                    "new_processed_path": move_result["processed_supplement_path"],
                    "new_original_path": move_result["original_supplement_path"],
                    "new_screen_shot_path": move_result.get("screen_shot_path", "")  # Use get to avoid KeyError
                }
            except Exception as e:
                print(f"❌ 移动表格文件失败 {table_file}: {e}")
        
        # Move document files to their final destination
        for document_file in document_files:
            # Find corresponding original file
            document_file_stem = Path(document_file).stem
            corresponding_original_file = ""
            
            for original_file in original_files:
                if Path(original_file).stem == document_file_stem:
                    corresponding_original_file = original_file
                    break
            
            try:
                move_result = move_supplement_files_to_final_destination(
                    document_file, corresponding_original_file, "document", village_name=state["village_name"]
                )
                print(f"✅ 文档文件已移动到最终位置: {Path(document_file).name}")
                
                # Store moved file info for later data.json update
                moved_files_info[Path(document_file).name] = {
                    "new_processed_path": move_result["processed_supplement_path"],
                    "new_original_path": move_result["original_supplement_path"]
                }
            except Exception as e:
                print(f"❌ 移动文档文件失败 {document_file}: {e}")
        
        # Update data.json entries with new file paths
        for location in data.keys():
            if isinstance(data[location], dict):
                # Update table file paths
                for file_key in data[location].get("表格", {}):
                    if file_key in moved_files_info:
                        if moved_files_info[file_key]["new_processed_path"]:
                            data[location]["表格"][file_key]["file_path"] = moved_files_info[file_key]["new_processed_path"]
                        if moved_files_info[file_key]["new_original_path"]:
                            data[location]["表格"][file_key]["original_file_path"] = moved_files_info[file_key]["new_original_path"]
                
                # Update document file paths
                for file_key in data[location].get("文档", {}):
                    if file_key in moved_files_info:
                        if moved_files_info[file_key]["new_processed_path"]:
                            data[location]["文档"][file_key]["file_path"] = moved_files_info[file_key]["new_processed_path"]
                        if moved_files_info[file_key]["new_original_path"]:
                            data[location]["文档"][file_key]["original_file_path"] = moved_files_info[file_key]["new_original_path"]
        
        # Save updated data.json with atomic write
        try:
            # Write to a temporary file first to prevent corruption
            temp_path = data_json_path.with_suffix('.json.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            # Atomic rename to replace the original file
            temp_path.replace(data_json_path)
            
            # Count total files across all locations
            total_table_files = sum(len(data[location]["表格"]) for location in data.keys() if isinstance(data[location], dict))
            total_document_files = sum(len(data[location]["文档"]) for location in data.keys() if isinstance(data[location], dict))
            
            print(f"✅ 已更新 data.json，表格文件 {total_table_files} 个，文档文件 {total_document_files} 个")
            
            # Log the files that were processed in this batch
            if table_files:
                print(f"📊 本批次处理的表格文件: {[Path(f).name for f in table_files]}")
            if document_files:
                print(f"📄 本批次处理的文档文件: {[Path(f).name for f in document_files]}")
            
            # Log current distribution by location
            print("📍 当前数据分布:")
            for location in data.keys():
                if isinstance(data[location], dict):
                    table_count = len(data[location]["表格"])
                    doc_count = len(data[location]["文档"])
                    print(f"  {location}: 表格 {table_count} 个, 文档 {doc_count} 个")
                
        except Exception as e:
            print(f"❌ 保存 data.json 时出错: {e}")
            # Clean up temp file if it exists
            temp_path = data_json_path.with_suffix('.json.tmp')
            if temp_path.exists():
                try:
                    temp_path.unlink()
                    print("🗑️ 临时文件已清理")
                except Exception as cleanup_error:
                    print(f"⚠️ 清理临时文件失败: {cleanup_error}")
        
        print("✅ _process_supplement 执行完成")
        print("=" * 50)
        
        # Return empty dict since we don't need to update state with messages
        return {}
    
        
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

    
    def _process_template(self, state: FileProcessState) -> FileProcessState:
        """This node will process the template files, it will analyze the template files and determine if it's a valid template"""
        
        print("\n🔍 开始执行: _process_template")
        print("=" * 50)
        
        template_files = state["uploaded_template_files_path"]
        print(f"📋 需要处理的模板文件数量: {len(template_files)}")
        
        # If multiple templates, ask user to choose
        if len(template_files) > 1:
            print("⚠️ 检测到多个模板文件，需要用户选择")
            template_names = [Path(f).name for f in template_files]
            template_list = "\n".join([f"  {i+1}. {name}" for i, name in enumerate(template_names)])
            question = f"""检测到多个模板文件，请选择要使用的模板：

📋 可用模板：
{template_list}

请输入序号（如：1）选择模板："""
            
            try:
                print("🤝 正在请求用户确认模板选择...")
                user_choice = self.request_user_clarification.invoke(
                    input = {"question": question,
                             "context": "系统需要确定使用哪个模板文件进行后续处理"}
                    )
                
                # Parse user choice
                try:
                    choice_index = int(user_choice.strip()) - 1
                    if 0 <= choice_index < len(template_files):
                        selected_template = template_files[choice_index]
                        # Remove non-selected templates
                        rejected_templates = [f for i, f in enumerate(template_files) if i != choice_index]
                        
                        # Delete rejected template files (both processed and original)
                        original_files = state.get("original_files_path", [])
                        for rejected_file in rejected_templates:
                            try:
                                # Delete processed template file
                                Path(rejected_file).unlink()
                                print(f"🗑️ 已删除未选中的处理模板: {Path(rejected_file).name}")
                                
                                # Find and delete corresponding original file
                                rejected_file_stem = Path(rejected_file).stem
                                for original_file in original_files:
                                    original_file_path = Path(original_file)
                                    if original_file_path.stem == rejected_file_stem:
                                        try:
                                            original_file_path.unlink()
                                            print(f"🗑️ 已删除未选中的原始模板: {original_file_path.name}")
                                            break
                                        except Exception as orig_error:
                                            print(f"❌ 删除原始模板文件出错: {orig_error}")
                                
                            except Exception as e:
                                print(f"❌ 删除模板文件出错: {e}")
                        
                        # Update state to only include selected template
                        template_files = [selected_template]
                        print(f"✅ 用户选择了模板: {Path(selected_template).name}")
                        
                    else:
                        print("❌ 无效的选择，使用第一个模板")
                        selected_template = template_files[0]
                        template_files = [selected_template]
                        
                except ValueError:
                    print("❌ 输入格式错误，使用第一个模板")
                    selected_template = template_files[0]
                    template_files = [selected_template]
                    
            except Exception as e:
                print(f"❌ 用户选择出错: {e}")
                selected_template = template_files[0]
                template_files = [selected_template]
        
        # Analyze the selected template for complexity
        template_file = template_files[0]
        print(f"🔍 正在分析模板复杂度: {Path(template_file).name}")
        
        try:
            source_path = Path(template_file)
            template_content = source_path.read_text(encoding='utf-8')
            
            # Create prompt to determine if template is complex or simple
            system_prompt = f"""你是一个表格结构分析专家，需要判断这个表格模板是复杂模板还是简单模板。

            判断标准：
            - **复杂模板**: 表格同时包含行表头和列表头，即既有行标题又有列标题的二维表格结构
            - **简单模板**: 表格只包含列表头或者只包含行表头，但是可以是多级表头，每行是独立的数据记录

            模板内容（HTML格式）：
            {template_content}

            请仔细分析表格结构，然后只回复以下选项之一：
            [Complex] - 如果是复杂模板（包含行表头和列表头）
            [Simple] - 如果是简单模板（只包含列表头）"""
            

            print("📤 正在调用LLM进行模板复杂度分析...")
            
            analysis_response = invoke_model(model_name="Pro/deepseek-ai/DeepSeek-V3", messages=[SystemMessage(content=system_prompt)])
            
            # Extract the classification from the response
            if "[Complex]" in analysis_response:
                template_type = "[Complex]"
            elif "[Simple]" in analysis_response:
                template_type = "[Simple]"
            else:
                template_type = "[Simple]"  # Default fallback
            
            # 将模板文件（包括原始文件）移动到最终位置
            # Find corresponding original file
            original_files = state.get("original_files_path", [])
            template_file_stem = Path(template_file).stem
            corresponding_original_file = ""
            
            for original_file in original_files:
                if Path(original_file).stem == template_file_stem:
                    corresponding_original_file = original_file
                    break
            
            # Move template files to final destination using session ID
            # Extract session ID from one of the file paths
            session_id = state["session_id"]  # Default session ID
            if template_file:
                # Extract session ID from the file path: conversations/session_id/user_uploaded_files/...
                template_path_parts = Path(template_file).parts
                if len(template_path_parts) >= 3 and template_path_parts[0] == "conversations":
                    session_id = template_path_parts[1]
            
            move_result = move_template_files_to_final_destination(
                template_file, corresponding_original_file, session_id
            )
            final_template_path = move_result["processed_template_path"]
            final_original_template_path = move_result["original_template_path"]
            print(f"📁 模板原始文件已移动到: {final_original_template_path}")
            print(f"📁 模板处理文件已移动到: {final_template_path}")
            if move_result["original_template_path"]:
                print(f"📁 模板原始文件已移动到: {move_result['original_template_path']}")
            else:
                print("⚠️ 未找到对应的原始模板文件")

            print(f"📥 模板分析结果: {template_type}")
            print("✅ _process_template 执行完成")
            print("=" * 50)

            return {"template_complexity": template_type,
                    "uploaded_template_files_path": [final_template_path]
                    }

        except Exception as e:
            print(f"❌ 模板分析LLM调用出错: {e}")
            # Default to Simple if analysis fails
            template_type = "[Simple]"
            print("⚠️ 模板分析失败，默认为简单模板")
            
            # Still try to move the template file (including original) even if LLM analysis fails
            original_files = state.get("original_files_path", [])
            template_file_stem = Path(template_file).stem
            corresponding_original_file = ""
            
            for original_file in original_files:
                if Path(original_file).stem == template_file_stem:
                    corresponding_original_file = original_file
                    break
            
            # Extract session ID from file path
            session_id = state["session_id"]  # Default session ID
            if template_file:
                template_path_parts = Path(template_file).parts
                if len(template_path_parts) >= 3 and template_path_parts[0] == "conversations":
                    session_id = template_path_parts[1]
            
            move_result = move_template_files_to_final_destination(
                template_file, corresponding_original_file, session_id
            )
            final_template_path = move_result["processed_template_path"]
            
            if move_result["original_template_path"]:
                print(f"📁 模板原始文件已移动到: {move_result['original_template_path']}")
            else:
                print("⚠️ 未找到对应的原始模板文件")
            
            print("✅ _process_template 执行完成")
            print("=" * 50)
            
            return {
                "template_complexity": template_type,
                "uploaded_template_files_path": [final_template_path]
            }

    def _summary_file_upload(self, state: FileProcessState) -> FileProcessState:
        """Summary node for file upload process"""
        print("\n🔍 开始执行: _summary_file_upload")
        print("=" * 50)
        
        # Log the final state summary
        print("📊 文件处理总结:")
        print(f"  - 上传文件总数: {len(state.get('upload_files_path', []))}")
        print(f"  - 新上传文件数: {len(state.get('new_upload_files_path', []))}")
        print(f"  - 模板文件数: {len(state.get('uploaded_template_files_path', []))}")
        print(f"  - 补充表格文件数: {len(state.get('supplement_files_path', {}).get('表格', []))}")
        print(f"  - 补充文档文件数: {len(state.get('supplement_files_path', {}).get('文档', []))}")
        print(f"  - 无关文件数: {len(state.get('irrelevant_files_path', []))}")
        print(f"  - 模板复杂度: {state.get('template_complexity', 'N/A')}")
        
        print("✅ _summary_file_upload 执行完成")
        print("=" * 50)
        
        return {}


    def run_file_process_agent(self, session_id: str = "1", upload_files_path: list[str] = [], village_name: str = "") -> FileProcessState:
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

            return final_state
        
        except Exception as e:
            print(f"❌ 执行过程中发生错误: {e}")
            return initial_state
