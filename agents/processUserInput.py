import sys
from pathlib import Path

# Add root project directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))



from typing import Dict, List, Optional, Any, TypedDict, Annotated
from datetime import datetime
from utils.modelRelated import invoke_model
from utils.file_process import (detect_and_process_file_paths)
from agents.fileProcessAgent import FileProcessAgent

import uuid
import json
import os

from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

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

load_dotenv()


class ProcessUserInputState(TypedDict):
    process_user_input_messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    upload_files_path: list[str] # Store all uploaded files
    text_input_validation: str  # Store validation result [Valid] or [Invalid]
    previous_AI_messages: list[BaseMessage]
    summary_message: str  # Add the missing field
    template_file_path: str
    template_complexity: str
    session_id: str
    current_node: str
    next_node: str
    village_name: str
    
class ProcessUserInputAgent:

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


    def _build_graph(self) -> StateGraph:
        """This function will build the graph for the process user input agent"""
        graph = StateGraph(ProcessUserInputState)
        graph.add_node("collect_user_input", self._collect_user_input)
        graph.add_node("file_process_agent", self._file_process_agent)
        graph.add_node("analyze_text_input", self._analyze_text_input)
        graph.add_node("clarification_tool_node", ToolNode(self.tools, messages_key = "process_user_input_messages"))
        graph.add_node("summary_user_input", self._summary_user_input)
        graph.add_node("decide_next_node", self._decide_next_node)
        graph.add_node("combine_summary_and_decide_next_node", self._combine_summary_and_decide_next_node)
        
        graph.add_edge(START, "collect_user_input")

        graph.add_conditional_edges(
            "collect_user_input",
            self._route_after_collect_user_input,
            {
                "file_process_agent": "file_process_agent",
                "analyze_text_input": "analyze_text_input",
            }
        )

        graph.add_conditional_edges("file_process_agent", self._route_after_file_process_agent)
        graph.add_conditional_edges("analyze_text_input", self._route_after_analyze_text_input)
        graph.add_edge("summary_user_input", "combine_summary_and_decide_next_node")
        graph.add_edge("decide_next_node", "combine_summary_and_decide_next_node")
        graph.add_edge("combine_summary_and_decide_next_node", END)
        return graph



    def create_initial_state(self, session_id: str, previous_AI_messages = None, 
                             current_node: str = "", village_name: str = "") -> ProcessUserInputState:
        """This function initializes the state of the process user input agent"""
        
        # Handle both single BaseMessage and list[BaseMessage] input
        processed_messages = None
        if previous_AI_messages is not None:
            if isinstance(previous_AI_messages, list):
                processed_messages = previous_AI_messages
                print(f"🔍 初始化: 接收到消息列表，包含 {len(previous_AI_messages)} 条消息")
            else:
                # It's a single message, convert to list
                processed_messages = [previous_AI_messages]
                print(f"🔍 初始化: 接收到单条消息，已转换为列表")
        else:
            print(f"🔍 初始化: 没有接收到previous_AI_messages")
        
        return {
            "process_user_input_messages": [],
            "user_input": "",
            "upload_files_path": [],
            "text_input_validation": None,
            "previous_AI_messages": processed_messages,
            "summary_message": "",
            "template_complexity": "",
            "template_file_path": "",
            "session_id": session_id,
            "current_node": current_node,
            "next_node": "collect_user_input",
            "village_name": village_name
        }


    def _collect_user_input(self, state: ProcessUserInputState) -> ProcessUserInputState:
        """This is the node where we get user's input"""
        print("\n🔍 开始执行: _collect_user_input")
        print("=" * 50)
        print("⌨️ 等待用户输入...")
        
        user_input = interrupt("用户：")
        
        print(f"📥 接收到用户输入: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
        user_upload_files = detect_and_process_file_paths(user_input)
        print(f"🔍 检测到的文件: {user_upload_files}")
        print("✅ _collect_user_input 执行完成")
        print("=" * 50)
        
        return {
            "process_user_input_messages": [HumanMessage(content=user_input)],
            "user_input": user_input,
            "upload_files_path": user_upload_files
        }



    def _route_after_collect_user_input(self, state: ProcessUserInputState) -> str:
        """This node act as a safety check node, it will analyze the user's input and determine if it's a valid input,
        based on the LLM's previous response, at the same time it will route the agent to the correct node"""
        
        upload_files_path = state["upload_files_path"]
        if upload_files_path:
            # Files detected - route to file_upload 
            # Note: We cannot modify state in routing functions, so file_upload node will re-detect files
            return "file_process_agent"
        
        # User didn't upload any new files, we will analyze the text input
        return "analyze_text_input"


    def _file_process_agent(self, state: ProcessUserInputState) -> ProcessUserInputState:
        """This node will route to the file process agent"""
        print("\n🔍 开始执行: _file_process_agent")
        print("=" * 50)
        
        file_process_agent = FileProcessAgent()
        file_process_agent_final_state = file_process_agent.run_file_process_agent(
            session_id=state["session_id"],
            upload_files_path=state["upload_files_path"],
            village_name=state["village_name"]
        )
        
        # Handle template file path - convert list to string if necessary
        template_files_list = file_process_agent_final_state.get("uploaded_template_files_path", [])
        if isinstance(template_files_list, list) and len(template_files_list) > 0:
            template_file_path = template_files_list[0]  # Take the first template file
        else:
            template_file_path = ""
            
        template_complexity = file_process_agent_final_state.get("template_complexity", "")
        print(f"🔍 模板文件路径: {template_file_path}")
        print(f"🔍 模板复杂度: {template_complexity}")

        return {"template_file_path": template_file_path,
                "template_complexity": template_complexity}

    def _route_after_file_process_agent(self, state: ProcessUserInputState) -> str:
        sends = []
        sends.append(Send("decide_next_node", state))
        sends.append(Send("summary_user_input", state))
        return sends

    def _analyze_text_input(self, state: ProcessUserInputState) -> ProcessUserInputState:
        """This node performs a safety check on user text input when all uploaded files are irrelevant.
        It validates if the user input contains meaningful table/Excel-related content.
        Returns [Valid] or [Invalid] based on the analysis."""
        
        print("\n🔍 开始执行: _analyze_text_input")
        print("=" * 50)
        
        user_input = state["user_input"]
        print(f"📝 正在分析用户文本输入: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
        
        if not user_input or user_input.strip() == "":
            print("❌ 用户输入为空")
            print("✅ _analyze_text_input 执行完成")
            print("=" * 50)
            return {
                "text_input_validation": "[Invalid]",
                "process_user_input_messages": [SystemMessage(content="❌ 用户输入为空，验证失败")]
            }
        
        # Create validation prompt for text input safety check
        # Get the previous AI message content safely
        previous_ai_content = ""
        try:
            if state.get("previous_AI_messages"):
                previous_ai_messages = state["previous_AI_messages"]
                print(f"🔍 previous_AI_messages 类型: {type(previous_ai_messages)}")
                
                # Handle both single message and list of messages
                if isinstance(previous_ai_messages, list):
                    if len(previous_ai_messages) > 0:
                        latest_message = previous_ai_messages[-1]
                        if hasattr(latest_message, 'content'):
                            previous_ai_content = latest_message.content
                        else:
                            previous_ai_content = str(latest_message)
                        print(f"📝 从消息列表提取内容，长度: {len(previous_ai_content)}")
                    else:
                        print("⚠️ 消息列表为空")
                else:
                    # It's a single message object
                    if hasattr(previous_ai_messages, 'content'):
                        previous_ai_content = previous_ai_messages.content
                    else:
                        previous_ai_content = str(previous_ai_messages)
                    print(f"📝 从单个消息提取内容，长度: {len(previous_ai_content)}")
            else:
                print("⚠️ 没有找到previous_AI_messages")
                
        except Exception as e:
            print(f"❌ 提取previous_AI_messages内容时出错: {e}")
            previous_ai_content = ""
            
        system_prompt = f"""
你是一位专业的输入验证专家，任务是判断用户的文本输入是否与**表格生成或 Excel 处理相关**，并且是否在当前对话上下文中具有实际意义。

你将获得以下两部分信息：
- 上一轮 AI 的回复（用于判断上下文是否连贯）
- 当前用户的输入内容

请根据以下标准进行判断：

【有效输入 [Valid]】满足以下任一条件即可视为有效：
- 明确提到生成表格、填写表格、Excel 处理、数据整理等相关操作
- 提出关于表格字段、数据格式、模板结构等方面的需求或提问
- 提供表格相关的数据内容、字段说明或规则
- 对上一轮 AI 的回复作出有意义的延续或回应（即使未直接提到表格）
- 即使存在错别字、语病、拼写错误，只要语义清晰合理，也视为有效

【无效输入 [Invalid]】符合以下任一情况即视为无效：
- 内容与表格/Excel 完全无关（如闲聊、情绪表达、与上下文跳脱）
- 明显为测试文本、随机字符或系统调试输入（如 "123"、"测试一下"、"哈啊啊啊" 等）
- 仅包含空白、表情符号、标点符号等无实际内容

【输出要求】
请你根据上述标准，**仅输出以下两种结果之一**（不添加任何其他内容）：
- [Valid]
- [Invalid]

【上一轮 AI 的回复】
{previous_ai_content}
"""


        try:
            print("📤 正在调用LLM进行文本输入验证...")
            # Get LLM validation
            print("系统提示词验证用户输入: \n" + system_prompt)
            user_input = "用户输入：" + user_input
            print("analyze_text_input时调用模型的输入: \n" + user_input)              
            validation_response = invoke_model(model_name="Pro/deepseek-ai/DeepSeek-V3", messages=[SystemMessage(content=system_prompt), HumanMessage(content=user_input)])
            # validation_response = self.llm_s.invoke([SystemMessage(content=system_prompt)])
            
            print(f"📥 验证响应: {validation_response}")
            
            if "[Valid]" in validation_response:
                validation_result = "[Valid]"
                status_message = "用户输入验证通过 - 内容与表格相关且有意义"
            elif "[Invalid]" in validation_response:
                validation_result = "[Invalid]"
                status_message = "用户输入验证失败 - 内容与表格无关或无意义"
            else:
                # Default to Invalid for safety
                validation_result = "[Invalid]"
                status_message = "用户输入验证失败 - 无法确定输入有效性，默认为无效"
                print(f"⚠️ 无法解析验证结果，LLM响应: {validation_response}")
            
            print(f"📊 验证结果: {validation_result}")
            print(f"📋 状态说明: {status_message}")
            
            # Create validation summary
            summary_message = f"""文本输入安全检查完成:
            
            **用户输入**: {user_input[:100]}{'...' if len(user_input) > 100 else ''}
            **验证结果**: {validation_result}
            **状态**: {status_message}"""
            
            print("✅ _analyze_text_input 执行完成")
            print("=" * 50)
            
            return {
                "text_input_validation": validation_result,
                "process_user_input_messages": [SystemMessage(content=summary_message)]
            }
                
        except Exception as e:
            print(f"❌ 验证文本输入时出错: {e}")
            
            # Default to Invalid for safety when there's an error
            error_message = f"""❌ 文本输入验证出错: {e}
            
            📄 **用户输入**: {user_input[:100]}{'...' if len(user_input) > 100 else ''}
            🔒 **安全措施**: 默认标记为无效输入"""
            
            print("✅ _analyze_text_input 执行完成 (出错)")
            print("=" * 50)
            
            return {
                "text_input_validation": "[Invalid]",
                "process_user_input_messages": [SystemMessage(content=error_message)]
            }



    def _route_after_analyze_text_input(self, state: ProcessUserInputState) -> str:
        """Route after text input validation based on [Valid] or [Invalid] result."""
        sends = []
        validation_result = state.get("text_input_validation", "[Invalid]")
        
        if validation_result == "[Valid]":
            sends.append(Send("decide_next_node", state))
            sends.append(Send("summary_user_input", state))
        else:
            # Text input is invalid, route back to collect user input
            sends.append(Send("collect_user_input", state))
        return sends
        


    def _decide_next_node(self, state: ProcessUserInputState) -> ProcessUserInputState:
        """这个节点调用大模型来决定下一步的路由"""
        print("\n🔍 开始执行: _decide_next_node")
        print("=" * 50)
        
        # First check if template complexity is available
        template_complexity = state.get("template_complexity", "")
        print(f"🔍 模板复杂度: {template_complexity}")
        
        if "[Complex]" in template_complexity:
            route_decision = "complex_template"
            print("📍 基于模板复杂度路由到: complex_template")
        elif "[Simple]" in template_complexity:
            route_decision = "simple_template"
            print("📍 基于模板复杂度路由到: simple_template")
        else:
            # If no template complexity, determine based on user input context
            user_input = state.get("user_input", "")
            
            # Get previous AI messages for context
            previous_ai_content = ""
            if state.get("previous_AI_messages"):
                previous_ai_messages = state["previous_AI_messages"]
                if isinstance(previous_ai_messages, list) and len(previous_ai_messages) > 0:
                    latest_message = previous_ai_messages[-1]
                    if hasattr(latest_message, 'content'):
                        previous_ai_content = latest_message.content
                    else:
                        previous_ai_content = str(latest_message)
                elif not isinstance(previous_ai_messages, list):
                    if hasattr(previous_ai_messages, 'content'):
                        previous_ai_content = previous_ai_messages.content
                    else:
                        previous_ai_content = str(previous_ai_messages)
            
            system_prompt = f"""你是一位智能工作流路由决策专家，负责根据用户输入和对话上下文，和当前节点，精确判断下一步应该执行的节点。

## 📋 路由决策规则

### 1. 当前节点为initial_collect_user_input时
- **判断逻辑**：分析用户输入是否包含明确的表格生成意图
- **路由规则**：
  - **包含明确表格生成意图**（如"生成表格"、"创建模板"、"填充表格"、"制作报表"等）→ `chat_with_user_to_determine_template`
  - **仅提供文件路径**或**模糊需求**或**无明确意图** → `initial_collect_user_input`（继续收集用户输入）
- **关键词示例**：
  - ✅ 触发路由：生成、创建、制作、填充、表格、模板、报表、统计、整理数据
  - ❌ 继续收集：仅文件路径、简单问候、不明确描述

### 2. 当前节点为design_excel_template时
- **触发条件**：用户输入涉及表格生成、模板设计、字段定义等需求
- **用户反馈判断**：
  - 满意/确认 → `generate_html_template`（开始生成HTML模板）
  - 修改建议/不满意 → `design_excel_template`（重新设计表格结构）

### 3. 当前节点为recall_relative_files时
- **触发条件**：需要查找或确认相关文件
- **用户反馈判断**：
  - 文件相关/确认使用 → `determine_the_mapping_of_headers`（进行字段映射）
  - 文件不相关/需要重新查找 → `recall_relative_files`（重新召回文件）

### 4. 当前节点为modify_generated_table时
- **触发条件**：用户想要修改生成的表格，并提出相应的需求
- **用户反馈判断**：
  - 需要对表格添加新的表头 → `reconstruct_table_structure`（重新修改表格结构）
  - 需要对表格进行删除或者过滤 → `filter_generated_table`（重新过滤表格）

## 当前节点
{state["current_node"]}

## 📊 上下文信息
**上一轮AI回复内容**：
{previous_ai_content}

**用户当前输入**：
{user_input}

## 🎯 判断要点
- **意图识别**：用户是否明确表达了表格生成需求？
- **内容分析**：仅文件路径 vs 明确的表格处理指令
- **上下文理解**：结合当前节点状态做出合理判断

## 📝 决策示例
**当前节点为initial_collect_user_input时：**
- 用户输入："帮我生成一个党员信息表格" → `chat_with_user_to_determine_template`
- 用户输入："d:\files\党员信息表.xlsx" → `initial_collect_user_input`
- 用户输入："你好" → `initial_collect_user_input`
- 用户输入："我要制作一个统计报表" → `chat_with_user_to_determine_template`

## 📤 输出要求
**严格按照以下格式输出，不得包含任何解释或额外文字**：
仅返回节点名称，如：`design_excel_template`"""
            
            try:
                print("当前节点为：" + state["current_node"])
                print("用户输入为：" + user_input)
                print("系统提示为：" + system_prompt)
                response = invoke_model(model_name="Pro/deepseek-ai/DeepSeek-V3", messages=[SystemMessage(content=system_prompt), HumanMessage(content=user_input)])
                route_decision = response.strip()
                print(f"📍 基于LLM决策路由到: {route_decision}")
            except Exception as e:
                print(f"❌ LLM路由决策失败: {e}")
                route_decision = "design_excel_template"  # 默认路由
                print(f"📍 使用默认路由: {route_decision}")
        
        print("✅ _decide_next_node 执行完成")
        print("=" * 50)
        
        return {"next_node": route_decision}
    
    def _summary_user_input(self, state: ProcessUserInputState) -> ProcessUserInputState:
        """Summary node that consolidates all information from this round."""
        
        print("\n🔍 开始执行: _summary_user_input")
        print("=" * 50)
        
        print(f"🔄 开始总结用户输入，当前消息数: {len(state.get('process_user_input_messages', []))}")
        
        # Extract content from all messages in this processing round
        process_user_input_messages_content = ("\n").join([item.content for item in state["process_user_input_messages"]])
        print(f"📝 处理的消息内容长度: {len(process_user_input_messages_content)} 字符")
        
        system_prompt = f"""
你是一位专业的用户输入分析专家，任务是根据当前轮次的历史对话内容，总结用户在信息收集过程中的所有有效输入。

【你的目标】
- 提取本轮对话中用户提供的所有有价值信息，包括但不限于：
  - 文件上传（如数据文件、模板文件等）；
  - 文本输入（如填写说明、政策信息、计算规则等）；
  - 对召回文件的判断（例如用户确认某些文件是否相关）；
- 注意：有时你被作为"确认节点"调用，任务是让用户判断文件是否相关，此时你需要总结的是"用户的判断结果"，而不是文件本身。
- 请基于上下文灵活判断哪些内容构成有价值的信息。
- 总结中请不要包含用户上传的无关信息内容，以及有效性验证
- 但是一定不要忽略曲解用户的意图

【输出格式】
仅返回以下 JSON 对象，不得包含任何额外解释或文本,不要包裹在```json中，直接返回json格式即可：
{{
  "summary": "对本轮用户提供的信息进行总结"
}}
"""

        try:
            user_input = "【历史对话】\n" + process_user_input_messages_content
            print("📤 正在调用LLM生成总结...")
            response = invoke_model(model_name="Pro/deepseek-ai/DeepSeek-V3", messages=[SystemMessage(content=system_prompt), HumanMessage(content=user_input)])
            print(f"📥 LLM总结响应长度: {len(response)} 字符")
            
            # Clean the response to handle markdown code blocks and malformed JSON
            cleaned_response = response.strip()
            
            # Remove markdown code blocks if present
            if '```json' in cleaned_response:
                print("🔍 检测到markdown代码块，正在清理...")
                # Extract content between ```json and ```
                start_marker = '```json'
                end_marker = '```'
                start_index = cleaned_response.find(start_marker)
                if start_index != -1:
                    start_index += len(start_marker)
                    end_index = cleaned_response.find(end_marker, start_index)
                    if end_index != -1:
                        cleaned_response = cleaned_response[start_index:end_index].strip()
                    else:
                        # If no closing ```, take everything after ```json
                        cleaned_response = cleaned_response[start_index:].strip()
            elif '```' in cleaned_response:
                print("🔍 检测到通用代码块，正在清理...")
                # Handle generic ``` blocks
                parts = cleaned_response.split('```')
                if len(parts) >= 3:
                    # Take the middle part (index 1)
                    cleaned_response = parts[1].strip()
            
            # If there are multiple JSON objects, take the first valid one
            if '}{' in cleaned_response:
                print("⚠️ 检测到多个JSON对象，取第一个")
                cleaned_response = cleaned_response.split('}{')[0] + '}'
            
            print(f"🔍 清理后的响应: {cleaned_response}")
            
            response_json = json.loads(cleaned_response)
            final_response = json.dumps(response_json, ensure_ascii=False)
            
            print(f"✅ 总结生成成功")
            print(f"📊 最终响应: {final_response}")
            print("✅ _summary_user_input 执行完成")
            print("=" * 50)
            
            return {"summary_message": final_response}
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {e}")
            print(f"❌ 原始响应: {repr(response)}")
            # Fallback response
            fallback_response = {
                "summary": "用户本轮提供了文件信息，但解析过程中出现错误"
            }
            final_fallback = json.dumps(fallback_response, ensure_ascii=False)
            print(f"🔄 使用备用响应: {final_fallback}")
            print("✅ _summary_user_input 执行完成 (备用)")
            print("=" * 50)
            return {"summary_message": final_fallback}

    def _combine_summary_and_decide_next_node(self, state: ProcessUserInputState) -> ProcessUserInputState:
        """Combine summary and decide next node results"""
        print("\n🔍 开始执行: _combine_summary_and_decide_next_node")
        print("=" * 50)
        
        summary_message = state.get("summary_message", "")
        next_node = state.get("next_node", "")
        
        print(f"📝 总结消息: {summary_message}")
        print(f"🔄 下一节点: {next_node}")
        
        # Parse the summary message to add next_node information
        try:
            if summary_message:
                summary_json = json.loads(summary_message)
                summary_json["next_node"] = next_node
                combined_summary = json.dumps(summary_json, ensure_ascii=False)
            else:
                # If no summary message, create a basic one
                combined_summary = json.dumps({
                    "summary": "用户输入处理完成",
                    "next_node": next_node
                }, ensure_ascii=False)
        except json.JSONDecodeError as e:
            print(f"❌ 解析summary_message时出错: {e}")
            # Fallback to basic structure
            combined_summary = json.dumps({
                "summary": "用户输入处理完成，但解析过程中出现错误",
                "next_node": next_node
            }, ensure_ascii=False)
        
        print(f"📊 合并后的总结: {combined_summary}")
        print("✅ _combine_summary_and_decide_next_node 执行完成")
        print("=" * 50)
        
        return {"summary_message": combined_summary}

    def run_process_user_input_agent(self, session_id: str = "1", previous_AI_messages: BaseMessage = None, 
                                     current_node: str = "", village_name: str = "") -> List:
        """This function runs the process user input agent using invoke method instead of streaming"""
        print("\n🚀 开始运行 ProcessUserInputAgent")
        print("=" * 60)
        
        initial_state = self.create_initial_state(session_id=session_id, previous_AI_messages=previous_AI_messages, 
                                                  current_node=current_node, village_name=village_name)
        config = {"configurable": {"thread_id": session_id}}
        
        print(f"📋 会话ID: {session_id}")
        print(f"📝 初始状态已创建")
        print("🔄 正在执行用户输入处理工作流...")
        
        try:
            # Use invoke instead of stream for simpler execution
            while True:
                final_state = self.graph.invoke(initial_state, config=config)
                if "__interrupt__" in final_state:
                    interrupt_value = final_state["__interrupt__"][0].value
                    print(f"💬 智能体: {interrupt_value}")
                    user_response = input("👤 请输入您的回复: ")
                    initial_state = Command(resume=user_response)
                    continue

                print("🎉执行完毕")
                summary_message = final_state.get("summary_message", "")
                template_file = final_state.get("template_file_path", "")
                print(f"🔍 返回信息测试summary: {summary_message}")
                print(f"🔍 返回信息测试template: {template_file}")
                combined_message = [summary_message, template_file]
                print(f"🔍 返回信息测试combined: {combined_message}")
                return combined_message
            
        except Exception as e:
            print(f"❌ 执行过程中发生错误: {e}")
            # Return empty results on error
            error_summary = json.dumps({
                "summary": f"处理用户输入时发生错误: {str(e)}",
                "next_node": "design_excel_template"
            }, ensure_ascii=False)
            return [error_summary, ""]



# Langgraph studio to export the compiled graph
agent = ProcessUserInputAgent()
graph = agent.graph


if __name__ == "__main__":
    agent = ProcessUserInputAgent()
    # save_graph_visualization(agent.graph, "process_user_input_graph.png")
    agent.run_process_user_input_agent(current_node="design_excel_template")