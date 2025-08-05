from typing import Dict, List, Optional, Any, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
import os
import time
import random
from pathlib import Path
import base64
from openai import RateLimitError, APIError
import requests

from utils.screen_shot import ExcelTableScreenshot


def _handle_rate_limit_with_backoff(func, max_retries: int = 6, base_delay: float = 1.0, max_delay: float = 60.0, silent_mode: bool = False):
    """
    Handle rate limit errors with exponential backoff retry logic.
    
    Args:
        func: Function to execute with retry logic
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay in seconds
        silent_mode: Whether to suppress logging output
        
    Returns:
        Function result on success
        
    Raises:
        Exception: Re-raises the last exception if all retries failed
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            
            # Check if this is a rate limit error
            is_rate_limit_error = False
            retry_after = None
            
            # Handle different types of rate limit errors
            if hasattr(e, 'status_code') and e.status_code == 429:
                is_rate_limit_error = True
                # Try to extract retry-after header
                if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                    retry_after = e.response.headers.get('retry-after')
            elif 'rate limit' in str(e).lower() or '429' in str(e) or 'too many requests' in str(e).lower():
                is_rate_limit_error = True
            elif isinstance(e, RateLimitError):
                is_rate_limit_error = True
                
            if not is_rate_limit_error or attempt >= max_retries:
                # Not a rate limit error or max retries reached
                break
                
            # Calculate delay with exponential backoff
            if retry_after:
                try:
                    delay = float(retry_after)
                    if not silent_mode:
                        print(f"⏳ Rate limit hit, server requested {delay}s wait (attempt {attempt + 1}/{max_retries + 1})")
                except (ValueError, TypeError):
                    delay = min(base_delay * (2 ** attempt), max_delay)
            else:
                # Exponential backoff with jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                # Add jitter to prevent thundering herd
                delay += random.uniform(0, delay * 0.1)
                
            if not silent_mode:
                print(f"⏳ Rate limit detected, waiting {delay:.1f}s before retry (attempt {attempt + 1}/{max_retries + 1})")
                
            time.sleep(delay)
    
    # All retries failed, raise the last exception
    if not silent_mode:
        print(f"❌ All {max_retries + 1} attempts failed due to rate limiting")
    raise last_exception


def invoke_model(model_name : str, messages : List[BaseMessage], temperature: float = 0.2, silent_mode: bool = False) -> str:
    """调用大模型 with automatic rate limit retry"""
    if not silent_mode:
        print(f"🚀 开始调用LLM: {model_name} (temperature={temperature})")
    
    def _make_api_call():
        start_time = time.time()
        
        if model_name.startswith("gpt-"):  # ChatGPT 系列模型
            if not silent_mode:
                print("🔍 使用 OpenAI ChatGPT 模型")
            base_url = "https://api.openai.com/v1"
            api_key = os.getenv("OPENAI_API_KEY")
        else:  # 其他模型，例如 deepseek, siliconflow...
            if not silent_mode:
                print("🔍 使用 SiliconFlow 模型")
            base_url = "https://api.siliconflow.cn/v1"
            api_key = os.getenv("SILICONFLOW_API_KEY")
        
        llm = ChatOpenAI(
            model = model_name,
            api_key=api_key, 
            base_url=base_url,
            streaming=not silent_mode,  # Disable streaming in silent mode
            temperature=temperature,
            timeout=200  # network timeout
        )

        full_response = ""
        total_tokens_used = {"input": 0, "output": 0, "total": 0}
        
        if silent_mode:
            # Silent mode: use invoke instead of stream
            response = llm.invoke(messages)
            full_response = response.content
            
            # Extract token usage from response
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                total_tokens_used["input"] = usage.get('input_tokens', 0)
                total_tokens_used["output"] = usage.get('output_tokens', 0)
                total_tokens_used["total"] = usage.get('total_tokens', 0)
        else:
            # Normal mode: use streaming
            for chunk in llm.stream(messages):
                chunk_content = chunk.content
                print(chunk_content, end="", flush=True)
                full_response += chunk_content
                
                # Extract token usage if available in chunk
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    usage = chunk.usage_metadata
                    total_tokens_used["input"] = usage.get('input_tokens', 0)
                    total_tokens_used["output"] = usage.get('output_tokens', 0)
                    total_tokens_used["total"] = usage.get('total_tokens', 0)
                
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Print timing and token usage only if not in silent mode
        if not silent_mode:
            print(f"\n⏱️ LLM调用完成，耗时: {execution_time:.2f}秒")
            if total_tokens_used["total"] > 0:
                print(f"📊 Token使用: 输入={total_tokens_used['input']:,} | 输出={total_tokens_used['output']:,} | 总计={total_tokens_used['total']:,}")
        
        return full_response
    
    # Use rate limit retry wrapper
    try:
        return _handle_rate_limit_with_backoff(_make_api_call, silent_mode=silent_mode)
    except Exception as e:
        if not silent_mode:
            print(f"\n❌ LLM调用最终失败，错误: {e}")
        raise


def invoke_model_with_tools(model_name : str, messages : List[BaseMessage], tools : List[str], temperature: float = 0.2) -> Any:
    """调用大模型并使用工具 with automatic rate limit retry"""
    print(f"🚀 开始调用LLM(带工具): {model_name} (temperature={temperature})")
    
    def _make_api_call_with_tools():
        start_time = time.time()
        
        if model_name.startswith("gpt-"):  # ChatGPT 系列模型
            print("🔍 使用 OpenAI ChatGPT 模型")
            base_url = "https://api.openai.com/v1"
            api_key = os.getenv("OPENAI_API_KEY")
        else:  # 其他模型，例如 deepseek, siliconflow...
            print("🔍 使用 SiliconFlow 模型")
            base_url = "https://api.siliconflow.cn/v1"
            api_key = os.getenv("SILICONFLOW_API_KEY")

        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            streaming=False,
            temperature=temperature,
            timeout=200
        )
        
        # 绑定工具到模型
        llm_with_tools = llm.bind_tools(tools)
        
        print("📤 正在调用LLM...")
        
        response = llm_with_tools.invoke(messages)
        
        print("📥 LLM响应接收完成")
        
        # 打印响应内容（如果有）
        if response.content:
            print(f"\n💬 LLM回复内容:")
            print(response.content)
        
        # Extract token usage information
        token_usage = {"input": 0, "output": 0, "total": 0, "reasoning": 0}
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            token_usage["input"] = usage.get('input_tokens', 0)
            token_usage["output"] = usage.get('output_tokens', 0)
            token_usage["total"] = usage.get('total_tokens', 0)
            
            # Check for reasoning tokens (for reasoning models like Qwen3-32B)
            if 'output_token_details' in usage and usage['output_token_details']:
                token_usage["reasoning"] = usage['output_token_details'].get('reasoning', 0)
        
        # 检查是否有工具调用
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"\n🔧 检测到 {len(response.tool_calls)} 个工具调用:")
            
            # 打印每个工具调用的详细信息
            for i, tool_call in enumerate(response.tool_calls):
                print(f"\n📋 工具调用 {i+1}:")
                print(f"   🔧 工具名称: {tool_call.get('name', 'unknown')}")
                
                # 提取工具参数
                args = tool_call.get('args', {})
                print(f"   📝 参数: {args}")
                
                # 如果是用户交互工具，特别显示问题
                if tool_call.get('name') == 'request_user_clarification':
                    question = args.get('question', '')
                    context = args.get('context', '')
                    if question:
                        print(f"\n💬 ⭐ 用户问题: {question}")
                        if context:
                            print(f"📖 上下文: {context}")
                elif tool_call.get('name') == '_collect_user_input':
                    print(f"\n🔄 将收集用户输入信息")
                    session_id = args.get('session_id', '')
                    if session_id:
                        print(f"📋 会话ID: {session_id}")
            
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"\n⏱️ LLM调用完成(带工具调用)，耗时: {execution_time:.2f}秒")
        else:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"\n⏱️ LLM调用完成(无工具调用)，耗时: {execution_time:.2f}秒")
        
        # Print token usage information
        if token_usage["total"] > 0:
            print(f"📊 Token使用: 输入={token_usage['input']:,} | 输出={token_usage['output']:,} | 总计={token_usage['total']:,}")
            if token_usage["reasoning"] > 0:
                print(f"🧠 推理Token: {token_usage['reasoning']:,} (内部推理过程)")
                visible_output = token_usage["output"] - token_usage["reasoning"]
                print(f"👀 可见输出Token: {visible_output:,}")
        else:
            print("⚠️ 未能获取Token使用信息")
        
        # 返回完整响应以便调用者处理
        return response
    
    # Use rate limit retry wrapper
    try:
        return _handle_rate_limit_with_backoff(_make_api_call_with_tools, silent_mode=False)
    except Exception as e:
        print(f"\n❌ LLM调用最终失败，错误: {e}")
        import traceback
        traceback.print_exc()
        raise


def invoke_model_with_screenshot(model_name : str, file_path : str, temperature: float = 0.2) -> Any:
    """调用大模型并使用截图 with automatic rate limit retry"""
    print(f"🚀 开始调用LLM(带截图): {model_name} (temperature={temperature})")

    path = Path(file_path)
    screen_shot_path = path.with_suffix(".png")

    file_name = path.name

    excelTableScreenshot = ExcelTableScreenshot()
    excelTableScreenshot.take_screenshot(file_path, screen_shot_path)

    with open(screen_shot_path, "rb") as image_file:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

    human_message = HumanMessage(content=[
    {
        "type": "text",
        "text": "请识别图片中的文字"
    },
    {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_base64}"
        }
    }
    ])

    system_message = SystemMessage(content=f"""
你是一位专业的表格结构分析专家，擅长从复杂的 Excel 或 HTML 表格中提取完整的多级表头结构，并结合数据内容辅助理解字段含义、层级和分类汇总关系。

请根据用户提供的表格，完成以下任务：

【任务目标】

1. 提取完整的表头层级结构：
   - 从表格主标题开始，逐层提取所有分类表头、字段表头；
   - 结合实际数据，辅助判断字段的含义、分类、层级归属，确保结构理解准确；
   - 使用嵌套的 key-value 结构，表达表头的层级关系；
   - 每一级表头都应清晰反映其子级字段或子分类，避免遗漏或误分类。

2. 采用「值 / 分解 / 规则」结构识别分类汇总关系：
   - 如果某个父级字段**自身有数据**，同时又包含多个子字段（如"保障人数"、"领取金额"），必须采用以下格式：

   {{
     "字段名": {{
       "值": [],        // 表示该字段自身的数据（即原表格中的单元格数据）
       "分解": {{        // 表示该字段下的子分类或子字段
         "子字段1": [],
         "子字段2": []
       }},
       "规则": "子字段1 + 子字段2"  // 如果该字段的值等于子字段的加总或有其他规则，请自动推理并写明规则。如果无规律可循，返回空字符串。
     }}
   }}

   - 如果父级字段只是分类（自身无数据），只输出 "分解"；
   - 如果某个字段既没有子分类，也没有子字段拆分，直接输出为：

   {{
     "字段名": []
   }}

3. 辅助推理字段的计算规则：
   - 检查父级字段的值与其子字段数据的关系，自动推理是否有加法、减法、比例、特殊逻辑等；
   - 如果发现规律，完整输出公式描述（如："子字段1 + 子字段2" 或 "(子字段1 + 子字段2) × 1.2"）；
   - 如果父级字段的值和子字段数据无明显关系，输出 "规则": ""。

4. 辅助判断字段含义：
   - 结合数据内容辅助判断字段用途，避免仅依赖表头文字表面拆分；
   - 识别重复字段、并列字段、合并单元格带来的结构层级；
   - 对于存在数据但表头描述模糊的情况，尽量根据数据判断其真实分类。

【输出格式要求】

- 严格输出为**标准 JSON 格式**，不能有 markdown、代码块标记或其他多余文字；
- 只输出表头结构，**不要输出表格总结、描述、用途说明或数据样例**；
- JSON 的根键名必须为 {file_name}（严格保留，不能更改）；
- 每一层级都以对象形式描述，遵循以下结构：

【输出示例】

{{
  "{file_name}": {{
    "表格结构": {{
      "序号": [],
      "户主姓名": [],
      "低保证号": [],
      "身份证号码": [],
      "保障人数": {{
        "值": [],
        "分解": {{
          "重点保障人数": [],
          "残疾人数": []
        }},
        "规则": "重点保障人数 + 残疾人数"
      }},
      "领取金额": {{
        "值": [],
        "分解": {{
          "家庭补差": [],
          "重点救助60元": [],
          "重点救助100元": [],
          "残疾人救助": []
        }},
        "规则": "家庭补差 + 重点救助60元 + 重点救助100元 + 残疾人救助"
      }},
      "领款人签字(章)": [],
      "领款时间": []
    }}
  }}
}}

【特别注意】

- 所有输出必须为严格的 JSON 结构，但不要将输出包裹在 ```json 和 ``` 中；
- 不允许输出表格总结、描述、元数据或用途说明；
- 不允许有 markdown、代码块标记或多余的格式符号；
- 保持层级结构清晰，完整保留所有分类表头、子表头和字段表头；
- **父级字段有数据时，必须采用 "值 / 分解 / 规则" 结构，确保数据与分类信息、计算关系都保留**；
- 必须结合数据辅助判断字段含义，确保分类、层级和汇总逻辑准确；
- 规则字段需自动推理子字段与父字段之间的计算关系，找不到规则时返回空字符串。
""")


    messages = [system_message, human_message]

    # Use the rate limit retry invoke_model (which already has retry logic)
    response = invoke_model(model_name, messages)

    return response
    
    
    