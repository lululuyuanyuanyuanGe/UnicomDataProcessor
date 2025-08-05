import sys
from pathlib import Path
import json

# Add root project directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))



from typing import Dict, List, Optional, Any, TypedDict, Annotated, Union

from utils.modelRelated import invoke_model, invoke_model_with_tools

from pathlib import Path
# Create an interactive chatbox using gradio
from dotenv import load_dotenv


from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# import other agents
from agents.processUserInput import ProcessUserInputAgent

load_dotenv()

def append_strings(left: list[str], right: Union[list[str], str]) -> list[str]:
    """Custom reducer to append strings to a list"""
    if isinstance(right, list):
        return left + right
    else:
        return left + [right]
    

@tool
def _collect_user_input(session_id: str, AI_question: str, village_name: str) -> str:
    """这是一个用来收集用户输入的工具，你需要调用这个工具来收集用户输入
    参数：
        session_id: 当前会话ID
        AI_question: 大模型的问题
        village_name: 当前村名
    返回：
        str: 总结后的用户输入信息
    """

    print(f"🔄 开始收集用户输入，当前会话ID: {session_id}")
    print(f"💬 AI问题: {AI_question}")
    
    processUserInputAgent = ProcessUserInputAgent()
    ai_message = AIMessage(content=AI_question)
    response = processUserInputAgent.run_process_user_input_agent(session_id = session_id, 
                                                                  previous_AI_messages = ai_message,
                                                                  village_name = village_name)
    print(f"🔄 返回响应: {response[:100]}...")
    return response
    

class FrontdeskState(TypedDict):
    chat_history: Annotated[list[str], append_strings]
    messages: Annotated[list[BaseMessage], add_messages]
    previous_node: str # Track the previous node
    session_id: str
    village_name: str


class FrontdeskAgent:
    """
    用于处理用户上传的模板，若未提供模板，和用户沟通确定表格结构
    """



    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.tools = [_collect_user_input]
        self.graph = self._build_graph()




    def _build_graph(self):
        """This function will build the graph of the frontdesk agent"""

        graph = StateGraph(FrontdeskState)

        graph.add_node("entry", self._entry_node)
        graph.add_node("collect_user_input", ToolNode(self.tools))
        graph.add_node("initial_collect_user_input", self._initial_collect_user_input)


        graph.add_edge(START, "entry")
        graph.add_edge("entry", "initial_collect_user_input")
        graph.add_conditional_edges("initial_collect_user_input", self._route_after_initial_collect_user_input,
                                    {
                                        "initial_collect_user_input": "initial_collect_user_input"
                                    })
        graph.add_conditional_edges("collect_user_input", self._route_after_collect_user_input)

        
        # Compile the graph to make it executable with stream() method
        # You can add checkpointer if needed: graph.compile(checkpointer=MemorySaver())
        return graph.compile()



    def _create_initial_state(self, session_id: str = "1", village_name: str = "") -> FrontdeskState:
        """This function will create the initial state of the frontdesk agent"""
        return {
            "chat_history": [],
            "messages": [],
            "session_id": session_id,
            "previous_node": "",
            "village_name": village_name
        }


    def _entry_node(self, state: FrontdeskState) -> FrontdeskState:
        """This is the starting node of our frontdesk agent"""
        print("\n🚀 开始执行: _entry_node")
        print("=" * 50)
        
        # Enrich this later, it should include a short description of the agent's ability and how to use it
        welcome_message = "你好，我是一个表格处理助手！"
        print(f"💬 欢迎消息: {welcome_message}")
        
        print("✅ _entry_node 执行完成")
        print("=" * 50)
        
        return {
            "messages": [AIMessage(content=welcome_message)],
        }
    

    def _initial_collect_user_input(self, state: FrontdeskState) -> FrontdeskState:
        """调用ProcessUserInputAgent来收集用户输入"""
        print("\n🔍 开始执行: _initial_collect_user_input")
        print("=" * 50)
        
        session_id = state["session_id"]
        previous_AI_messages = state["messages"][-1]
        
        print(f"📋 会话ID: {session_id}")
        print("🔄 正在调用ProcessUserInputAgent...")
        
        processUserInputAgent = ProcessUserInputAgent()
        summary_message = processUserInputAgent.run_process_user_input_agent(session_id = session_id, 
                                                                             previous_AI_messages = previous_AI_messages,
                                                                             village_name = state["village_name"],
                                                                             current_node = "initial_collect_user_input")
        print(f"📥 原始返回信息：{summary_message}")
        
        # Handle the case where summary_message might be None
        if summary_message is None or len(summary_message) < 2:
            error_msg = "用户输入处理失败，请重新输入"
            print(f"❌ {error_msg}")
            print("✅ _initial_collect_user_input 执行完成(错误)")
            print("=" * 50)
            return {
                "messages": [AIMessage(content=error_msg)],
                "template_file_path": summary_message[1],
                "previous_node": "initial_collect_user_input"
            }
            
        print(f"📊 返回信息JSON dump：{json.dumps(summary_message[0])}")
        
        print("✅ _initial_collect_user_input 执行完成")
        print("=" * 50)
        print("tempalte_file_paath初始化: ", summary_message[1])
        return {
            "messages": [AIMessage(content=summary_message[0])],
            "template_file_path": summary_message[1],
        }
        
    def _route_after_initial_collect_user_input(self, state: FrontdeskState) -> str:
        """初始调用ProcessUserInputAgent后，根据返回信息决定下一步的流程"""
        print("\n🔀 开始执行: _route_after_initial_collect_user_input")
        print("=" * 50)
        
        content = state['messages'][-1].content
        print(f"📋 state测试: {content}")
        
        # Check if content is JSON or plain text error message
        try:
            summary_message = json.loads(content)
            print(f"📊 summary_message测试: {summary_message}")
            next_node = summary_message.get("next_node", "previous_node")
            print(f"🔄 路由决定: {next_node}")
            
            print("✅ _route_after_initial_collect_user_input 执行完成")
            print("=" * 50)
                
            if next_node == "complex_template":
                # Complex template handling not implemented yet, fallback to simple template
                print("⚠️ 复杂模板处理暂未实现，转为简单模板处理")
                return "simple_template_handle"
            elif next_node == "simple_template":
                return "simple_template_handle"
            else:
                return next_node  # Fallback to previous node
                
        except json.JSONDecodeError:
            # Content is plain text error message, not JSON
            print("❌ 内容不是有效的JSON，可能是错误消息")
            print("🔄 路由到 chat_with_user_to_determine_template 重新开始")
            print("✅ _route_after_initial_collect_user_input 执行完成")
            print("=" * 50)
            return "chat_with_user_to_determine_template"
        

    def _route_after_collect_user_input(self, state: FrontdeskState) -> str:
        """This node will route the agent to the next node based on the summary message from the ProcessUserInputAgent"""
        print("\n🔀 开始执行: _route_after_collect_user_input")
        print("=" * 50)
        
        latest_message = state["messages"][-1]
        
        # This is a regular message, try to parse as JSON for routing
        summary_message_str = latest_message.content
        print(f"📋 原始内容: {summary_message_str}")
        
        try:
            summary_message_json = json.loads(summary_message_str)
            summary_message = json.loads(summary_message_json[0])
            print(f"📊 summary_message测试: {summary_message}")
            next_node = summary_message.get("next_node", "previous_node")
            print(f"🔄 路由决定: {next_node}")
            
            print("✅ _route_after_collect_user_input 执行完成")
            print("=" * 50)
                
            if next_node == "complex_template":
                # Complex template handling not implemented yet, fallback to simple template
                print("⚠️ 复杂模板处理暂未实现，转为简单模板处理")
                return "simple_template_handle"
            elif next_node == "simple_template":
                return "simple_template_handle"
            else:
                return state.get("previous_node", "entry")  # Fallback to previous node
                
        except json.JSONDecodeError:
            # Content is plain text error message, not JSON
            print("❌ 内容不是有效的JSON，可能是错误消息")
            print("🔄 路由到 chat_with_user_to_determine_template 重新开始")
            print("✅ _route_after_collect_user_input 执行完成")
            print("=" * 50)
            return "chat_with_user_to_determine_template"
            

    
    def run_frontdesk_agent(self, session_id: str = "1", village_name: str = "") -> None:
        """This function will run the frontdesk agent using stream method with interrupt handling"""
        print("\n🚀 启动 FrontdeskAgent")
        print("=" * 60)
        
        initial_state = self._create_initial_state(session_id, village_name)
        config = {"configurable": {"thread_id": session_id}}
        current_state = initial_state

        while True:
            try:
                print(f"\n🔄 执行状态图，当前会话ID: {session_id}")
                print("-" * 50)
                
                final_state = self.graph.invoke(current_state, config = config)
                if "__interrupt__" in final_state:
                    interrupt_value = final_state["__interrupt__"][0].value
                    print(f"💬 智能体: {interrupt_value}")
                    user_response = input("👤 请输入您的回复: ")
                    current_state = Command(resume=user_response)
                    continue
                print("FrontdeskAgent执行完毕")
                break
                
            except Exception as e:
                print(f"❌ 执行过程中发生错误: {e}")
                print(f"错误类型: {type(e).__name__}")
                print("-" * 50)
                break

            

frontdesk_agent = FrontdeskAgent()
graph = frontdesk_agent.graph



if __name__ == "__main__":

    frontdesk_agent = FrontdeskAgent()
    frontdesk_agent.run_frontdesk_agent(village_name="燕云村")