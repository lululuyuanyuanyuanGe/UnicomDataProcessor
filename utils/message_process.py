from __future__ import annotations
from bs4 import BeautifulSoup, Tag
from pathlib import Path

from typing import TypedDict, Annotated, List
import re
import os
from pathlib import Path
import subprocess
import chardet

from openai import OpenAI
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage


def build_BaseMessage_type(messages:list[dict], file_paths : list[str] = None) -> list[BaseMessage]:
    """"将消息队列转换成LangChain的消息模板"""
    langchain_messages = []
    for msg in messages:
        if msg["role"] == "system":
            langchain_messages.append(SystemMessage(content = msg["content"]))
        elif msg["role"] == "user":
            # 判断是否为复杂输入(包含文件)
            if isinstance(msg["content"], list):
                # 将用户文本输入存储在 contenxt_text
                contexnt_text = next((item["text"] for item in msg["content"] if item["type"] == "text"), "")
                file_refs = [item["file_id"] for item in msg["content"] if item["type"] == "input_file"]
                user_input = F"{contexnt_text} + input files list: {' '.join(file_refs)}"
                human_msg = HumanMessage(
                    content= user_input,
                    additional_kargs = {
                        "filer_ids": file_refs,
                        "multimodal_content": msg["content"]
                    }
                )
                langchain_messages.append(human_msg)
        else:
            langchain_messages.append(HumanMessage(content=msg["content"]))

    return langchain_messages

def filter_out_system_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """辅助函数过滤消息队列中的系统提示词消息"""
    return [message for message in messages if not isinstance(message, SystemMessage)]
