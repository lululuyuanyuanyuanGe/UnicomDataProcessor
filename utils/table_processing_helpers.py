"""
Helper functions for table processing in fileProcessAgent
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


def extract_headers_from_response(response: str) -> list[str]:
    """Extract headers from LLM response"""
    try:
        # Try to parse JSON response first
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


def parse_llm_table_response(response: str) -> Dict[str, any]:
    """
    Parse LLM response to extract table name and headers
    
    Expected fixed response format:
    {
        "filename.xls": {
            "表格结构": {
                "表头1": [],
                "表头2": [],
                "表头3": []
            }
        }
    }
    """
    try:
        # Try to parse as JSON first
        if '{' in response and '}' in response:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    
                    # Extract filename (the top-level key) and remove suffix
                    table_name = ""
                    headers = []
                    
                    # Get the first (and should be only) key which is the filename
                    filename_key = next(iter(data.keys()))
                    if filename_key:
                        # Remove file extension to get clean table name
                        table_name = Path(filename_key).stem
                        
                        # Navigate to 表格结构 to get headers
                        file_data = data[filename_key]
                        if isinstance(file_data, dict) and "表格结构" in file_data:
                            table_structure = file_data["表格结构"]
                            if isinstance(table_structure, dict):
                                headers = list(table_structure.keys())
                    
                    if table_name or headers:
                        return {
                            "table_name": table_name,
                            "headers": headers,
                            "success": True
                        }
                        
                except json.JSONDecodeError as je:
                    print(f"JSON解析失败: {je}")
                except Exception as pe:
                    print(f"处理JSON数据失败: {pe}")
        
        # Fallback: try to extract from plain text (keep original fallback logic)
        lines = response.split('\n')
        table_name = ""
        headers = []
        
        for line in lines:
            line = line.strip()
            # Look for table name patterns
            if any(keyword in line for keyword in ["表格名", "文件名", "表名", "名称"]) and ":" in line:
                table_name = line.split(':', 1)[1].strip().strip('"\'')
            # Look for headers patterns
            elif any(keyword in line for keyword in ["表头", "字段", "列名"]) and ":" in line:
                header_text = line.split(':', 1)[1].strip()
                # Split by comma, semicolon, or other delimiters
                headers = [h.strip().strip('"\'') for h in re.split(r'[,，;；]', header_text)]
        
        return {
            "table_name": table_name,
            "headers": headers,
            "success": bool(table_name or headers)
        }
        
    except Exception as e:
        print(f"解析LLM响应失败: {e}")
        return {
            "table_name": "",
            "headers": [],
            "success": False,
            "error": str(e)
        }


def generate_fallback_table_name(filename: str) -> str:
    """Generate fallback table name from filename"""
    try:
        # Try to extract from filename first
        filename_stem = Path(filename).stem
        
        # Remove common suffixes and numbers
        cleaned_name = re.sub(r'_\d{8}_\d{6}$', '', filename_stem)  # Remove timestamp
        cleaned_name = re.sub(r'\d+$', '', cleaned_name)  # Remove trailing numbers
        
        # Fallback to cleaned filename
        if len(cleaned_name) > 0 and not cleaned_name.isdigit():
            return cleaned_name
        else:
            return f"表格_{datetime.now().strftime('%m%d')}"
            
    except Exception as e:
        print(f"⚠️ 生成后备表格名失败: {e}")
        return f"未知表格_{datetime.now().strftime('%m%d%H%M')}"


def create_table_description(table_name: str, headers: List[str]) -> str:
    """Create table description for embedding"""
    if headers:
        return f"{table_name} 包含表头：{','.join(headers)}"
    else:
        return table_name