def clean_json_response(response: str) -> str:
    """
    Clean JSON response by removing markdown code blocks and handling multiple JSON objects.
    
    Args:
        response (str): Raw response from LLM that may contain markdown code blocks
        
    Returns:
        str: Cleaned JSON string
    """
    if not response:
        return ""
    
    cleaned_response = response.strip()
    
    # Remove markdown code blocks if present
    if '```json' in cleaned_response:
        print("🔍 检测到JSON代码块，正在清理...")
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
    
    # Handle multiple JSON objects with proper JSON parsing instead of naive string splitting
    import json
    
    # Try to parse as single JSON first
    try:
        json.loads(cleaned_response)
        # If successful, it's a single valid JSON object
        print(f"🔍 清理后的JSON响应长度: {len(cleaned_response)} 字符")
        return cleaned_response
    except json.JSONDecodeError:
        # If failed, try to extract the first valid JSON object
        print("⚠️ 检测到可能的多个JSON对象或格式问题，尝试提取第一个有效JSON")
        
        # Try to find the first complete JSON object
        brace_count = 0
        start_pos = -1
        
        for i, char in enumerate(cleaned_response):
            if char == '{':
                if start_pos == -1:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    # Found a complete JSON object
                    potential_json = cleaned_response[start_pos:i+1]
                    try:
                        json.loads(potential_json)
                        print(f"🔍 提取的JSON响应长度: {len(potential_json)} 字符")
                        return potential_json
                    except json.JSONDecodeError:
                        # Continue looking for the next complete JSON object
                        start_pos = -1
                        continue
        
        # If no valid JSON found, return the original cleaned response
        print("⚠️ 无法提取有效的JSON对象，返回原始清理后的响应")
        print(f"🔍 清理后的JSON响应长度: {len(cleaned_response)} 字符")
        return cleaned_response


def clean_html_response(response: str) -> str:
    """
    Clean HTML response by removing markdown code blocks.
    
    Args:
        response (str): Raw response from LLM that may contain markdown code blocks
        
    Returns:
        str: Cleaned HTML string
    """
    if not response:
        return ""
    
    cleaned_response = response.strip()
    
    # Remove markdown code blocks if present
    if '```html' in cleaned_response:
        print("🔍 检测到HTML代码块，正在清理...")
        # Extract content between ```html and ```
        start_marker = '```html'
        end_marker = '```'
        start_index = cleaned_response.find(start_marker)
        if start_index != -1:
            start_index += len(start_marker)
            end_index = cleaned_response.find(end_marker, start_index)
            if end_index != -1:
                cleaned_response = cleaned_response[start_index:end_index].strip()
            else:
                # If no closing ```, take everything after ```html
                cleaned_response = cleaned_response[start_index:].strip()
    elif '```' in cleaned_response:
        print("🔍 检测到通用代码块，正在清理...")
        # Handle generic ``` blocks
        parts = cleaned_response.split('```')
        if len(parts) >= 3:
            # Take the middle part (index 1)
            cleaned_response = parts[1].strip()
    
    print(f"🔍 清理后的HTML响应长度: {len(cleaned_response)} 字符")
    return cleaned_response


def clean_code_response(response: str, code_type: str = "code") -> str:
    """
    Generic function to clean code responses by removing markdown code blocks.
    
    Args:
        response (str): Raw response from LLM that may contain markdown code blocks
        code_type (str): Type of code block to look for (e.g., "python", "javascript", "sql")
        
    Returns:
        str: Cleaned code string
    """
    if not response:
        return ""
    
    cleaned_response = response.strip()
    
    # Remove markdown code blocks if present
    specific_marker = f'```{code_type}'
    if specific_marker in cleaned_response:
        print(f"🔍 检测到{code_type}代码块，正在清理...")
        # Extract content between ```{code_type} and ```
        start_marker = specific_marker
        end_marker = '```'
        start_index = cleaned_response.find(start_marker)
        if start_index != -1:
            start_index += len(start_marker)
            end_index = cleaned_response.find(end_marker, start_index)
            if end_index != -1:
                cleaned_response = cleaned_response[start_index:end_index].strip()
            else:
                # If no closing ```, take everything after ```{code_type}
                cleaned_response = cleaned_response[start_index:].strip()
    elif '```' in cleaned_response:
        print("🔍 检测到通用代码块，正在清理...")
        # Handle generic ``` blocks
        parts = cleaned_response.split('```')
        if len(parts) >= 3:
            # Take the middle part (index 1)
            cleaned_response = parts[1].strip()
    
    print(f"🔍 清理后的{code_type}响应长度: {len(cleaned_response)} 字符")
    return cleaned_response
