import json
import csv
import os
import sys
from bs4 import BeautifulSoup
from pathlib import Path

# Add root project directory to sys.path if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Safe print function that handles encoding issues
def safe_print(*args, **kwargs):
    """Print function that handles Unicode encoding issues on Windows"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Convert all args to ASCII-safe versions
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                # Replace problematic characters with safe alternatives
                safe_arg = arg.encode('ascii', errors='replace').decode('ascii')
                safe_args.append(safe_arg)
            else:
                safe_args.append(str(arg))
        print(*safe_args, **kwargs)

from utils.file_process import read_txt_file


def generate_header_html(json_data: dict) -> str:
    """
    Generate HTML table structure from JSON data.
    Supports mixed structure with simple fields (arrays) and complex fields (with 值/分解/规则).
    
    Args:
        json_data: Dictionary containing 表格标题 and 表格结构
        
    Returns:
        str: HTML table code
    """
    try:
        # Parse JSON if it's a string
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        
        table_title = data.get("表格标题", "表格")
        table_structure = data.get("表格结构", {})
        
        safe_print(f"Processing table structure: {table_structure}")
        
        # Analyze structure and build column layout information
        columns_layout = []  # [(field_name, parent_name, level, has_subfields, has_parent_value)]
        column_spans = {}  # field_name -> span count
        parent_value_fields = set()  # track fields that have their own value cells
        max_levels = 1
        
        def analyze_field(field_name, field_value, parent_name=None, level=1):
            nonlocal max_levels
            max_levels = max(max_levels, level)
            
            if isinstance(field_value, list):
                # Simple field: field_name -> []
                columns_layout.append((field_name, parent_name, level, False, False))
                column_spans[field_name] = 1
                return 1
                
            elif isinstance(field_value, dict) and "分解" in field_value:
                # Complex field with sub-fields in "分解"
                fenjie = field_value.get("分解", {})
                zhi = field_value.get("值", [])
                
                # Check if parent field has its own value (non-empty "值" array)
                has_parent_value = isinstance(zhi, list) and len(zhi) > 0 and any(str(item).strip() for item in zhi)
                
                if fenjie:
                    # This is a parent field with sub-fields
                    columns_layout.append((field_name, parent_name, level, True, has_parent_value))
                    
                    if has_parent_value:
                        parent_value_fields.add(field_name)
                    
                    # Process sub-fields
                    total_sub_span = 0
                    for sub_field_name, sub_field_value in fenjie.items():
                        sub_span = analyze_field(sub_field_name, sub_field_value, field_name, level + 1)
                        total_sub_span += sub_span
                    
                    # If parent has its own value, add 1 to the span count
                    parent_span = total_sub_span + (1 if has_parent_value else 0)
                    column_spans[field_name] = parent_span
                    return parent_span
                else:
                    # Complex field but no sub-fields (treat as simple)
                    columns_layout.append((field_name, parent_name, level, False, False))
                    column_spans[field_name] = 1
                    return 1
            else:
                # Unknown structure, treat as simple
                columns_layout.append((field_name, parent_name, level, False, False))
                column_spans[field_name] = 1
                return 1
        
        # Analyze all top-level fields
        for field_name, field_value in table_structure.items():
            analyze_field(field_name, field_value)
        
        # Count total columns (leaf fields + parent value fields count as actual columns)
        total_columns = 0
        for field_name, parent_name, level, has_subfields, has_parent_value in columns_layout:
            if not has_subfields:  # Leaf field
                total_columns += 1
            elif has_parent_value:  # Parent field with its own value cell
                total_columns += 1
        
        safe_print(f"Table analysis:")
        safe_print(f"   - Total columns: {total_columns}")
        safe_print(f"   - Max levels: {max_levels}")
        safe_print(f"   - Column layout: {columns_layout}")
        safe_print(f"   - Column spans: {column_spans}")
        safe_print(f"   - Parent value fields: {parent_value_fields}")
        
        # Generate colgroup for each actual column
        colgroup_html = "\n".join([f"<colgroup></colgroup>" for _ in range(total_columns)])
        
        # Generate complete HTML structure
        html_lines = [
            "<html><body><table>",
            colgroup_html,
            # Title row
            f'<tr><td colspan="{total_columns}"><b>{table_title}</b></td></tr>'
        ]
        
        # Generate header rows based on levels
        if max_levels > 1:
            # Multi-level header structure
            for current_level in range(1, max_levels + 1):
                row_html = ["<tr>"]
                
                if current_level < max_levels:
                    # Non-final levels: show parent fields with colspan and simple fields with rowspan
                    for field_name, parent_name, level, has_subfields, has_parent_value in columns_layout:
                        if level == current_level:
                            if has_subfields:
                                # Parent field - use colspan for all its sub-fields
                                span = column_spans[field_name]
                                row_html.append(f'<td colspan="{span}"><b>{field_name}</b></td>')
                            else:
                                # Simple field - needs rowspan to cover remaining levels
                                if parent_name is None:
                                    levels_to_span = max_levels - current_level + 1
                                    row_html.append(f'<td rowspan="{levels_to_span}"><b>{field_name}</b></td>')
                else:
                    # Final level: show all data cells (leaf fields + parent value fields)
                    for field_name, parent_name, level, has_subfields, has_parent_value in columns_layout:
                        if level == current_level:
                            # Leaf field at final level
                            if not has_subfields:
                                row_html.append(f'<td><b>{field_name}</b></td>')
                        elif has_subfields and has_parent_value and level < current_level:
                            # Parent field with value - add its value cell at final level
                            row_html.append(f'<td><b>{field_name}</b></td>')
                
                row_html.append("</tr>")
                html_lines.append("\n".join(row_html))
        
        else:
            # Single-level structure (all fields are simple)
            field_row = ["<tr>"]
            
            for field_name, _, _, has_subfields, has_parent_value in columns_layout:
                if not has_subfields:  # Only leaf fields
                    field_row.append(f'<td><b>{field_name}</b></td>')
            
            field_row.append("</tr>")
            html_lines.append("\n".join(field_row))
        
        # Add empty data row that matches the total column count
        empty_row = ["<tr>"]
        for _ in range(total_columns):
            empty_row.append("<td><br/></td>")
        empty_row.append("</tr>")
        html_lines.append("\n".join(empty_row))
        
        html_lines.append("</table></body></html>")
        
        result_html = "\n".join(html_lines)
        safe_print(f"HTML generation successful, length: {len(result_html)} characters")
        return result_html
        
    except Exception as e:
        # Fallback simple structure
        error_msg = f"<html><body><table><tr><td><b>表格生成错误: {str(e)}</b></td></tr></table></body></html>"
        safe_print(f"HTML生成失败: {e}")
        import traceback
        safe_print(f"错误详情: {traceback.format_exc()}")
        return error_msg


def extract_empty_row_html_code_based(template_file_path: str) -> str:
    """
    Extract empty row HTML template from template file using code-based approach.
    
    Args:
        template_file_path: Path to HTML template file
        
    Returns:
        str: HTML code for empty row template
    """
    print("\n🔄 开始执行: extract_empty_row_html_code_based")
    print("=" * 50)
    
    try:
        template_file_content = read_txt_file(template_file_path)
        print(f"📄 读取模板文件: {template_file_path}")
        
        # Parse HTML content
        soup = BeautifulSoup(template_file_content, 'html.parser')
        table = soup.find('table')
        
        if not table:
            print("❌ 未找到table元素")
            return ""
        
        # Find all rows
        rows = table.find_all('tr')
        print(f"📋 找到 {len(rows)} 行")
        
        # Look for empty row (contains <br/> or is mostly empty)
        empty_row = None
        for row in rows:
            cells = row.find_all('td')
            if cells and len(cells) > 1:  # Skip single-cell title rows
                # Check if this row has empty cells or <br/> tags
                empty_cell_count = 0
                for cell in cells:
                    if cell.find('br') or cell.get_text().strip() == '':
                        empty_cell_count += 1
                
                # If most cells are empty, this is likely our empty row template
                if empty_cell_count >= len(cells) - 1:  # Allow one cell to have content (like sequence number)
                    empty_row = row
                    break
        
        if empty_row:
            # Clean up the empty row - ensure all cells except first are empty
            cells = empty_row.find_all('td')
            for i, cell in enumerate(cells):
                if i == 0:
                    # First cell might have sequence number, make it empty
                    cell.clear()
                    cell.string = ""
                else:
                    # Other cells should be empty with <br/>
                    cell.clear()
                    cell.append(soup.new_tag('br'))
            
            empty_row_html = str(empty_row)
            print(f"✅ 找到空行模板: {empty_row_html}")
            print("✅ extract_empty_row_html_code_based 执行完成")
            print("=" * 50)
            return empty_row_html
        else:
            # If no empty row found, create one based on the table structure
            print("⚠️ 未找到空行，基于表头创建空行")
            header_row = None
            for row in rows:
                cells = row.find_all('td')
                if cells and len(cells) > 1 and not any(cell.get('colspan') for cell in cells):
                    header_row = row
                    break
            
            if header_row:
                # Create empty row based on header structure
                new_row = soup.new_tag('tr')
                header_cells = header_row.find_all('td')
                for i in range(len(header_cells)):
                    new_cell = soup.new_tag('td')
                    if i == 0:
                        new_cell.string = ""
                    else:
                        new_cell.append(soup.new_tag('br'))
                    new_row.append(new_cell)
                
                empty_row_html = str(new_row)
                print(f"✅ 创建空行模板: {empty_row_html}")
                print("✅ extract_empty_row_html_code_based 执行完成")
                print("=" * 50)
                return empty_row_html
            else:
                print("❌ 无法创建空行模板")
                return ""
    
    except Exception as e:
        print(f"❌ extract_empty_row_html_code_based 执行失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return ""


def extract_headers_html_code_based(template_file_path: str) -> str:
    """
    Extract headers HTML from template file using code-based approach.
    
    Args:
        template_file_path: Path to HTML template file
        
    Returns:
        str: HTML code for headers section
    """
    print("\n🔄 开始执行: extract_headers_html_code_based")
    print("=" * 50)
    
    try:
        template_file_content = read_txt_file(template_file_path)
        print(f"📄 读取模板文件: {template_file_path}")
        
        # Parse HTML content
        soup = BeautifulSoup(template_file_content, 'html.parser')
        
        # Find the table
        table = soup.find('table')
        if not table:
            print("❌ 未找到table元素")
            return ""
        
        # Get all rows
        rows = table.find_all('tr')
        print(f"📋 找到 {len(rows)} 行")
        
        # Find the first empty row (data row)
        first_empty_row_index = None
        for i, row in enumerate(rows):
            cells = row.find_all('td')
            if cells and len(cells) > 1:
                # Check if this row has empty cells or <br/> tags
                empty_cell_count = 0
                for cell in cells:
                    if cell.find('br') or cell.get_text().strip() == '':
                        empty_cell_count += 1
                
                # If most cells are empty, this is likely our first data row
                if empty_cell_count >= len(cells) - 1:
                    first_empty_row_index = i
                    break
        
        if first_empty_row_index is None:
            print("⚠️ 未找到空行，使用所有行作为表头")
            first_empty_row_index = len(rows)
        
        # Build header HTML
        header_parts = []
        header_parts.append("<html><body><table>")
        
        # Add colgroup if present
        colgroups = soup.find_all('colgroup')
        for colgroup in colgroups:
            header_parts.append(str(colgroup))
        
        # Add all rows before the first empty row
        for i in range(first_empty_row_index):
            header_parts.append(str(rows[i]))
        
        headers_html = '\n'.join(header_parts)
        print(f"✅ 提取表头HTML (包含 {first_empty_row_index} 行)")
        print("✅ extract_headers_html_code_based 执行完成")
        print("=" * 50)
        return headers_html
        
    except Exception as e:
        print(f"❌ extract_headers_html_code_based 执行失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return ""


def extract_footer_html_code_based(template_file_path: str) -> str:
    """
    Extract footer HTML from template file using code-based approach.
    
    Args:
        template_file_path: Path to HTML template file
        
    Returns:
        str: HTML code for footer section
    """
    print("\n🔄 开始执行: extract_footer_html_code_based")
    print("=" * 50)
    
    try:
        template_file_content = read_txt_file(template_file_path)
        print(f"📄 读取模板文件: {template_file_path}")
        
        # Parse HTML content
        soup = BeautifulSoup(template_file_content, 'html.parser')
        
        # Find the table
        table = soup.find('table')
        if not table:
            print("❌ 未找到table元素")
            return ""
        
        # Get all rows
        rows = table.find_all('tr')
        print(f"📋 找到 {len(rows)} 行")
        
        # Find the last empty row (data row)
        last_empty_row_index = None
        for i in range(len(rows) - 1, -1, -1):
            row = rows[i]
            cells = row.find_all('td')
            if cells and len(cells) > 1:
                # Check if this row has empty cells or <br/> tags
                empty_cell_count = 0
                for cell in cells:
                    if cell.find('br') or cell.get_text().strip() == '':
                        empty_cell_count += 1
                
                # If most cells are empty, this is likely our last data row
                if empty_cell_count >= len(cells) - 1:
                    last_empty_row_index = i
                    break
        
        if last_empty_row_index is None:
            print("⚠️ 未找到空行，无页脚")
            return "</table></body></html>"
        
        # Build footer HTML
        footer_parts = []
        
        # Add all rows after the last empty row
        for i in range(last_empty_row_index + 1, len(rows)):
            footer_parts.append(str(rows[i]))
        
        # Close the HTML structure
        footer_parts.append("</table></body></html>")
        
        footer_html = '\n'.join(footer_parts)
        print(f"✅ 提取页脚HTML (包含 {len(rows) - last_empty_row_index - 1} 行)")
        print("✅ extract_footer_html_code_based 执行完成")
        print("=" * 50)
        return footer_html
        
    except Exception as e:
        print(f"❌ extract_footer_html_code_based 执行失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return ""


def is_valid_csv_row(row_text: str) -> bool:
    """
    Check if a row contains valid CSV data.
    
    Args:
        row_text: The text row to check
        
    Returns:
        bool: True if row appears to be valid CSV data
    """
    if not row_text or not row_text.strip():
        return False
    
    # Skip obvious non-CSV content
    non_csv_indicators = [
        '===', '---', '***', '+++', '###',  # Separator lines
        '推理过程', '最终答案', '分析', '结论',  # Analysis text
        'Error', 'Exception', 'Traceback',  # Error messages
        '步骤', '过程', '思考', '判断',  # Process text
        '根据', '因为', '所以', '由于',  # Logic text
    ]
    
    for indicator in non_csv_indicators:
        if indicator in row_text:
            return False
    
    # Check if it looks like CSV (has commas and reasonable structure)
    try:
        # Try to parse as CSV
        csv_reader = csv.reader([row_text])
        fields = next(csv_reader)
        
        # Should have multiple fields
        if len(fields) < 2:
            return False
        
        # Fields shouldn't be too long (likely prose text)
        for field in fields:
            if len(field.strip()) > 200:  # Reasonable field length limit
                return False
        
        return True
        
    except:
        return False


def parse_csv_row_safely(row_text: str) -> list:
    """
    Safely parse a CSV row, handling various edge cases.
    
    Args:
        row_text: The CSV row text to parse
        
    Returns:
        list: List of field values, or empty list if parsing fails
    """
    try:
        # First try proper CSV parsing
        csv_reader = csv.reader([row_text])
        fields = next(csv_reader)
        
        # Strip whitespace from each field
        fields = [field.strip() for field in fields]
        
        return fields
        
    except:
        # Fallback to simple comma split
        try:
            fields = row_text.split(',')
            fields = [field.strip() for field in fields]
            return fields
        except:
            return []


def transform_data_to_html_code_based(csv_file_path: str, empty_row_html: str, session_id: str, template_file_path: str = None) -> str:
    """
    Transform CSV data to HTML using code-based approach with robust error handling.
    
    Args:
        csv_file_path: Path to CSV file
        empty_row_html: HTML template for empty row
        session_id: Session ID for logging
        
    Returns:
        str: Generated HTML rows
    """
    print("\n🔄 开始执行: transform_data_to_html_code_based")
    print("=" * 50)
    
    try:
        # Check if CSV file exists
        if not os.path.exists(csv_file_path):
            print(f"❌ CSV文件不存在: {csv_file_path}")
            return ""
        
        # Read CSV data
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_lines = file.read().strip().split('\n')
        
        print(f"📊 读取到 {len(csv_lines)} 行原始数据")
        
        # Parse the empty row HTML template
        soup = BeautifulSoup(empty_row_html, 'html.parser')
        template_row = soup.find('tr')
        
        if not template_row:
            print("❌ 无法在模板中找到<tr>元素")
            return ""
        
        # Get all td elements in the template
        template_cells = template_row.find_all('td')
        expected_columns = len(template_cells)
        print(f"📋 模板列数: {expected_columns}")
        
        # Check if template has "序号" column and detect its position
        sequence_column_index = None
        if template_file_path and os.path.exists(template_file_path):
            try:
                with open(template_file_path, 'r', encoding='utf-8') as f:
                    template_content = f.read()
                
                # Parse template to find "序号" column
                template_soup = BeautifulSoup(template_content, 'html.parser')
                # Find all table rows to locate header structure
                all_rows = template_soup.find_all('tr')
                
                for row in all_rows:
                    cells = row.find_all(['td', 'th'])
                    for i, cell in enumerate(cells):
                        cell_text = cell.get_text(strip=True)
                        if '序号' in cell_text and sequence_column_index is None:
                            sequence_column_index = i
                            print(f"🔢 检测到序号列，位置: {sequence_column_index}")
                            break
                    if sequence_column_index is not None:
                        break
                        
            except Exception as e:
                print(f"⚠️ 读取模板文件失败: {e}")
        
        filled_rows = []
        valid_row_count = 0
        skipped_row_count = 0
        
        for row_index, csv_line in enumerate(csv_lines):
            # Skip empty lines
            if not csv_line.strip():
                skipped_row_count += 1
                continue
            
            # Check if row is valid CSV
            if not is_valid_csv_row(csv_line):
                print(f"⚠️ 跳过非CSV行 {row_index + 1}: {csv_line[:50]}...")
                skipped_row_count += 1
                continue
            
            # Parse CSV row safely
            row_data = parse_csv_row_safely(csv_line)
            if not row_data:
                print(f"⚠️ 无法解析行 {row_index + 1}: {csv_line[:50]}...")
                skipped_row_count += 1
                continue
            
            valid_row_count += 1
            
            # Create a new row based on the template
            new_row = BeautifulSoup(empty_row_html, 'html.parser').find('tr')
            cells = new_row.find_all('td')
            
            # Log sequence column detection
            if sequence_column_index is not None:
                if sequence_column_index == 0:
                    print(f"🔢 检测到序号列在第一列，将跳过CSV第一列数据，使用自动编号")
                else:
                    print(f"🔢 检测到序号列在第{sequence_column_index + 1}列，将使用自动编号")
            
            # Fill in the data
            csv_data_pointer = 0  # Track which CSV column we should use next
            
            for i, cell in enumerate(cells):
                if sequence_column_index is not None and i == sequence_column_index:
                    # Use auto-enumeration for sequence column
                    if cell.find('br'):
                        cell.clear()
                    cell.string = str(valid_row_count)
                    if valid_row_count <= 5:  # Only log first 5 rows to avoid spam
                        print(f"🔢 序号列 (列{i}) 自动编号: {valid_row_count}")
                    # Don't increment csv_data_pointer for sequence column
                else:
                    # Use CSV data for non-sequence columns
                    # Skip the first CSV column if it corresponds to the sequence column
                    if sequence_column_index == 0 and csv_data_pointer == 0:
                        csv_data_pointer = 1  # Skip the first CSV column (original 序号 data)
                        if valid_row_count <= 3:  # Log for first few rows
                            print(f"🔢 跳过CSV第1列数据 '{row_data[0]}' (原序号数据)")
                    
                    if csv_data_pointer < len(row_data):
                        # Replace <br/> or empty content with actual data
                        if cell.find('br'):
                            cell.clear()
                        cell_value = row_data[csv_data_pointer] if row_data[csv_data_pointer] else ''
                        cell.string = cell_value
                        if valid_row_count <= 3:  # Log for first few rows
                            print(f"🔢 模板列{i} ← CSV列{csv_data_pointer}: '{cell_value}'")
                        csv_data_pointer += 1
                    else:
                        # If we have fewer data fields than template columns, fill with empty
                        if cell.find('br'):
                            cell.clear()
                        cell.string = ''
            
            # No inline styles needed - CSS handles all styling
            filled_rows.append(str(new_row))
            
            # Progress indicator for large datasets
            if valid_row_count % 100 == 0:
                print(f"✅ 已处理 {valid_row_count} 行有效数据")
        
        combined_html = '\n'.join(filled_rows)
        
        print(f"🎉 处理完成:")
        print(f"   - 总行数: {len(csv_lines)}")
        print(f"   - 有效行数: {valid_row_count}")
        print(f"   - 跳过行数: {skipped_row_count}")
        print(f"   - 生成HTML长度: {len(combined_html)} 字符")
        if sequence_column_index is not None:
            print(f"   - 序号列处理: 第{sequence_column_index + 1}列使用自动编号 (1-{valid_row_count})")
        else:
            print("   - 序号列处理: 未检测到序号列")
        
        # Save a sample to file for debugging
        if session_id:
            sample_output_path = f"conversations/{session_id}/output/sample_filled_rows.html"
            os.makedirs(os.path.dirname(sample_output_path), exist_ok=True)
            with open(sample_output_path, 'w', encoding='utf-8') as f:
                f.write(combined_html[:5000])  # Save first 5000 chars as sample
            print(f"📝 样本HTML已保存到: {sample_output_path}")
        
        print("✅ transform_data_to_html_code_based 执行完成")
        print("=" * 50)
        
        return combined_html
        
    except Exception as e:
        print(f"❌ transform_data_to_html_code_based 执行失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return ""


def combine_html_parts(headers_html: str, data_html: str, footer_html: str) -> str:
    """
    Combine HTML parts with enhanced modern styling.
    Dynamically detects header structure instead of assuming fixed levels.
    
    Args:
        headers_html: HTML for headers
        data_html: HTML for data rows
        footer_html: HTML for footer
        
    Returns:
        str: Complete HTML document
    """
    print("\n🔄 开始执行: combine_html_parts")
    print("=" * 50)
    
    try:
        # Parse headers to detect actual header structure
        header_row_count = 0
        if headers_html:
            soup = BeautifulSoup(headers_html, 'html.parser')
            table = soup.find('table')
            if table:
                rows = table.find_all('tr')
                header_row_count = len(rows)
                print(f"📋 检测到 {header_row_count} 行表头")
        
        # Generate dynamic CSS for headers based on actual structure
        header_css_rules = []
        
        if header_row_count > 0:
            # First row (main title) - always dark blue
            header_css_rules.append(f"""
        /* 主标题行 */
        table tr:first-child td {{
            background-color: #2c3e50 !important;
            color: white !important;
            font-weight: 600;
            font-size: 16px;
            text-align: center;
            padding: 18px 15px;
            border: 1px solid #2c3e50;
            border-right: 2px solid #1a252f;
        }}
        
        table tr:first-child td:last-child {{
            border-right: 1px solid #2c3e50;
        }}""")
        
        if header_row_count > 1:
            # Second row (category headers) - medium dark blue
            header_css_rules.append(f"""
        /* 分类标题行 */
        table tr:nth-child(2) td {{
            background-color: #34495e !important;
            color: white !important;
            font-weight: 600;
            font-size: 14px;
            text-align: center;
            padding: 14px 12px;
            border: 1px solid #34495e;
            border-right: 2px solid #2c3e50;
        }}
        
        table tr:nth-child(2) td:last-child {{
            border-right: 1px solid #34495e;
        }}""")
        
        if header_row_count > 2:
            # Third row (field headers) - light gray
            header_css_rules.append(f"""
        /* 字段标题行 */
        table tr:nth-child(3) td {{
            background-color: #ecf0f1 !important;
            color: #2c3e50 !important;
            font-weight: 600;
            font-size: 13px;
            text-align: center;
            padding: 12px 10px;
            border: 1px solid #bdc3c7;
            border-right: 2px solid #95a5a6;
        }}
        
        table tr:nth-child(3) td:last-child {{
            border-right: 1px solid #bdc3c7;
        }}""")
        
        # For additional header rows (if any), apply similar styling
        if header_row_count > 3:
            for i in range(4, header_row_count + 1):
                header_css_rules.append(f"""
        /* 第{i}行表头 */
        table tr:nth-child({i}) td {{
            background-color: #f8f9fa !important;
            color: #2c3e50 !important;
            font-weight: 600;
            font-size: 13px;
            text-align: center;
            padding: 12px 10px;
            border: 1px solid #bdc3c7;
            border-right: 2px solid #95a5a6;
        }}
        
        table tr:nth-child({i}) td:last-child {{
            border-right: 1px solid #bdc3c7;
        }}""")
        
        # Generate selector for data rows (everything after header rows)
        data_row_selector = f"table tr:nth-child(n+{header_row_count + 1}):not(:last-child)"
        if header_row_count == 0:
            data_row_selector = "table tr:not(:last-child)"
        
        # Create complete HTML document with professional formal styling
        complete_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>表格报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
            background-color: #f8f9fa;
            padding: 30px 20px;
            color: #333;
            line-height: 1.6;
        }}
        
        .table-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            margin: 0 auto;
            width: 100%;
            max-width: none;
            padding: 25px;
            border: 1px solid #e0e0e0;
        }}
        
        .table-wrapper {{
            overflow-x: auto;
            overflow-y: hidden;
            width: 100%;
            border-radius: 4px;
        }}
        
        table {{
            width: 100%;
            min-width: 800px;
            border-collapse: collapse;
            margin: 0;
            background: white;
            font-size: 14px;
        }}
        
        {chr(10).join(header_css_rules)}
        
        /* 数据行基础样式 */
        {data_row_selector} {{
            background-color: white;
            transition: background-color 0.2s ease;
        }}
        
        /* 交替行颜色 */
        {data_row_selector}:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        
        /* 数据行悬停效果 */
        {data_row_selector}:hover {{
            background-color: #e3f2fd;
        }}
        
        /* 数据单元格样式 */
        {data_row_selector} td {{
            padding: 12px 10px;
            text-align: left;
            border: 1px solid #e0e0e0;
            font-size: 13px;
            color: #333;
            font-weight: 400;
            vertical-align: top;
        }}
        
        /* 底部边框增强 */
        table tr:last-child td {{
            border-bottom: 2px solid #2c3e50;
        }}
        
        /* 响应式设计 */
        @media (max-width: 768px) {{
            body {{
                padding: 15px 10px;
            }}
            
            .table-container {{
                padding: 15px;
            }}
            
            table {{
                min-width: 600px;
                font-size: 13px;
            }}
            
            {data_row_selector} td {{
                padding: 10px 8px;
                font-size: 12px;
            }}
        }}
        
        @media (max-width: 480px) {{
            body {{
                padding: 10px 5px;
            }}
            
            .table-container {{
                padding: 10px;
            }}
            
            table {{
                min-width: 500px;
                font-size: 12px;
            }}
            
            {data_row_selector} td {{
                padding: 8px 6px;
                font-size: 11px;
            }}
        }}
        
        /* 自定义滚动条 */
        .table-wrapper::-webkit-scrollbar {{
            height: 8px;
        }}
        
        .table-wrapper::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 4px;
        }}
        
        .table-wrapper::-webkit-scrollbar-thumb {{
            background: #c1c1c1;
            border-radius: 4px;
        }}
        
        .table-wrapper::-webkit-scrollbar-thumb:hover {{
            background: #a8a8a8;
        }}
        
        /* 打印样式 */
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            
            .table-container {{
                box-shadow: none;
                border: none;
                padding: 0;
            }}
            
            table {{
                min-width: auto;
            }}
        }}
    </style>
</head>
<body>
    <div class="table-container">
        <div class="table-wrapper">
            {headers_html}
            {data_html}
            {footer_html}
        </div>
    </div>
</body>
</html>"""
        
        print(f"✅ 生成完整HTML文档 (表头行数: {header_row_count})")
        print("✅ combine_html_parts 执行完成")
        print("=" * 50)
        
        return complete_html
        
    except Exception as e:
        print(f"❌ combine_html_parts 执行失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return ""
