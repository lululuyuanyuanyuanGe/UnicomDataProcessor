# -*- coding: utf-8 -*-
import re
import os

def parse_sql_create_table(sql_statement):
    """
    Parse SQL CREATE TABLE statement to extract table name and column headers.
    
    Args:
        sql_statement (str): SQL CREATE TABLE statement
        
    Returns:
        tuple: (table_comment, column_headers) where table_comment is the table name
               from COMMENT and column_headers is a list of Chinese column descriptions
    """
    # Extract table comment (table name)
    comment_match = re.search(r"COMMENT='([^']+)'", sql_statement)
    table_name = comment_match.group(1) if comment_match else None
    
    # Extract column definitions
    # Find content between parentheses after CREATE TABLE
    table_def_match = re.search(r'CREATE TABLE[^(]+\((.*)\)\s*ENGINE', sql_statement, re.DOTALL)
    if not table_def_match:
        return None, []
    
    table_def = table_def_match.group(1)
    
    # Split by lines and extract Chinese column headers from COMMENT
    lines = table_def.split('\n')
    column_headers = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines, PRIMARY KEY, and other constraints
        if not line or line.startswith('PRIMARY KEY') or line.startswith('KEY') or line.startswith('INDEX'):
            continue
        
        # Extract column name and comment
        column_match = re.match(r'`([^`]+)`', line)
        comment_match = re.search(r"COMMENT '([^']+)'", line)
        
        if column_match and comment_match:
            column_name = column_match.group(1)
            chinese_header = comment_match.group(1)
            
            # Skip id column as it's usually auto-increment
            if column_name.lower() != 'id':
                column_headers.append(chinese_header)
    
    return table_name, column_headers

def update_data_txt(table_name, column_headers, data_file_path="D:\\asianInfo\\dataProcessor\\agents\\data.txt"):
    """
    Update the data.txt file with extracted table information in format: table_name:header1,header2,header3
    
    Args:
        table_name (str): Name of the table (from COMMENT)
        column_headers (list): List of Chinese column descriptions
        data_file_path (str): Path to data.txt file
    """
    if table_name and column_headers:
        # Format as table_name:header1,header2,header3
        headers_str = ",".join(column_headers)
        line = f"{table_name}:{headers_str}\n"
        
        # Append to file
        with open(data_file_path, 'a', encoding='utf-8') as f:
            f.write(line)
        
        print(f"Added table with {len(column_headers)} columns to data.txt")

def process_sql_statement(sql_statement):
    """
    Process a SQL CREATE TABLE statement and update data.txt.
    
    Args:
        sql_statement (str): SQL CREATE TABLE statement
    """
    table_name, column_headers = parse_sql_create_table(sql_statement)
    
    if table_name and column_headers:
        update_data_txt(table_name, column_headers)
        try:
            headers_str = ",".join(column_headers)
            print(f"{table_name}:{headers_str}")
        except UnicodeEncodeError:
            print("Table: [Chinese characters - encoding issue]")
        return table_name, column_headers
    else:
        print("Failed to parse SQL statement")
        return None, None


if __name__ == "__main__":
    with open("D:\\asianInfo\\dataProcessor\\utils\\sql_queries.txt", "r", encoding="utf-8") as f:
        sql_statements = f.read()
        sql_statements = sql_statements.split("!")
    for sql_statement in sql_statements:
        process_sql_statement(sql_statement)