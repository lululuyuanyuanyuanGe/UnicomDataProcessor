import sys
import os
import time
import pickle
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.modelRelated import invoke_embedding_model
from dotenv import load_dotenv

load_dotenv()

with open("D:\\asianInfo\\dataProcessor\\agents\\data.txt", "r", encoding="utf-8") as f:
    texts = f.read()
    tables = [line.strip() for line in texts.split("\n") if line.strip()]  # Filter empty lines
    tables_with_description = []
    for table in tables:
        if ":" in table:  # Check if the line has the expected format
            table_name, table_content = table.split(":", 1)  # Split only on first occurrence
            tables_with_description.append(f"{table_name} 包含表头：{table_content}")
        else:
            print(f"跳过格式错误的行: {table}")

print(f"找到 {len(tables_with_description)} 个表格需要处理")

# Process tables in batches of 5
batch_size = 5
all_embeddings = []
table_names = []

# Extract table names for metadata
for table in tables:
    if ":" in table:
        table_name, _ = table.split(":", 1)
        table_names.append(table_name)

for i in range(0, len(tables_with_description), batch_size):
    batch = tables_with_description[i:i+batch_size]
    print(f"正在处理第 {i//batch_size + 1} 批: {len(batch)} 个表格")
    print("batch里的内容: ", batch)
    # Call embedding model for this batch
    try:
        embeddings = invoke_embedding_model(model_name="Qwen/Qwen3-Embedding-8B", texts=batch)
        all_embeddings.extend(embeddings)
        print(f"成功处理第 {i//batch_size + 1} 批")
    except Exception as e:
        print(f"处理第 {i//batch_size + 1} 批时出错: {e}")
        continue

print(f"🎉 完成处理 {len(all_embeddings)} 个嵌入向量，对应 {len(tables_with_description)} 个表格")
print(f"📊 嵌入向量维度: {len(all_embeddings[0]) if all_embeddings else '不可用'}")

# Save embeddings in 3 different formats
if all_embeddings:
    import numpy as np
    
    # Create table info dictionary
    table_info = {}
    for table in tables:
        if ":" in table:
            table_name, table_content = table.split(":", 1)
            table_info[table_name] = table_content
    
    # 1. Save in pickle format
    try:
        embedding_data = {
            'embeddings': all_embeddings,
            'table_names': table_names,
            'table_descriptions': tables_with_description,
            'table_info': table_info
        }
        with open('embedded_tables/table_embeddings.pkl', 'wb') as f:
            pickle.dump(embedding_data, f)
        print("✅ 已保存嵌入向量到 table_embeddings.pkl")
    except Exception as e:
        print(f"❌ 保存pickle格式嵌入向量时出错: {e}")
    
    # 2. Save in JSON format (metadata only, embeddings too large for JSON)
    try:
        metadata = {
            'model': 'Qwen/Qwen3-Embedding-8B',
            'total_tables': len(tables_with_description),
            'table_info': table_info,
            'table_names': table_names,
            'table_descriptions': tables_with_description,
            'embedding_dimension': len(all_embeddings[0]) if all_embeddings else 0,
            'timestamp': time.time()
        }
        with open('embedded_tables/table_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print("✅ 已保存元数据到 table_metadata.json")
    except Exception as e:
        print(f"❌ 保存JSON格式元数据时出错: {e}")
    
    # 3. Save in numpy format
    try:
        # Convert to numpy arrays
        embeddings_array = np.array(all_embeddings)
        table_names_array = np.array(table_names, dtype=object)
        
        # Save embeddings array only
        np.save('embedded_tables/table_embeddings.npy', embeddings_array)
        
        # Save multiple arrays in compressed format
        np.savez_compressed('embedded_tables/table_embeddings.npz', 
                           embeddings=embeddings_array,
                           table_names=table_names_array,
                           table_descriptions=np.array(tables_with_description, dtype=object))
        
        print("✅ 已保存嵌入向量到 table_embeddings.npy 和 table_embeddings.npz")
    except Exception as e:
        print(f"❌ 保存numpy格式嵌入向量时出错: {e}")
        
    print("\n📁 生成的文件:")
    print("  - table_embeddings.pkl (完整数据，pickle格式)")
    print("  - table_metadata.json (仅元数据)")
    print("  - table_embeddings.npy (嵌入向量，numpy数组格式)")
    print("  - table_embeddings.npz (嵌入向量+元数据，压缩numpy格式)")
else:
    print("❌ 未生成嵌入向量，跳过文件保存")








