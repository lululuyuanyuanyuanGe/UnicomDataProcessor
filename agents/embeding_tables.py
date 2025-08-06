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
            tables_with_description.append(f"数据表： {table_name} 包含表头：{table_content}")
        else:
            print(f"⚠️ Skipping malformed line: {table}")

print(f"📋 Found {len(tables_with_description)} tables to process")

# Process tables in batches of 5
batch_size = 5
all_embeddings = []
table_names = []

# Extract table names for metadata
for table in tables:
    if "：" in table:
        table_name, _ = table.split("：", 1)
        table_names.append(table_name)

for i in range(0, len(tables_with_description), batch_size):
    batch = tables_with_description[i:i+batch_size]
    print(f"Processing batch {i//batch_size + 1}: {len(batch)} tables")
    
    # Call embedding model for this batch
    try:
        embeddings = invoke_embedding_model(model_name="Qwen/Qwen3-Embedding-8B", texts=batch)
        all_embeddings.extend(embeddings)
        print(f"✅ Successfully processed batch {i//batch_size + 1}")
    except Exception as e:
        print(f"❌ Error processing batch {i//batch_size + 1}: {e}")
        continue

print(f"🎉 Completed processing {len(all_embeddings)} embeddings for {len(tables_with_description)} tables")
print(f"📊 Embedding dimensions: {len(all_embeddings[0]) if all_embeddings else 'N/A'}")

# Save embeddings in 3 different formats
if all_embeddings:
    import numpy as np
    
    # Create table info dictionary
    table_info = {}
    for table in tables:
        if "：" in table:
            table_name, table_content = table.split("：", 1)
            table_info[table_name] = table_content
    
    # 1. Save in pickle format
    try:
        embedding_data = {
            'embeddings': all_embeddings,
            'table_names': table_names,
            'table_descriptions': tables_with_description,
            'table_info': table_info
        }
        with open('table_embeddings.pkl', 'wb') as f:
            pickle.dump(embedding_data, f)
        print("✅ Saved embeddings to table_embeddings.pkl")
    except Exception as e:
        print(f"❌ Error saving embeddings in pickle: {e}")
    
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
        with open('table_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print("✅ Saved metadata to table_metadata.json")
    except Exception as e:
        print(f"❌ Error saving metadata in JSON: {e}")
    
    # 3. Save in numpy format
    try:
        # Convert to numpy arrays
        embeddings_array = np.array(all_embeddings)
        table_names_array = np.array(table_names, dtype=object)
        
        # Save embeddings array only
        np.save('table_embeddings.npy', embeddings_array)
        
        # Save multiple arrays in compressed format
        np.savez_compressed('table_embeddings.npz', 
                           embeddings=embeddings_array,
                           table_names=table_names_array,
                           table_descriptions=np.array(tables_with_description, dtype=object))
        
        print("✅ Saved embeddings to table_embeddings.npy and table_embeddings.npz")
    except Exception as e:
        print(f"❌ Error saving embeddings in numpy format: {e}")
        
    print("\n📁 Generated files:")
    print("  - table_embeddings.pkl (full data with pickle)")
    print("  - table_metadata.json (metadata only)")
    print("  - table_embeddings.npy (embeddings as numpy array)")
    print("  - table_embeddings.npz (embeddings + metadata as compressed numpy)")
else:
    print("❌ No embeddings generated, skipping file save")








