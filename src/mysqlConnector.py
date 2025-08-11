import sys
import os
import MySQLdb
import json
import re
import time
import pickle
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from utils.modelRelated import invoke_embedding_model


class DatabaseManager:
    """Database management system for reading MySQL schema and creating embeddings"""
    
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.connect()
    
    def connect(self):
        """Establish database connection using singleton settings"""
        try:
            self.connection = MySQLdb.connect(
                host=settings.chatbi_address,
                port=settings.chatbi_port,
                user=settings.chatbi_user,
                passwd=settings.chatbi_password,
                db=settings.chatbi_database,
                charset='utf8mb4'
            )
            self.cursor = self.connection.cursor()
            print(f"Successfully connected to database: {settings.chatbi_database}")
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        try:
            if self.cursor:
                self.cursor.close()
        except:
            pass  # Connection might already be closed
        
        try:
            if self.connection:
                self.connection.close()
        except:
            pass  # Connection might already be closed
            
        print("Database connection closed")
    
    def fetch_all_tables(self) -> List[str]:
        """Get all table names from database (excluding views)"""
        try:
            # Get all tables and filter out views
            self.cursor.execute("SHOW FULL TABLES WHERE Table_Type = 'BASE TABLE'")
            tables = [table[0] for table in self.cursor.fetchall()]
            print(f"Found {len(tables)} tables in database (excluding views)")
            return tables
        except Exception as e:
            print(f"Error fetching tables: {e}")
            # Fallback to simple SHOW TABLES if the above fails
            try:
                self.cursor.execute("SHOW TABLES")
                all_objects = [table[0] for table in self.cursor.fetchall()]
                # Filter out views manually by checking if they end with '_view'
                tables = [table for table in all_objects if not table.lower().endswith('_view')]
                print(f"Found {len(tables)} tables using fallback method")
                return tables
            except Exception as e2:
                print(f"Error with fallback method: {e2}")
                return []
    
    def get_table_ddl(self, table_name: str) -> Optional[str]:
        """Get DDL (CREATE TABLE statement) for a specific table"""
        try:
            self.cursor.execute(f"SHOW CREATE TABLE `{table_name}`")
            result = self.cursor.fetchone()
            if result:
                return result[1]  # DDL is in the second column
            return None
        except Exception as e:
            print(f"Error getting DDL for table {table_name}: {e}")
            return None
    
    def parse_ddl(self, ddl_text: str, table_name: str) -> Dict[str, any]:
        """Parse DDL to extract Chinese and English names for table and columns"""
        table_info = {
            "english_table_name": table_name,
            "chinese_table_name": "",
            "chinese_headers": [],
            "english_headers": [],
            "header_count": 0
        }
        
        # Extract table comment (Chinese name) - look at the end of CREATE TABLE statement
        table_comment_match = re.search(r"COMMENT='([^']*)'(?:\s*;?)$", ddl_text, re.IGNORECASE | re.MULTILINE)
        if table_comment_match:
            table_info["chinese_table_name"] = table_comment_match.group(1)
        else:
            # Use English name as fallback
            table_info["chinese_table_name"] = table_name
        
        # Extract column definitions and comments - improved regex pattern
        # This pattern matches: `column_name` type COMMENT 'comment text'
        column_pattern = r"`([^`]+)`\s+[^,\n]*?COMMENT\s+'([^']+)'"
        columns = re.findall(column_pattern, ddl_text, re.IGNORECASE | re.DOTALL)
        
        chinese_headers = []
        english_headers = []
        
        for column_name, comment in columns:
            # Skip the auto-increment ID column unless we want to include it
            if column_name.lower() == 'id':
                continue
                
            english_headers.append(column_name)
            # Use comment as Chinese name (it should always exist given our regex)
            chinese_headers.append(comment)
        
        table_info["chinese_headers"] = chinese_headers
        table_info["english_headers"] = english_headers
        table_info["header_count"] = len(chinese_headers)
        
        return table_info
    
    def load_data_json(self) -> Dict:
        """Load existing data.json file"""
        data_json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "data.json")
        try:
            with open(data_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            # Create basic structure if file doesn't exist
            return {"ChatBI": {}}
        except Exception as e:
            print(f"Error loading data.json: {e}")
            return {"ChatBI": {}}
    
    def save_data_json(self, data: Dict):
        """Save data to data.json file"""
        data_json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "data.json")
        try:
            with open(data_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print("Successfully updated data.json")
        except Exception as e:
            print(f"Error saving data.json: {e}")
    
    def update_data_json(self):
        """Update data.json with database schema information"""
        print("Starting database schema extraction...")
        
        # Load existing data
        data = self.load_data_json()
        
        # Clear and recreate database tables section to remove any old view entries
        data["ChatBI"]["数据库表格"] = {}
        
        # Get all tables
        tables = self.fetch_all_tables()
        
        for table_name in tables:
            print(f"Processing table: {table_name}")
            
            # Get DDL
            ddl = self.get_table_ddl(table_name)
            if not ddl:
                continue
            
            # Parse DDL
            table_info = self.parse_ddl(ddl, table_name)
            
            # Use Chinese name as key
            chinese_table_name = table_info["chinese_table_name"]
            
            # Create entry in data.json following existing pattern
            data["ChatBI"]["数据库表格"][chinese_table_name] = {
                "chinese_headers": table_info["chinese_headers"],
                "english_table_name": table_info["english_table_name"],
                "english_headers": table_info["english_headers"], 
                "header_count": table_info["header_count"],
            }
            
            print(f"   Table processed - {table_info['header_count']} columns")
        
        # Save updated data
        self.save_data_json(data)
        print(f"Successfully processed {len(tables)} database tables")
    
    def create_table_embeddings(self):
        """Generate embeddings for database tables and save to pickle file"""
        print("Starting database table embedding generation...")
        
        # Load data.json to get table information
        data = self.load_data_json()
        
        if "数据库表格" not in data["ChatBI"] or not data["ChatBI"]["数据库表格"]:
            print("No database tables found in data.json. Run update_data_json() first.")
            return
        
        # Prepare table descriptions for embedding (Chinese names only)
        tables_with_description = []
        table_names = []
        
        for chinese_table_name, table_info in data["ChatBI"]["数据库表格"].items():
            headers_str = ",".join(table_info["chinese_headers"])
            description = f"{chinese_table_name} 包含表头：{headers_str}"
            tables_with_description.append(description)
            table_names.append(chinese_table_name)
        
        print(f"Found {len(tables_with_description)} database tables to process")
        
        # Process tables in batches of 5 using concurrent processing
        batch_size = 5
        all_embeddings = []
        
        # Create batches
        batches = []
        for i in range(0, len(tables_with_description), batch_size):
            batch = tables_with_description[i:i+batch_size]
            batches.append((i//batch_size + 1, batch))
        
        def process_batch(batch_data):
            batch_num, batch = batch_data
            print(f"Processing batch {batch_num}: {len(batch)} tables")
            try:
                embeddings = invoke_embedding_model(model_name="Qwen/Qwen3-Embedding-8B", texts=batch)
                print(f"Successfully processed batch {batch_num}")
                return batch_num, embeddings
            except Exception as e:
                print(f"Error processing batch {batch_num}: {e}")
                return batch_num, None
        
        # Use ThreadPoolExecutor for concurrent processing
        max_workers = min(5, len(batches))  # Limit concurrent requests to avoid rate limiting
        batch_results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {executor.submit(process_batch, batch_data): batch_data[0] for batch_data in batches}
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    batch_num_result, embeddings = future.result()
                    if embeddings is not None:
                        batch_results[batch_num_result] = embeddings
                except Exception as e:
                    print(f"Batch {batch_num} generated an exception: {e}")
        
        # Combine results in order
        for batch_num in sorted(batch_results.keys()):
            all_embeddings.extend(batch_results[batch_num])
        
        print(f"Completed processing {len(all_embeddings)} embeddings for {len(tables_with_description)} database tables")
        print(f"Embedding dimension: {len(all_embeddings[0]) if all_embeddings else 'N/A'}")
        
        # Save embeddings (following existing pattern)
        if all_embeddings:
            
            # Create table info dictionary
            table_info = {}
            for chinese_name, info in data["ChatBI"]["数据库表格"].items():
                headers_str = ",".join(info["chinese_headers"])
                table_info[chinese_name] = headers_str
            
            # Save in pickle format (following existing structure)
            try:
                embedding_data = {
                    'embeddings': all_embeddings,
                    'table_names': table_names,
                    'table_descriptions': tables_with_description,
                    'table_info': table_info,
                    'source': 'database_schema'
                }
                
                # Ensure embedded_tables directory exists
                embedded_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embedded_tables")
                os.makedirs(embedded_dir, exist_ok=True)
                
                pickle_path = os.path.join(embedded_dir, "database_table_embeddings.pkl")
                with open(pickle_path, 'wb') as f:
                    pickle.dump(embedding_data, f)
                print("Saved database table embeddings to database_table_embeddings.pkl")
            except Exception as e:
                print(f"Error saving pickle embeddings: {e}")
            
            # Save metadata in JSON format
            try:
                metadata = {
                    'model': 'Qwen/Qwen3-Embedding-8B',
                    'total_tables': len(tables_with_description),
                    'table_info': table_info,
                    'table_names': table_names,
                    'table_descriptions': tables_with_description,
                    'table_embeddings': [embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding) for embedding in all_embeddings],
                    'embedding_dimension': len(all_embeddings[0]) if all_embeddings else 0,
                    'timestamp': time.time(),
                    'source': 'database_schema'
                }
                
                metadata_path = os.path.join(embedded_dir, "database_table_metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    # Custom JSON formatting to keep embeddings compact
                    metadata_copy = metadata.copy()
                    embeddings = metadata_copy.pop('table_embeddings')
                    
                    # Write metadata without embeddings first
                    json_str = json.dumps(metadata_copy, ensure_ascii=False, indent=2)
                    
                    # Insert embeddings with custom formatting
                    json_str = json_str[:-1]  # Remove closing brace
                    json_str += ',\n  "table_embeddings": [\n'
                    
                    for i, embedding in enumerate(embeddings):
                        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                        json_str += f'    {json.dumps(embedding_list, ensure_ascii=False)}'
                        if i < len(embeddings) - 1:
                            json_str += ','
                        json_str += '\n'
                    
                    json_str += '  ]\n}'
                    f.write(json_str)
                print("Saved database table metadata to database_table_metadata.json")
            except Exception as e:
                print(f"Error saving JSON metadata: {e}")
        else:
            print("No embeddings generated, skipping file save")
    
    def process_all_tables(self):
        """Complete workflow: extract schema and create embeddings"""
        try:
            print("Starting complete database table processing workflow...")
            
            # Step 1: Update data.json with schema information
            self.update_data_json()
            
            # Step 2: Create embeddings
            self.create_table_embeddings()
            
            print("Database table processing completed!")
            
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            self.disconnect()


if __name__ == "__main__":
    # Example usage
    db_manager = DatabaseManager()
    try:
        db_manager.process_all_tables()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db_manager.disconnect()