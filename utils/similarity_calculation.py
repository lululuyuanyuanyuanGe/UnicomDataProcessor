import sys
import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.modelRelated import invoke_embedding_model


class TableSimilarityCalculator:
    """
    A class for calculating similarity between table embeddings and finding best matches.
    Handles loading existing embeddings, computing cosine similarity with percentage scoring,
    and formatting results for user decision making.
    """
    
    def __init__(self, embeddings_path: str = "embedded_tables/table_embeddings.pkl"):
        """
        Initialize the similarity calculator with embeddings data.
        
        Args:
            embeddings_path: Path to the pickle file containing table embeddings
        """
        self.embeddings_path = Path(embeddings_path)
        self.embeddings = None
        self.table_names = None
        self.table_descriptions = None
        self.table_info = None
        self.embedding_model = "Qwen/Qwen3-Embedding-8B"

        # Load existing embeddings on initialization
        self.load_existing_embeddings()
    
    def load_existing_embeddings(self) -> bool:
        """
        Load existing table embeddings from pickle file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.embeddings_path.exists():
                print(f"Embeddings file not found: {self.embeddings_path}")
                return False
            
            print(f"正在加载表格嵌入: {self.embeddings_path}")
            
            with open(self.embeddings_path, 'rb') as f:
                data = pickle.load(f)
            
            # Validate data structure
            required_keys = ['embeddings', 'table_names', 'table_descriptions', 'table_info']
            for key in required_keys:
                if key not in data:
                    print(f"Missing key in embeddings data: {key}")
                    return False
            
            self.embeddings = np.array(data['embeddings'])
            self.table_names = data['table_names']
            self.table_descriptions = data['table_descriptions']
            self.table_info = data['table_info']
            
            print(f"成功加载 {len(self.embeddings)} 个表格嵌入向量")
            print(f"嵌入向量维度: {self.embeddings.shape[1] if len(self.embeddings.shape) > 1 else '不可用'}")
            
            return True
            
        except Exception as e:
            print(f"加载嵌入向量时出错: {e}")
            return False
    
    def calculate_cosine_similarity_sklearn(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity using scikit-learn (most efficient and reliable)
        """
        try:
            # Reshape vectors to 2D arrays (required by sklearn)
            v1 = np.array(vector1).reshape(1, -1)
            v2 = np.array(vector2).reshape(1, -1)
            
            # Calculate similarity (returns values between -1 and 1)
            similarity = cosine_similarity(v1, v2)[0][0]
            
            # Convert to percentage (0-100)
            percentage = max(0.0, min(100.0, (similarity + 1) * 50))  # Map [-1,1] to [0,100]
            
            return percentage
            
        except Exception as e:
            print(f"计算余弦相似度时出错: {e}")
            return 0.0
    
    def find_similar_tables(self, new_embedding: List[float], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Find the most similar tables to a new table embedding.
        
        Args:
            new_embedding: Embedding vector for the new table
            top_n: Number of top matches to return
            
        Returns:
            List of dictionaries with similarity results including percentages
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            print("没有加载已存在的嵌入向量")
            return []
        
        try:
            similarities = []
            new_vector = np.array(new_embedding)
            
            # Calculate similarity with each existing table
            for i, existing_embedding in enumerate(self.embeddings):
                similarity_percentage = self.calculate_cosine_similarity_sklearn(new_vector, existing_embedding)
                
                # Get table information
                table_name = self.table_names[i] if i < len(self.table_names) else f"表格_{i}"
                table_description = self.table_descriptions[i] if i < len(self.table_descriptions) else "无描述"
                
                similarities.append({
                    'index': i,
                    'table_name': table_name,
                    'description': table_description,
                    'similarity_percentage': similarity_percentage,
                    'similarity_formatted': f"{similarity_percentage:.1f}%"
                })
            
            # Sort by similarity percentage (descending) and return top_n
            similarities.sort(key=lambda x: x['similarity_percentage'], reverse=True)
            return similarities[:top_n]
            
        except Exception as e:
            print(f"查找相似表格时出错: {e}")
            return []
    
    def format_similarity_results(self, similarity_results: List[Dict[str, Any]]) -> str:
        """
        Format similarity results for user-friendly display.
        
        Args:
            similarity_results: List of similarity result dictionaries
            
        Returns:
            str: Formatted results string
        """
        if not similarity_results:
            return "未找到相似的表格。"
        
        formatted_output = "\n" + "="*80 + "\n"
        formatted_output += "最相似的表格结果\n"
        formatted_output += "="*80 + "\n"
        
        for i, result in enumerate(similarity_results, 1):
            formatted_output += f"\n#{i} - 相似度: {result['similarity_formatted']}\n"
            
            # Clean table name - remove "数据表：" prefix if present
            table_name = result['table_name']
            formatted_output += f"表格名称: {table_name}\n"
            
            # Extract headers from description for preview
            description = result['description']
            
            # Remove "数据表：" prefix from description if present
            clean_description = description
            if clean_description.startswith("数据表： "):
                clean_description = clean_description[4:].strip()  # Remove "数据表： "
            
            if '包含表头：' in clean_description:
                headers_part = clean_description.split('包含表头：')[1] if '包含表头：' in clean_description else clean_description
                headers = [h.strip() for h in headers_part.split(',')]
                headers_preview = ', '.join(headers)  # Show first 5 headers
                formatted_output += f"表头: {headers_preview}\n"
            else:
                formatted_output += f"描述: {clean_description}\n"
            formatted_output += "-" * 40 + "\n"
        
        return formatted_output
    
    def embed_new_table_description(self, description: str) -> Optional[List[float]]:
        """
        Generate embedding for a new table description.
        
        Args:
            description: Table description string to embed
            
        Returns:
            List[float]: Embedding vector or None if failed
        """
        try:
            try:
                print(f"正在为以下内容生成嵌入向量: {description}")
            except UnicodeEncodeError:
                print("正在生成嵌入向量: [包含中文字符]")
            
            # Use the same embedding model as existing tables
            embeddings = invoke_embedding_model(
                model_name=self.embedding_model, 
                texts=[description],  # Pass as list for batch processing
                silent_mode=True  # Use silent mode to avoid Unicode console issues
            )
            
            if embeddings and len(embeddings) > 0:
                return embeddings[0]  # Return first (and only) embedding
            else:
                print("模型未返回嵌入向量")
                return None
                
        except Exception as e:
            try:
                print(f"生成嵌入向量时出错: {e}")
            except UnicodeEncodeError:
                print("生成嵌入向量时出错: [编码问题]")
            return None
    
    def get_best_matches(self, new_table_description: str, top_n: int = 3) -> Dict[str, Any]:
        """
        Main entry point to find best matching tables for a new table description.
        
        Args:
            new_table_description: Description of the new table to match
            top_n: Number of top matches to return
            
        Returns:
            Dict containing similarity results and formatted output
        """
        try:
            print(f"\n正在为新表格查找最佳匹配...")
            try:
                print(f"表格描述: {new_table_description}")
            except UnicodeEncodeError:
                print("表格描述: [包含中文字符]")
            
            # Generate embedding for new table
            new_embedding = self.embed_new_table_description(new_table_description)
            if new_embedding is None:
                return {
                    'success': False,
                    'error': '无法为新表格生成嵌入向量',
                    'matches': [],
                    'formatted_output': '错误: 无法生成嵌入向量'
                }
            
            # Find similar tables
            similarity_results = self.find_similar_tables(new_embedding, top_n)
            
            if not similarity_results:
                return {
                    'success': False,
                    'error': '未找到相似表格',
                    'matches': [],
                    'formatted_output': '数据库中未找到相似的表格。'
                }
            
            # Format results for display
            formatted_output = self.format_similarity_results(similarity_results)
            
            return {
                'success': True,
                'matches': similarity_results,
                'formatted_output': formatted_output,
                'top_match': similarity_results[0] if similarity_results else None
            }
            
        except Exception as e:
            error_msg = f"查找最佳匹配时出错: {e}"
            try:
                print(error_msg)
            except UnicodeEncodeError:
                print("查找最佳匹配时出错: [编码问题]")
                error_msg = "发生编码错误"
            return {
                'success': False,
                'error': error_msg,
                'matches': [],
                'formatted_output': f'错误: {error_msg}'
            }
    
    def get_table_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a table by its index.
        
        Args:
            index: Index of the table in the embeddings array
            
        Returns:
            Dict with table information or None if not found
        """
        try:
            if index < 0 or index >= len(self.table_names):
                return None
            
            return {
                'index': index,
                'table_name': self.table_names[index],
                'description': self.table_descriptions[index],
                'embedding': self.embeddings[index].tolist(),
                'info': self.table_info.get(self.table_names[index], {})
            }
            
        except Exception as e:
            print(f"根据索引获取表格时出错: {e}")
            return None