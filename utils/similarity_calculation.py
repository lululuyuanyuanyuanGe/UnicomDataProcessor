import sys
import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

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
            
            print(f"Loading existing embeddings from: {self.embeddings_path}")
            
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
            
            print(f"Successfully loaded {len(self.embeddings)} table embeddings")
            print(f"Embedding dimension: {self.embeddings.shape[1] if len(self.embeddings.shape) > 1 else 'N/A'}")
            
            return True
            
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False
    
    def calculate_cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors and return as percentage.
        
        Args:
            vector1: First vector (numpy array)
            vector2: Second vector (numpy array)
            
        Returns:
            float: Similarity percentage (0.0-100.0)
        """
        try:
            # Ensure vectors are numpy arrays
            v1 = np.array(vector1).flatten()
            v2 = np.array(vector2).flatten()
            
            # Check dimension compatibility
            if v1.shape != v2.shape:
                print(f"Dimension mismatch: {v1.shape} vs {v2.shape}")
                return 0.0
            
            # Handle zero vectors
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            cosine_sim = dot_product / (norm1 * norm2)
            
            # Convert to percentage and ensure it's in valid range
            percentage = max(0.0, min(100.0, cosine_sim * 100))
            
            return percentage
            
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
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
            print("No existing embeddings loaded")
            return []
        
        try:
            similarities = []
            new_vector = np.array(new_embedding)
            
            # Calculate similarity with each existing table
            for i, existing_embedding in enumerate(self.embeddings):
                similarity_percentage = self.calculate_cosine_similarity(new_vector, existing_embedding)
                
                # Get table information
                table_name = self.table_names[i] if i < len(self.table_names) else f"Table_{i}"
                table_description = self.table_descriptions[i] if i < len(self.table_descriptions) else "No description"
                
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
            print(f"Error finding similar tables: {e}")
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
            return "No similar tables found."
        
        formatted_output = "\n" + "="*80 + "\n"
        formatted_output += "TOP SIMILAR TABLES FOUND\n"
        formatted_output += "="*80 + "\n"
        
        for i, result in enumerate(similarity_results, 1):
            formatted_output += f"\n#{i} - Similarity: {result['similarity_formatted']}\n"
            formatted_output += f"Table: {result['table_name']}\n"
            
            # Extract headers from description for preview
            description = result['description']
            if 'headers:' in description.lower():
                headers_part = description.split(':')[1] if ':' in description else description
                headers = [h.strip() for h in headers_part.split(',')]
                headers_preview = ', '.join(headers[:5])  # Show first 5 headers
                if len(headers) > 5:
                    headers_preview += f"... (+{len(headers)-5} more)"
                formatted_output += f"Headers: {headers_preview}\n"
            
            formatted_output += f"Description: {description}\n"
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
                print(f"Generating embedding for: {description}")
            except UnicodeEncodeError:
                print("Generating embedding for: [Contains Chinese characters]")
            
            # Use the same embedding model as existing tables
            # Suppress all output to avoid Unicode encoding issues
            import sys
            import os
            from contextlib import redirect_stdout, redirect_stderr
            
            with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
                embeddings = invoke_embedding_model(
                    model_name=self.embedding_model, 
                    texts=[description],  # Pass as list for batch processing
                    silent_mode=True  # Use silent mode to avoid Unicode console issues
                )
            
            if embeddings and len(embeddings) > 0:
                return embeddings[0]  # Return first (and only) embedding
            else:
                print("No embeddings returned from model")
                return None
                
        except Exception as e:
            try:
                print(f"Error generating embedding: {e}")
            except UnicodeEncodeError:
                print("Error generating embedding: [Unicode encoding issue]")
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
            print(f"\nFinding best matches for new table...")
            try:
                print(f"Description: {new_table_description}")
            except UnicodeEncodeError:
                print("Description: [Contains Chinese characters]")
            
            # Generate embedding for new table
            new_embedding = self.embed_new_table_description(new_table_description)
            if new_embedding is None:
                return {
                    'success': False,
                    'error': 'Failed to generate embedding for new table',
                    'matches': [],
                    'formatted_output': 'Error: Could not generate embedding'
                }
            
            # Find similar tables
            similarity_results = self.find_similar_tables(new_embedding, top_n)
            
            if not similarity_results:
                return {
                    'success': False,
                    'error': 'No similar tables found',
                    'matches': [],
                    'formatted_output': 'No similar tables found in the database.'
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
            error_msg = f"Error in get_best_matches: {e}"
            try:
                print(error_msg)
            except UnicodeEncodeError:
                print("Error in get_best_matches: [Unicode encoding issue]")
                error_msg = "Unicode encoding error occurred"
            return {
                'success': False,
                'error': error_msg,
                'matches': [],
                'formatted_output': f'Error: {error_msg}'
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
            print(f"Error getting table by index: {e}")
            return None