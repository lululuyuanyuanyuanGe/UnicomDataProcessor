import sys
import asyncio
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.mysqlConnector import DatabaseManager
from api.models import DatabaseVectorizationResponse, ProcessingStatus

logger = logging.getLogger(__name__)


class DatabaseVectorizationService:
    """Service for handling database vectorization operations"""
    
    def __init__(self):
        self.active_tasks: Dict[str, Dict] = {}
    
    async def vectorize_database(
        self,
        force_refresh: bool = False,
        specific_tables: Optional[List[str]] = None
    ) -> DatabaseVectorizationResponse:
        """
        Vectorize database tables using the existing DatabaseManager
        
        Args:
            force_refresh: Whether to force refresh all embeddings
            specific_tables: Optional list of specific tables to vectorize
            
        Returns:
            DatabaseVectorizationResponse with vectorization results
        """
        task_id = f"vectorization_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        try:
            logger.info(f"Starting database vectorization task: {task_id}")
            
            # Update task status
            self.active_tasks[task_id] = {
                "status": "processing",
                "start_time": datetime.now(),
                "force_refresh": force_refresh,
                "specific_tables": specific_tables
            }
            
            # Run vectorization in executor to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._run_vectorization,
                force_refresh,
                specific_tables
            )
            
            # Update task status
            self.active_tasks[task_id]["status"] = "completed" if result["success"] else "failed"
            self.active_tasks[task_id]["end_time"] = datetime.now()
            
            # Format response
            if result["success"]:
                response = DatabaseVectorizationResponse(
                    status=ProcessingStatus(
                        status="completed",
                        message=f"Successfully vectorized {result.get('total_tables', 0)} database tables",
                        progress=100
                    ),
                    total_tables=result.get('total_tables', 0),
                    processed_tables=result.get('processed_tables', []),
                    failed_tables=result.get('failed_tables', []),
                    embedding_info=result.get('embedding_info', {})
                )
                logger.info(f"Database vectorization completed for task: {task_id}")
            else:
                response = DatabaseVectorizationResponse(
                    status=ProcessingStatus(
                        status="failed",
                        message=f"Vectorization failed: {result.get('error', 'Unknown error')}",
                        progress=0
                    ),
                    embedding_info={"error": result.get('error', 'Unknown error')}
                )
                logger.error(f"Database vectorization failed for task: {task_id}")
            
            return response
            
        except Exception as e:
            logger.error(f"Database vectorization failed for task {task_id}: {e}", exc_info=True)
            
            # Update task status
            self.active_tasks[task_id] = {
                "status": "failed",
                "error": str(e),
                "end_time": datetime.now()
            }
            
            return DatabaseVectorizationResponse(
                status=ProcessingStatus(
                    status="failed",
                    message=f"Vectorization failed: {str(e)}",
                    progress=0
                ),
                embedding_info={"error": str(e)}
            )
    
    def _run_vectorization(
        self,
        force_refresh: bool,
        specific_tables: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Run database vectorization synchronously
        
        Args:
            force_refresh: Whether to force refresh all embeddings
            specific_tables: Optional list of specific tables to vectorize
            
        Returns:
            Vectorization result dictionary
        """
        db_manager = None
        try:
            logger.info("Initializing DatabaseManager...")
            db_manager = DatabaseManager()
            
            # Track processed tables
            processed_tables = []
            failed_tables = []
            
            if specific_tables:
                logger.info(f"Processing specific tables: {specific_tables}")
                # For specific tables, we would need to modify DatabaseManager
                # to support selective processing. For now, process all tables
                # and filter the results.
                logger.warning("Specific table processing not yet implemented, processing all tables")
            
            # Step 1: Update data.json with schema information
            logger.info("Updating database schema information...")
            db_manager.update_data_json()
            
            # Step 2: Create embeddings
            logger.info("Creating table embeddings...")
            db_manager.create_table_embeddings()
            
            # Get information about processed tables
            data = db_manager.load_data_json()
            database_tables = data.get("ChatBI", {}).get("数据库表格", {})
            
            processed_tables = list(database_tables.keys())
            total_tables = len(processed_tables)
            
            # Get embedding information
            embedding_info = {
                "model": "Qwen/Qwen3-Embedding-8B",
                "total_embeddings": total_tables,
                "embedding_dimension": 1024,  # Default dimension for the model
                "source": "database_schema"
            }
            
            # Try to get actual embedding info from saved files
            try:
                import json
                metadata_path = project_root / "embedded_tables" / "database_table_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    embedding_info.update({
                        "embedding_dimension": metadata.get("embedding_dimension", 1024),
                        "total_embeddings": metadata.get("total_tables", total_tables),
                        "model": metadata.get("model", "Qwen/Qwen3-Embedding-8B")
                    })
            except Exception as e:
                logger.warning(f"Failed to read embedding metadata: {e}")
            
            logger.info(f"Successfully processed {total_tables} database tables")
            
            return {
                "success": True,
                "total_tables": total_tables,
                "processed_tables": processed_tables,
                "failed_tables": failed_tables,
                "embedding_info": embedding_info
            }
            
        except Exception as e:
            logger.error(f"DatabaseManager execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "total_tables": 0,
                "processed_tables": [],
                "failed_tables": [],
                "embedding_info": {}
            }
        finally:
            # Clean up database connection
            if db_manager:
                try:
                    db_manager.disconnect()
                except Exception as e:
                    logger.warning(f"Failed to disconnect database: {e}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get the status of a vectorization task"""
        return self.active_tasks.get(task_id)
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Clean up old task data"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        expired_tasks = []
        for task_id, task_data in self.active_tasks.items():
            start_time = task_data.get("start_time")
            if start_time and start_time.timestamp() < cutoff_time:
                expired_tasks.append(task_id)
        
        for task_id in expired_tasks:
            del self.active_tasks[task_id]
            logger.info(f"Cleaned up expired task: {task_id}")


# Global service instance
db_service = DatabaseVectorizationService()