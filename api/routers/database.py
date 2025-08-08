from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from typing import Optional
import logging

from api.models import DatabaseVectorizationRequest, DatabaseVectorizationResponse, ProcessingStatus
from api.services.db_service import db_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/revectorize-database", response_model=DatabaseVectorizationResponse)
async def revectorize_database(
    background_tasks: BackgroundTasks,
    request: Optional[DatabaseVectorizationRequest] = None
):
    """
    Trigger database re-vectorization using the existing mysqlConnector.
    
    This endpoint:
    1. Connects to the MySQL database and extracts table schemas
    2. Updates the data.json with current database structure
    3. Generates embeddings for all database tables using the embedding model
    4. Saves embeddings for similarity matching with uploaded files
    
    Args:
        request: Optional request parameters including:
            - force_refresh: Whether to force refresh all embeddings (default: False)
            - specific_tables: Optional list of specific tables to vectorize
            
    Returns:
        DatabaseVectorizationResponse with processing results and embedding information
    """
    
    try:
        # Use default values if no request provided
        if request is None:
            request = DatabaseVectorizationRequest()
        
        logger.info(f"Received database vectorization request: force_refresh={request.force_refresh}, specific_tables={request.specific_tables}")
        
        # Validate specific_tables if provided
        if request.specific_tables is not None:
            if len(request.specific_tables) > 100:  # Reasonable limit
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Too many tables specified. Maximum 100 tables allowed per request."
                )
            
            if not all(isinstance(table, str) and table.strip() for table in request.specific_tables):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="All table names must be non-empty strings."
                )
        
        # Process vectorization through the service
        result = await db_service.vectorize_database(
            force_refresh=request.force_refresh,
            specific_tables=request.specific_tables
        )
        
        # Schedule cleanup of old tasks in the background
        background_tasks.add_task(db_service.cleanup_old_tasks)
        
        logger.info(f"Database vectorization completed with status: {result.status.status}")
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in database vectorization: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during database vectorization: {str(e)}"
        )


@router.get("/database-info")
async def get_database_info():
    """
    Get information about the current database structure and embeddings.
    
    Returns:
        Information about database tables, embeddings, and vectorization status
    """
    try:
        from pathlib import Path
        import json
        
        project_root = Path(__file__).resolve().parent.parent.parent
        
        # Get data.json information
        data_json_path = project_root / "src" / "data.json"
        database_info = {
            "database_tables": {},
            "total_tables": 0,
            "embedding_info": {},
            "last_updated": None
        }
        
        # Load database table information
        if data_json_path.exists():
            try:
                with open(data_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                database_tables = data.get("ChatBI", {}).get("数据库表格", {})
                database_info["database_tables"] = {
                    name: {
                        "header_count": info.get("header_count", 0),
                        "english_name": info.get("english_table_name", ""),
                        "chinese_headers": info.get("chinese_headers", [])[:5]  # Show first 5 headers only
                    }
                    for name, info in database_tables.items()
                }
                database_info["total_tables"] = len(database_tables)
                
            except Exception as e:
                logger.warning(f"Failed to read data.json: {e}")
        
        # Get embedding information
        metadata_path = project_root / "embedded_tables" / "database_table_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                database_info["embedding_info"] = {
                    "model": metadata.get("model", "unknown"),
                    "total_embeddings": metadata.get("total_tables", 0),
                    "embedding_dimension": metadata.get("embedding_dimension", 0),
                    "last_updated": metadata.get("timestamp")
                }
                
                if metadata.get("timestamp"):
                    from datetime import datetime
                    database_info["last_updated"] = datetime.fromtimestamp(metadata["timestamp"]).isoformat()
                
            except Exception as e:
                logger.warning(f"Failed to read embedding metadata: {e}")
        
        return database_info
        
    except Exception as e:
        logger.error(f"Error getting database info: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve database information"
        )


@router.get("/database-tables")
async def list_database_tables():
    """
    List all database tables that can be vectorized.
    
    Returns:
        List of database table names with basic information
    """
    try:
        logger.info("Attempting to connect to database to list tables...")
        
        # Import here to avoid issues if database is not available
        from src.mysqlConnector import DatabaseManager
        
        db_manager = None
        try:
            db_manager = DatabaseManager()
            tables = db_manager.fetch_all_tables()
            
            return {
                "total_tables": len(tables),
                "tables": tables,
                "status": "connected"
            }
            
        finally:
            if db_manager:
                db_manager.disconnect()
        
    except Exception as e:
        logger.warning(f"Failed to connect to database: {e}")
        
        # Fallback: try to get table list from existing data.json
        try:
            from pathlib import Path
            import json
            
            project_root = Path(__file__).resolve().parent.parent.parent
            data_json_path = project_root / "src" / "data.json"
            
            if data_json_path.exists():
                with open(data_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                database_tables = data.get("ChatBI", {}).get("数据库表格", {})
                table_names = list(database_tables.keys())
                
                return {
                    "total_tables": len(table_names),
                    "tables": table_names,
                    "status": "from_cache",
                    "warning": "Database connection failed, showing cached table list"
                }
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
        
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot connect to database and no cached data available: {str(e)}"
        )