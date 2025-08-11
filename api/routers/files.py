from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from typing import List
import logging
from pathlib import Path

from api.models import FileProcessRequest, FileProcessWithIDsRequest, FileProcessResponse
from api.services.file_service import file_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Allowed base paths for security (mounted volumes in Docker)
ALLOWED_BASE_PATHS = [
    "/data",
    "/uploads", 
    "/shared",
    "/mnt/data",
    "/app/data"  # For development
]


def validate_file_path(file_path: str) -> str:
    """
    Validate file path for security and existence
    
    Args:
        file_path: File path to validate
        
    Returns:
        Validated absolute file path
        
    Raises:
        HTTPException: If path is invalid or not allowed
    """
    try:
        # Convert to Path object for better handling
        path = Path(file_path)
        
        # Resolve to absolute path
        resolved_path = path.resolve()
        
        # Check if path is within allowed directories
        path_allowed = False
        for allowed_base in ALLOWED_BASE_PATHS:
            try:
                resolved_path.relative_to(Path(allowed_base).resolve())
                path_allowed = True
                break
            except ValueError:
                continue
        
        if not path_allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"File path not in allowed directories. Allowed: {ALLOWED_BASE_PATHS}"
            )
        
        # Check if file exists
        if not resolved_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {file_path}"
            )
        
        # Check if it's actually a file (not a directory)
        if not resolved_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Path is not a file: {file_path}"
            )
        
        # Check file size (optional limit: 500MB per file)
        file_size = resolved_path.stat().st_size
        max_file_size = 500 * 1024 * 1024  # 500MB
        if file_size > max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large: {file_size} bytes. Maximum: {max_file_size} bytes"
            )
        
        return str(resolved_path)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error validating file path {file_path}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file path: {str(e)}"
        )


@router.post("/process-files", response_model=FileProcessResponse)
async def process_files(
    request: FileProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    Process files using direct file paths (Docker volume mount approach).
    
    This endpoint:
    1. Receives a list of file paths that exist on the server filesystem
    2. Validates that paths are within allowed directories (Docker volume mounts)
    3. Processes them through the existing FileProcessAgent workflow
    4. Returns processing results including table data and similarity scores
    
    Args:
        request: FileProcessRequest containing:
            - file_paths: List of absolute file paths to process
            - village_name: Optional village name for file organization
            
    Returns:
        FileProcessResponse with processing results, table information, and summary
        
    Note:
        Files must be accessible within Docker container via volume mounts.
        Allowed base paths: /data, /uploads, /shared, /mnt/data
    """
    
    # Validate input
    if not request.file_paths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file paths provided for processing"
        )
    
    if len(request.file_paths) > 50:  # Reasonable limit
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many files. Maximum 50 files allowed per request."
        )
    
    try:
        logger.info(f"Received file processing request: {len(request.file_paths)} files, village_name='{request.village_name}'")
        
        # Validate all file paths
        validated_paths = []
        for file_path in request.file_paths:
            try:
                validated_path = validate_file_path(file_path)
                validated_paths.append(validated_path)
                logger.info(f"Validated file path: {file_path} -> {validated_path}")
            except HTTPException as e:
                logger.error(f"File validation failed for {file_path}: {e.detail}")
                # Continue with other files or fail completely?
                # For now, fail the entire request if any file is invalid
                raise
        
        # Convert validated paths to files_data format (for backward compatibility, use dummy file IDs)
        files_data = {path: f"api_file_{i}" for i, path in enumerate(validated_paths)}
        
        # Process files through the service
        result = await file_service.process_files(
            files_data=files_data,
            village_name=request.village_name or ""
        )
        
        # Schedule cleanup of old sessions in the background
        background_tasks.add_task(file_service.cleanup_old_sessions)
        
        logger.info(f"File processing completed for session: {result.session_id}")
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in file processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during file processing: {str(e)}"
        )


@router.post("/process-files-with-ids", response_model=FileProcessResponse)
async def process_files_with_ids(
    request: FileProcessWithIDsRequest,
    background_tasks: BackgroundTasks
):
    """
    Process files using direct file paths with custom file IDs (Docker volume mount approach).
    
    This endpoint:
    1. Receives a mapping of file paths to file IDs
    2. Validates that paths are within allowed directories (Docker volume mounts)
    3. Processes them through the existing FileProcessAgent workflow with file IDs preserved
    4. Returns processing results including table data and similarity scores
    
    Args:
        request: FileProcessWithIDsRequest containing:
            - files_data: Dict mapping file paths to file IDs
            - village_name: Optional village name for file organization
            
    Returns:
        FileProcessResponse with processing results, table information, and summary
        
    Note:
        Files must be accessible within Docker container via volume mounts.
        Allowed base paths: /data, /uploads, /shared, /mnt/data
    """
    
    # Validate input
    if not request.files_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file paths provided for processing"
        )
    
    if len(request.files_data) > 50:  # Reasonable limit
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many files. Maximum 50 files allowed per request."
        )
    
    try:
        logger.info(f"Received file processing request with IDs: {len(request.files_data)} files, village_name='{request.village_name}'")
        
        # Validate all file paths
        validated_files_data = {}
        for file_path, file_id in request.files_data.items():
            try:
                validated_path = validate_file_path(file_path)
                validated_files_data[validated_path] = file_id
                logger.info(f"Validated file path: {file_path} -> {validated_path} (ID: {file_id})")
            except HTTPException as e:
                logger.error(f"File validation failed for {file_path}: {e.detail}")
                # Continue with other files or fail completely?
                # For now, fail the entire request if any file is invalid
                raise
        
        # Process files through the service
        result = await file_service.process_files(
            files_data=validated_files_data,
            village_name=request.village_name or ""
        )
        
        # Schedule cleanup of old sessions in the background
        background_tasks.add_task(file_service.cleanup_old_sessions)
        
        logger.info(f"File processing with IDs completed for session: {result.session_id}")
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in file processing with IDs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during file processing: {str(e)}"
        )


@router.get("/process-files/status/{session_id}")
async def get_processing_status(session_id: str):
    """
    Get the status of a file processing session.
    
    Args:
        session_id: The processing session ID returned from the process-files endpoint
        
    Returns:
        Current processing status and progress information
    """
    try:
        session_data = file_service.get_session_status(session_id)
        
        if not session_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Processing session {session_id} not found"
            )
        
        return {
            "session_id": session_id,
            "status": session_data.get("status", "unknown"),
            "start_time": session_data.get("start_time"),
            "end_time": session_data.get("end_time"),
            "total_files": session_data.get("total_files", 0),
            "error": session_data.get("error")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session status"
        )


@router.get("/validate-paths")
async def validate_paths(file_paths: List[str]):
    """
    Validate a list of file paths without processing them.
    
    Useful for checking if files are accessible before starting processing.
    
    Args:
        file_paths: List of file paths to validate
        
    Returns:
        Validation results for each path
    """
    results = []
    
    for file_path in file_paths:
        result = {
            "path": file_path,
            "valid": False,
            "error": None,
            "resolved_path": None,
            "file_size": None
        }
        
        try:
            validated_path = validate_file_path(file_path)
            file_size = Path(validated_path).stat().st_size
            
            result.update({
                "valid": True,
                "resolved_path": validated_path,
                "file_size": file_size
            })
            
        except HTTPException as e:
            result["error"] = e.detail
        except Exception as e:
            result["error"] = str(e)
        
        results.append(result)
    
    return {
        "validation_results": results,
        "total_files": len(file_paths),
        "valid_files": sum(1 for r in results if r["valid"]),
        "invalid_files": sum(1 for r in results if not r["valid"]),
        "allowed_base_paths": ALLOWED_BASE_PATHS
    }