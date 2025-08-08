from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status, BackgroundTasks
from typing import List, Optional
import logging

from api.models import FileProcessResponse, ProcessingStatus, ErrorResponse
from api.services.file_service import file_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/process-files", response_model=FileProcessResponse)
async def process_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Files to process"),
    village_name: Optional[str] = Form(default="", description="Village name (optional)")
):
    """
    Process uploaded files using the FileProcessAgent workflow.
    
    This endpoint:
    1. Receives multiple files from the client
    2. Processes them through the existing FileProcessAgent
    3. Returns processing results including table data and similarity scores
    4. Handles file classification (tables, documents, irrelevant)
    
    Args:
        files: List of files to process (Excel, CSV, Word documents, text files, etc.)
        village_name: Optional village name parameter for file organization
        
    Returns:
        FileProcessResponse with processing results, table information, and summary
    """
    
    # Validate input
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided for processing"
        )
    
    if len(files) > 20:  # Reasonable limit
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many files. Maximum 20 files allowed per request."
        )
    
    # Validate file sizes (50MB per file limit)
    max_file_size = 50 * 1024 * 1024  # 50MB
    for file in files:
        if hasattr(file, 'size') and file.size and file.size > max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File {file.filename} is too large. Maximum size is 50MB."
            )
    
    try:
        logger.info(f"Received file processing request: {len(files)} files, village_name='{village_name}'")
        
        # Read file contents
        file_contents = []
        filenames = []
        
        for file in files:
            try:
                content = await file.read()
                file_contents.append(content)
                filenames.append(file.filename or f"unnamed_{len(filenames)}")
                logger.info(f"Read file: {file.filename} ({len(content)} bytes)")
            except Exception as e:
                logger.error(f"Failed to read file {file.filename}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to read file {file.filename}: {str(e)}"
                )
        
        # Process files through the service
        result = await file_service.process_files(
            uploaded_files=file_contents,
            filenames=filenames,
            village_name=village_name
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