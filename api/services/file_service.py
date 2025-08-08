import sys
import asyncio
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import tempfile
import os
import shutil
import logging

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.fileProcessAgent import FileProcessAgent
from api.models import FileProcessResponse, ProcessingStatus, TableInfo

logger = logging.getLogger(__name__)


class FileProcessingService:
    """Service for handling file processing operations"""
    
    def __init__(self):
        self.agent = FileProcessAgent()
        self.active_sessions: Dict[str, Dict] = {}
    
    async def process_files(
        self,
        uploaded_files: List[bytes],
        filenames: List[str],
        village_name: Optional[str] = ""
    ) -> FileProcessResponse:
        """
        Process uploaded files using the existing FileProcessAgent
        
        Args:
            uploaded_files: List of file contents as bytes
            filenames: List of original filenames
            village_name: Optional village name parameter
            
        Returns:
            FileProcessResponse with processing results
        """
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        try:
            logger.info(f"Starting file processing session: {session_id}")
            
            # Update session status
            self.active_sessions[session_id] = {
                "status": "processing",
                "start_time": datetime.now(),
                "total_files": len(uploaded_files)
            }
            
            # Save uploaded files to temporary directory
            temp_dir = Path(tempfile.mkdtemp())
            saved_file_paths = []
            
            try:
                for file_content, filename in zip(uploaded_files, filenames):
                    # Save file to temporary location
                    temp_file_path = temp_dir / filename
                    with open(temp_file_path, 'wb') as f:
                        f.write(file_content)
                    saved_file_paths.append(str(temp_file_path))
                    logger.info(f"Saved uploaded file: {filename}")
                
                # Process files using existing FileProcessAgent
                logger.info("Invoking FileProcessAgent...")
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._run_file_processing,
                    session_id,
                    saved_file_paths,
                    village_name or ""
                )
                
                # Parse and format results
                response = self._format_response(session_id, result, saved_file_paths)
                
                # Update session status
                self.active_sessions[session_id]["status"] = "completed"
                self.active_sessions[session_id]["end_time"] = datetime.now()
                
                logger.info(f"File processing completed for session: {session_id}")
                return response
                
            finally:
                # Clean up temporary files
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp directory: {cleanup_error}")
        
        except Exception as e:
            logger.error(f"File processing failed for session {session_id}: {e}", exc_info=True)
            
            # Update session status
            self.active_sessions[session_id] = {
                "status": "failed",
                "error": str(e),
                "end_time": datetime.now()
            }
            
            return FileProcessResponse(
                session_id=session_id,
                status=ProcessingStatus(
                    status="failed",
                    message=f"Processing failed: {str(e)}",
                    progress=0
                ),
                summary={"error": str(e)}
            )
    
    def _run_file_processing(self, session_id: str, file_paths: List[str], village_name: str) -> Dict:
        """
        Run the file processing agent synchronously
        
        Args:
            session_id: Processing session ID
            file_paths: List of file paths to process
            village_name: Village name parameter
            
        Returns:
            Processing result dictionary
        """
        try:
            # Use the existing FileProcessAgent
            result = self.agent.run_file_process_agent(
                session_id=session_id,
                upload_files_path=file_paths,
                village_name=village_name
            )
            
            return {
                "success": True,
                "result": result,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"FileProcessAgent execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    def _format_response(
        self,
        session_id: str,
        agent_result: Dict,
        original_file_paths: List[str]
    ) -> FileProcessResponse:
        """
        Format the FileProcessAgent result into API response format
        
        Args:
            session_id: Processing session ID
            agent_result: Result from FileProcessAgent
            original_file_paths: Original uploaded file paths
            
        Returns:
            Formatted FileProcessResponse
        """
        try:
            if not agent_result.get("success", False):
                return FileProcessResponse(
                    session_id=session_id,
                    status=ProcessingStatus(
                        status="failed",
                        message=agent_result.get("error", "Unknown error"),
                        progress=0
                    )
                )
            
            # Extract information from agent result
            result_data = agent_result.get("result", {})
            
            # Get file lists from the final state
            table_files = []
            processed_files = []
            irrelevant_files = []
            
            # Check if we have the final state data
            if isinstance(result_data, dict):
                # Extract table information
                table_files_data = result_data.get("table_files_path", [])
                for table_entry in table_files_data:
                    if isinstance(table_entry, dict):
                        # Try to extract table info from processed results
                        table_info = TableInfo(
                            chinese_table_name=self._extract_table_name(table_entry),
                            headers=self._extract_headers(table_entry),
                            header_count=len(self._extract_headers(table_entry))
                        )
                        table_files.append(table_info)
                
                # Extract processed and irrelevant files
                processed_files_data = result_data.get("new_upload_files_processed_path", [])
                irrelevant_files_data = result_data.get("irrelevant_files_path", [])
                
                processed_files = [
                    Path(f.get("path", "") if isinstance(f, dict) else str(f)).name
                    for f in processed_files_data
                ]
                
                irrelevant_files = [
                    Path(f.get("path", "") if isinstance(f, dict) else str(f)).name
                    for f in irrelevant_files_data
                ]
            
            # Create summary
            summary = {
                "total_files": len(original_file_paths),
                "tables_found": len(table_files),
                "processed_files": len(processed_files),
                "irrelevant_files": len(irrelevant_files),
                "session_duration": "completed"
            }
            
            return FileProcessResponse(
                session_id=session_id,
                status=ProcessingStatus(
                    status="completed",
                    message=f"Successfully processed {len(original_file_paths)} files",
                    progress=100
                ),
                processed_files=processed_files,
                table_files=table_files,
                irrelevant_files=irrelevant_files,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Failed to format response: {e}", exc_info=True)
            return FileProcessResponse(
                session_id=session_id,
                status=ProcessingStatus(
                    status="failed",
                    message=f"Failed to format response: {str(e)}",
                    progress=0
                ),
                summary={"error": str(e)}
            )
    
    def _extract_table_name(self, table_entry: Dict) -> str:
        """Extract table name from table entry"""
        if isinstance(table_entry, dict):
            path = table_entry.get("path", "")
            if path:
                return Path(path).stem
        return "Unknown Table"
    
    def _extract_headers(self, table_entry: Dict) -> List[str]:
        """Extract headers from table entry"""
        # This is a placeholder - the actual headers would come from
        # the processed table data in the FileProcessAgent result
        return []
    
    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get the status of a processing session"""
        return self.active_sessions.get(session_id)
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old session data"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        expired_sessions = []
        for session_id, session_data in self.active_sessions.items():
            start_time = session_data.get("start_time")
            if start_time and start_time.timestamp() < cutoff_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")


# Global service instance
file_service = FileProcessingService()