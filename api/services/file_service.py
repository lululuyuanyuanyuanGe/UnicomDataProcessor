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

from src.fileProcessAgent import FileProcessAgent
from api.models import FileProcessResponse, ProcessingStatus, TableInfo

logger = logging.getLogger(__name__)


class FileProcessingService:
    """Service for handling file processing operations with direct file paths"""
    
    def __init__(self):
        self.agent = FileProcessAgent()
        self.active_sessions: Dict[str, Dict] = {}
    
    async def process_files(
        self,
        files_data: Dict[str, str],
        village_name: Optional[str] = ""
    ) -> FileProcessResponse:
        """
        Process files using direct file paths (Docker volume mount approach)
        
        Args:
            files_data: Dict mapping file paths to file IDs {file_path: file_id}
            village_name: Optional village name parameter
            
        Returns:
            FileProcessResponse with processing results
        """
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        try:
            logger.info(f"Starting file processing session: {session_id}")
            file_paths = list(files_data.keys())
            logger.info(f"Processing {len(file_paths)} files: {[Path(p).name for p in file_paths]}")
            
            # Update session status
            self.active_sessions[session_id] = {
                "status": "processing",
                "start_time": datetime.now(),
                "total_files": len(file_paths),
                "input_files": file_paths.copy()
            }
            
            # Validate all files exist (should already be validated, but double-check)
            validated_files_data = {}
            for file_path, file_id in files_data.items():
                path = Path(file_path)
                if not path.exists():
                    logger.error(f"File not found during processing: {file_path}")
                    continue
                validated_files_data[str(path.resolve())] = file_id
            
            if not validated_files_data:
                raise ValueError("No valid files found for processing")
            
            # Process files using existing FileProcessAgent directly
            logger.info("Invoking FileProcessAgent with direct file paths...")
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._run_file_processing,
                session_id,
                validated_files_data,
                village_name or ""
            )
            
            # Parse and format results
            response = self._format_response(session_id, result, file_paths, list(validated_files_data.keys()))
            
            # Update session status
            self.active_sessions[session_id]["status"] = "completed"
            self.active_sessions[session_id]["end_time"] = datetime.now()
            
            logger.info(f"File processing completed for session: {session_id}")
            return response
                
        except Exception as e:
            logger.error(f"File processing failed for session {session_id}: {e}", exc_info=True)
            
            # Update session status
            self.active_sessions[session_id] = {
                "status": "failed",
                "error": str(e),
                "end_time": datetime.now(),
                "input_files": file_paths.copy()
            }
            
            return FileProcessResponse(
                session_id=session_id,
                status=ProcessingStatus(
                    status="failed",
                    message=f"Processing failed: {str(e)}",
                    progress=0
                ),
                input_files=file_paths,
                summary={"error": str(e)}
            )
    
    def _run_file_processing(self, session_id: str, files_data: Dict[str, str], village_name: str) -> Dict:
        """
        Run the file processing agent synchronously with direct file paths
        
        Args:
            session_id: Processing session ID
            files_data: Dict mapping file paths to file IDs
            village_name: Village name parameter
            
        Returns:
            Processing result dictionary
        """
        try:
            logger.info(f"FileProcessAgent processing {len(files_data)} files")
            
            # Use the existing FileProcessAgent directly with files data
            # The FileProcessAgent expects a dict mapping file paths to file IDs
            result = self.agent.run_file_process_agent(
                session_id=session_id,
                upload_files_data=files_data,  # Direct file paths mapped to file IDs
                village_name=village_name
            )
            
            logger.info("FileProcessAgent completed successfully")
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
        original_file_paths: List[str],
        validated_file_paths: List[str]
    ) -> FileProcessResponse:
        """
        Format the FileProcessAgent result into API response format
        
        Args:
            session_id: Processing session ID
            agent_result: Result from FileProcessAgent
            original_file_paths: Original input file paths
            validated_file_paths: Validated file paths that were processed
            
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
                    ),
                    input_files=original_file_paths
                )
            
            # Extract information from agent result
            result_data = agent_result.get("result", {})
            
            # Get file lists from the final state
            table_files = []
            processed_files = []
            irrelevant_files = []
            
            # Check if we have the final state data
            if isinstance(result_data, dict):
                # Extract table information from processed results
                processed_table_results = result_data.get("processed_table_results", [])
                for table_entry in processed_table_results:
                    if isinstance(table_entry, dict) and table_entry.get("success", False):
                        table_info = TableInfo(
                            chinese_table_name=table_entry.get("chinese_table_name", "Unknown"),
                            headers=table_entry.get("headers", []),
                            header_count=len(table_entry.get("headers", [])),
                            similarity_scores=self._extract_similarity_scores(table_entry)
                        )
                        table_files.append(table_info)
                        
                        # Add to processed files
                        file_path = table_entry.get("file_path", "")
                        if file_path:
                            processed_files.append(file_path)
                
                # Extract irrelevant files
                irrelevant_files_data = result_data.get("irrelevant_files_path", [])
                irrelevant_files = [
                    f.get("path", "") if isinstance(f, dict) else str(f)
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
                input_files=original_file_paths,
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
                input_files=original_file_paths,
                summary={"error": str(e)}
            )
    
    def _extract_similarity_scores(self, table_entry: Dict) -> Optional[List[Dict[str, Any]]]:
        """Extract similarity scores from table entry"""
        similarity_match = table_entry.get("similarity_match", {})
        if isinstance(similarity_match, dict):
            top_matches = similarity_match.get("top_matches", [])
            if top_matches:
                return top_matches
        return None
    
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