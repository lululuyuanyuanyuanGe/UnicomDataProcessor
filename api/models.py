from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# Request Models
class FileProcessRequest(BaseModel):
    """Request model for file processing endpoint"""
    file_paths: List[str] = Field(..., description="List of file paths to process")
    village_name: Optional[str] = Field(default="", description="Village name (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_paths": [
                    "/data/燕云村残疾人名单.xlsx",
                    "/data/村民信息表.csv",
                    "/uploads/财务报表.xlsx"
                ],
                "village_name": "燕云村"
            }
        }


class DatabaseVectorizationRequest(BaseModel):
    """Request model for database vectorization endpoint"""
    force_refresh: Optional[bool] = Field(default=False, description="Force refresh all embeddings")
    specific_tables: Optional[List[str]] = Field(default=None, description="Specific tables to vectorize")
    
    class Config:
        json_schema_extra = {
            "example": {
                "force_refresh": False,
                "specific_tables": ["用户表", "产品表"]
            }
        }


# Response Models
class ProcessingStatus(BaseModel):
    """Processing status information"""
    status: str = Field(description="Processing status: pending, processing, completed, failed")
    message: str = Field(description="Status message")
    progress: Optional[int] = Field(default=None, description="Progress percentage (0-100)")


class TableInfo(BaseModel):
    """Table information model"""
    chinese_table_name: str
    english_table_name: Optional[str] = None
    headers: List[str]
    header_count: int
    similarity_scores: Optional[List[Dict[str, Any]]] = None


class FileProcessResponse(BaseModel):
    """Response model for file processing endpoint"""
    session_id: str
    status: ProcessingStatus
    input_files: List[str] = Field(default_factory=list, description="Original input file paths")
    processed_files: List[str] = Field(default_factory=list, description="Successfully processed file paths")
    table_files: List[TableInfo] = Field(default_factory=list, description="Files identified as tables")
    irrelevant_files: List[str] = Field(default_factory=list, description="Files identified as irrelevant")
    summary: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "20250108_123456",
                "status": {
                    "status": "completed",
                    "message": "Successfully processed 3 files",
                    "progress": 100
                },
                "input_files": ["/data/file1.xlsx", "/data/file2.csv", "/data/readme.txt"],
                "processed_files": ["/data/file1.xlsx", "/data/file2.csv"],
                "table_files": [{
                    "chinese_table_name": "燕云村残疾人名单",
                    "english_table_name": "disability_list",
                    "headers": ["姓名", "年龄", "残疾类型"],
                    "header_count": 3,
                    "similarity_scores": []
                }],
                "irrelevant_files": ["/data/readme.txt"],
                "summary": {
                    "total_files": 3,
                    "tables_found": 2,
                    "irrelevant_files": 1
                }
            }
        }


class DatabaseVectorizationResponse(BaseModel):
    """Response model for database vectorization endpoint"""
    status: ProcessingStatus
    total_tables: int = 0
    processed_tables: List[str] = Field(default_factory=list)
    failed_tables: List[str] = Field(default_factory=list)
    embedding_info: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": {
                    "status": "completed",
                    "message": "Successfully vectorized 15 database tables",
                    "progress": 100
                },
                "total_tables": 15,
                "processed_tables": ["用户表", "产品表", "订单表"],
                "failed_tables": [],
                "embedding_info": {
                    "model": "Qwen/Qwen3-Embedding-8B",
                    "embedding_dimension": 1024,
                    "total_embeddings": 15
                }
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    services: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "services": {
                    "database": "connected",
                    "file_processor": "ready",
                    "embedding_service": "ready"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid file format provided",
                "details": {"supported_formats": ["xlsx", "csv", "txt", "docx"]},
                "timestamp": "2025-01-08T12:34:56"
            }
        }