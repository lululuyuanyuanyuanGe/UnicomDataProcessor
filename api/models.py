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


class FileProcessWithIDsRequest(BaseModel):
    """Request model for file processing endpoint with file IDs"""
    files_data: Dict[str, str] = Field(..., description="Mapping of file paths to file IDs")
    village_name: Optional[str] = Field(default="", description="Village name (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "files_data": {
                    "/data/燕云村残疾人名单.xlsx": "file_001",
                    "/data/村民信息表.csv": "file_002",
                    "/uploads/财务报表.xlsx": "file_003"
                },
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


class SimilarityMatch(BaseModel):
    """Similarity match information"""
    index: int
    table_name: str
    description: str
    similarity_percentage: float
    similarity_formatted: str


class SimilarityResult(BaseModel):
    """Complete similarity result with top matches and best match"""
    top_matches: List[SimilarityMatch] = Field(default_factory=list)
    best_match: Optional[SimilarityMatch] = None


class UploadedFileInfo(BaseModel):
    """Individual uploaded file information matching uploaded_files.json structure"""
    excel_name: str
    timestamp: str
    file_id: Optional[str] = None
    headers: List[str]
    original_file_path: str
    table_description: str
    similarity_match: SimilarityResult


class UploadedFilesResponse(BaseModel):
    """Response model returning uploaded_files.json contents"""
    files: List[UploadedFileInfo] = Field(description="List of uploaded file information")
    total_files: int = Field(description="Total number of files")
    
    class Config:
        json_schema_extra = {
            "example": {
                "files": [
                    {
                        "excel_name": "党员信息",
                        "timestamp": "2025-08-13T14:13:00.996033",
                        "file_id": "file_001",
                        "headers": ["姓名", "性别", "民族", "学历"],
                        "original_file_path": "uploaded_files\\党员信息.xlsx",
                        "table_description": "党员信息 包含表头：姓名,性别,民族,学历",
                        "similarity_match": {
                            "top_matches": [
                                {
                                    "index": 2,
                                    "table_name": "党员信息表",
                                    "description": "党员信息表，包含各村数据",
                                    "similarity_percentage": 94.04,
                                    "similarity_formatted": "94.0%"
                                }
                            ],
                            "best_match": {
                                "index": 2,
                                "table_name": "党员信息表",
                                "description": "党员信息表，包含各村数据",
                                "similarity_percentage": 94.04,
                                "similarity_formatted": "94.0%"
                            }
                        }
                    }
                ],
                "total_files": 1
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