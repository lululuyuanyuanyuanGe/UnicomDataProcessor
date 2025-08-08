import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import logging

from api.models import HealthCheckResponse, ErrorResponse
from api.routers import files, database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Asian Information Data Processor API",
    description="API for processing Asian information data files and managing database vectorization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(files.router, prefix="/api", tags=["File Processing"])
app.include_router(database.router, prefix="/api", tags=["Database Vectorization"])


@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connectivity
        from src.mysqlConnector import DatabaseManager
        db_status = "connected"
        try:
            db = DatabaseManager()
            db.disconnect()
        except Exception as e:
            db_status = f"error: {str(e)[:50]}"
            logger.warning(f"Database health check failed: {e}")
        
        return HealthCheckResponse(
            status="healthy",
            version="1.0.0",
            services={
                "database": db_status,
                "file_processor": "ready",
                "embedding_service": "ready"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            details={"status_code": exc.status_code}
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An internal server error occurred",
            details={"exception_type": type(exc).__name__}
        ).dict()
    )


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Asian Information Data Processor API starting up...")
    
    # Create necessary directories
    from pathlib import Path
    temp_dir = project_root / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    uploaded_files_dir = project_root / "uploaded_files"
    uploaded_files_dir.mkdir(exist_ok=True)
    
    logger.info("API startup completed")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Asian Information Data Processor API shutting down...")


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )