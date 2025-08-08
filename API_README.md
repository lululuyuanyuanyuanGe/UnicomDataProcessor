# Asian Information Data Processor API

A FastAPI-based REST API for processing Asian information data files using direct file path access via Docker volume mounts.

## Features

- **Direct File Path Processing**: Access files directly from server filesystem via Docker volumes
- **Database Vectorization**: Generate embeddings for database tables for similarity matching
- **Automatic Classification**: Classify files as tables, documents, or irrelevant content
- **Similarity Matching**: Find similar tables using vector embeddings
- **Docker Volume Mounts**: Secure file access through configured volume mounts
- **High Performance**: No file transfer overhead, direct filesystem access

## API Endpoints

### File Processing

#### `POST /api/process-files`
Process files using direct file paths (Docker volume mount approach).

**Request Body:**
```json
{
  "file_paths": [
    "/data/燕云村残疾人名单.xlsx",
    "/data/村民信息表.csv",
    "/uploads/财务报表.xlsx"
  ],
  "village_name": "燕云村"
}
```

**Response:**
```json
{
  "session_id": "20250108_123456_abc123",
  "status": {
    "status": "completed",
    "message": "Successfully processed 3 files",
    "progress": 100
  },
  "input_files": ["/data/file1.xlsx", "/data/file2.csv", "/data/readme.txt"],
  "processed_files": ["/data/file1.xlsx", "/data/file2.csv"],
  "table_files": [{
    "chinese_table_name": "燕云村残疾人名单",
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
```

#### `GET /api/process-files/status/{session_id}`
Get the processing status of a session.

#### `GET /api/validate-paths`
Validate file paths without processing them.

**Query Parameters:**
- `file_paths`: List of file paths to validate

**Response:**
```json
{
  "validation_results": [
    {
      "path": "/data/test.xlsx",
      "valid": true,
      "resolved_path": "/data/test.xlsx",
      "file_size": 1024,
      "error": null
    }
  ],
  "total_files": 1,
  "valid_files": 1,
  "invalid_files": 0,
  "allowed_base_paths": ["/data", "/uploads", "/shared"]
}
```

### Database Vectorization

#### `POST /api/revectorize-database`
Trigger database re-vectorization using mysqlConnector.

**Request Body:**
```json
{
  "force_refresh": false,
  "specific_tables": ["用户表", "产品表"]
}
```

**Response:**
```json
{
  "status": {
    "status": "completed",
    "message": "Successfully vectorized 15 database tables",
    "progress": 100
  },
  "total_tables": 15,
  "processed_tables": ["用户表", "产品表"],
  "embedding_info": {
    "model": "Qwen/Qwen3-Embedding-8B",
    "embedding_dimension": 1024,
    "total_embeddings": 15
  }
}
```

#### `GET /api/database-info`
Get information about the current database structure and embeddings.

#### `GET /api/database-tables`
List all database tables that can be vectorized.

### Health Check

#### `GET /api/health`
Health check endpoint for monitoring.

## Quick Start

### Docker Deployment (Recommended)

1. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials and volume mount paths
   ```

   **Key Configuration:**
   ```bash
   # Database
   CHAT_BI_ADDR=your_mysql_host
   CHAT_BI_USER=your_db_user
   CHAT_BI_PASSWORD=your_db_password
   
   # Volume Mounts - IMPORTANT: Set these to your actual data paths
   HOST_DATA_PATH=D:\asianInfo\ExcelAssist      # Your main data directory
   HOST_UPLOADS_PATH=D:\uploads                  # User uploads directory
   HOST_SHARED_PATH=D:\shared                    # Shared files directory
   ```

2. **Deploy with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

3. **Verify deployment:**
   ```bash
   curl http://localhost:8000/api/health
   ```

4. **View API documentation:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Manual Docker Build

1. **Build the image:**
   ```bash
   docker build -t dataprocessor-api .
   ```

2. **Run with volume mounts:**
   ```bash
   docker run -p 8000:8000 \
     -e CHAT_BI_ADDR=your_db_host \
     -e CHAT_BI_USER=your_db_user \
     -e CHAT_BI_PASSWORD=your_db_password \
     -e CHAT_BI_DB=your_db_name \
     -e SILICONFLOW_API_KEY=your_api_key \
     -v /host/data:/data \
     -v /host/uploads:/uploads \
     -v /host/shared:/shared \
     -v ./uploaded_files:/app/uploaded_files \
     -v ./embedded_tables:/app/embedded_tables \
     dataprocessor-api
   ```

### Local Development

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Create test directories:**
   ```bash
   mkdir -p /app/data /app/uploads /app/shared
   # Or adjust ALLOWED_BASE_PATHS in api/routers/files.py for local paths
   ```

4. **Start the API server:**
   ```bash
   python run_api.py
   ```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `CHAT_BI_ADDR` | MySQL database host | Yes |
| `CHAT_BI_PORT` | MySQL database port | No (default: 3306) |
| `CHAT_BI_USER` | MySQL database username | Yes |
| `CHAT_BI_PASSWORD` | MySQL database password | Yes |
| `CHAT_BI_DB` | MySQL database name | Yes |
| `SILICONFLOW_API_KEY` | API key for LLM services | Yes |

## File Support

The API supports processing the following file types:

- **Spreadsheets**: .xlsx, .xls, .xlsm, .ods, .csv
- **Documents**: .docx, .doc, .pptx, .ppt
- **Text files**: .txt, .md, .json, .xml, .html, .py, .js, .css, .sql
- **Images**: .jpg, .jpeg, .png, .gif, .bmp, .tiff (metadata extraction)
- **Other formats**: Basic file information extraction

## API Usage Examples

### Python Example

```python
import aiohttp
import asyncio
import json

async def process_files():
    async with aiohttp.ClientSession() as session:
        # Process files using direct file paths
        request_data = {
            "file_paths": [
                "/data/燕云村残疾人名单.xlsx",
                "/data/村民信息表.csv"
            ],
            "village_name": "测试村"
        }
        
        async with session.post(
            'http://localhost:8000/api/process-files',
            json=request_data,
            headers={'Content-Type': 'application/json'}
        ) as response:
            result = await response.json()
            print(f"Session ID: {result['session_id']}")
            print(f"Status: {result['status']['status']}")
            print(f"Tables found: {len(result['table_files'])}")

async def validate_paths():
    async with aiohttp.ClientSession() as session:
        # Validate file paths before processing
        paths_to_check = ["/data/test.xlsx", "/uploads/data.csv"]
        
        async with session.get(
            f'http://localhost:8000/api/validate-paths',
            params={'file_paths': paths_to_check}
        ) as response:
            result = await response.json()
            print(f"Valid files: {result['valid_files']}/{result['total_files']}")

asyncio.run(process_files())
```

### cURL Examples

```bash
# Process files with direct paths
curl -X POST "http://localhost:8000/api/process-files" \
     -H "Content-Type: application/json" \
     -d '{
       "file_paths": ["/data/test.xlsx", "/uploads/data.csv"],
       "village_name": "测试村"
     }'

# Validate file paths
curl -X GET "http://localhost:8000/api/validate-paths" \
     -G -d "file_paths=/data/test.xlsx" -d "file_paths=/uploads/data.csv"

# Vectorize database
curl -X POST "http://localhost:8000/api/revectorize-database" \
     -H "Content-Type: application/json" \
     -d '{"force_refresh": false}'

# Health check
curl http://localhost:8000/api/health
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

async function processFiles() {
    try {
        const response = await axios.post('http://localhost:8000/api/process-files', {
            file_paths: [
                '/data/燕云村残疾人名单.xlsx',
                '/data/村民信息表.csv'
            ],
            village_name: '测试村'
        });
        
        console.log('Processing completed:', response.data);
        console.log('Tables found:', response.data.table_files.length);
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

processFiles();
```

## Architecture

- **FastAPI**: Modern web framework with automatic API documentation
- **Uvicorn**: ASGI server for production deployment
- **Pydantic**: Data validation and serialization
- **Async Processing**: Non-blocking file processing and database operations
- **Docker**: Containerized deployment ready for Linux servers

## Monitoring

- Health check endpoint at `/api/health`
- Structured logging for debugging and monitoring
- Request/response validation with detailed error messages
- Progress tracking for long-running operations

## Security Considerations

- File size limits (50MB per file, max 20 files per request)
- Input validation for all parameters
- Error handling without exposing internal details
- CORS configuration for web clients

## Troubleshooting

1. **Database connection issues**: Check environment variables and network connectivity
2. **File processing failures**: Verify file formats and sizes
3. **Memory issues**: Limit concurrent processing or increase container resources
4. **API timeout**: Increase timeout values for large file processing

For more details, see the auto-generated API documentation at `/docs` when running the server.