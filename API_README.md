# Asian Information Data Processor API

A FastAPI-based REST API for processing Asian information data files and managing database vectorization.

## Features

- **File Processing**: Upload and process various file types (Excel, CSV, Word documents, text files)
- **Database Vectorization**: Generate embeddings for database tables for similarity matching
- **Automatic Classification**: Classify uploaded files as tables, documents, or irrelevant content
- **Similarity Matching**: Find similar tables using vector embeddings
- **Docker Support**: Ready for containerized deployment on Linux servers

## API Endpoints

### File Processing

#### `POST /api/process-files`
Process uploaded files through the FileProcessAgent workflow.

**Parameters:**
- `files`: List of files to upload (multipart/form-data)
- `village_name`: Optional village name for file organization

**Response:**
```json
{
  "session_id": "20250108_123456_abc123",
  "status": {
    "status": "completed",
    "message": "Successfully processed 3 files",
    "progress": 100
  },
  "processed_files": ["file1.xlsx", "file2.csv"],
  "table_files": [{
    "chinese_table_name": "燕云村残疾人名单",
    "headers": ["姓名", "年龄", "残疾类型"],
    "header_count": 3
  }],
  "irrelevant_files": ["readme.txt"],
  "summary": {
    "total_files": 3,
    "tables_found": 2,
    "irrelevant_files": 1
  }
}
```

#### `GET /api/process-files/status/{session_id}`
Get the processing status of a session.

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

### Local Development

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your database and API credentials
   ```

3. **Start the API server:**
   ```bash
   python start_api.py
   ```

4. **Test the API:**
   ```bash
   python test_api.py
   ```

5. **View API documentation:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Docker Deployment

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

2. **View logs:**
   ```bash
   docker-compose logs -f
   ```

3. **Stop the service:**
   ```bash
   docker-compose down
   ```

### Manual Docker Build

1. **Build the image:**
   ```bash
   docker build -t dataprocessor-api .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 \
     -e CHAT_BI_ADDR=your_db_host \
     -e CHAT_BI_USER=your_db_user \
     -e CHAT_BI_PASSWORD=your_db_password \
     -e CHAT_BI_DB=your_db_name \
     -e SILICONFLOW_API_KEY=your_api_key \
     -v ./uploaded_files:/app/uploaded_files \
     -v ./embedded_tables:/app/embedded_tables \
     dataprocessor-api
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

async def process_files():
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('village_name', '测试村')
        
        with open('sample.xlsx', 'rb') as f:
            data.add_field('files', f.read(), filename='sample.xlsx')
        
        async with session.post('http://localhost:8000/api/process-files', 
                               data=data) as response:
            result = await response.json()
            print(result)

asyncio.run(process_files())
```

### cURL Example

```bash
# Process files
curl -X POST "http://localhost:8000/api/process-files" \
     -F "files=@sample.xlsx" \
     -F "village_name=测试村"

# Vectorize database
curl -X POST "http://localhost:8000/api/revectorize-database" \
     -H "Content-Type: application/json" \
     -d '{"force_refresh": false}'

# Health check
curl http://localhost:8000/api/health
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