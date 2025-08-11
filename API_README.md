# 亚洲信息数据处理器 API

基于 FastAPI 的 REST API，用于通过 Docker 卷挂载直接访问文件路径来处理亚洲信息数据文件。

## 功能特性

- **直接文件路径处理**：通过 Docker 卷直接从服务器文件系统访问文件
- **数据库向量化**：为数据库表生成嵌入向量用于相似性匹配
- **自动分类**：将文件分类为表格、文档或无关内容
- **相似性匹配**：使用向量嵌入查找相似表格
- **文件 ID 支持**：支持带自定义文件 ID 的文件处理

## 核心 API 接口

### 文件处理

#### `POST /api/process-files`
使用文件路径列表处理文件（向后兼容）

**请求体：**
```json
{
  "file_paths": [
    "/data/七田村党员名册.xlsx",
    "/data/村民信息表.csv"
  ],
  "village_name": "七田村"
}
```

#### `POST /api/process-files-with-ids`
使用文件路径和自定义文件 ID 处理文件（推荐）

**请求体：**
```json
{
  "files_data": {
    "/data/七田村党员名册.xlsx": "file_001",
    "/data/村民信息表.csv": "file_002"
  },
  "village_name": "七田村"
}
```

**响应示例：**
```json
{
  "session_id": "20250811_123456_abc123",
  "status": {
    "status": "completed",
    "message": "成功处理 2 个文件",
    "progress": 100
  },
  "table_files": [{
    "chinese_table_name": "七田村党员名册",
    "headers": ["姓名", "性别", "年龄"],
    "header_count": 3,
    "similarity_scores": []
  }],
  "summary": {
    "total_files": 2,
    "tables_found": 1,
    "irrelevant_files": 1
  }
}
```

#### `GET /api/process-files/status/{session_id}`
获取处理会话状态

#### `GET /api/validate-paths`
验证文件路径（不实际处理文件）

### 数据库向量化

#### `POST /api/revectorize-database`
触发数据库重新向量化

**请求体：**
```json
{
  "force_refresh": false,
  "specific_tables": ["用户表", "产品表"]
}
```

### 健康检查

#### `GET /api/health`
API 健康状态检查

## 环境配置

### 必需的环境变量

在项目根目录创建 `.env` 文件，配置以下变量：

```bash
# 数据库配置（必需）
CHAT_BI_ADDR=your_mysql_host          # MySQL 主机地址
CHAT_BI_PORT=3306                     # MySQL 端口（可选，默认 3306）
CHAT_BI_USER=your_db_user            # MySQL 用户名
CHAT_BI_PASSWORD=your_db_password    # MySQL 密码
CHAT_BI_DB=your_database_name        # 数据库名称

# AI 模型服务配置（必需）
SILICONFLOW_API_KEY=your_api_key     # SiliconFlow API 密钥

# Docker 卷挂载路径（Docker 部署时需要）
HOST_FILES_PATH=D:\asianInfo\data     # 主数据目录路径
```

### 环境变量说明

| 变量名 | 描述 | 必需 | 默认值 |
|--------|------|------|--------|
| `CHAT_BI_ADDR` | MySQL 数据库主机地址 | 是 | - |
| `CHAT_BI_PORT` | MySQL 数据库端口 | 否 | 3306 |
| `CHAT_BI_USER` | MySQL 数据库用户名 | 是 | - |
| `CHAT_BI_PASSWORD` | MySQL 数据库密码 | 是 | - |
| `CHAT_BI_DB` | MySQL 数据库名称 | 是 | - |
| `SILICONFLOW_API_KEY` | AI 模型服务 API 密钥 | 是 | - |
| `HOST_FILES_PATH` | 主数据目录（Docker 用） | Docker 部署时需要 | - |

## 快速开始

### Docker 部署（推荐）

1. **配置环境变量：**
   ```bash
   cp .env.example .env
   # 编辑 .env 文件设置你的配置
   ```

2. **启动服务：**
   ```bash
   docker-compose up -d
   ```

3. **验证部署：**
   ```bash
   curl http://localhost:8000/api/health
   ```

4. **查看 API 文档：**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### 本地开发

1. **安装依赖：**
   ```bash
   uv sync
   ```

2. **配置环境变量：**
   ```bash
   cp .env.example .env
   # 编辑 .env 文件
   ```

3. **启动 API 服务器：**
   ```bash
   python run_api.py
   ```

## 支持的文件格式

- **电子表格**：.xlsx, .xls, .xlsm, .ods, .csv
- **文档**：.docx, .doc, .pptx, .ppt
- **文本文件**：.txt, .md, .json, .xml, .html
- **其他格式**：基础文件信息提取

## 使用示例

### Python 示例

```python
import requests

# 使用文件 ID 处理文件（推荐）
response = requests.post('http://localhost:8000/api/process-files-with-ids', 
    json={
        "files_data": {
            "/data/党员名册.xlsx": "file_001",
            "/data/村民信息.csv": "file_002"
        },
        "village_name": "测试村"
    }
)

result = response.json()
print(f"处理状态: {result['status']['status']}")
print(f"找到表格: {len(result['table_files'])} 个")
```

### cURL 示例

```bash
# 处理带文件 ID 的文件
curl -X POST "http://localhost:8000/api/process-files-with-ids" \
     -H "Content-Type: application/json" \
     -d '{
       "files_data": {
         "/data/test.xlsx": "file_001",
         "/uploads/data.csv": "file_002"
       },
       "village_name": "测试村"
     }'

# 健康检查
curl http://localhost:8000/api/health
```

## 数据存储

处理后的文件数据存储在：
- **原始文件**：`uploaded_files/` 目录
- **处理结果**：`src/uploaded_files.json`（包含文件 ID、表头、相似度匹配等信息）

## 故障排除

1. **数据库连接失败**：检查环境变量和网络连通性
2. **文件处理失败**：验证文件格式和大小限制
3. **API 超时**：增加大文件处理的超时值
4. **权限问题**：确保 Docker 卷挂载路径有正确权限

更多详细信息请查看运行时的自动生成 API 文档：`/docs`