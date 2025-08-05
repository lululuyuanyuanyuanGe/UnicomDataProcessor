# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based data processing system focused on file processing, analysis, and content management for Asian information data. The system uses LangGraph for workflow orchestration and integrates with various AI models for document analysis.

## Development Setup

### Python Environment
- Python 3.13+ required
- Uses `uv` for dependency management
- Virtual environment located in `.venv/`

### Dependencies Installation
```cmd
# Install dependencies using uv
uv sync

# Activate virtual environment on Windows
.venv\Scripts\activate
```

### Environment Configuration
- Copy `.env.example` to `.env` if it exists
- Configure API keys and model endpoints in `.env`

## Core Architecture

### Main Components

1. **FileProcessAgent** (`agents/fileProcessAgent.py`)
   - Core workflow orchestration using LangGraph
   - Handles file upload, analysis, classification, and processing
   - States: file upload → analysis → routing → processing → summary
   - Supports parallel processing of different file types

2. **Utilities** (`utils/`)
   - `file_process.py`: File handling, path detection, content extraction
   - `modelRelated.py`: AI model integration with rate limiting and retry logic
   - `screen_shot.py`: Excel table screenshot generation
   - `html_generator.py`: HTML content generation
   - `message_process.py`: Message processing utilities
   - `clean_response.py`: Response cleaning utilities

3. **Data Storage**
   - `agents/data.json`: Main data storage for processed files and metadata
   - Location-based organization with templates and supplements

### File Processing Workflow

1. **Upload**: Detects and validates file paths from user input
2. **Analysis**: Classifies files into template, supplement (tables/documents), or irrelevant
3. **Processing**: 
   - Templates: Complexity analysis and final destination storage
   - Supplements: Content analysis, summarization, and database storage
   - Irrelevant: Cleanup and deletion
4. **Storage**: Organized by location/village name with metadata tracking

## Common Development Commands

### Running the Application
```cmd
# Run file processing agent
python agents\fileProcessAgent.py

# Process specific files (Windows paths)
python -c "from agents.fileProcessAgent import FileProcessAgent; agent = FileProcessAgent(); agent.run_file_process_agent(session_id='test', upload_files_path=['C:\\path\\to\\file'], village_name='location')"
```

### Testing
```cmd
# No specific test framework configured - check for test files in project
python -m pytest
```

### Code Quality
```cmd
# Check for linting configuration
python -m flake8 .
python -m black .
```

## Key Features

- **Multi-format file support**: Excel, text, document processing
- **AI-powered analysis**: Document classification and content summarization
- **Parallel processing**: Concurrent file analysis for performance
- **Location-based organization**: Data organized by geographic location
- **Template complexity detection**: Distinguishes between simple and complex table templates
- **Robust error handling**: Rate limiting, retries, and fallback mechanisms

## File Organization

- Templates stored in organized directory structure
- Supplements categorized as tables (表格) or documents (文档)
- Original files preserved alongside processed versions
- Staging areas for temporary processing

## Model Integration

- Supports multiple AI models (DeepSeek-V3, Qwen2.5-VL, etc.)
- Rate limiting with exponential backoff
- Screenshot-based analysis for Excel files
- Configurable model endpoints through environment variables