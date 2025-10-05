# RAG Repository Analyzer - REST API Docker Container

A production-ready REST API for analyzing GitHub repositories using Retrieval Augmented Generation (RAG). This system converts the original Jupyter notebook workflow into a containerized web service that can analyze repositories and answer questions about them using AI.

## 🚀 Features

- **REST API Interface**: Clean endpoints for repository analysis and querying
- **Docker Container**: Production-ready containerized deployment
- **Dual Embedding System**: Uses OpenAI text-embedding-ada-002 + Microsoft CodeBERT
- **In-Memory Vector Storage**: ChromaDB for fast similarity search
- **Async Processing**: Non-blocking API operations with proper error handling
- **Security**: Non-root container user and proper resource management
- **Health Monitoring**: Built-in health checks and status endpoints

## 📋 API Endpoints

### `POST /analyze`
Analyze a GitHub repository and prepare it for queries.

**Request:**
```json
{
  "repository": "username/repo_name"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Repository username/repo_name analyzed successfully",
  "repository": "username/repo_name"
}
```

### `POST /query`
Query the analyzed repository with a question.

**Request:**
```json
{
  "question": "What is this repository about?"
}
```

**Response:**
```json
{
  "answer": "This repository is a Visual Studio Code editor...",
  "sources": [
    {
      "file_name": "README.md",
      "content_type": "text",
      "score": 0.85,
      "content": "Visual Studio Code is a lightweight but powerful..."
    }
  ]
}
```

### `GET /health`
Health check endpoint for container monitoring.

**Response:**
```json
{
  "status": "healthy"
}
```

### `POST /analyze-and-query`
Analyze a GitHub repository and immediately query it with a question in a single call.

**Request:**
```json
{
  "repository": "username/repo_name",
  "question": "What is this repository about?"
}
```

**Response:**
```json
{
  "status": "success",
  "repository": "username/repo_name",
  "answer": "This repository is a Visual Studio Code editor...",
  "sources": [
    {
      "file_name": "README.md",
      "content_type": "text",
      "score": 0.85,
      "content": "Visual Studio Code is a lightweight but powerful..."
    }
  ],
  "message": "Repository 'username/repo_name' analyzed and question answered successfully"
}
```

### `GET /status`
Check if a repository is loaded and ready for queries.

**Response:**
```json
{
  "repository": "microsoft/vscode",
  "ready": true,
  "message": "Repository 'microsoft/vscode' is ready for queries"
}
```

## 🐳 Docker Usage

### Quick Start

1. **Build the container:**
```bash
docker build -t rag-analyzer .
```

2. **Run with environment variables:**
```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="your-openai-key" \
  -e GITHUB_API_KEY="your-github-key" \
  rag-analyzer
```

3. **Test the API:**
```bash
# Health check
curl http://localhost:8000/health

# Analyze a repository
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"repository": "microsoft/vscode"}'

# Query the repository
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this repository about?"}'

# OR use the new convenience endpoint (analyze + query in one call)
curl -X POST http://localhost:8000/analyze-and-query \
  -H "Content-Type: application/json" \
  -d '{"repository": "microsoft/vscode", "question": "What is this repository about?"}'
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key for embeddings and completions |
| `GITHUB_API_KEY` | Yes | Your GitHub API key for repository access |
| `PORT` | No | API server port (default: 8000) |
| `HOST` | No | API server host (default: 0.0.0.0) |

### Using Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  rag-analyzer:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GITHUB_API_KEY=${GITHUB_API_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

Then run:
```bash
docker-compose up -d
```

## 🛠️ Development Setup

### Local Development

1. **Clone and setup:**
```bash
git clone <repository-url>
cd RAG-Repository-Analyzer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Create environment file:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run the API server:**
```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

5. **Test the API:**
```bash
python test_api.py --quick  # Quick tests only
python test_api.py          # Full test suite
```

### API Testing

The included test scripts provide comprehensive testing:

```bash
# Quick tests (health, status, error handling)
python test_api.py --quick

# Full test suite (includes repository analysis)
python test_api.py --repository microsoft/vscode

# Test the new analyze-and-query endpoint
python test_analyze_and_query.py

# Custom API endpoint
python test_api.py --url http://localhost:8080
```

## 🏗️ System Architecture

### Container Architecture
- **Base Image**: Python 3.12-slim for minimal footprint
- **Multi-stage Build**: Separates build dependencies from runtime
- **Security**: Non-root user (raguser) for container security
- **Pre-cached Models**: CodeBERT model downloaded during build
- **Health Checks**: Built-in container health monitoring

### API Architecture
- **FastAPI Framework**: Modern async Python web framework
- **Pydantic Models**: Request/response validation and documentation
- **Global State Management**: Single repository analysis per container session
- **Thread Pool Execution**: CPU-bound tasks run in background threads
- **Structured Logging**: Comprehensive logging for debugging

### RAG Pipeline
1. **Repository Analysis**: GitHub API → File extraction → Content chunking
2. **Embedding Generation**: OpenAI (text) + CodeBERT (code) → Vector embeddings  
3. **Vector Storage**: ChromaDB in-memory collections for fast retrieval
4. **Query Processing**: Question → Embedding → Similarity search → RAG prompt → AI response

## 📊 Performance Considerations

- **Memory Usage**: ~2-4GB RAM depending on repository size
- **Processing Time**: 2-10 minutes for repository analysis (varies by size)
- **Concurrent Requests**: Single repository per container instance
- **Model Loading**: CodeBERT loaded once during container startup
- **API Rate Limits**: Respects OpenAI and GitHub API limitations

## 🔧 Configuration

### Docker Build Arguments
```bash
# Custom Python version
docker build --build-arg PYTHON_VERSION=3.11 -t rag-analyzer .

# Skip model pre-download (faster builds, slower startup)
docker build --build-arg SKIP_MODEL_DOWNLOAD=1 -t rag-analyzer .
```

### Runtime Configuration
- **Single Worker**: Optimized for memory efficiency
- **Async Operations**: Non-blocking I/O for API calls
- **Resource Cleanup**: Proper shutdown handling
- **Error Recovery**: Graceful handling of API failures

## 📖 API Documentation

Once the container is running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🐛 Troubleshooting

### Common Issues

**Container won't start:**
- Check API keys are set correctly
- Ensure port 8000 is available
- Check Docker logs: `docker logs <container-id>`

**Repository analysis fails:**
- Verify GitHub API key has access to the repository
- Check repository exists and is public
- Monitor rate limits in API responses

**Out of memory errors:**
- Increase Docker memory limit
- Try smaller repositories first
- Monitor memory usage: `docker stats`

**OpenAI API errors:**
- Verify API key is valid and has credits
- Check OpenAI service status
- Review rate limiting in logs

### Debug Mode

Run with debug logging:
```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="your-key" \
  -e GITHUB_API_KEY="your-key" \
  -e LOG_LEVEL="DEBUG" \
  rag-analyzer
```

## 📝 Migration from Notebook

This REST API replaces the original Jupyter notebook workflow:

| Notebook Component | API Equivalent |
|-------------------|----------------|
| `helper.ipynb` cells 1-22 | `POST /analyze` endpoint |
| `helper.ipynb` cells 92-102 | `POST /query` endpoint |
| Manual execution | Automated REST API |
| Local state | Containerized service |

The core algorithms and models remain identical, ensuring consistent results between the notebook and API versions.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure Docker builds successfully
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- **OpenAI** for embedding and completion APIs
- **Microsoft** for CodeBERT model
- **ChromaDB** for vector database
- **FastAPI** for web framework
