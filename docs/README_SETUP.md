# DeepCode-Insight Setup Guide

## 🚀 Quick Setup

### 1. Clone và Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd DeepCode-Insight

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

#### Create .env file
```bash
# Copy template
cp .env.template .env

# Edit .env file với your API keys
nano .env
```

#### Required Environment Variables
```bash
# OpenAI API Configuration (Required)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Google API Configuration (Optional)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-pro

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=deepcode_context
```

### 3. Start Qdrant Vector Database

```bash
# Start Qdrant with Docker
docker compose up -d

# Verify Qdrant is running
curl http://localhost:6333/
```

### 4. Test Configuration

```bash
# Test config setup
python config.py

# Run basic tests
python test_rag_simple.py

# Run real data tests (requires OpenAI API key)
python test_rag_real_data.py
```

## 🔧 Configuration Options

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (required) |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | LLM model for responses |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION` | `deepcode_context` | Collection name |

### Performance Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `1024` | Code chunk size |
| `CHUNK_OVERLAP` | `200` | Chunk overlap size |
| `VECTOR_DIMENSION` | `1536` | Embedding dimension |
| `MAX_CHUNKS_PER_FILE` | `50` | Max chunks per file |

### Repository Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_FILE_PATTERNS` | `*.py,*.java,*.js,*.ts,*.go,*.rs` | File patterns to index |
| `MAX_FILE_SIZE_MB` | `10` | Max file size to process |
| `EXCLUDE_PATTERNS` | `__pycache__,node_modules,.git,*.pyc,*.class` | Patterns to exclude |

## 🧪 Testing

### Test Hierarchy

1. **Component Tests** (No API required)
   ```bash
   python test_rag_simple.py
   ```

2. **Real Data Tests** (Requires OpenAI API)
   ```bash
   python test_rag_real_data.py
   ```

3. **Demo with Mock Data**
   ```bash
   python demo_rag_context.py
   ```

### Test Coverage

- ✅ Configuration loading và validation
- ✅ Qdrant connection và operations
- ✅ Code chunking với AST analysis
- ✅ Vector indexing và retrieval
- ✅ Real OpenAI embeddings
- ✅ LLM response generation
- ✅ Performance metrics

## 🔍 Usage Examples

### Basic Usage

```python
from deepcode_insight.agents.rag_context import RAGContextAgent

# Initialize with config
rag_agent = RAGContextAgent()

# Index code
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

rag_agent.index_code_file(code, "fibonacci.py", "python")

# Query
results = rag_agent.query("How to calculate fibonacci?", top_k=3)
print(f"Found {results['total_results']} relevant chunks")
```

### Advanced Usage

```python
# Query with LLM response
context = rag_agent.query_with_context(
    "Explain error handling patterns",
    top_k=5,
    generate_response=True
)

print(context["response"])

# Repository indexing
from deepcode_insight.agents.code_fetcher import CodeFetcherAgent

code_fetcher = CodeFetcherAgent()
results = rag_agent.index_repository(
    code_fetcher,
    "https://github.com/user/repo",
    file_patterns=["*.py", "*.java"]
)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. OpenAI API Errors
```bash
# Check API key
echo $OPENAI_API_KEY

# Test API connection
python -c "from openai import OpenAI; print(OpenAI().models.list())"
```

#### 2. Qdrant Connection Issues
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker compose restart

# Check logs
docker compose logs qdrant
```

#### 3. Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 4. Configuration Issues
```bash
# Validate config
python config.py

# Check environment variables
python -c "from config import config; config.print_config()"
```

### Performance Issues

#### Slow Indexing
- Reduce `CHUNK_SIZE` for faster processing
- Increase `MAX_CHUNKS_PER_FILE` limit
- Use smaller embedding model

#### High Memory Usage
- Reduce `VECTOR_DIMENSION`
- Process files in smaller batches
- Clear collection periodically

#### Poor Search Quality
- Increase `CHUNK_OVERLAP` for better context
- Use larger embedding model
- Adjust chunking strategy

## 📊 Monitoring

### Collection Statistics
```python
stats = rag_agent.get_collection_stats()
print(f"Total points: {stats['total_points']}")
print(f"Languages: {stats['indexed_languages']}")
```

### Performance Metrics
```python
# Query performance
import time
start = time.time()
results = rag_agent.query("test query")
duration = time.time() - start
print(f"Query took {duration:.3f}s")
```

## 🔄 Updates và Maintenance

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Clear Cache
```python
rag_agent.clear_collection()
```

### Backup Data
```bash
# Backup Qdrant data
docker compose exec qdrant tar -czf /qdrant/storage/backup.tar.gz /qdrant/storage
```

## 🎯 Next Steps

1. **Set up your API keys** in `.env` file
2. **Start Qdrant** với Docker
3. **Run tests** để verify setup
4. **Index your code** với RAGContextAgent
5. **Query và explore** your codebase

## 📞 Support

- Check logs trong `docker compose logs`
- Run diagnostic tests
- Review configuration với `python config.py`
- Check GitHub issues for known problems 