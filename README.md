# DeepCode-Insight

🚀 **Advanced Code Analysis & RAG-powered Code Intelligence Platform**

DeepCode-Insight là một platform phân tích code tiên tiến sử dụng AST parsing, static analysis và RAG (Retrieval-Augmented Generation) để cung cấp insights sâu sắc về codebase.

## ✨ Key Features

### 🔍 **Static Code Analysis**
- **Multi-language support**: Python, Java, JavaScript, TypeScript
- **AST-based parsing**: Deep code structure analysis
- **Quality metrics**: Complexity, maintainability, security issues
- **Custom rules engine**: Extensible rule system

### 🧠 **RAG-powered Code Intelligence**
- **Semantic code search**: Find code by meaning, not just keywords
- **Vector embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **LLM integration**: GPT-3.5-turbo for intelligent responses
- **Context-aware queries**: Get explanations with relevant code context

### 🗄️ **Vector Database Integration**
- **Qdrant vector store**: High-performance similarity search
- **Code chunking**: Intelligent code segmentation with AST analysis
- **Metadata enrichment**: Functions, classes, complexity indicators
- **Repository indexing**: Full codebase indexing support

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd DeepCode-Insight

# Install dependencies
pip install -r requirements.txt

# Setup environment
python scripts/create_env.py
```

### 2. Configuration

Edit `.env` file với your API keys:

```bash
# Required for full functionality
OPENAI_API_KEY=your_openai_api_key_here

# Optional
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Start Infrastructure

```bash
# Start Qdrant vector database
docker compose up -d
```

### 4. Run Demo

```bash
# Quick start demo
python scripts/quick_start.py

# Real data tests
python tests/test_rag_real_data.py
```

## 📁 Project Structure

```
DeepCode-Insight/
├── deepcode_insight/           # Core package
│   ├── agents/                 # AI agents
│   │   ├── rag_context.py     # RAG context agent
│   │   ├── ast_parser.py      # AST parsing agent
│   │   └── static_analyzer.py # Static analysis agent
│   ├── parsers/               # Code parsers
│   └── utils/                 # Utilities
├── docs/                      # Documentation
│   ├── README_SETUP.md        # Setup guide
│   ├── README_RAG.md          # RAG usage guide
│   └── setup_api_keys.md      # API keys setup
├── tests/                     # Test suite
│   ├── test_rag_real_data.py  # Real data tests
│   ├── test_rag_simple.py     # Component tests
│   └── demo_rag_context.py    # Demo scripts
├── scripts/                   # Utility scripts
│   ├── quick_start.py         # Quick start demo
│   ├── setup_env.py           # Environment setup
│   └── create_env.py          # Create .env file
├── config.py                  # Configuration management
├── requirements.txt           # Dependencies
└── docker-compose.yml         # Qdrant setup
```

## 🔧 Core Components

### RAGContextAgent
```python
from deepcode_insight.agents.rag_context import RAGContextAgent

# Initialize
rag_agent = RAGContextAgent()

# Index code
rag_agent.index_code_file(code, "file.py", "python")

# Semantic search
results = rag_agent.query("How to handle errors?", top_k=5)

# LLM-powered responses
context = rag_agent.query_with_context(
    "Explain async patterns", 
    generate_response=True
)
```

### Static Analyzer
```python
from deepcode_insight.agents.static_analyzer import StaticAnalyzer

# Analyze code
analyzer = StaticAnalyzer()
results = analyzer.analyze_file("path/to/file.py")

# Get metrics
metrics = analyzer.get_quality_metrics(results)
```

### AST Parser
```python
from deepcode_insight.agents.ast_parser import ASTParsingAgent

# Parse code structure
parser = ASTParsingAgent()
ast_data = parser.parse_code(code, "file.py")

# Extract functions, classes, imports
functions = ast_data['functions']
classes = ast_data['classes']
```

## 🧪 Testing

### Component Tests (No API required)
```bash
python tests/test_rag_simple.py
```

### Real Data Tests (Requires OpenAI API)
```bash
python tests/test_rag_real_data.py
```

### Demo with Mock Data
```bash
python tests/demo_rag_context.py
```

## 📊 Performance Metrics

### Real Data Test Results
- **✅ Indexing**: 29 chunks với real OpenAI embeddings
- **✅ Search**: 100% query success rate
- **✅ Quality**: High relevance semantic search
- **✅ Performance**: Sub-second query responses
- **✅ LLM Integration**: Context-aware response generation

### Supported Languages
- **Python**: Full AST support, advanced analysis
- **Java**: Class/method detection, Spring annotations
- **JavaScript/TypeScript**: Modern syntax support
- **Go, Rust**: Basic support

## 🔒 Security & Configuration

### Environment Variables
- **API Keys**: Stored securely in `.env` file
- **Configuration**: Centralized config management
- **Validation**: Automatic config validation
- **Security**: Sensitive data masking

### Best Practices
- `.env` file excluded from Git
- API key validation
- Error handling và fallbacks
- Production-ready defaults

## 📚 Documentation

- **[Setup Guide](docs/README_SETUP.md)**: Complete installation và configuration
- **[RAG Guide](docs/README_RAG.md)**: RAG usage và examples
- **[API Keys Setup](docs/setup_api_keys.md)**: Detailed API configuration
- **[Implementation Summary](docs/RAG_IMPLEMENTATION_SUMMARY.md)**: Technical details

## 🛠️ Development

### Requirements
- Python 3.8+
- Docker (for Qdrant)
- OpenAI API key
- 4GB+ RAM recommended

### Dependencies
- **LlamaIndex**: RAG framework
- **Qdrant**: Vector database
- **OpenAI**: Embeddings và LLM
- **Tree-sitter**: Code parsing
- **FastAPI**: Web framework (future)

## 🎯 Use Cases

### Code Analysis
- **Quality Assessment**: Identify code smells, complexity issues
- **Security Scanning**: Detect potential vulnerabilities
- **Refactoring Support**: Find improvement opportunities

### Code Intelligence
- **Semantic Search**: "Find error handling patterns"
- **Code Explanation**: "Explain this async function"
- **Best Practices**: "Show me caching examples"

### Repository Management
- **Codebase Overview**: Understand large repositories
- **Knowledge Extraction**: Document code patterns
- **Onboarding**: Help new developers understand code

## 🚀 Future Roadmap

### Phase 1 (Current)
- ✅ RAG-powered code search
- ✅ Multi-language AST parsing
- ✅ Vector database integration
- ✅ OpenAI LLM integration

### Phase 2 (Next)
- 🔄 Web UI interface
- 🔄 GitHub integration
- 🔄 Advanced analytics dashboard
- 🔄 Team collaboration features

### Phase 3 (Future)
- 📋 Code generation capabilities
- 📋 Automated refactoring suggestions
- 📋 CI/CD integration
- 📋 Enterprise features

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **LlamaIndex**: RAG framework
- **Qdrant**: Vector database
- **OpenAI**: AI capabilities
- **Tree-sitter**: Code parsing

---

**Built with ❤️ for developers who want to understand their code better** 