# Enhanced LLMOrchestratorAgent

## Tổng quan

Enhanced LLMOrchestratorAgent là phiên bản nâng cấp của LLMOrchestratorAgent với các tính năng mới:

1. **RAG Context Integration**: Tích hợp với RAGContextAgent để lấy context từ codebase
2. **Chain-of-Thought Prompting**: Sử dụng Chain-of-Thought reasoning cho analysis sâu hơn
3. **Multi-LLM Provider Support**: Hỗ trợ Ollama, OpenAI, và Gemini APIs thông qua abstract interface

## Tính năng chính

### 1. RAG Context Integration

```python
from deepcode_insight.agents.llm_orchestrator import LLMOrchestratorAgent
from deepcode_insight.agents.rag_context import RAGContextAgent

# Tạo RAG context agent
rag_agent = RAGContextAgent()

# Tạo LLM orchestrator với RAG
orchestrator = LLMOrchestratorAgent(
    provider="ollama",
    model="codellama",
    rag_context_agent=rag_agent,
    enable_rag=True
)
```

### 2. Chain-of-Thought Prompting

Chain-of-Thought prompting được enable mặc định và thực hiện analysis theo 4 bước:

1. **Code Structure Analysis**: Phân tích kiến trúc và organization
2. **Issue Impact Assessment**: Đánh giá mức độ nghiêm trọng của issues
3. **Risk Evaluation**: Đánh giá rủi ro và dependencies
4. **Solution Strategy**: Đề xuất approach để giải quyết

### 3. Multi-LLM Provider Support

#### Ollama (Local)
```python
orchestrator = LLMOrchestratorAgent(
    provider="ollama",
    model="codellama",
    base_url="http://localhost:11434"
)
```

#### OpenAI
```python
orchestrator = LLMOrchestratorAgent(
    provider="openai",
    model="gpt-4",
    api_key="your-openai-api-key"
)
```

#### Google Gemini
```python
orchestrator = LLMOrchestratorAgent(
    provider="gemini",
    model="gemini-pro",
    api_key="your-gemini-api-key"
)
```

## Cách sử dụng

### Basic Usage

```python
from deepcode_insight.agents.llm_orchestrator import create_llm_orchestrator_agent

# Tạo agent với default settings (Ollama + RAG + Chain-of-Thought)
agent = create_llm_orchestrator_agent()

# Sample state từ LangGraph
state = {
    'static_analysis_results': {
        'static_issues': {
            'missing_docstrings': [
                {'message': 'Function missing docstring', 'line': 10}
            ]
        },
        'metrics': {
            'code_quality_score': 75.5,
            'maintainability_index': 68.2
        }
    },
    'code_content': 'def calculate(): pass',
    'filename': 'example.py'
}

# Process findings
result = agent.process_findings(state)
```

### Advanced Configuration

```python
# Custom configuration với OpenAI và disabled RAG
agent = LLMOrchestratorAgent(
    provider="openai",
    model="gpt-4",
    enable_rag=False,
    enable_chain_of_thought=True,
    api_key="your-api-key",
    temperature=0.3,
    max_tokens=1000
)
```

### Với Custom RAG Agent

```python
from deepcode_insight.agents.rag_context import RAGContextAgent

# Setup custom RAG agent
rag_agent = RAGContextAgent(
    qdrant_host="localhost",
    qdrant_port=6333,
    collection_name="custom_collection"
)

# Index some code
rag_agent.index_code_file(code_content, "example.py")

# Create orchestrator với custom RAG
agent = LLMOrchestratorAgent(
    provider="gemini",
    model="gemini-pro",
    rag_context_agent=rag_agent,
    enable_rag=True
)
```

## Output Format

Enhanced LLMOrchestratorAgent trả về analysis results với format mở rộng:

```python
{
    'filename': 'example.py',
    'summary': 'Overall summary của code quality...',
    'detailed_analysis': 'Chain-of-Thought analysis...',
    'priority_issues': [
        {
            'type': 'missing_docstring',
            'description': 'Function lacks documentation',
            'reason': 'Impacts maintainability',
            'impact_level': 'Medium'
        }
    ],
    'recommendations': [
        {
            'action': 'Add docstrings',
            'impact': 'Improves documentation',
            'effort': 'Low',
            'priority': 'High'
        }
    ],
    'solution_suggestions': [  # New: Chain-of-Thought solutions
        {
            'solution': 'Implement docstring templates',
            'implementation': 'Use IDE snippets',
            'benefit': 'Consistent documentation'
        }
    ],
    'code_quality_assessment': 'Detailed quality assessment...',
    'improvement_suggestions': [
        {
            'action': 'Refactor complex functions',
            'implementation': 'Split into smaller methods',
            'benefit': 'Better maintainability',
            'timeline': '1-2 days'
        }
    ],
    'rag_context_used': True,  # New: RAG context indicator
    'llm_metadata': {
        'provider': 'OpenAIProvider',
        'model_used': 'gpt-4',
        'analysis_type': 'enhanced_code_review_with_rag_and_cot',
        'rag_enabled': True,
        'chain_of_thought_enabled': True
    }
}
```

## Environment Variables

Để sử dụng các LLM providers khác nhau, set các environment variables sau:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Gemini
export GEMINI_API_KEY="your-gemini-api-key"

# Ollama (optional, defaults)
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_API_KEY="optional-api-key"

# RAG/Qdrant settings
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
export QDRANT_COLLECTION="deepcode_insight"
```

## Performance Considerations

### RAG Context
- RAG context retrieval thêm ~200-500ms latency
- Có thể disable bằng `enable_rag=False` nếu không cần
- Vector database cần được setup trước (Qdrant)

### Chain-of-Thought
- Chain-of-Thought prompting tăng token usage ~30-50%
- Có thể disable bằng `enable_chain_of_thought=False`
- Cải thiện chất lượng analysis đáng kể

### LLM Provider Performance
- **Ollama**: Fastest (local), free, limited model options
- **OpenAI**: Good balance, paid, excellent quality
- **Gemini**: Good quality, competitive pricing, Google ecosystem

## Error Handling

Enhanced agent có robust error handling:

```python
try:
    result = agent.process_findings(state)
    if 'error' in result.get('llm_analysis', {}):
        print(f"Analysis error: {result['llm_analysis']['error']}")
except Exception as e:
    print(f"Agent error: {e}")
```

## Testing

Chạy tests cho enhanced agent:

```bash
# Run specific test file
pytest deepcode_insight/tests/test_enhanced_llm_orchestrator.py -v

# Run với coverage
pytest deepcode_insight/tests/test_enhanced_llm_orchestrator.py --cov=deepcode_insight.agents.llm_orchestrator

# Run integration tests
pytest deepcode_insight/tests/ -k "enhanced" -v
```

## Migration từ Old LLMOrchestratorAgent

### Before (Old)
```python
from deepcode_insight.agents.llm_orchestrator import LLMOrchestratorAgent

agent = LLMOrchestratorAgent(
    model="codellama",
    base_url="http://localhost:11434"
)
```

### After (Enhanced)
```python
from deepcode_insight.agents.llm_orchestrator import LLMOrchestratorAgent

agent = LLMOrchestratorAgent(
    provider="ollama",  # Specify provider
    model="codellama",
    base_url="http://localhost:11434"  # Provider-specific kwargs
)
```

## Best Practices

1. **Use RAG for large codebases**: Enable RAG khi analyze projects với nhiều files
2. **Chain-of-Thought for complex analysis**: Keep enabled cho detailed analysis
3. **Choose appropriate LLM**: 
   - Ollama cho development/testing
   - OpenAI cho production quality
   - Gemini cho cost-effective solution
4. **Monitor token usage**: Track costs với paid providers
5. **Cache RAG context**: Reuse RAG agent instances cho multiple analyses

## Troubleshooting

### Common Issues

1. **RAG initialization fails**
   ```
   Solution: Check Qdrant server is running và accessible
   ```

2. **OpenAI API errors**
   ```
   Solution: Verify API key và check rate limits
   ```

3. **Ollama connection issues**
   ```
   Solution: Ensure Ollama server is running: `ollama serve`
   ```

4. **Memory issues với large codebases**
   ```
   Solution: Reduce chunk size hoặc disable RAG
   ```

## Future Enhancements

- [ ] Support for more LLM providers (Anthropic Claude, etc.)
- [ ] Advanced RAG strategies (hierarchical, multi-modal)
- [ ] Caching layer cho LLM responses
- [ ] Streaming responses cho real-time feedback
- [ ] Custom prompt templates
- [ ] Multi-language Chain-of-Thought prompts 