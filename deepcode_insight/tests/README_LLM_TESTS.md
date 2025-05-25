# LLM Caller Test Suite

Comprehensive test suite cho `utils.llm_caller` module sử dụng pytest và pytest-mock.

## 📁 Cấu trúc Files

```
tests/
├── conftest.py                 # Pytest configuration và shared fixtures
├── test_llm_caller.py         # Main test file cho llm_caller module
├── test_prompt_formatting.py  # Tests cho prompt formatting và templates
└── README_LLM_TESTS.md        # Documentation này
```

## 🧪 Test Coverage

### `test_llm_caller.py` - Main Tests

#### **TestOllamaLLMCaller**
- ✅ Initialization với default values
- ✅ Initialization với custom values  
- ✅ Environment variables handling
- ✅ String model support

#### **TestMakeRequest**
- ✅ Successful HTTP requests
- ✅ Error handling (404, 401, 500)
- ✅ Connection errors với retry logic
- ✅ Timeout handling
- ✅ Generic request exceptions
- ✅ Retry then success scenarios

#### **TestGenerate**
- ✅ Basic prompt generation
- ✅ Code snippet formatting
- ✅ System prompt handling
- ✅ Custom parameters (temperature, top_p, max_tokens, stream)
- ✅ JSON decode error handling

#### **TestChat**
- ✅ Basic chat messages
- ✅ Custom parameters
- ✅ Empty message content handling

#### **TestAnalyzeCode**
- ✅ General analysis
- ✅ Bug detection
- ✅ Optimization suggestions
- ✅ Documentation generation
- ✅ Invalid analysis type fallback

#### **TestHealthAndModels**
- ✅ Health check success/failure
- ✅ Models listing
- ✅ HTTP errors và request exceptions

#### **TestConvenienceFunctions**
- ✅ `create_llm_caller()` function
- ✅ `quick_analyze_code()` function

#### **TestOllamaModel**
- ✅ Enum values verification

#### **TestOllamaResponse**
- ✅ Response object creation
- ✅ Optional fields handling

#### **TestOllamaAPIError**
- ✅ Error creation với details
- ✅ Status codes và response text

#### **TestIntegration**
- ✅ Full workflow mocking

### `test_prompt_formatting.py` - Prompt Tests

#### **TestPromptFormatting**
- ✅ Basic prompt formatting
- ✅ Code snippet integration
- ✅ Multiline code handling
- ✅ Special characters preservation
- ✅ System prompt separation

#### **TestAnalysisPromptTemplates**
- ✅ General analysis prompts
- ✅ Bug analysis prompts
- ✅ Optimization prompts
- ✅ Documentation prompts
- ✅ Language-specific system prompts
- ✅ Temperature settings

#### **TestChatPromptFormatting**
- ✅ Chat messages formatting
- ✅ Code trong chat messages
- ✅ Empty messages handling

#### **TestPromptParameterHandling**
- ✅ Temperature parameter
- ✅ Top-p parameter
- ✅ Max tokens parameter
- ✅ Stream parameter
- ✅ Combined parameters

## 🔧 Dependencies

```bash
pip install pytest pytest-mock pytest-cov pytest-html requests
```

## 🚀 Chạy Tests

### Basic Usage

```bash
# Chạy tất cả tests
python run_llm_tests.py

# Chạy specific test file
pytest tests/test_llm_caller.py -v

# Chạy với coverage
pytest tests/ --cov=utils.llm_caller --cov-report=html
```

### Sử dụng Test Runner Script

```bash
# Full test suite
python run_llm_tests.py all

# Basic tests only
python run_llm_tests.py basic

# Prompt formatting tests
python run_llm_tests.py prompt

# Coverage analysis
python run_llm_tests.py coverage

# Generate detailed report
python run_llm_tests.py report
```

### Test Categories

```bash
# Chạy specific test class
pytest tests/test_llm_caller.py::TestGenerate -v

# Chạy tests theo marker
pytest -m integration -v

# Chạy parametrized tests
pytest -k "parametrize" -v

# Chạy mock verification
pytest -k "mock" -v
```

## 🎯 Mock Strategy

### HTTP Requests Mocking

```python
# Mock successful response
mock_response = Mock()
mock_response.status_code = 200
mock_response.json.return_value = {"response": "test"}
mocker.patch("requests.post", return_value=mock_response)
```

### Method Mocking

```python
# Mock internal methods
mock_make_request = mocker.patch.object(llm, "_make_request", return_value=mock_response)

# Verify calls
mock_make_request.assert_called_once_with("api/generate", payload)
```

### Environment Variables

```python
# Mock env vars
monkeypatch.setenv("OLLAMA_BASE_URL", "http://test:8080")
monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
```

## 📊 Test Fixtures

### Shared Fixtures (conftest.py)

- **`mock_requests_response`** - Factory cho mock HTTP responses
- **`sample_ollama_response_data`** - Sample Ollama API response
- **`sample_chat_response_data`** - Sample chat response
- **`sample_models_response_data`** - Sample models list
- **`sample_code_snippets`** - Code snippets cho testing
- **`mock_environment_variables`** - Environment variable mocking
- **`clean_environment_variables`** - Clean env vars
- **`reset_logging`** - Reset logging configuration

### Custom Assertions

```python
# Validate OllamaResponse
pytest.assert_ollama_response_valid(response)

# Validate OllamaAPIError
pytest.assert_api_error_valid(error)
```

## 🔍 Test Scenarios

### Error Handling Tests

```python
def test_connection_error_with_retry(self, mocker):
    """Test connection error với retry logic"""
    mocker.patch("requests.post", side_effect=ConnectionError("Connection failed"))
    
    with pytest.raises(OllamaAPIError) as exc_info:
        self.llm._make_request("api/generate", {})
    
    assert "Không thể kết nối" in exc_info.value.message
```

### Prompt Formatting Tests

```python
def test_prompt_with_code_snippet_formatting(self, mocker):
    """Test formatting prompt với code snippet"""
    # ... setup mocks ...
    
    self.llm.generate(prompt, code_snippet=code_snippet)
    
    # Verify prompt structure
    assert "Code snippet:" in formatted_prompt
    assert code_snippet in formatted_prompt
```

### Parametrized Tests

```python
@pytest.mark.parametrize("analysis_type,expected_keyword", [
    ("general", "phân tích code snippet"),
    ("bugs", "lỗi tiềm ẩn"),
    ("optimization", "tối ưu hóa performance"),
])
def test_analyze_code_prompts(analysis_type, expected_keyword, mocker):
    # Test implementation
```

## 📈 Coverage Goals

- **Line Coverage**: > 95%
- **Branch Coverage**: > 90%
- **Function Coverage**: 100%

### Current Coverage Areas

✅ **Fully Covered:**
- Initialization và configuration
- HTTP request handling
- Error handling và retry logic
- Prompt formatting
- Response parsing
- Environment variables

✅ **Well Covered:**
- Parameter validation
- Model enum handling
- Convenience functions

⚠️ **Needs Attention:**
- Edge cases trong JSON parsing
- Network timeout scenarios
- Memory handling với large responses

## 🐛 Common Test Patterns

### Mock HTTP Response

```python
def test_example(self, mocker):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "test"}
    
    mocker.patch("requests.post", return_value=mock_response)
    
    result = self.llm.generate("test prompt")
    assert result.response == "test"
```

### Test Error Conditions

```python
def test_error_handling(self, mocker):
    mocker.patch("requests.post", side_effect=ConnectionError("Failed"))
    
    with pytest.raises(OllamaAPIError) as exc_info:
        self.llm.generate("test")
    
    assert "connection" in exc_info.value.message.lower()
```

### Verify Method Calls

```python
def test_method_calls(self, mocker):
    mock_method = mocker.patch.object(self.llm, "_make_request")
    
    self.llm.generate("test", temperature=0.5)
    
    # Verify call arguments
    call_args = mock_method.call_args
    payload = call_args[0][1]
    assert payload["options"]["temperature"] == 0.5
```

## 🔄 Continuous Integration

### GitHub Actions Example

```yaml
- name: Run LLM Caller Tests
  run: |
    pip install pytest pytest-mock pytest-cov
    python run_llm_tests.py coverage
    
- name: Upload Coverage
  uses: codecov/codecov-action@v1
  with:
    file: ./coverage.xml
```

### Pre-commit Hooks

```yaml
- repo: local
  hooks:
    - id: llm-tests
      name: LLM Caller Tests
      entry: python run_llm_tests.py basic
      language: system
      pass_filenames: false
```

## 📝 Writing New Tests

### Test Class Template

```python
class TestNewFeature:
    """Test class cho new feature"""
    
    def setup_method(self):
        self.llm = OllamaLLMCaller()
    
    def test_feature_basic(self, mocker):
        """Test basic functionality"""
        # Setup mocks
        mock_response = Mock()
        mock_response.json.return_value = {"response": "test"}
        mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        # Execute
        result = self.llm.new_feature("input")
        
        # Assert
        assert result.response == "test"
    
    def test_feature_error_handling(self, mocker):
        """Test error conditions"""
        mocker.patch.object(self.llm, "_make_request", side_effect=Exception("Error"))
        
        with pytest.raises(OllamaAPIError):
            self.llm.new_feature("input")
```

### Fixture Template

```python
@pytest.fixture
def sample_new_data():
    """Fixture cho new feature data"""
    return {
        "field1": "value1",
        "field2": "value2"
    }
```

## 🎯 Best Practices

1. **Mock External Dependencies**: Always mock HTTP requests và external services
2. **Test Error Conditions**: Test both success và failure scenarios
3. **Verify Call Arguments**: Check that methods are called với correct parameters
4. **Use Descriptive Names**: Test names should clearly describe what's being tested
5. **Keep Tests Independent**: Each test should be able to run independently
6. **Use Fixtures**: Share common test data via fixtures
7. **Test Edge Cases**: Include boundary conditions và edge cases
8. **Maintain Coverage**: Aim for high test coverage but focus on quality

## 🚨 Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure project root is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Mock Not Working:**
```python
# Use correct patch target
mocker.patch("utils.llm_caller.requests.post")  # ✅ Correct
mocker.patch("requests.post")  # ❌ May not work
```

**Fixture Not Found:**
```python
# Ensure conftest.py is in tests directory
# Check fixture scope and naming
```

### Debug Tips

```bash
# Run with verbose output
pytest -v -s tests/test_llm_caller.py

# Run specific test với debugging
pytest -v -s tests/test_llm_caller.py::TestGenerate::test_basic_prompt

# Show local variables on failure
pytest --tb=long tests/
```

## 📚 References

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-mock Documentation](https://pytest-mock.readthedocs.io/)
- [Python Mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md) 