"""
Pytest configuration và shared fixtures cho test suite.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock

# Thêm project root vào Python path để import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Fixture cung cấp đường dẫn đến project root"""
    return project_root


@pytest.fixture
def mock_requests_response():
    """Fixture tạo mock response cho requests"""
    def _create_response(status_code=200, json_data=None, text=""):
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.text = text
        if json_data:
            mock_response.json.return_value = json_data
        return mock_response
    return _create_response


@pytest.fixture
def sample_ollama_response_data():
    """Fixture cung cấp sample response data từ Ollama API"""
    return {
        "response": "This is a sample response from Ollama",
        "model": "codellama",
        "created_at": "2024-01-01T00:00:00.000Z",
        "done": True,
        "total_duration": 1500000000,  # 1.5 seconds in nanoseconds
        "load_duration": 100000000,   # 0.1 seconds
        "prompt_eval_count": 25,
        "prompt_eval_duration": 200000000,  # 0.2 seconds
        "eval_count": 50,
        "eval_duration": 1200000000  # 1.2 seconds
    }


@pytest.fixture
def sample_chat_response_data():
    """Fixture cung cấp sample chat response data"""
    return {
        "message": {
            "role": "assistant",
            "content": "This is a chat response from Ollama"
        },
        "model": "codellama",
        "created_at": "2024-01-01T00:00:00.000Z",
        "done": True,
        "total_duration": 1000000000,
        "eval_count": 30
    }


@pytest.fixture
def sample_models_response_data():
    """Fixture cung cấp sample models list response"""
    return {
        "models": [
            {
                "name": "codellama:latest",
                "modified_at": "2024-01-01T00:00:00.000Z",
                "size": 3800000000,
                "digest": "sha256:abc123"
            },
            {
                "name": "codellama:7b",
                "modified_at": "2024-01-01T00:00:00.000Z",
                "size": 3800000000,
                "digest": "sha256:def456"
            },
            {
                "name": "llama2:13b",
                "modified_at": "2024-01-01T00:00:00.000Z",
                "size": 7000000000,
                "digest": "sha256:ghi789"
            }
        ]
    }


@pytest.fixture
def sample_code_snippets():
    """Fixture cung cấp các code snippets để test"""
    return {
        "python_fibonacci": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
        """,
        
        "python_bubble_sort": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
        """,
        
        "javascript_async": """
async function fetchUserData(userId) {
    const response = await fetch(`/api/users/${userId}`);
    const userData = await response.json();
    return userData;
}
        """,
        
        "python_with_bug": """
def divide_numbers(a, b):
    return a / b  # Potential division by zero
        """,
        
        "python_inefficient": """
def find_duplicates(lst):
    duplicates = []
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            if lst[i] == lst[j] and lst[i] not in duplicates:
                duplicates.append(lst[i])
    return duplicates
        """
    }


@pytest.fixture
def mock_environment_variables(monkeypatch):
    """Fixture để mock environment variables"""
    def _set_env_vars(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setenv(key, value)
    return _set_env_vars


@pytest.fixture
def clean_environment_variables(monkeypatch):
    """Fixture để clean environment variables"""
    # Remove common env vars that might affect tests
    env_vars_to_clean = [
        "OLLAMA_BASE_URL",
        "OLLAMA_API_KEY"
    ]
    
    for var in env_vars_to_clean:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def reset_logging():
    """Fixture để reset logging configuration cho mỗi test"""
    import logging
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Reset to default level
    root_logger.setLevel(logging.WARNING)


# Pytest configuration
def pytest_configure(config):
    """Pytest configuration hook"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "network: mark test as requiring network access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if "slow" in item.name.lower() or "timeout" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark network tests
        if any(keyword in item.name.lower() for keyword in ["network", "connection", "http"]):
            item.add_marker(pytest.mark.network)


# Custom assertions
def assert_ollama_response_valid(response):
    """Custom assertion để kiểm tra OllamaResponse validity"""
    from ..utils.llm_caller import OllamaResponse
    
    assert isinstance(response, OllamaResponse)
    assert isinstance(response.response, str)
    assert isinstance(response.model, str)
    assert isinstance(response.created_at, str)
    assert isinstance(response.done, bool)
    
    # Optional fields có thể là None hoặc int
    if response.total_duration is not None:
        assert isinstance(response.total_duration, int)
    if response.eval_count is not None:
        assert isinstance(response.eval_count, int)


def assert_api_error_valid(error):
    """Custom assertion để kiểm tra OllamaAPIError validity"""
    from ..utils.llm_caller import OllamaAPIError
    
    assert isinstance(error, OllamaAPIError)
    assert isinstance(error.message, str)
    assert len(error.message) > 0
    
    if error.status_code is not None:
        assert isinstance(error.status_code, int)
        assert 100 <= error.status_code < 600  # Valid HTTP status codes
    
    if error.response_text is not None:
        assert isinstance(error.response_text, str)


# Make custom assertions available globally
pytest.assert_ollama_response_valid = assert_ollama_response_valid
pytest.assert_api_error_valid = assert_api_error_valid 