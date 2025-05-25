"""
Tests cho utils.llm_caller module.
Sử dụng pytest-mock để mock HTTP requests và test các chức năng chính.
"""

import pytest
import json
import os
from unittest.mock import Mock, patch
from requests.exceptions import ConnectionError, Timeout, RequestException

from ..utils.llm_caller import (
    OllamaLLMCaller,
    OllamaModel,
    OllamaResponse,
    OllamaAPIError,
    create_llm_caller,
    quick_analyze_code
)


class TestOllamaLLMCaller:
    """Test class cho OllamaLLMCaller"""
    
    def setup_method(self):
        """Setup cho mỗi test method"""
        self.base_url = "http://localhost:11434"
        self.model = "codellama"
        self.llm = OllamaLLMCaller(
            base_url=self.base_url,
            model=OllamaModel.CODELLAMA,
            timeout=30,
            max_retries=2
        )
    
    def test_init_default_values(self):
        """Test khởi tạo với giá trị mặc định"""
        llm = OllamaLLMCaller()
        assert llm.base_url == "http://localhost:11434"
        assert llm.model == "codellama"
        assert llm.timeout == 120
        assert llm.max_retries == 3
        assert "Content-Type" in llm.headers
        assert llm.headers["Content-Type"] == "application/json"
    
    def test_init_with_custom_values(self):
        """Test khởi tạo với giá trị tùy chỉnh"""
        custom_url = "http://custom-server:8080"
        llm = OllamaLLMCaller(
            base_url=custom_url,
            model=OllamaModel.CODELLAMA_7B,
            timeout=60,
            max_retries=5
        )
        assert llm.base_url == custom_url
        assert llm.model == "codellama:7b"
        assert llm.timeout == 60
        assert llm.max_retries == 5
    
    def test_init_with_env_variables(self, monkeypatch):
        """Test khởi tạo với environment variables"""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-server:9999")
        monkeypatch.setenv("OLLAMA_API_KEY", "test-api-key")
        
        llm = OllamaLLMCaller()
        assert llm.base_url == "http://env-server:9999"
        assert llm.api_key == "test-api-key"
        assert "Authorization" in llm.headers
        assert llm.headers["Authorization"] == "Bearer test-api-key"
    
    def test_init_with_string_model(self):
        """Test khởi tạo với string model"""
        llm = OllamaLLMCaller(model="custom-model")
        assert llm.model == "custom-model"


class TestMakeRequest:
    """Test class cho _make_request method"""
    
    def setup_method(self):
        self.llm = OllamaLLMCaller(max_retries=2)
    
    def test_successful_request(self, mocker):
        """Test request thành công"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "test response"}
        
        mock_post = mocker.patch("requests.post", return_value=mock_response)
        
        payload = {"model": "codellama", "prompt": "test"}
        result = self.llm._make_request("api/generate", payload)
        
        assert result == mock_response
        mock_post.assert_called_once_with(
            f"{self.llm.base_url}/api/generate",
            json=payload,
            headers=self.llm.headers,
            timeout=self.llm.timeout
        )
    
    def test_404_error_model_not_found(self, mocker):
        """Test lỗi 404 - model không tìm thấy"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Model not found"
        
        mocker.patch("requests.post", return_value=mock_response)
        
        with pytest.raises(OllamaAPIError) as exc_info:
            self.llm._make_request("api/generate", {})
        
        assert "không tìm thấy" in exc_info.value.message
        assert exc_info.value.status_code == 404
    
    def test_401_error_unauthorized(self, mocker):
        """Test lỗi 401 - unauthorized"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        
        mocker.patch("requests.post", return_value=mock_response)
        
        with pytest.raises(OllamaAPIError) as exc_info:
            self.llm._make_request("api/generate", {})
        
        assert "Unauthorized" in exc_info.value.message
        assert exc_info.value.status_code == 401
    
    def test_connection_error_with_retry(self, mocker):
        """Test connection error với retry logic"""
        mock_post = mocker.patch("requests.post", side_effect=ConnectionError("Connection failed"))
        
        with pytest.raises(OllamaAPIError) as exc_info:
            self.llm._make_request("api/generate", {})
        
        assert "Không thể kết nối" in exc_info.value.message
        # Verify retry logic: initial + 2 retries = 3 calls
        assert mock_post.call_count == 3
    
    def test_timeout_error_with_retry(self, mocker):
        """Test timeout error với retry logic"""
        mocker.patch("requests.post", side_effect=Timeout("Request timeout"))
        
        with pytest.raises(OllamaAPIError) as exc_info:
            self.llm._make_request("api/generate", {})
        
        assert "timeout" in exc_info.value.message
    
    def test_generic_request_exception(self, mocker):
        """Test generic request exception"""
        mocker.patch("requests.post", side_effect=RequestException("Generic error"))
        
        with pytest.raises(OllamaAPIError) as exc_info:
            self.llm._make_request("api/generate", {})
        
        assert "Request error" in exc_info.value.message
    
    def test_retry_then_success(self, mocker):
        """Test retry logic với success sau vài lần thất bại"""
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        
        mock_response_error = Mock()
        mock_response_error.status_code = 500
        mock_response_error.text = "Server error"
        
        # First call fails, second call succeeds
        mocker.patch("requests.post", side_effect=[mock_response_error, mock_response_success])
        
        result = self.llm._make_request("api/generate", {})
        assert result == mock_response_success


class TestGenerate:
    """Test class cho generate method"""
    
    def setup_method(self):
        self.llm = OllamaLLMCaller()
    
    def test_generate_basic_prompt(self, mocker):
        """Test generate với prompt cơ bản"""
        mock_response_data = {
            "response": "Generated response",
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True,
            "total_duration": 1000000,
            "eval_count": 10
        }
        
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        
        mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        result = self.llm.generate("Test prompt")
        
        assert isinstance(result, OllamaResponse)
        assert result.response == "Generated response"
        assert result.model == "codellama"
        assert result.done is True
        assert result.total_duration == 1000000
        assert result.eval_count == 10
    
    def test_generate_with_code_snippet(self, mocker):
        """Test generate với code snippet"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Code analysis",
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        prompt = "Analyze this code"
        code_snippet = "def hello(): print('hello')"
        
        result = self.llm.generate(prompt, code_snippet=code_snippet)
        
        # Verify payload contains formatted prompt with code
        call_args = mock_make_request.call_args
        payload = call_args[0][1]  # Second argument is payload
        
        assert "Analyze this code" in payload["prompt"]
        assert "Code snippet:" in payload["prompt"]
        assert "def hello(): print('hello')" in payload["prompt"]
        assert payload["model"] == "codellama"
    
    def test_generate_with_system_prompt(self, mocker):
        """Test generate với system prompt"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Response with system",
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        result = self.llm.generate(
            "User prompt",
            system_prompt="You are a helpful assistant"
        )
        
        call_args = mock_make_request.call_args
        payload = call_args[0][1]
        
        assert payload["system"] == "You are a helpful assistant"
        assert payload["prompt"] == "User prompt"
    
    def test_generate_with_custom_parameters(self, mocker):
        """Test generate với custom parameters"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Custom response",
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        result = self.llm.generate(
            "Test prompt",
            temperature=0.5,
            top_p=0.8,
            max_tokens=100,
            stream=True
        )
        
        call_args = mock_make_request.call_args
        payload = call_args[0][1]
        
        assert payload["options"]["temperature"] == 0.5
        assert payload["options"]["top_p"] == 0.8
        assert payload["options"]["num_predict"] == 100
        assert payload["stream"] is True
    
    def test_generate_json_decode_error(self, mocker):
        """Test generate với JSON decode error"""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        with pytest.raises(OllamaAPIError) as exc_info:
            self.llm.generate("Test prompt")
        
        assert "parse JSON response" in exc_info.value.message


class TestChat:
    """Test class cho chat method"""
    
    def setup_method(self):
        self.llm = OllamaLLMCaller()
    
    def test_chat_basic(self, mocker):
        """Test chat với messages cơ bản"""
        mock_response_data = {
            "message": {"content": "Chat response"},
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        result = self.llm.chat(messages)
        
        assert isinstance(result, OllamaResponse)
        assert result.response == "Chat response"
        
        call_args = mock_make_request.call_args
        payload = call_args[0][1]
        assert payload["messages"] == messages
        assert payload["model"] == "codellama"
    
    def test_chat_with_parameters(self, mocker):
        """Test chat với custom parameters"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": "Response"},
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        messages = [{"role": "user", "content": "Test"}]
        
        result = self.llm.chat(
            messages,
            temperature=0.3,
            max_tokens=50
        )
        
        call_args = mock_make_request.call_args
        payload = call_args[0][1]
        
        assert payload["options"]["temperature"] == 0.3
        assert payload["options"]["num_predict"] == 50
    
    def test_chat_empty_message_content(self, mocker):
        """Test chat với empty message content"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        result = self.llm.chat([{"role": "user", "content": "Test"}])
        
        assert result.response == ""  # Should handle missing message content


class TestAnalyzeCode:
    """Test class cho analyze_code method"""
    
    def setup_method(self):
        self.llm = OllamaLLMCaller()
    
    def test_analyze_code_general(self, mocker):
        """Test analyze_code với general analysis"""
        mock_response = OllamaResponse(
            response="Code analysis result",
            model="codellama",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        
        mock_generate = mocker.patch.object(self.llm, "generate", return_value=mock_response)
        
        code = "def test(): pass"
        result = self.llm.analyze_code(code, "general", "python")
        
        assert result == mock_response
        
        # Verify generate was called with correct parameters
        call_args = mock_generate.call_args
        assert "phân tích code snippet" in call_args[1]["prompt"]
        assert call_args[1]["code_snippet"] == code
        assert "Ngôn ngữ lập trình: python" in call_args[1]["system_prompt"]
        assert call_args[1]["temperature"] == 0.3
    
    def test_analyze_code_bugs(self, mocker):
        """Test analyze_code với bugs analysis"""
        mock_response = OllamaResponse(
            response="Bug analysis",
            model="codellama",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        
        mock_generate = mocker.patch.object(self.llm, "generate", return_value=mock_response)
        
        result = self.llm.analyze_code("code", "bugs")
        
        call_args = mock_generate.call_args
        assert "lỗi tiềm ẩn" in call_args[1]["prompt"]
    
    def test_analyze_code_optimization(self, mocker):
        """Test analyze_code với optimization analysis"""
        mock_response = OllamaResponse(
            response="Optimization suggestions",
            model="codellama",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        
        mock_generate = mocker.patch.object(self.llm, "generate", return_value=mock_response)
        
        result = self.llm.analyze_code("code", "optimization")
        
        call_args = mock_generate.call_args
        assert "tối ưu hóa performance" in call_args[1]["prompt"]
    
    def test_analyze_code_documentation(self, mocker):
        """Test analyze_code với documentation analysis"""
        mock_response = OllamaResponse(
            response="Documentation",
            model="codellama",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        
        mock_generate = mocker.patch.object(self.llm, "generate", return_value=mock_response)
        
        result = self.llm.analyze_code("code", "documentation")
        
        call_args = mock_generate.call_args
        assert "documentation" in call_args[1]["prompt"]
    
    def test_analyze_code_invalid_type(self, mocker):
        """Test analyze_code với invalid analysis type"""
        mock_response = OllamaResponse(
            response="General analysis",
            model="codellama",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        
        mock_generate = mocker.patch.object(self.llm, "generate", return_value=mock_response)
        
        result = self.llm.analyze_code("code", "invalid_type")
        
        # Should fallback to general analysis
        call_args = mock_generate.call_args
        assert "phân tích code snippet" in call_args[1]["prompt"]


class TestHealthAndModels:
    """Test class cho health check và list models"""
    
    def setup_method(self):
        self.llm = OllamaLLMCaller()
    
    def test_check_health_success(self, mocker):
        """Test health check thành công"""
        mock_response = Mock()
        mock_response.status_code = 200
        
        mock_get = mocker.patch("requests.get", return_value=mock_response)
        
        result = self.llm.check_health()
        
        assert result is True
        mock_get.assert_called_once_with(
            f"{self.llm.base_url}/api/tags",
            headers=self.llm.headers,
            timeout=10
        )
    
    def test_check_health_failure(self, mocker):
        """Test health check thất bại"""
        mock_response = Mock()
        mock_response.status_code = 500
        
        mocker.patch("requests.get", return_value=mock_response)
        
        result = self.llm.check_health()
        assert result is False
    
    def test_check_health_exception(self, mocker):
        """Test health check với exception"""
        mocker.patch("requests.get", side_effect=ConnectionError("Connection failed"))
        
        result = self.llm.check_health()
        assert result is False
    
    def test_list_models_success(self, mocker):
        """Test list models thành công"""
        mock_response_data = {
            "models": [
                {"name": "codellama:7b"},
                {"name": "llama2:13b"},
                {"name": "codellama:34b"}
            ]
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        
        mock_get = mocker.patch("requests.get", return_value=mock_response)
        
        result = self.llm.list_models()
        
        expected_models = ["codellama:7b", "llama2:13b", "codellama:34b"]
        assert result == expected_models
        
        mock_get.assert_called_once_with(
            f"{self.llm.base_url}/api/tags",
            headers=self.llm.headers,
            timeout=30
        )
    
    def test_list_models_http_error(self, mocker):
        """Test list models với HTTP error"""
        mock_response = Mock()
        mock_response.status_code = 500
        
        mocker.patch("requests.get", return_value=mock_response)
        
        with pytest.raises(OllamaAPIError) as exc_info:
            self.llm.list_models()
        
        assert "Không thể lấy danh sách models" in exc_info.value.message
    
    def test_list_models_request_exception(self, mocker):
        """Test list models với request exception"""
        mocker.patch("requests.get", side_effect=ConnectionError("Connection failed"))
        
        with pytest.raises(OllamaAPIError) as exc_info:
            self.llm.list_models()
        
        assert "Lỗi khi lấy danh sách models" in exc_info.value.message


class TestConvenienceFunctions:
    """Test class cho convenience functions"""
    
    def test_create_llm_caller_default(self):
        """Test create_llm_caller với giá trị mặc định"""
        llm = create_llm_caller()
        
        assert isinstance(llm, OllamaLLMCaller)
        assert llm.model == "codellama"
        assert llm.base_url == "http://localhost:11434"
    
    def test_create_llm_caller_custom(self):
        """Test create_llm_caller với giá trị tùy chỉnh"""
        custom_url = "http://custom:8080"
        llm = create_llm_caller(
            model=OllamaModel.CODELLAMA_7B,
            base_url=custom_url
        )
        
        assert llm.model == "codellama:7b"
        assert llm.base_url == custom_url
    
    def test_quick_analyze_code(self, mocker):
        """Test quick_analyze_code function"""
        mock_response = OllamaResponse(
            response="Quick analysis result",
            model="codellama",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        
        # Mock the create_llm_caller and analyze_code
        mock_llm = Mock()
        mock_llm.analyze_code.return_value = mock_response
        
        mocker.patch("utils.llm_caller.create_llm_caller", return_value=mock_llm)
        
        code = "def test(): pass"
        result = quick_analyze_code(code, "bugs")
        
        assert result == "Quick analysis result"
        mock_llm.analyze_code.assert_called_once_with(code, "bugs")


class TestOllamaModel:
    """Test class cho OllamaModel enum"""
    
    def test_model_values(self):
        """Test các giá trị của OllamaModel enum"""
        assert OllamaModel.CODELLAMA.value == "codellama"
        assert OllamaModel.CODELLAMA_7B.value == "codellama:7b"
        assert OllamaModel.CODELLAMA_13B.value == "codellama:13b"
        assert OllamaModel.CODELLAMA_34B.value == "codellama:34b"
        assert OllamaModel.LLAMA2.value == "llama2"
        assert OllamaModel.LLAMA2_7B.value == "llama2:7b"
        assert OllamaModel.LLAMA2_13B.value == "llama2:13b"


class TestOllamaResponse:
    """Test class cho OllamaResponse dataclass"""
    
    def test_response_creation(self):
        """Test tạo OllamaResponse object"""
        response = OllamaResponse(
            response="Test response",
            model="codellama",
            created_at="2024-01-01T00:00:00Z",
            done=True,
            total_duration=1000000,
            eval_count=10
        )
        
        assert response.response == "Test response"
        assert response.model == "codellama"
        assert response.done is True
        assert response.total_duration == 1000000
        assert response.eval_count == 10
    
    def test_response_optional_fields(self):
        """Test OllamaResponse với optional fields"""
        response = OllamaResponse(
            response="Test",
            model="codellama",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        
        assert response.total_duration is None
        assert response.load_duration is None
        assert response.eval_count is None


class TestOllamaAPIError:
    """Test class cho OllamaAPIError exception"""
    
    def test_error_creation_basic(self):
        """Test tạo OllamaAPIError cơ bản"""
        error = OllamaAPIError("Test error message")
        
        assert error.message == "Test error message"
        assert error.status_code is None
        assert error.response_text is None
        assert str(error) == "Test error message"
    
    def test_error_creation_with_details(self):
        """Test tạo OllamaAPIError với chi tiết"""
        error = OllamaAPIError(
            "API Error",
            status_code=500,
            response_text="Internal Server Error"
        )
        
        assert error.message == "API Error"
        assert error.status_code == 500
        assert error.response_text == "Internal Server Error"


class TestIntegration:
    """Integration tests"""
    
    def test_full_workflow_mock(self, mocker):
        """Test full workflow với mock"""
        # Mock successful health check
        mock_health_response = Mock()
        mock_health_response.status_code = 200
        
        # Mock successful models list
        mock_models_response = Mock()
        mock_models_response.status_code = 200
        mock_models_response.json.return_value = {
            "models": [{"name": "codellama"}]
        }
        
        # Mock successful generate
        mock_generate_response = Mock()
        mock_generate_response.status_code = 200
        mock_generate_response.json.return_value = {
            "response": "Code looks good!",
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_get = mocker.patch("requests.get")
        mock_post = mocker.patch("requests.post")
        
        # Setup mock responses in order
        mock_get.side_effect = [mock_health_response, mock_models_response]
        mock_post.return_value = mock_generate_response
        
        # Test workflow
        llm = create_llm_caller()
        
        # Check health
        assert llm.check_health() is True
        
        # List models
        models = llm.list_models()
        assert models == ["codellama"]
        
        # Analyze code
        result = llm.analyze_code("def hello(): pass", "general")
        assert result.response == "Code looks good!"
        
        # Verify all calls were made
        assert mock_get.call_count == 2
        assert mock_post.call_count == 1


# Fixtures
@pytest.fixture
def sample_code():
    """Fixture cung cấp sample code để test"""
    return """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""


@pytest.fixture
def mock_ollama_response():
    """Fixture cung cấp mock OllamaResponse"""
    return OllamaResponse(
        response="Mock response",
        model="codellama",
        created_at="2024-01-01T00:00:00Z",
        done=True,
        total_duration=1000000,
        eval_count=10
    )


# Parametrized tests
@pytest.mark.parametrize("analysis_type,expected_keyword", [
    ("general", "phân tích code snippet"),
    ("bugs", "lỗi tiềm ẩn"),
    ("optimization", "tối ưu hóa performance"),
    ("documentation", "documentation"),
])
def test_analyze_code_prompts(analysis_type, expected_keyword, mocker):
    """Test các prompts khác nhau cho analyze_code"""
    llm = OllamaLLMCaller()
    
    mock_response = OllamaResponse(
        response="Analysis result",
        model="codellama",
        created_at="2024-01-01T00:00:00Z",
        done=True
    )
    
    mock_generate = mocker.patch.object(llm, "generate", return_value=mock_response)
    
    llm.analyze_code("test code", analysis_type)
    
    call_args = mock_generate.call_args
    assert expected_keyword in call_args[1]["prompt"]


@pytest.mark.parametrize("model_enum,expected_string", [
    (OllamaModel.CODELLAMA, "codellama"),
    (OllamaModel.CODELLAMA_7B, "codellama:7b"),
    (OllamaModel.LLAMA2, "llama2"),
])
def test_model_enum_to_string(model_enum, expected_string):
    """Test chuyển đổi từ enum sang string"""
    llm = OllamaLLMCaller(model=model_enum)
    assert llm.model == expected_string


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 