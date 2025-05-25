"""
Tests cho prompt formatting và template generation trong llm_caller.
"""

import pytest
from unittest.mock import Mock

from ..utils.llm_caller import OllamaLLMCaller, OllamaModel, OllamaResponse


class TestPromptFormatting:
    """Test class cho prompt formatting"""
    
    def setup_method(self):
        self.llm = OllamaLLMCaller()
    
    def test_basic_prompt_formatting(self, mocker):
        """Test formatting prompt cơ bản"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Test response",
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        prompt = "Explain this function"
        self.llm.generate(prompt)
        
        call_args = mock_make_request.call_args
        payload = call_args[0][1]
        
        assert payload["prompt"] == prompt
        assert payload["model"] == "codellama"
    
    def test_prompt_with_code_snippet_formatting(self, mocker):
        """Test formatting prompt với code snippet"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Code explanation",
            "model": "codellama", 
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        prompt = "What does this function do?"
        code_snippet = "def hello_world():\n    print('Hello, World!')"
        
        self.llm.generate(prompt, code_snippet=code_snippet)
        
        call_args = mock_make_request.call_args
        payload = call_args[0][1]
        formatted_prompt = payload["prompt"]
        
        # Verify prompt structure
        assert prompt in formatted_prompt
        assert "Code snippet:" in formatted_prompt
        assert "```" in formatted_prompt
        assert code_snippet in formatted_prompt
        
        # Verify order: prompt first, then code
        prompt_index = formatted_prompt.index(prompt)
        code_index = formatted_prompt.index(code_snippet)
        assert prompt_index < code_index
    
    def test_prompt_with_multiline_code(self, mocker, sample_code_snippets):
        """Test formatting với multiline code snippet"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Analysis",
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z", 
            "done": True
        }
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        prompt = "Analyze this function"
        code = sample_code_snippets["python_fibonacci"]
        
        self.llm.generate(prompt, code_snippet=code)
        
        call_args = mock_make_request.call_args
        payload = call_args[0][1]
        formatted_prompt = payload["prompt"]
        
        # Verify multiline code is properly formatted
        assert "def fibonacci(n):" in formatted_prompt
        assert "return fibonacci(n-1) + fibonacci(n-2)" in formatted_prompt
        assert formatted_prompt.count("```") == 2  # Opening and closing
    
    def test_prompt_with_special_characters(self, mocker):
        """Test formatting với special characters trong code"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Response",
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        prompt = "Explain this regex"
        code_snippet = 'pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"'
        
        self.llm.generate(prompt, code_snippet=code_snippet)
        
        call_args = mock_make_request.call_args
        payload = call_args[0][1]
        formatted_prompt = payload["prompt"]
        
        # Verify special characters are preserved
        assert code_snippet in formatted_prompt
        assert "\\." in formatted_prompt
        assert "[a-zA-Z0-9._%+-]+" in formatted_prompt
    
    def test_system_prompt_formatting(self, mocker):
        """Test system prompt formatting"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Response with system",
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        user_prompt = "Help me with this code"
        system_prompt = "You are an expert Python developer. Provide clear, concise explanations."
        code = "print('hello')"
        
        self.llm.generate(
            user_prompt,
            code_snippet=code,
            system_prompt=system_prompt
        )
        
        call_args = mock_make_request.call_args
        payload = call_args[0][1]
        
        # Verify system prompt is separate from main prompt
        assert payload["system"] == system_prompt
        assert system_prompt not in payload["prompt"]
        assert user_prompt in payload["prompt"]
        assert code in payload["prompt"]


class TestAnalysisPromptTemplates:
    """Test class cho analysis prompt templates"""
    
    def setup_method(self):
        self.llm = OllamaLLMCaller()
    
    def test_general_analysis_prompt(self, mocker):
        """Test general analysis prompt template"""
        mock_response = OllamaResponse(
            response="General analysis",
            model="codellama",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        
        mock_generate = mocker.patch.object(self.llm, "generate", return_value=mock_response)
        
        code = "def test(): pass"
        self.llm.analyze_code(code, "general")
        
        call_args = mock_generate.call_args
        prompt = call_args[1]["prompt"]
        
        assert "phân tích code snippet" in prompt
        assert "cấu trúc" in prompt
        assert "logic" in prompt
        assert "chất lượng code" in prompt
    
    def test_bugs_analysis_prompt(self, mocker):
        """Test bugs analysis prompt template"""
        mock_response = OllamaResponse(
            response="Bug analysis",
            model="codellama",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        
        mock_generate = mocker.patch.object(self.llm, "generate", return_value=mock_response)
        
        code = "def divide(a, b): return a/b"
        self.llm.analyze_code(code, "bugs")
        
        call_args = mock_generate.call_args
        prompt = call_args[1]["prompt"]
        
        assert "lỗi tiềm ẩn" in prompt
        assert "bugs" in prompt
        assert "vấn đề" in prompt
    
    def test_optimization_analysis_prompt(self, mocker):
        """Test optimization analysis prompt template"""
        mock_response = OllamaResponse(
            response="Optimization suggestions",
            model="codellama",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        
        mock_generate = mocker.patch.object(self.llm, "generate", return_value=mock_response)
        
        code = "for i in range(len(arr)): print(arr[i])"
        self.llm.analyze_code(code, "optimization")
        
        call_args = mock_generate.call_args
        prompt = call_args[1]["prompt"]
        
        assert "tối ưu hóa performance" in prompt
        assert "cải thiện" in prompt
    
    def test_documentation_analysis_prompt(self, mocker):
        """Test documentation analysis prompt template"""
        mock_response = OllamaResponse(
            response="Documentation",
            model="codellama",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        
        mock_generate = mocker.patch.object(self.llm, "generate", return_value=mock_response)
        
        code = "def calculate(x, y): return x * y + 1"
        self.llm.analyze_code(code, "documentation")
        
        call_args = mock_generate.call_args
        prompt = call_args[1]["prompt"]
        
        assert "documentation" in prompt
        assert "comments" in prompt
    
    def test_system_prompt_with_language(self, mocker):
        """Test system prompt với language specification"""
        mock_response = OllamaResponse(
            response="Python analysis",
            model="codellama",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        
        mock_generate = mocker.patch.object(self.llm, "generate", return_value=mock_response)
        
        code = "def hello(): print('hello')"
        self.llm.analyze_code(code, "general", "python")
        
        call_args = mock_generate.call_args
        system_prompt = call_args[1]["system_prompt"]
        
        assert "chuyên gia phân tích code" in system_prompt
        assert "Ngôn ngữ lập trình: python" in system_prompt
        assert "chi tiết" in system_prompt
        assert "chính xác" in system_prompt
    
    def test_system_prompt_without_language(self, mocker):
        """Test system prompt không có language specification"""
        mock_response = OllamaResponse(
            response="General analysis",
            model="codellama",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        
        mock_generate = mocker.patch.object(self.llm, "generate", return_value=mock_response)
        
        code = "function test() { return true; }"
        self.llm.analyze_code(code, "general")
        
        call_args = mock_generate.call_args
        system_prompt = call_args[1]["system_prompt"]
        
        assert "chuyên gia phân tích code" in system_prompt
        assert "Ngôn ngữ lập trình:" not in system_prompt
    
    def test_temperature_setting_for_analysis(self, mocker):
        """Test temperature setting cho code analysis"""
        mock_response = OllamaResponse(
            response="Analysis",
            model="codellama",
            created_at="2024-01-01T00:00:00Z",
            done=True
        )
        
        mock_generate = mocker.patch.object(self.llm, "generate", return_value=mock_response)
        
        code = "x = 1 + 1"
        self.llm.analyze_code(code, "general")
        
        call_args = mock_generate.call_args
        temperature = call_args[1]["temperature"]
        
        # Code analysis should use lower temperature for consistency
        assert temperature == 0.3


class TestChatPromptFormatting:
    """Test class cho chat prompt formatting"""
    
    def setup_method(self):
        self.llm = OllamaLLMCaller()
    
    def test_chat_messages_formatting(self, mocker):
        """Test chat messages formatting"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": "Chat response"},
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        self.llm.chat(messages)
        
        call_args = mock_make_request.call_args
        payload = call_args[0][1]
        
        assert payload["messages"] == messages
        assert payload["model"] == "codellama"
        assert "prompt" not in payload  # Chat uses messages, not prompt
    
    def test_chat_with_code_in_messages(self, mocker):
        """Test chat với code trong messages"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": "Code explanation"},
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        code_message = """Can you explain this code?

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
```"""
        
        messages = [
            {"role": "user", "content": code_message}
        ]
        
        self.llm.chat(messages)
        
        call_args = mock_make_request.call_args
        payload = call_args[0][1]
        
        # Verify code is preserved in message content
        assert "def factorial(n):" in payload["messages"][0]["content"]
        assert "```python" in payload["messages"][0]["content"]
    
    def test_empty_messages_list(self, mocker):
        """Test với empty messages list"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": "Empty response"},
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        messages = []
        self.llm.chat(messages)
        
        call_args = mock_make_request.call_args
        payload = call_args[0][1]
        
        assert payload["messages"] == []


class TestPromptParameterHandling:
    """Test class cho parameter handling trong prompts"""
    
    def setup_method(self):
        self.llm = OllamaLLMCaller()
    
    def test_temperature_parameter(self, mocker):
        """Test temperature parameter"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Response",
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        self.llm.generate("Test", temperature=0.5)
        
        call_args = mock_make_request.call_args
        payload = call_args[0][1]
        
        assert payload["options"]["temperature"] == 0.5
    
    def test_top_p_parameter(self, mocker):
        """Test top_p parameter"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Response",
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        self.llm.generate("Test", top_p=0.8)
        
        call_args = mock_make_request.call_args
        payload = call_args[0][1]
        
        assert payload["options"]["top_p"] == 0.8
    
    def test_max_tokens_parameter(self, mocker):
        """Test max_tokens parameter"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Response",
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        self.llm.generate("Test", max_tokens=100)
        
        call_args = mock_make_request.call_args
        payload = call_args[0][1]
        
        assert payload["options"]["num_predict"] == 100
    
    def test_stream_parameter(self, mocker):
        """Test stream parameter"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Response",
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        self.llm.generate("Test", stream=True)
        
        call_args = mock_make_request.call_args
        payload = call_args[0][1]
        
        assert payload["stream"] is True
    
    def test_all_parameters_combined(self, mocker):
        """Test tất cả parameters combined"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Response",
            "model": "codellama",
            "created_at": "2024-01-01T00:00:00Z",
            "done": True
        }
        
        mock_make_request = mocker.patch.object(self.llm, "_make_request", return_value=mock_response)
        
        self.llm.generate(
            "Test prompt",
            code_snippet="def test(): pass",
            system_prompt="You are helpful",
            temperature=0.3,
            top_p=0.9,
            max_tokens=200,
            stream=False
        )
        
        call_args = mock_make_request.call_args
        payload = call_args[0][1]
        
        # Verify all parameters are set correctly
        assert "Test prompt" in payload["prompt"]
        assert "def test(): pass" in payload["prompt"]
        assert payload["system"] == "You are helpful"
        assert payload["options"]["temperature"] == 0.3
        assert payload["options"]["top_p"] == 0.9
        assert payload["options"]["num_predict"] == 200
        assert payload["stream"] is False


# Parametrized tests cho different prompt scenarios
@pytest.mark.parametrize("prompt,code,expected_parts", [
    ("Explain this", "print('hello')", ["Explain this", "Code snippet:", "print('hello')"]),
    ("What does this do?", "x = 1 + 1", ["What does this do?", "Code snippet:", "x = 1 + 1"]),
    ("Analyze", "def f(): return True", ["Analyze", "Code snippet:", "def f(): return True"]),
])
def test_prompt_formatting_scenarios(prompt, code, expected_parts, mocker):
    """Test different prompt formatting scenarios"""
    llm = OllamaLLMCaller()
    
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": "Response",
        "model": "codellama",
        "created_at": "2024-01-01T00:00:00Z",
        "done": True
    }
    
    mock_make_request = mocker.patch.object(llm, "_make_request", return_value=mock_response)
    
    llm.generate(prompt, code_snippet=code)
    
    call_args = mock_make_request.call_args
    formatted_prompt = call_args[0][1]["prompt"]
    
    for part in expected_parts:
        assert part in formatted_prompt


@pytest.mark.parametrize("analysis_type,language,expected_in_system", [
    ("general", "python", "Ngôn ngữ lập trình: python"),
    ("bugs", "javascript", "Ngôn ngữ lập trình: javascript"),
    ("optimization", None, "chuyên gia phân tích code"),
    ("documentation", "java", "Ngôn ngữ lập trình: java"),
])
def test_system_prompt_language_scenarios(analysis_type, language, expected_in_system, mocker):
    """Test system prompt với different languages"""
    llm = OllamaLLMCaller()
    
    mock_response = OllamaResponse(
        response="Analysis",
        model="codellama",
        created_at="2024-01-01T00:00:00Z",
        done=True
    )
    
    mock_generate = mocker.patch.object(llm, "generate", return_value=mock_response)
    
    llm.analyze_code("test code", analysis_type, language)
    
    call_args = mock_generate.call_args
    system_prompt = call_args[1]["system_prompt"]
    
    assert expected_in_system in system_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 