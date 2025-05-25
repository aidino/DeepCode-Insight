"""
Utility module để gọi API Ollama local (CodeLlama) qua HTTP.
Hỗ trợ gửi prompt và code snippet, xử lý API keys và errors.
"""

import os
import json
import logging
import requests
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

# Thiết lập logging
logger = logging.getLogger(__name__)


class OllamaModel(Enum):
    """Enum cho các model Ollama được hỗ trợ"""
    CODELLAMA = "codellama:7b-instruct-q4_K_M"
    CODELLAMA_7B = "codellama:7b"
    CODELLAMA_13B = "codellama:13b"
    CODELLAMA_34B = "codellama:34b"
    LLAMA2 = "llama2"
    LLAMA2_7B = "llama2:7b"
    LLAMA2_13B = "llama2:13b"


@dataclass
class OllamaResponse:
    """Dataclass để chứa response từ Ollama API"""
    response: str
    model: str
    created_at: str
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaAPIError(Exception):
    """Custom exception cho Ollama API errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response_text: Optional[str] = None):
        self.message = message
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(self.message)


class OllamaLLMCaller:
    """
    Class để gọi Ollama API local qua HTTP.
    Hỗ trợ gửi prompt và code snippet, xử lý authentication và errors.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Union[str, OllamaModel] = OllamaModel.CODELLAMA,
        timeout: int = 120,
        max_retries: int = 3
    ):
        """
        Khởi tạo OllamaLLMCaller.
        
        Args:
            base_url: URL của Ollama server (mặc định từ env var OLLAMA_BASE_URL hoặc http://localhost:11434)
            model: Model để sử dụng (mặc định CodeLlama)
            timeout: Timeout cho requests (giây)
            max_retries: Số lần retry tối đa khi gặp lỗi
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model.value if isinstance(model, OllamaModel) else model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # API key (nếu cần thiết cho authentication)
        self.api_key = os.getenv("OLLAMA_API_KEY")
        
        # Headers mặc định
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Thêm API key vào headers nếu có
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        logger.info(f"Khởi tạo OllamaLLMCaller với base_url: {self.base_url}, model: {self.model}")
    
    def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> requests.Response:
        """
        Thực hiện HTTP request với retry logic.
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            
        Returns:
            requests.Response object
            
        Raises:
            OllamaAPIError: Khi gặp lỗi API
        """
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Gửi request đến {url} (lần thử {attempt + 1})")
                response = requests.post(
                    url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 404:
                    raise OllamaAPIError(
                        f"Model '{self.model}' không tìm thấy. Hãy đảm bảo model đã được pull.",
                        status_code=response.status_code,
                        response_text=response.text
                    )
                elif response.status_code == 401:
                    raise OllamaAPIError(
                        "Unauthorized. Kiểm tra API key.",
                        status_code=response.status_code,
                        response_text=response.text
                    )
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    if attempt == self.max_retries:
                        raise OllamaAPIError(error_msg, response.status_code, response.text)
                    logger.warning(f"Request thất bại (lần thử {attempt + 1}): {error_msg}")
                    
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Không thể kết nối đến Ollama server tại {self.base_url}"
                if attempt == self.max_retries:
                    raise OllamaAPIError(f"{error_msg}: {str(e)}")
                logger.warning(f"Connection error (lần thử {attempt + 1}): {str(e)}")
                
            except requests.exceptions.Timeout as e:
                error_msg = f"Request timeout sau {self.timeout} giây"
                if attempt == self.max_retries:
                    raise OllamaAPIError(f"{error_msg}: {str(e)}")
                logger.warning(f"Timeout error (lần thử {attempt + 1}): {str(e)}")
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Request error: {str(e)}"
                if attempt == self.max_retries:
                    raise OllamaAPIError(error_msg)
                logger.warning(f"Request error (lần thử {attempt + 1}): {str(e)}")
    
    def generate(
        self,
        prompt: str,
        code_snippet: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> OllamaResponse:
        """
        Gửi prompt và code snippet đến Ollama API để generate response.
        
        Args:
            prompt: Prompt chính
            code_snippet: Code snippet để phân tích (optional)
            system_prompt: System prompt (optional)
            temperature: Temperature cho generation (0.0 - 1.0)
            top_p: Top-p sampling parameter
            max_tokens: Số token tối đa để generate
            stream: Có stream response hay không
            
        Returns:
            OllamaResponse object chứa kết quả
            
        Raises:
            OllamaAPIError: Khi gặp lỗi API
        """
        # Xây dựng prompt đầy đủ
        full_prompt = prompt
        
        if code_snippet:
            full_prompt = f"{prompt}\n\nCode snippet:\n```\n{code_snippet}\n```"
        
        # Payload cho API request
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p
            }
        }
        
        # Thêm system prompt nếu có
        if system_prompt:
            payload["system"] = system_prompt
        
        # Thêm max_tokens nếu có
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        logger.info(f"Gửi generate request với model: {self.model}")
        logger.debug(f"Prompt length: {len(full_prompt)} characters")
        
        try:
            response = self._make_request("api/generate", payload)
            response_data = response.json()
            
            # Parse response thành OllamaResponse object
            ollama_response = OllamaResponse(
                response=response_data.get("response", ""),
                model=response_data.get("model", self.model),
                created_at=response_data.get("created_at", ""),
                done=response_data.get("done", True),
                total_duration=response_data.get("total_duration"),
                load_duration=response_data.get("load_duration"),
                prompt_eval_count=response_data.get("prompt_eval_count"),
                prompt_eval_duration=response_data.get("prompt_eval_duration"),
                eval_count=response_data.get("eval_count"),
                eval_duration=response_data.get("eval_duration")
            )
            
            logger.info(f"Nhận được response thành công. Response length: {len(ollama_response.response)} characters")
            return ollama_response
            
        except json.JSONDecodeError as e:
            raise OllamaAPIError(f"Không thể parse JSON response: {str(e)}")
    
    def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> OllamaResponse:
        """
        Gửi chat messages đến Ollama API.
        
        Args:
            messages: List các messages theo format [{"role": "user", "content": "..."}]
            temperature: Temperature cho generation
            top_p: Top-p sampling parameter
            max_tokens: Số token tối đa để generate
            stream: Có stream response hay không
            
        Returns:
            OllamaResponse object chứa kết quả
            
        Raises:
            OllamaAPIError: Khi gặp lỗi API
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        logger.info(f"Gửi chat request với {len(messages)} messages")
        
        try:
            response = self._make_request("api/chat", payload)
            response_data = response.json()
            
            # Extract content từ message response
            message_content = ""
            if "message" in response_data and "content" in response_data["message"]:
                message_content = response_data["message"]["content"]
            
            ollama_response = OllamaResponse(
                response=message_content,
                model=response_data.get("model", self.model),
                created_at=response_data.get("created_at", ""),
                done=response_data.get("done", True),
                total_duration=response_data.get("total_duration"),
                load_duration=response_data.get("load_duration"),
                prompt_eval_count=response_data.get("prompt_eval_count"),
                prompt_eval_duration=response_data.get("prompt_eval_duration"),
                eval_count=response_data.get("eval_count"),
                eval_duration=response_data.get("eval_duration")
            )
            
            logger.info(f"Nhận được chat response thành công. Response length: {len(ollama_response.response)} characters")
            return ollama_response
            
        except json.JSONDecodeError as e:
            raise OllamaAPIError(f"Không thể parse JSON response: {str(e)}")
    
    def analyze_code(
        self,
        code_snippet: str,
        analysis_type: str = "general",
        language: Optional[str] = None
    ) -> OllamaResponse:
        """
        Phân tích code snippet với prompt được tối ưu hóa.
        
        Args:
            code_snippet: Code cần phân tích
            analysis_type: Loại phân tích ("general", "bugs", "optimization", "documentation")
            language: Ngôn ngữ lập trình (optional, sẽ auto-detect nếu không có)
            
        Returns:
            OllamaResponse object chứa kết quả phân tích
        """
        # Prompts cho các loại phân tích khác nhau
        analysis_prompts = {
            "general": "Hãy phân tích code snippet sau và đưa ra nhận xét về cấu trúc, logic, và chất lượng code:",
            "bugs": "Hãy tìm kiếm các lỗi tiềm ẩn, bugs, hoặc vấn đề trong code snippet sau:",
            "optimization": "Hãy đề xuất các cách tối ưu hóa performance và cải thiện code snippet sau:",
            "documentation": "Hãy tạo documentation và comments cho code snippet sau:"
        }
        
        prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])
        
        # System prompt cho code analysis
        system_prompt = f"""Bạn là một chuyên gia phân tích code. 
        Hãy đưa ra phân tích chi tiết, chính xác và hữu ích.
        {f'Ngôn ngữ lập trình: {language}' if language else ''}
        Định dạng response của bạn một cách rõ ràng và dễ đọc."""
        
        return self.generate(
            prompt=prompt,
            code_snippet=code_snippet,
            system_prompt=system_prompt,
            temperature=0.3  # Lower temperature cho code analysis
        )
    
    def check_health(self) -> bool:
        """
        Kiểm tra xem Ollama server có hoạt động không.
        
        Returns:
            True nếu server hoạt động, False nếu không
        """
        try:
            url = f"{self.base_url.rstrip('/')}/api/tags"
            response = requests.get(url, headers=self.headers, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check thất bại: {str(e)}")
            return False
    
    def list_models(self) -> list:
        """
        Lấy danh sách các models có sẵn trên Ollama server.
        
        Returns:
            List các model names
            
        Raises:
            OllamaAPIError: Khi gặp lỗi API
        """
        try:
            url = f"{self.base_url.rstrip('/')}/api/tags"
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                models = [model.get("name", "") for model in data.get("models", [])]
                logger.info(f"Tìm thấy {len(models)} models: {models}")
                return models
            else:
                raise OllamaAPIError(f"Không thể lấy danh sách models: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise OllamaAPIError(f"Lỗi khi lấy danh sách models: {str(e)}")


# Convenience functions
def create_llm_caller(
    model: Union[str, OllamaModel] = OllamaModel.CODELLAMA,
    base_url: Optional[str] = None
) -> OllamaLLMCaller:
    """
    Tạo OllamaLLMCaller instance với cấu hình mặc định.
    
    Args:
        model: Model để sử dụng
        base_url: URL của Ollama server
        
    Returns:
        OllamaLLMCaller instance
    """
    return OllamaLLMCaller(model=model, base_url=base_url)


def quick_analyze_code(code_snippet: str, analysis_type: str = "general") -> str:
    """
    Phân tích code nhanh với cấu hình mặc định.
    
    Args:
        code_snippet: Code cần phân tích
        analysis_type: Loại phân tích
        
    Returns:
        Kết quả phân tích dưới dạng string
        
    Raises:
        OllamaAPIError: Khi gặp lỗi API
    """
    caller = create_llm_caller()
    response = caller.analyze_code(code_snippet, analysis_type)
    return response.response


# Example usage
if __name__ == "__main__":
    # Thiết lập logging
    logging.basicConfig(level=logging.INFO)
    
    # Tạo LLM caller
    llm = create_llm_caller(model=OllamaModel.CODELLAMA)
    
    # Kiểm tra health
    if not llm.check_health():
        print("❌ Ollama server không hoạt động!")
        exit(1)
    
    print("✅ Ollama server hoạt động bình thường")
    
    # Lấy danh sách models
    try:
        models = llm.list_models()
        print(f"📋 Models có sẵn: {models}")
    except OllamaAPIError as e:
        print(f"❌ Lỗi khi lấy danh sách models: {e.message}")
    
    # Test code analysis
    sample_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
    """
    
    try:
        print("\n🔍 Phân tích code sample...")
        result = llm.analyze_code(sample_code, "optimization")
        print(f"📝 Kết quả phân tích:\n{result.response}")
    except OllamaAPIError as e:
        print(f"❌ Lỗi khi phân tích code: {e.message}") 