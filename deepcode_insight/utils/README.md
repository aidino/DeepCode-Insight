# Utils Package

Package chứa các utility functions và classes hỗ trợ cho DeepCode-Insight project.

## Modules

### `llm_caller.py`

Module để gọi API Ollama local (CodeLlama) qua HTTP, hỗ trợ gửi prompt và code snippet với xử lý API keys và errors.

## Cài đặt Dependencies

Đảm bảo bạn đã cài đặt các dependencies cần thiết:

```bash
pip install requests
```

## Thiết lập Ollama

1. **Cài đặt Ollama:**
   ```bash
   # macOS
   brew install ollama
   
   # Hoặc download từ https://ollama.ai/
   ```

2. **Chạy Ollama server:**
   ```bash
   ollama serve
   ```

3. **Pull CodeLlama model:**
   ```bash
   ollama pull codellama
   # Hoặc các variants khác:
   # ollama pull codellama:7b
   # ollama pull codellama:13b
   # ollama pull codellama:34b
   ```

## Sử dụng

### Basic Usage

```python
from utils.llm_caller import create_llm_caller, OllamaModel

# Tạo LLM caller
llm = create_llm_caller(model=OllamaModel.CODELLAMA)

# Kiểm tra server
if llm.check_health():
    print("✅ Ollama server hoạt động")
else:
    print("❌ Ollama server không hoạt động")
```

### Phân tích Code

```python
code_snippet = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

# Phân tích optimization
result = llm.analyze_code(code_snippet, "optimization", "python")
print(result.response)

# Các loại phân tích khác:
# - "general": Phân tích tổng quát
# - "bugs": Tìm lỗi tiềm ẩn
# - "optimization": Đề xuất tối ưu hóa
# - "documentation": Tạo documentation
```

### Chat Interface

```python
messages = [
    {"role": "system", "content": "Bạn là chuyên gia Python"},
    {"role": "user", "content": "Làm thế nào để tối ưu hóa Python code?"}
]

response = llm.chat(messages)
print(response.response)
```

### Generate với Prompt tùy chỉnh

```python
response = llm.generate(
    prompt="Explain this code:",
    code_snippet=your_code,
    system_prompt="You are a code expert",
    temperature=0.7
)
print(response.response)
```

### Quick Functions

```python
from utils.llm_caller import quick_analyze_code

# Phân tích nhanh
result = quick_analyze_code(code_snippet, "bugs")
print(result)
```

## Environment Variables

Bạn có thể cấu hình thông qua environment variables:

```bash
# URL của Ollama server (mặc định: http://localhost:11434)
export OLLAMA_BASE_URL=http://your-server:11434

# API key (nếu cần thiết)
export OLLAMA_API_KEY=your-api-key
```

## Error Handling

Module cung cấp custom exception `OllamaAPIError` để xử lý lỗi:

```python
from utils.llm_caller import OllamaAPIError

try:
    result = llm.analyze_code(code)
except OllamaAPIError as e:
    print(f"API Error: {e.message}")
    if e.status_code:
        print(f"Status Code: {e.status_code}")
    if e.response_text:
        print(f"Response: {e.response_text}")
```

## Models được hỗ trợ

```python
from utils.llm_caller import OllamaModel

# Các models có sẵn:
OllamaModel.CODELLAMA      # codellama (mặc định)
OllamaModel.CODELLAMA_7B   # codellama:7b
OllamaModel.CODELLAMA_13B  # codellama:13b
OllamaModel.CODELLAMA_34B  # codellama:34b
OllamaModel.LLAMA2         # llama2
OllamaModel.LLAMA2_7B      # llama2:7b
OllamaModel.LLAMA2_13B     # llama2:13b
```

## Response Object

`OllamaResponse` dataclass chứa thông tin chi tiết về response:

```python
response = llm.generate("Hello")

print(response.response)          # Nội dung response
print(response.model)             # Model được sử dụng
print(response.total_duration)    # Thời gian xử lý (nanoseconds)
print(response.eval_count)        # Số tokens generated
print(response.done)              # Trạng thái hoàn thành
```

## Demo và Examples

Chạy file example để xem demo đầy đủ:

```bash
python utils/example_usage.py
```

## Troubleshooting

### Ollama server không hoạt động
- Đảm bảo Ollama đã được cài đặt và chạy: `ollama serve`
- Kiểm tra port 11434 có bị chiếm không
- Kiểm tra firewall settings

### Model không tìm thấy
- Pull model trước khi sử dụng: `ollama pull codellama`
- Kiểm tra danh sách models: `ollama list`

### Connection timeout
- Tăng timeout parameter khi khởi tạo OllamaLLMCaller
- Kiểm tra network connectivity

### Memory issues
- Sử dụng model nhỏ hơn (7B thay vì 34B)
- Đảm bảo đủ RAM cho model

## API Reference

### Classes

- **`OllamaLLMCaller`**: Main class để gọi Ollama API
- **`OllamaModel`**: Enum các models được hỗ trợ
- **`OllamaResponse`**: Dataclass chứa response data
- **`OllamaAPIError`**: Custom exception cho API errors

### Functions

- **`create_llm_caller()`**: Tạo LLM caller với cấu hình mặc định
- **`quick_analyze_code()`**: Phân tích code nhanh

### Methods

- **`generate()`**: Generate text từ prompt
- **`chat()`**: Chat interface với messages
- **`analyze_code()`**: Phân tích code với prompts tối ưu
- **`check_health()`**: Kiểm tra server health
- **`list_models()`**: Lấy danh sách models có sẵn 