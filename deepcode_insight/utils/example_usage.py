#!/usr/bin/env python3
"""
Example usage của OllamaLLMCaller.
Demo các chức năng chính của module llm_caller.
"""

import os
import sys
import logging
from pathlib import Path

# Thêm project root vào Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.llm_caller import (
    OllamaLLMCaller,
    OllamaModel,
    OllamaAPIError,
    create_llm_caller,
    quick_analyze_code
)

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_basic_usage():
    """Demo cách sử dụng cơ bản"""
    print("🚀 Demo Basic Usage")
    print("=" * 50)
    
    try:
        # Tạo LLM caller với model mặc định
        llm = create_llm_caller(model=OllamaModel.CODELLAMA)
        
        # Kiểm tra health
        print("🔍 Kiểm tra Ollama server...")
        if not llm.check_health():
            print("❌ Ollama server không hoạt động!")
            print("💡 Hãy đảm bảo Ollama đã được cài đặt và chạy:")
            print("   - Cài đặt: https://ollama.ai/")
            print("   - Chạy: ollama serve")
            print("   - Pull model: ollama pull codellama")
            return False
        
        print("✅ Ollama server hoạt động bình thường")
        
        # Lấy danh sách models
        print("\n📋 Lấy danh sách models...")
        models = llm.list_models()
        print(f"Models có sẵn: {models}")
        
        return True
        
    except OllamaAPIError as e:
        print(f"❌ Lỗi API: {e.message}")
        if e.status_code:
            print(f"   Status code: {e.status_code}")
        return False
    except Exception as e:
        print(f"❌ Lỗi không mong đợi: {str(e)}")
        return False


def demo_code_analysis():
    """Demo phân tích code"""
    print("\n🔍 Demo Code Analysis")
    print("=" * 50)
    
    # Sample code để phân tích
    sample_codes = {
        "Python - Fibonacci (có vấn đề performance)": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Sử dụng
result = fibonacci(30)
print(f"Fibonacci(30) = {result}")
        """,
        
        "Python - Bubble Sort": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# Test
numbers = [64, 34, 25, 12, 22, 11, 90]
sorted_numbers = bubble_sort(numbers)
print(sorted_numbers)
        """,
        
        "JavaScript - Async Function": """
async function fetchUserData(userId) {
    const response = await fetch(`/api/users/${userId}`);
    const userData = await response.json();
    return userData;
}

// Sử dụng
fetchUserData(123).then(user => {
    console.log(user.name);
});
        """
    }
    
    try:
        llm = create_llm_caller(model=OllamaModel.CODELLAMA)
        
        for title, code in sample_codes.items():
            print(f"\n📝 Phân tích: {title}")
            print("-" * 40)
            
            # Phân tích optimization
            print("🔧 Phân tích optimization:")
            result = llm.analyze_code(code, "optimization", "python" if "Python" in title else "javascript")
            print(f"{result.response[:300]}...")
            
            print(f"\n⏱️  Thời gian xử lý: {result.total_duration/1000000:.2f}ms" if result.total_duration else "")
            print(f"📊 Tokens generated: {result.eval_count}" if result.eval_count else "")
            
    except OllamaAPIError as e:
        print(f"❌ Lỗi khi phân tích code: {e.message}")
    except Exception as e:
        print(f"❌ Lỗi không mong đợi: {str(e)}")


def demo_chat_interface():
    """Demo chat interface"""
    print("\n💬 Demo Chat Interface")
    print("=" * 50)
    
    try:
        llm = create_llm_caller(model=OllamaModel.CODELLAMA)
        
        # Conversation messages
        messages = [
            {
                "role": "system",
                "content": "Bạn là một chuyên gia lập trình Python. Hãy trả lời ngắn gọn và hữu ích."
            },
            {
                "role": "user", 
                "content": "Làm thế nào để tối ưu hóa performance của Python code?"
            }
        ]
        
        print("🤖 Gửi chat message...")
        response = llm.chat(messages, temperature=0.7)
        
        print(f"💭 Response:\n{response.response}")
        print(f"\n⏱️  Thời gian: {response.total_duration/1000000:.2f}ms" if response.total_duration else "")
        
    except OllamaAPIError as e:
        print(f"❌ Lỗi chat: {e.message}")
    except Exception as e:
        print(f"❌ Lỗi không mong đợi: {str(e)}")


def demo_quick_functions():
    """Demo các convenience functions"""
    print("\n⚡ Demo Quick Functions")
    print("=" * 50)
    
    sample_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)
    """
    
    try:
        print("🔍 Sử dụng quick_analyze_code()...")
        result = quick_analyze_code(sample_code, "bugs")
        print(f"📝 Kết quả:\n{result[:200]}...")
        
    except OllamaAPIError as e:
        print(f"❌ Lỗi: {e.message}")
    except Exception as e:
        print(f"❌ Lỗi không mong đợi: {str(e)}")


def demo_error_handling():
    """Demo error handling"""
    print("\n🚨 Demo Error Handling")
    print("=" * 50)
    
    # Test với model không tồn tại
    try:
        print("🧪 Test với model không tồn tại...")
        llm = OllamaLLMCaller(model="nonexistent-model")
        llm.generate("Hello world")
        
    except OllamaAPIError as e:
        print(f"✅ Caught expected error: {e.message}")
        print(f"   Status code: {e.status_code}")
    
    # Test với URL không hợp lệ
    try:
        print("\n🧪 Test với URL không hợp lệ...")
        llm = OllamaLLMCaller(base_url="http://invalid-url:9999")
        llm.check_health()
        
    except OllamaAPIError as e:
        print(f"✅ Caught expected error: {e.message}")


def demo_environment_variables():
    """Demo sử dụng environment variables"""
    print("\n🌍 Demo Environment Variables")
    print("=" * 50)
    
    print("📋 Environment variables được hỗ trợ:")
    print("   - OLLAMA_BASE_URL: URL của Ollama server")
    print("   - OLLAMA_API_KEY: API key (nếu cần)")
    
    print(f"\n🔍 Giá trị hiện tại:")
    print(f"   OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL', 'Not set (sử dụng default)')}")
    print(f"   OLLAMA_API_KEY: {'Set' if os.getenv('OLLAMA_API_KEY') else 'Not set'}")
    
    print(f"\n💡 Để thiết lập:")
    print(f"   export OLLAMA_BASE_URL=http://your-ollama-server:11434")
    print(f"   export OLLAMA_API_KEY=your-api-key")


def main():
    """Main function để chạy tất cả demos"""
    print("🎯 OllamaLLMCaller Demo")
    print("=" * 60)
    
    # Demo basic usage trước
    if not demo_basic_usage():
        print("\n❌ Không thể kết nối đến Ollama server. Dừng demo.")
        return
    
    # Chạy các demos khác
    demo_environment_variables()
    demo_code_analysis()
    demo_chat_interface()
    demo_quick_functions()
    demo_error_handling()
    
    print("\n✅ Demo hoàn thành!")
    print("\n📚 Để sử dụng trong code của bạn:")
    print("```python")
    print("from utils.llm_caller import create_llm_caller, OllamaModel")
    print("")
    print("# Tạo LLM caller")
    print("llm = create_llm_caller(model=OllamaModel.CODELLAMA)")
    print("")
    print("# Phân tích code")
    print("result = llm.analyze_code(your_code, 'optimization')")
    print("print(result.response)")
    print("```")


if __name__ == "__main__":
    main() 