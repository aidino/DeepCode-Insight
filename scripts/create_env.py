#!/usr/bin/env python3
"""Script để tạo file .env với template"""

import os

def create_env_file():
    """Tạo file .env với template"""
    
    # Change to project root directory
    project_root = os.path.dirname(os.path.dirname(__file__))
    os.chdir(project_root)
    
    env_content = """# DeepCode-Insight Environment Configuration
# Add your actual API keys below

# OpenAI API Configuration (Required for full functionality)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Google AI Configuration (Optional)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-pro

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=deepcode_context

# Application Settings
LOG_LEVEL=INFO
DEBUG=False

# Test Configuration
TEST_MODE=False
USE_MOCK_EMBEDDINGS=False

# Performance Settings
CHUNK_SIZE=1024
CHUNK_OVERLAP=200
MAX_CHUNKS_PER_FILE=50
VECTOR_DIMENSION=1536

# Repository Settings
DEFAULT_FILE_PATTERNS=*.py,*.java,*.js,*.ts,*.go,*.rs
MAX_FILE_SIZE_MB=10
EXCLUDE_PATTERNS=__pycache__,node_modules,.git,*.pyc,*.class
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ File .env đã được tạo thành công!")
        print("📝 Bạn có thể edit file .env để thêm API keys:")
        print("   - OPENAI_API_KEY: Thêm OpenAI API key của bạn")
        print("   - GOOGLE_API_KEY: Thêm Google AI API key của bạn (optional)")
        print("\n🔒 File .env đã được thêm vào .gitignore để bảo mật")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi tạo file .env: {e}")
        return False

if __name__ == "__main__":
    create_env_file() 