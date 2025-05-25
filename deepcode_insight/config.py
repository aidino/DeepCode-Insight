"""Configuration management cho DeepCode-Insight"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    """Configuration class để quản lý tất cả settings"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    OPENAI_EMBEDDING_MODEL: str = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
    
    # Google API Configuration
    GOOGLE_API_KEY: Optional[str] = os.getenv('GOOGLE_API_KEY')
    GOOGLE_MODEL: str = os.getenv('GOOGLE_MODEL', 'gemini-pro')
    
    # Qdrant Configuration
    QDRANT_HOST: str = os.getenv('QDRANT_HOST', 'localhost')
    QDRANT_PORT: int = int(os.getenv('QDRANT_PORT', '6333'))
    QDRANT_COLLECTION: str = os.getenv('QDRANT_COLLECTION', 'deepcode_context')
    
    # Application Settings
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Test Configuration
    TEST_MODE: bool = os.getenv('TEST_MODE', 'False').lower() == 'true'
    USE_MOCK_EMBEDDINGS: bool = os.getenv('USE_MOCK_EMBEDDINGS', 'False').lower() == 'true'
    
    # Performance Settings
    CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', '1024'))
    CHUNK_OVERLAP: int = int(os.getenv('CHUNK_OVERLAP', '200'))
    MAX_CHUNKS_PER_FILE: int = int(os.getenv('MAX_CHUNKS_PER_FILE', '50'))
    VECTOR_DIMENSION: int = int(os.getenv('VECTOR_DIMENSION', '1536'))
    
    # Repository Settings
    DEFAULT_FILE_PATTERNS: list = os.getenv('DEFAULT_FILE_PATTERNS', '*.py,*.java,*.js,*.ts,*.go,*.rs').split(',')
    MAX_FILE_SIZE_MB: int = int(os.getenv('MAX_FILE_SIZE_MB', '10'))
    EXCLUDE_PATTERNS: list = os.getenv('EXCLUDE_PATTERNS', '__pycache__,node_modules,.git,*.pyc,*.class').split(',')
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        errors = []
        
        if not cls.OPENAI_API_KEY and not cls.USE_MOCK_EMBEDDINGS:
            errors.append("OPENAI_API_KEY is required when not using mock embeddings")
        
        if cls.QDRANT_PORT < 1 or cls.QDRANT_PORT > 65535:
            errors.append("QDRANT_PORT must be between 1 and 65535")
        
        if cls.CHUNK_SIZE < 100:
            errors.append("CHUNK_SIZE must be at least 100")
        
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration (hiding sensitive data)"""
        print("🔧 DeepCode-Insight Configuration:")
        print(f"  OpenAI API Key: {'✅ Set' if cls.OPENAI_API_KEY else '❌ Not set'}")
        print(f"  OpenAI Model: {cls.OPENAI_MODEL}")
        print(f"  OpenAI Embedding Model: {cls.OPENAI_EMBEDDING_MODEL}")
        print(f"  Google API Key: {'✅ Set' if cls.GOOGLE_API_KEY else '❌ Not set'}")
        print(f"  Google Model: {cls.GOOGLE_MODEL}")
        print(f"  Qdrant Host: {cls.QDRANT_HOST}:{cls.QDRANT_PORT}")
        print(f"  Qdrant Collection: {cls.QDRANT_COLLECTION}")
        print(f"  Log Level: {cls.LOG_LEVEL}")
        print(f"  Debug Mode: {cls.DEBUG}")
        print(f"  Test Mode: {cls.TEST_MODE}")
        print(f"  Use Mock Embeddings: {cls.USE_MOCK_EMBEDDINGS}")
        print(f"  Chunk Size: {cls.CHUNK_SIZE}")
        print(f"  Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"  Vector Dimension: {cls.VECTOR_DIMENSION}")

# Global config instance
config = Config()

def setup_environment():
    """Setup environment variables nếu chưa có"""
    env_template = """# DeepCode-Insight Environment Configuration
# Copy this content to a .env file and fill in your actual API keys

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Google API Configuration  
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
    
    if not os.path.exists('.env'):
        print("📝 Creating .env template file...")
        with open('.env.template', 'w') as f:
            f.write(env_template)
        print("✅ Created .env.template - copy to .env and fill in your API keys")
    
    return env_template

if __name__ == "__main__":
    setup_environment()
    config.print_config()
    if config.validate():
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors") 