"""
Utils package cho DeepCode-Insight project.
Chứa các utility functions và classes hỗ trợ.
"""

from .llm_caller import (
    OllamaLLMCaller,
    OllamaModel,
    OllamaResponse,
    OllamaAPIError,
    create_llm_caller,
    quick_analyze_code
)

from .llm_interface import (
    BaseLLMProvider,
    LLMProvider,
    LLMResponse,
    LLMProviderFactory,
    create_llm_provider
)

__all__ = [
    # LLM Caller (Ollama)
    "OllamaLLMCaller",
    "OllamaModel", 
    "OllamaResponse",
    "OllamaAPIError",
    "create_llm_caller",
    "quick_analyze_code",
    
    # LLM Interface (Multi-provider)
    "BaseLLMProvider",
    "LLMProvider",
    "LLMResponse", 
    "LLMProviderFactory",
    "create_llm_provider",
]

__version__ = "1.0.0" 