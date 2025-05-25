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

__all__ = [
    "OllamaLLMCaller",
    "OllamaModel", 
    "OllamaResponse",
    "OllamaAPIError",
    "create_llm_caller",
    "quick_analyze_code"
]

__version__ = "1.0.0" 