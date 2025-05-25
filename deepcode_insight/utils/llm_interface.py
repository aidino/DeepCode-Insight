"""
Abstract interface cho LLM providers để hỗ trợ Ollama, OpenAI, Gemini APIs
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

# Third-party imports
import requests
import openai
import google.generativeai as genai

# Local imports
from .llm_caller import OllamaLLMCaller, OllamaModel, OllamaResponse, OllamaAPIError


logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Enum cho các LLM providers được hỗ trợ"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class LLMResponse:
    """Unified response format cho tất cả LLM providers"""
    response: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """Abstract base class cho tất cả LLM providers"""
    
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate(self, 
                prompt: str,
                system_prompt: Optional[str] = None,
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                **kwargs) -> LLMResponse:
        """Generate response từ prompt"""
        pass
    
    @abstractmethod
    def check_health(self) -> bool:
        """Check if LLM service is available"""
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """Get list of available models"""
        pass


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider implementation"""
    
    def __init__(self, 
                 model: str = "codellama",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 120,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.ollama_caller = OllamaLLMCaller(
            base_url=base_url,
            model=model,
            timeout=timeout
        )
    
    def generate(self, 
                prompt: str,
                system_prompt: Optional[str] = None,
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                **kwargs) -> LLMResponse:
        """Generate response using Ollama"""
        try:
            # Combine system prompt with main prompt if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            
            ollama_response = self.ollama_caller.generate(
                prompt=full_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return LLMResponse(
                response=ollama_response.response,
                model=self.model,
                provider="ollama",
                usage={
                    "prompt_tokens": ollama_response.prompt_eval_count,
                    "completion_tokens": ollama_response.eval_count,
                    "total_tokens": (ollama_response.prompt_eval_count or 0) + (ollama_response.eval_count or 0)
                },
                metadata={
                    "total_duration": ollama_response.total_duration,
                    "load_duration": ollama_response.load_duration
                }
            )
        except OllamaAPIError as e:
            self.logger.error(f"Ollama API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Ollama provider error: {e}")
            raise
    
    def check_health(self) -> bool:
        """Check Ollama service health"""
        return self.ollama_caller.check_health()
    
    def list_models(self) -> List[str]:
        """List available Ollama models"""
        return self.ollama_caller.list_models()


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation"""
    
    def __init__(self, 
                 model: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 **kwargs):
        super().__init__(model, **kwargs)
        
        # Set API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def generate(self, 
                prompt: str,
                system_prompt: Optional[str] = None,
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                **kwargs) -> LLMResponse:
        """Generate response using OpenAI"""
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return LLMResponse(
                response=response.choices[0].message.content,
                model=self.model,
                provider="openai",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id
                }
            )
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
    
    def check_health(self) -> bool:
        """Check OpenAI service health"""
        try:
            # Simple test call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            self.logger.error(f"OpenAI health check failed: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """List available OpenAI models"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data if "gpt" in model.id]
        except Exception as e:
            self.logger.error(f"Failed to list OpenAI models: {e}")
            return []


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider implementation"""
    
    def __init__(self, 
                 model: str = "gemini-pro",
                 api_key: Optional[str] = None,
                 **kwargs):
        super().__init__(model, **kwargs)
        
        # Set API key
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model)
    
    def generate(self, 
                prompt: str,
                system_prompt: Optional[str] = None,
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                **kwargs) -> LLMResponse:
        """Generate response using Gemini"""
        try:
            # Combine system prompt with main prompt if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            return LLMResponse(
                response=response.text,
                model=self.model,
                provider="gemini",
                usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else None,
                    "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else None,
                    "total_tokens": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else None
                },
                metadata={
                    "finish_reason": response.candidates[0].finish_reason if response.candidates else None
                }
            )
        except Exception as e:
            self.logger.error(f"Gemini API error: {e}")
            raise
    
    def check_health(self) -> bool:
        """Check Gemini service health"""
        try:
            # Simple test call
            response = self.client.generate_content("Hello")
            return True
        except Exception as e:
            self.logger.error(f"Gemini health check failed: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """List available Gemini models"""
        try:
            models = genai.list_models()
            return [model.name for model in models if "gemini" in model.name.lower()]
        except Exception as e:
            self.logger.error(f"Failed to list Gemini models: {e}")
            return []


class LLMProviderFactory:
    """Factory class để tạo LLM providers"""
    
    @staticmethod
    def create_provider(provider: Union[str, LLMProvider], 
                       model: str,
                       **kwargs) -> BaseLLMProvider:
        """
        Create LLM provider instance
        
        Args:
            provider: Provider type (ollama, openai, gemini)
            model: Model name
            **kwargs: Additional provider-specific arguments
            
        Returns:
            BaseLLMProvider instance
        """
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        
        if provider == LLMProvider.OLLAMA:
            return OllamaProvider(model=model, **kwargs)
        elif provider == LLMProvider.OPENAI:
            return OpenAIProvider(model=model, **kwargs)
        elif provider == LLMProvider.GEMINI:
            return GeminiProvider(model=model, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")


# Convenience functions
def create_llm_provider(provider: str = "ollama",
                       model: str = "codellama",
                       **kwargs) -> BaseLLMProvider:
    """Convenience function để tạo LLM provider"""
    return LLMProviderFactory.create_provider(provider, model, **kwargs)


def get_available_providers() -> List[str]:
    """Get list of available LLM providers"""
    return [provider.value for provider in LLMProvider] 