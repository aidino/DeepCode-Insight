"""
Core interfaces and abstract base classes for DeepCode-Insight

Định nghĩa các interfaces chung để đảm bảo consistency và extensibility
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from enum import Enum


class AnalysisLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    UNKNOWN = "unknown"


class AnalysisResult:
    """Standardized analysis result container"""
    
    def __init__(self, 
                 filename: str,
                 language: AnalysisLanguage,
                 success: bool = True,
                 error: Optional[str] = None):
        self.filename = filename
        self.language = language
        self.success = success
        self.error = error
        self.issues: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
        self.suggestions: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_issue(self, issue_type: str, message: str, line: Optional[int] = None, 
                  severity: str = "warning", **kwargs):
        """Add an analysis issue"""
        issue = {
            'type': issue_type,
            'message': message,
            'line': line,
            'severity': severity,
            'filename': self.filename,
            **kwargs
        }
        self.issues.append(issue)
    
    def add_metric(self, name: str, value: Any):
        """Add a metric"""
        self.metrics[name] = value
    
    def add_suggestion(self, suggestion: str):
        """Add a suggestion"""
        self.suggestions.append(suggestion)


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str):
        self.name = name
        self._logger = None
    
    @property
    def logger(self):
        """Lazy logger initialization"""
        if self._logger is None:
            import logging
            self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger
    
    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the agent state and return updated state"""
        pass
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Validate input state - override in subclasses"""
        return True
    
    def handle_error(self, error: Exception, state: Dict[str, Any]) -> Dict[str, Any]:
        """Standard error handling"""
        self.logger.error(f"Error in {self.name}: {error}")
        state['error'] = str(error)
        state['processing_status'] = 'error'
        return state


class CodeAnalyzer(ABC):
    """Abstract base class for code analyzers"""
    
    @abstractmethod
    def analyze(self, code: str, filename: str) -> AnalysisResult:
        """Analyze code and return results"""
        pass
    
    @abstractmethod
    def supports_language(self, language: AnalysisLanguage) -> bool:
        """Check if analyzer supports the given language"""
        pass


class CodeParser(ABC):
    """Abstract base class for code parsers"""
    
    @abstractmethod
    def parse(self, code: str, language: AnalysisLanguage) -> Any:
        """Parse code and return AST or similar structure"""
        pass
    
    @abstractmethod
    def supports_language(self, language: AnalysisLanguage) -> bool:
        """Check if parser supports the given language"""
        pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available"""
        pass


class ReportGenerator(ABC):
    """Abstract base class for report generators"""
    
    @abstractmethod
    def generate_report(self, analysis_results: List[AnalysisResult], 
                       format_type: str = "markdown") -> str:
        """Generate a report from analysis results"""
        pass
    
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Return list of supported report formats"""
        pass


class ContextProvider(ABC):
    """Abstract base class for context providers (RAG, etc.)"""
    
    @abstractmethod
    def add_context(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Add content to the context store"""
        pass
    
    @abstractmethod
    def query_context(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Query the context store"""
        pass
    
    @abstractmethod
    def clear_context(self) -> bool:
        """Clear the context store"""
        pass


class ConfigurationError(Exception):
    """Configuration related errors"""
    pass


class AnalysisError(Exception):
    """Analysis related errors"""
    pass


class ParsingError(Exception):
    """Parsing related errors"""
    pass 