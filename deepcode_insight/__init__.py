"""
DeepCode-Insight: AI-Powered Code Analysis Tool

A comprehensive code analysis tool using LangGraph workflow
for static analysis, LLM-powered insights, and professional reporting.
"""

__version__ = "1.0.0"
__author__ = "DeepCode-Insight Team"
__email__ = "team@deepcode-insight.com"

# Core components
from .core.state import AgentState, DEFAULT_AGENT_STATE
from .core.graph import create_analysis_workflow

# Main agents
from .agents.static_analyzer import StaticAnalysisAgent
from .agents.rag_context import RAGContextAgent
from .agents.llm_orchestrator import LLMOrchestratorAgent
from .agents.diagram_generator import DiagramGenerationAgent
from .agents.solution_suggester import SolutionSuggestionAgent
from .agents.reporter import ReportingAgent
from .agents.code_fetcher import CodeFetcherAgent

# Parsers
from .parsers.ast_parser import ASTParsingAgent

# Utils
from .utils.llm_interface import create_llm_provider, LLMProvider
from .utils.llm_caller import create_llm_caller, OllamaModel

__all__ = [
    # Core
    "AgentState",
    "DEFAULT_AGENT_STATE", 
    "create_analysis_workflow",
    
    # Agents
    "StaticAnalysisAgent",
    "RAGContextAgent", 
    "LLMOrchestratorAgent",
    "DiagramGenerationAgent",
    "SolutionSuggestionAgent",
    "ReportingAgent",
    "CodeFetcherAgent",
    
    # Parsers
    "ASTParsingAgent",
    
    # Utils
    "create_llm_provider",
    "LLMProvider",
    "create_llm_caller",
    "OllamaModel",
] 