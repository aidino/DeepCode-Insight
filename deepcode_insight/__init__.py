"""
DeepCode-Insight: AI-Powered Code Analysis Tool

A comprehensive code analysis tool using LangGraph workflow
for static analysis, LLM-powered insights, and professional reporting.
"""

__version__ = "1.0.0"
__author__ = "DeepCode-Insight Team"

from .core.state import AgentState, DEFAULT_AGENT_STATE
from .core.graph import create_analysis_workflow

__all__ = [
    "AgentState",
    "DEFAULT_AGENT_STATE", 
    "create_analysis_workflow"
] 