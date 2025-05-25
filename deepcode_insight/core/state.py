"""
State định nghĩa cho DeepCode-Insight LangGraph Workflow
Định nghĩa cấu trúc dữ liệu được truyền giữa các agents trong complete analysis pipeline.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """
    State được sử dụng bởi graph để lưu trữ thông tin 
    giữa các lần thực thi của các agents trong analysis workflow.
    """
    # ===== Input Parameters =====
    # Repository URL để analyze
    repo_url: Optional[str]
    
    # Pull Request ID (optional)
    pr_id: Optional[str]
    
    # Target file path (for single file analysis)
    target_file: Optional[str]
    
    # ===== Processing State =====
    # Agent hiện tại đang xử lý
    current_agent: str
    
    # Status của processing pipeline
    processing_status: str  # 'started', 'code_fetched', 'static_analysis_completed', etc.
    
    # Error information nếu có
    error: Optional[str]
    
    # ===== Code Content =====
    # Raw code content được fetch
    code_content: Optional[str]
    
    # Filename của file đang analyze
    filename: Optional[str]
    
    # Repository information
    repository_info: Optional[Dict[str, Any]]
    
    # PR diff information (nếu có)
    pr_diff: Optional[Dict[str, Any]]
    
    # ===== Analysis Results =====
    # Kết quả từ StaticAnalysisAgent
    static_analysis_results: Optional[Dict[str, Any]]
    
    # Kết quả từ LLMOrchestratorAgent
    llm_analysis: Optional[Dict[str, Any]]
    
    # ===== Final Output =====
    # Report được generate bởi ReportingAgent
    report: Optional[Dict[str, Any]]
    
    # ===== Workflow Control =====
    # Flag để biết khi nào workflow hoàn thành
    finished: bool
    
    # Workflow configuration
    config: Optional[Dict[str, Any]]


class SimpleAgentState(TypedDict):
    """
    Simplified state cho basic demo workflow (backward compatibility)
    """
    # Danh sách các message được truyền giữa agents
    messages: List[str]
    
    # Thông tin về agent hiện tại đang xử lý
    current_agent: str
    
    # Số lượng lần message đã được xử lý
    message_count: int
    
    # Flag để biết khi nào dừng
    finished: bool


# Default state values
DEFAULT_AGENT_STATE = {
    'repo_url': None,
    'pr_id': None,
    'target_file': None,
    'current_agent': 'user_interaction',
    'processing_status': 'initialized',
    'error': None,
    'code_content': None,
    'filename': None,
    'repository_info': None,
    'pr_diff': None,
    'static_analysis_results': None,
    'llm_analysis': None,
    'report': None,
    'finished': False,
    'config': None
} 