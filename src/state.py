"""
State định nghĩa cho LangGraph Demo
Định nghĩa cấu trúc dữ liệu được truyền giữa các nodes trong graph.
"""

from typing import TypedDict, List
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """
    State được sử dụng bởi graph để lưu trữ thông tin 
    giữa các lần thực thi của các nodes/agents.
    """
    # Danh sách các message được truyền giữa agents
    messages: List[str]
    
    # Thông tin về agent hiện tại đang xử lý
    current_agent: str
    
    # Số lượng lần message đã được xử lý
    message_count: int
    
    # Flag để biết khi nào dừng
    finished: bool 