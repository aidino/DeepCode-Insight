"""
LangGraph Demo với hai agents truyền message cho nhau
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from .state import AgentState


def agent_1(state: AgentState) -> Dict[str, Any]:
    """
    Agent 1: Nhận message và xử lý, sau đó gửi lại cho Agent 2
    """
    messages = state.get("messages", [])
    message_count = state.get("message_count", 0)
    
    print(f"🤖 Agent 1 đang xử lý message số {message_count + 1}")
    
    # Thêm message mới từ Agent 1
    new_message = f"Xin chào từ Agent 1! (Lần thứ {message_count + 1})"
    messages.append(new_message)
    
    print(f"📝 Agent 1 nói: {new_message}")
    
    return {
        "messages": messages,
        "current_agent": "agent_2",
        "message_count": message_count + 1,
        "finished": message_count >= 4  # Dừng sau 5 lần trao đổi
    }


def agent_2(state: AgentState) -> Dict[str, Any]:
    """
    Agent 2: Nhận message từ Agent 1 và phản hồi
    """
    messages = state.get("messages", [])
    message_count = state.get("message_count", 0)
    
    print(f"🦾 Agent 2 đang xử lý message số {message_count + 1}")
    
    # Thêm message mới từ Agent 2
    new_message = f"Chào Agent 1! Tôi đã nhận được tin nhắn của bạn. (Lần thứ {message_count + 1})"
    messages.append(new_message)
    
    print(f"📝 Agent 2 trả lời: {new_message}")
    
    return {
        "messages": messages,
        "current_agent": "agent_1",
        "message_count": message_count + 1,
        "finished": message_count >= 4  # Dừng sau 5 lần trao đổi
    }


def should_continue(state: AgentState) -> str:
    """
    Quyết định xem có tiếp tục hay dừng lại
    """
    if state.get("finished", False):
        print("🏁 Kết thúc cuộc trò chuyện!")
        return END
    
    current_agent = state.get("current_agent", "agent_1")
    print(f"➡️  Chuyển đến {current_agent}")
    return current_agent


def create_graph() -> StateGraph:
    """
    Tạo và cấu hình LangGraph workflow
    """
    # Khởi tạo graph với AgentState
    workflow = StateGraph(AgentState)
    
    # Thêm các nodes (agents)
    workflow.add_node("agent_1", agent_1)
    workflow.add_node("agent_2", agent_2)
    
    # Thiết lập entry point
    workflow.add_edge(START, "agent_1")
    
    # Thêm conditional edge để quyết định luồng
    workflow.add_conditional_edges(
        "agent_1",
        should_continue,
        {
            "agent_2": "agent_2",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "agent_2", 
        should_continue,
        {
            "agent_1": "agent_1",
            END: END
        }
    )
    
    # Compile graph
    return workflow.compile()


def run_demo():
    """
    Chạy demo với initial state
    """
    print("🚀 Bắt đầu LangGraph Demo - Hai Agents Trò Chuyện")
    print("=" * 50)
    
    # Tạo graph
    graph = create_graph()
    
    # Initial state
    initial_state: AgentState = {
        "messages": [],
        "current_agent": "agent_1",
        "message_count": 0,
        "finished": False
    }
    
    # Chạy graph
    result = graph.invoke(initial_state)
    
    print("\n" + "=" * 50)
    print("📋 Tóm tắt cuộc trò chuyện:")
    for i, message in enumerate(result["messages"], 1):
        print(f"{i}. {message}")
    
    print(f"\n📊 Tổng số message: {result['message_count']}")
    print("✅ Demo hoàn thành!")


if __name__ == "__main__":
    run_demo() 