"""
Tests cho LangGraph Demo
Kiểm tra logic của graph và state management.
"""

import pytest
import sys
import os

# Thêm src vào Python path để import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ..core.state import AgentState
from ..core.graph import agent_1, agent_2, should_continue, create_graph


class TestAgentFunctions:
    """Test các agent functions riêng lẻ"""
    
    def test_agent_1_initial_state(self):
        """Test Agent 1 với initial state"""
        initial_state: AgentState = {
            "messages": [],
            "current_agent": "agent_1",
            "message_count": 0,
            "finished": False
        }
        
        result = agent_1(initial_state)
        
        # Kiểm tra message được thêm vào
        assert len(result["messages"]) == 1
        assert "Xin chào từ Agent 1! (Lần thứ 1)" in result["messages"][0]
        
        # Kiểm tra state updates
        assert result["current_agent"] == "agent_2"
        assert result["message_count"] == 1
        assert result["finished"] == False
    
    def test_agent_2_receives_message(self):
        """Test Agent 2 nhận và phản hồi message"""
        state_with_message: AgentState = {
            "messages": ["Xin chào từ Agent 1! (Lần thứ 1)"],
            "current_agent": "agent_2",
            "message_count": 1,
            "finished": False
        }
        
        result = agent_2(state_with_message)
        
        # Kiểm tra Agent 2 thêm message
        assert len(result["messages"]) == 2
        assert "Chào Agent 1!" in result["messages"][1]
        
        # Kiểm tra state updates
        assert result["current_agent"] == "agent_1"
        assert result["message_count"] == 2
        assert result["finished"] == False
    
    def test_agent_1_finishing_condition(self):
        """Test Agent 1 khi đạt điều kiện kết thúc"""
        state_near_end: AgentState = {
            "messages": ["msg1", "msg2", "msg3", "msg4"],
            "current_agent": "agent_1",
            "message_count": 4,
            "finished": False
        }
        
        result = agent_1(state_near_end)
        
        # Kiểm tra finished flag được set
        assert result["finished"] == True
        assert result["message_count"] == 5
        assert len(result["messages"]) == 5
    
    def test_agent_2_finishing_condition(self):
        """Test Agent 2 khi đạt điều kiện kết thúc"""
        state_near_end: AgentState = {
            "messages": ["msg1", "msg2", "msg3", "msg4"],
            "current_agent": "agent_2",
            "message_count": 4,
            "finished": False
        }
        
        result = agent_2(state_near_end)
        
        # Kiểm tra finished flag được set
        assert result["finished"] == True
        assert result["message_count"] == 5
        assert len(result["messages"]) == 5


class TestShouldContinue:
    """Test function should_continue logic"""
    
    def test_should_continue_when_not_finished(self):
        """Test tiếp tục khi chưa finished"""
        state: AgentState = {
            "messages": ["msg1"],
            "current_agent": "agent_2",
            "message_count": 1,
            "finished": False
        }
        
        result = should_continue(state)
        assert result == "agent_2"
    
    def test_should_end_when_finished(self):
        """Test kết thúc khi finished = True"""
        state: AgentState = {
            "messages": ["msg1", "msg2", "msg3", "msg4", "msg5"],
            "current_agent": "agent_1",
            "message_count": 5,
            "finished": True
        }
        
        result = should_continue(state)
        assert result == "__end__"  # END constant value
    
    def test_default_agent_fallback(self):
        """Test fallback khi không có current_agent"""
        state: AgentState = {
            "messages": [],
            "current_agent": "",
            "message_count": 0,
            "finished": False
        }
        
        result = should_continue(state)
        assert result == ""  # Returns empty string when current_agent is empty


class TestFullGraph:
    """Test toàn bộ graph workflow"""
    
    def test_complete_workflow(self):
        """Test workflow hoàn chỉnh từ đầu đến cuối"""
        graph = create_graph()
        
        initial_state: AgentState = {
            "messages": [],
            "current_agent": "agent_1",
            "message_count": 0,
            "finished": False
        }
        
        # Chạy graph
        result = graph.invoke(initial_state)
        
        # Kiểm tra kết quả cuối cùng
        assert result["finished"] == True
        assert result["message_count"] == 5
        assert len(result["messages"]) == 5
        
        # Kiểm tra messages xen kẽ giữa agent 1 và 2
        agent_1_messages = [msg for msg in result["messages"] if "Xin chào từ Agent 1" in msg]
        agent_2_messages = [msg for msg in result["messages"] if "Chào Agent 1! Tôi đã nhận được" in msg]
        
        assert len(agent_1_messages) == 3  # Agent 1 nói 3 lần
        assert len(agent_2_messages) == 2  # Agent 2 trả lời 2 lần
    
    def test_message_alternating_pattern(self):
        """Test pattern của messages xen kẽ giữa các agents"""
        graph = create_graph()
        
        initial_state: AgentState = {
            "messages": [],
            "current_agent": "agent_1", 
            "message_count": 0,
            "finished": False
        }
        
        result = graph.invoke(initial_state)
        
        # Kiểm tra pattern: Agent1, Agent2, Agent1, Agent2, Agent1
        messages = result["messages"]
        
        assert "Agent 1" in messages[0]  # Đầu tiên là Agent 1
        assert "Chào Agent 1" in messages[1]  # Sau đó Agent 2 trả lời
        assert "Agent 1" in messages[2]  # Lại Agent 1
        assert "Chào Agent 1" in messages[3]  # Lại Agent 2
        assert "Agent 1" in messages[4]  # Cuối cùng Agent 1
    
    def test_message_count_increments(self):
        """Test message_count tăng đều qua từng step"""
        graph = create_graph()
        
        initial_state: AgentState = {
            "messages": [],
            "current_agent": "agent_1",
            "message_count": 0,
            "finished": False
        }
        
        result = graph.invoke(initial_state)
        
        # Kiểm tra từng message có số thứ tự đúng
        for i, message in enumerate(result["messages"], 1):
            assert f"(Lần thứ {i})" in message
    
    def test_graph_state_consistency(self):
        """Test tính nhất quán của state qua các bước"""
        graph = create_graph()
        
        initial_state: AgentState = {
            "messages": [],
            "current_agent": "agent_1",
            "message_count": 0,
            "finished": False
        }
        
        result = graph.invoke(initial_state)
        
        # Kiểm tra state cuối cùng nhất quán
        assert len(result["messages"]) == result["message_count"]
        assert result["finished"] == True
        
        # Kiểm tra không có messages trống
        assert all(len(msg.strip()) > 0 for msg in result["messages"])
        
        # Kiểm tra tất cả messages đều có format đúng
        for msg in result["messages"]:
            assert "Lần thứ" in msg
            assert "(" in msg and ")" in msg


class TestEdgeCases:
    """Test các edge cases"""
    
    def test_empty_messages_list(self):
        """Test với danh sách messages trống"""
        state: AgentState = {
            "messages": [],
            "current_agent": "agent_1",
            "message_count": 0,
            "finished": False
        }
        
        result = agent_1(state)
        assert len(result["messages"]) == 1
        assert result["message_count"] == 1
    
    def test_state_missing_keys(self):
        """Test với state thiếu một số keys (sử dụng defaults)"""
        minimal_state = {}
        
        # Agent functions should handle missing keys gracefully
        result = agent_1(minimal_state)
        
        assert "messages" in result
        assert "current_agent" in result
        assert "message_count" in result
        assert "finished" in result


if __name__ == "__main__":
    # Chạy tests khi file được execute trực tiếp
    pytest.main([__file__, "-v"]) 