"""
Tests cho LangGraph Demo với Pytest Markers
Sử dụng markers để phân loại tests: unit, integration, edge_case
"""

import pytest
import sys
import os

# Thêm src vào Python path để import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.state import AgentState
from src.graph import agent_1, agent_2, should_continue, create_graph


@pytest.mark.unit
class TestAgentFunctionsUnit:
    """Unit tests cho các agent functions"""
    
    def test_agent_1_basic_functionality(self):
        """Test basic functionality của Agent 1"""
        state: AgentState = {
            "messages": [],
            "current_agent": "agent_1",
            "message_count": 0,
            "finished": False
        }
        
        result = agent_1(state)
        
        assert len(result["messages"]) == 1
        assert "Xin chào từ Agent 1!" in result["messages"][0]
        assert result["current_agent"] == "agent_2"
        assert result["message_count"] == 1
    
    def test_agent_2_basic_functionality(self):
        """Test basic functionality của Agent 2"""
        state: AgentState = {
            "messages": ["Previous message"],
            "current_agent": "agent_2",
            "message_count": 1,
            "finished": False
        }
        
        result = agent_2(state)
        
        assert len(result["messages"]) == 2
        assert "Chào Agent 1!" in result["messages"][1]
        assert result["current_agent"] == "agent_1"
        assert result["message_count"] == 2


@pytest.mark.unit
class TestShouldContinueUnit:
    """Unit tests cho should_continue function"""
    
    def test_continue_logic(self):
        """Test logic tiếp tục conversation"""
        state: AgentState = {
            "messages": ["msg1"],
            "current_agent": "agent_2",
            "message_count": 1,
            "finished": False
        }
        
        result = should_continue(state)
        assert result == "agent_2"
    
    def test_end_logic(self):
        """Test logic kết thúc conversation"""
        state: AgentState = {
            "messages": ["msg1", "msg2", "msg3"],
            "current_agent": "agent_1",
            "message_count": 3,
            "finished": True
        }
        
        result = should_continue(state)
        assert result == "__end__"


@pytest.mark.integration
class TestFullWorkflow:
    """Integration tests cho toàn bộ workflow"""
    
    def test_end_to_end_conversation(self):
        """Test conversation từ đầu đến cuối"""
        graph = create_graph()
        
        initial_state: AgentState = {
            "messages": [],
            "current_agent": "agent_1",
            "message_count": 0,
            "finished": False
        }
        
        result = graph.invoke(initial_state)
        
        # Verify final state
        assert result["finished"] == True
        assert result["message_count"] == 5
        assert len(result["messages"]) == 5
    
    def test_conversation_flow(self):
        """Test luồng conversation có đúng pattern không"""
        graph = create_graph()
        
        initial_state: AgentState = {
            "messages": [],
            "current_agent": "agent_1",
            "message_count": 0,
            "finished": False
        }
        
        result = graph.invoke(initial_state)
        messages = result["messages"]
        
        # Verify alternating pattern
        assert "Xin chào từ Agent 1!" in messages[0]  # Agent 1 first
        assert "Chào Agent 1!" in messages[1]        # Agent 2 responds
        assert "Xin chào từ Agent 1!" in messages[2]  # Agent 1 again
        assert "Chào Agent 1!" in messages[3]        # Agent 2 responds  
        assert "Xin chào từ Agent 1!" in messages[4]  # Agent 1 final


@pytest.mark.edge_case
class TestEdgeCases:
    """Edge case tests"""
    
    def test_agent_with_empty_state(self):
        """Test agent với state trống"""
        empty_state = {}
        
        result = agent_1(empty_state)
        
        # Should handle empty state gracefully
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["message_count"] == 1
    
    def test_agent_with_large_message_count(self):
        """Test với message_count lớn"""
        state: AgentState = {
            "messages": ["msg"] * 100,
            "current_agent": "agent_1",
            "message_count": 100,
            "finished": False
        }
        
        result = agent_1(state)
        
        # Should still work correctly
        assert len(result["messages"]) == 101
        assert result["message_count"] == 101
        assert result["finished"] == True  # Should finish due to large count
    
    def test_should_continue_with_invalid_agent(self):
        """Test should_continue với invalid agent name"""
        state: AgentState = {
            "messages": [],
            "current_agent": "invalid_agent",
            "message_count": 0,
            "finished": False
        }
        
        result = should_continue(state)
        assert result == "invalid_agent"  # Should return whatever is set


# Utility function để chạy specific marker groups
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tests với specific markers')
    parser.add_argument('--marker', type=str, default='',
                       help='Marker để chạy (unit, integration, edge_case)')
    
    args = parser.parse_args()
    
    if args.marker:
        pytest.main([__file__, "-v", "-m", args.marker])
    else:
        pytest.main([__file__, "-v"]) 