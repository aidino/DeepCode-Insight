"""
Tests cho Complete LangGraph Workflow
Kiểm tra end-to-end workflow từ UserInteraction đến ReportingAgent
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ..core.graph import create_analysis_workflow, run_analysis_demo
from ..core.state import AgentState, DEFAULT_AGENT_STATE


class TestCompleteWorkflow:
    """Test complete analysis workflow"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory cho reports"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_python_code(self):
        """Sample Python code để analyze"""
        return '''
def calculate_sum(a, b):
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y
        
def very_long_function_name_that_exceeds_the_recommended_line_length_limit_and_should_be_flagged():
    pass
'''
    
    def test_complete_workflow_with_code_content(self, sample_python_code):
        """Test complete workflow với provided code content"""
        
        # Create graph
        graph = create_analysis_workflow()
        
        # Initial state với code content
        initial_state: AgentState = {
            **DEFAULT_AGENT_STATE,
            'code_content': sample_python_code,
            'filename': 'test_workflow.py'
        }
        
        # Mock LLM để avoid real API calls
        with patch('deepcode_insight.agents.llm_orchestrator.create_llm_orchestrator_agent') as mock_create_llm:
            mock_agent = Mock()
            mock_agent.check_llm_health.return_value = False  # Skip LLM
            mock_create_llm.return_value = mock_agent
            
            # Run workflow
            result = graph.invoke(initial_state)
            
            # Verify workflow completion
            assert result['finished'] == True
            assert result['processing_status'] == 'report_generated'
            
            # Verify each stage
            assert result['code_content'] == sample_python_code
            assert result['filename'] == 'test_workflow.py'
            assert 'static_analysis_results' in result
            assert result['llm_analysis'] is None  # Skipped due to mock
            assert 'report' in result
            
            # Verify static analysis results
            static_results = result['static_analysis_results']
            assert 'static_issues' in static_results
            assert 'metrics' in static_results
            assert 'suggestions' in static_results
            
            # Verify report generation
            report = result['report']
            assert 'filename' in report
            assert 'content' in report
            assert 'output_path' in report
            assert os.path.exists(report['output_path'])
    
    def test_workflow_with_llm_analysis(self, sample_python_code):
        """Test workflow với mocked LLM analysis"""
        
        graph = create_analysis_workflow()
        
        initial_state: AgentState = {
            **DEFAULT_AGENT_STATE,
            'code_content': sample_python_code,
            'filename': 'test_llm_workflow.py'
        }
        
        # Mock LLM với successful analysis
        with patch('deepcode_insight.agents.llm_orchestrator.create_llm_orchestrator_agent') as mock_create_llm:
            mock_agent = Mock()
            mock_agent.check_llm_health.return_value = True
            
            # Mock LLM analysis result
            mock_llm_result = {
                **initial_state,
                'current_agent': 'llm_orchestrator',
                'processing_status': 'llm_analysis_completed',
                'llm_analysis': {
                    'filename': 'test_llm_workflow.py',
                    'summary': 'Test LLM analysis summary',
                    'detailed_analysis': 'Detailed test analysis',
                    'priority_issues': [],
                    'recommendations': [],
                    'code_quality_assessment': 'Good quality',
                    'improvement_suggestions': [],
                    'llm_metadata': {
                        'model_used': 'test_model',
                        'analysis_type': 'test'
                    }
                }
            }
            
            mock_agent.process_findings.return_value = mock_llm_result
            mock_create_llm.return_value = mock_agent
            
            # Run workflow
            result = graph.invoke(initial_state)
            
            # Verify LLM analysis was included
            assert result['finished'] == True
            assert result['processing_status'] == 'report_generated'
            assert result['llm_analysis'] is not None
            assert result['llm_analysis']['summary'] == 'Test LLM analysis summary'
            
            # Verify report includes LLM analysis
            report_content = result['report']['content']
            assert 'Test LLM analysis summary' in report_content
            assert 'AI-Powered Analysis' in report_content
    
    def test_workflow_error_handling_no_input(self):
        """Test workflow error handling khi không có input"""
        
        graph = create_analysis_workflow()
        
        # Empty initial state
        initial_state: AgentState = {
            **DEFAULT_AGENT_STATE
            # No code_content or repo_url
        }
        
        # Run workflow
        result = graph.invoke(initial_state)
        
        # Should stop at user_interaction với error
        assert result['finished'] == True
        assert result['processing_status'] == 'error'
        assert 'Either repo_url or code_content must be provided' in result['error']
        assert result['current_agent'] == 'user_interaction'
    
    def test_workflow_error_handling_invalid_code(self):
        """Test workflow error handling với invalid code"""
        
        graph = create_analysis_workflow()
        
        # Initial state với empty code
        initial_state: AgentState = {
            **DEFAULT_AGENT_STATE,
            'code_content': '',  # Empty code
            'filename': 'empty.py'
        }
        
        # Run workflow
        result = graph.invoke(initial_state)
        
        # Should complete but với minimal results
        assert result['finished'] == True
        # Static analysis should handle empty code gracefully
        assert 'static_analysis_results' in result
    
    def test_workflow_with_repository_url(self):
        """Test workflow với repository URL (mocked)"""
        
        graph = create_analysis_workflow()
        
        initial_state: AgentState = {
            **DEFAULT_AGENT_STATE,
            'repo_url': 'https://github.com/test/repo',
            'target_file': 'test.py'
        }
        
        # Mock CodeFetcherAgent
        with patch('deepcode_insight.agents.code_fetcher.CodeFetcherAgent') as mock_fetcher_class:
            mock_fetcher = Mock()
            mock_fetcher.get_repository_info.return_value = {'name': 'test-repo'}
            mock_fetcher.list_repository_files.return_value = ['test.py', 'README.md']
            mock_fetcher.get_file_content.return_value = 'def test():\n    pass'
            mock_fetcher_class.return_value = mock_fetcher
            
            # Mock LLM để skip
            with patch('deepcode_insight.agents.llm_orchestrator.create_llm_orchestrator_agent') as mock_create_llm:
                mock_agent = Mock()
                mock_agent.check_llm_health.return_value = False
                mock_create_llm.return_value = mock_agent
                
                # Run workflow
                result = graph.invoke(initial_state)
                
                # Verify workflow completion
                assert result['finished'] == True
                assert result['processing_status'] == 'report_generated'
                assert result['code_content'] == 'def test():\n    pass'
                assert result['filename'] == 'test.py'
                assert result['repository_info']['name'] == 'test-repo'
                
                # Verify CodeFetcher was called
                mock_fetcher.get_repository_info.assert_called_once()
                mock_fetcher.get_file_content.assert_called_once()
                mock_fetcher.cleanup.assert_called_once()
    
    def test_workflow_state_management(self, sample_python_code):
        """Test state management throughout workflow"""
        
        graph = create_analysis_workflow()
        
        initial_state: AgentState = {
            **DEFAULT_AGENT_STATE,
            'code_content': sample_python_code,
            'filename': 'state_test.py',
            'config': {'test_mode': True}
        }
        
        # Mock LLM để skip
        with patch('deepcode_insight.agents.llm_orchestrator.create_llm_orchestrator_agent') as mock_create_llm:
            mock_agent = Mock()
            mock_agent.check_llm_health.return_value = False
            mock_create_llm.return_value = mock_agent
            
            # Run workflow
            result = graph.invoke(initial_state)
            
            # Verify state preservation
            assert result['config']['test_mode'] == True
            assert result['filename'] == 'state_test.py'
            
            # Verify state progression
            assert result['current_agent'] == 'reporter'  # Final agent
            assert result['processing_status'] == 'report_generated'
            
            # Verify all intermediate results are preserved
            assert 'code_content' in result
            assert 'static_analysis_results' in result
            assert 'report' in result
    
    def test_workflow_routing_logic(self, sample_python_code):
        """Test conditional routing logic"""
        
        graph = create_analysis_workflow()
        
        # Test successful path
        initial_state: AgentState = {
            **DEFAULT_AGENT_STATE,
            'code_content': sample_python_code,
            'filename': 'routing_test.py'
        }
        
        # Mock LLM để skip
        with patch('deepcode_insight.agents.llm_orchestrator.create_llm_orchestrator_agent') as mock_create_llm:
            mock_agent = Mock()
            mock_agent.check_llm_health.return_value = False
            mock_create_llm.return_value = mock_agent
            
            result = graph.invoke(initial_state)
            
            # Should go through all stages
            assert result['finished'] == True
            assert result['processing_status'] == 'report_generated'
            
        # Test error path
        error_state: AgentState = {
            **DEFAULT_AGENT_STATE
            # No input - should stop early
        }
        
        error_result = graph.invoke(error_state)
        
        # Should stop at user_interaction
        assert error_result['finished'] == True
        assert error_result['processing_status'] == 'error'
        assert error_result['current_agent'] == 'user_interaction'


class TestWorkflowDemo:
    """Test workflow demo functions"""
    
    def test_run_analysis_demo(self, capsys):
        """Test run_analysis_demo function"""
        
        # Mock LLM để avoid real calls
        with patch('deepcode_insight.agents.llm_orchestrator.create_llm_orchestrator_agent') as mock_create_llm:
            mock_agent = Mock()
            mock_agent.check_llm_health.return_value = False
            mock_create_llm.return_value = mock_agent
            
            # Run demo
            run_analysis_demo()
            
            # Capture output
            captured = capsys.readouterr()
            
            # Verify demo output
            assert "DeepCode-Insight Analysis Workflow" in captured.out
            assert "UserInteractionAgent" in captured.out
            assert "StaticAnalysisAgent" in captured.out
            assert "ReportingAgent" in captured.out
            assert "Workflow completed successfully" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 