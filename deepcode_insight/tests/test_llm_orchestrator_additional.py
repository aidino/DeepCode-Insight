"""
Additional test cases cho LLMOrchestratorAgent
Covers edge cases và specific scenarios
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ..agents.llm_orchestrator import (
    LLMOrchestratorAgent, 
    create_llm_orchestrator_agent, 
    llm_orchestrator_node
)
from ..utils.llm_caller import OllamaModel, OllamaResponse, OllamaAPIError


class TestLLMOrchestratorEdgeCases:
    """Test edge cases và special scenarios"""
    
    @pytest.fixture
    def agent(self, mocker):
        """Create agent với mocked LLM caller"""
        mock_llm_caller = Mock()
        mock_llm_caller.model = "codellama"
        
        with patch('deepcode_insight.agents.llm_orchestrator.OllamaLLMCaller', return_value=mock_llm_caller):
            agent = LLMOrchestratorAgent()
        
        return agent
    
    def test_process_findings_empty_static_issues(self, agent):
        """Test processing với empty static issues"""
        state = {
            'static_analysis_results': {
                'filename': 'clean_code.py',
                'static_issues': {},  # No issues
                'metrics': {
                    'code_quality_score': 95,
                    'maintainability_index': 90
                },
                'suggestions': []
            },
            'code_content': 'def hello():\n    """Perfect function"""\n    return "Hello"',
            'filename': 'clean_code.py'
        }
        
        # Mock successful LLM responses
        mock_responses = [
            Mock(response="Code quality is excellent"),  # summary
            Mock(response="No critical issues found"),  # detailed
            Mock(response="No priority issues identified"),  # priority
            Mock(response="- Maintain current standards - Keep quality high - Low"),  # recommendations
            Mock(response="Code quality is excellent"),  # quality
            Mock(response="1. Continue good practices - Keep current approach - Maintain quality")  # improvements
        ]
        agent.llm_caller.generate.side_effect = mock_responses
        
        result = agent.process_findings(state)
        
        assert result['processing_status'] == 'llm_analysis_completed'
        assert result['llm_analysis']['filename'] == 'clean_code.py'
        assert result['llm_analysis']['summary'] == "Code quality is excellent"
    
    def test_process_findings_large_number_of_issues(self, agent):
        """Test processing với large number of issues"""
        # Create many issues
        many_issues = {
            'missing_docstrings': [
                {'type': 'missing_function_docstring', 'name': f'func{i}', 'line': i*5, 'message': f'Function func{i} missing docstring'}
                for i in range(20)
            ],
            'complex_functions': [
                {'type': 'high_complexity', 'name': f'complex_func{i}', 'line': i*10, 'complexity': 15+i, 'message': f'Function complex_func{i} too complex'}
                for i in range(15)
            ],
            'code_smells': [
                {'type': 'long_line', 'line': i*2, 'length': 120+i, 'message': f'Line {i*2} too long'}
                for i in range(25)
            ]
        }
        
        state = {
            'static_analysis_results': {
                'filename': 'problematic_code.py',
                'static_issues': many_issues,
                'metrics': {
                    'code_quality_score': 35,
                    'maintainability_index': 25
                },
                'suggestions': [f'Fix issue type {i}' for i in range(10)]
            },
            'code_content': 'def problematic_function():\n    pass',
            'filename': 'problematic_code.py'
        }
        
        # Mock LLM responses
        mock_responses = [
            Mock(response="Code has significant quality issues"),
            Mock(response="Multiple critical problems detected"),
            Mock(response="1. High Complexity - Multiple functions - Critical\n2. Missing Docs - Many functions - Important"),
            Mock(response="- Refactor immediately - Reduce complexity - High\n- Add documentation - Improve readability - Medium"),
            Mock(response="Code quality is poor, needs major refactoring"),
            Mock(response="1. Start with complexity - Break down functions - Critical priority")
        ]
        agent.llm_caller.generate.side_effect = mock_responses
        
        result = agent.process_findings(state)
        
        assert result['processing_status'] == 'llm_analysis_completed'
        assert 'significant quality issues' in result['llm_analysis']['summary']
        assert len(result['llm_analysis']['priority_issues']) >= 1
    
    def test_process_findings_malformed_state(self, agent):
        """Test processing với malformed state structure"""
        malformed_state = {
            'static_analysis_results': {
                'filename': 'test.py',
                'static_issues': {
                    'missing_docstrings': [
                        {'incomplete': 'data'}  # Missing required fields
                    ]
                },
                'metrics': {
                    'code_quality_score': 'invalid_score'  # Wrong type
                },
                'suggestions': 'not_a_list'  # Wrong type
            },
            'code_content': None,  # None instead of string
            'filename': 'test.py'
        }
        
        # Mock LLM responses that handle malformed data gracefully
        mock_responses = [
            Mock(response="Analysis completed despite data issues"),
            Mock(response="Some data formatting issues detected"),
            Mock(response="1. Data Quality - Malformed input - Fix data structure"),
            Mock(response="- Validate input data - Ensure proper formatting - High"),
            Mock(response="Data quality issues affect analysis"),
            Mock(response="1. Fix data validation - Add input checks - High priority")
        ]
        agent.llm_caller.generate.side_effect = mock_responses
        
        result = agent.process_findings(malformed_state)
        
        # Should still complete but may have warnings in analysis
        assert result['processing_status'] == 'llm_analysis_completed'
        assert result['llm_analysis']['filename'] == 'test.py'
    
    def test_analyze_findings_with_unicode_content(self, agent):
        """Test analysis với Unicode characters trong code"""
        unicode_static_results = {
            'filename': 'unicode_test.py',
            'static_issues': {
                'missing_docstrings': [
                    {'type': 'missing_function_docstring', 'name': 'tính_toán', 'line': 5, 'message': 'Hàm tính_toán thiếu docstring'}
                ]
            },
            'metrics': {
                'code_quality_score': 75,
                'maintainability_index': 70
            },
            'suggestions': ['Thêm docstring cho hàm tính_toán']
        }
        
        unicode_code = '''
def tính_toán(số_a, số_b):
    """Hàm tính toán với Unicode"""
    kết_quả = số_a + số_b
    return kết_quả

# Comment với tiếng Việt và emoji 🚀
class XửLýDữLiệu:
    def __init__(self):
        self.dữ_liệu = []
'''
        
        # Mock LLM responses
        mock_responses = [
            Mock(response="Code sử dụng Unicode characters properly"),
            Mock(response="Unicode handling looks good"),
            Mock(response="1. Unicode Support - Good implementation - Maintain standards"),
            Mock(response="- Continue Unicode best practices - Good internationalization - Low"),
            Mock(response="Unicode implementation is solid"),
            Mock(response="1. Maintain Unicode standards - Keep current approach - Good practice")
        ]
        agent.llm_caller.generate.side_effect = mock_responses
        
        result = agent.analyze_findings_with_llm(unicode_static_results, unicode_code, 'unicode_test.py')
        
        assert result['filename'] == 'unicode_test.py'
        assert 'Unicode' in result['summary']
    
    def test_analyze_findings_with_very_long_code(self, agent):
        """Test analysis với very long code content"""
        long_code = '\n'.join([f'def function_{i}():\n    return {i}' for i in range(1000)])
        
        static_results = {
            'filename': 'long_file.py',
            'static_issues': {
                'missing_docstrings': [
                    {'type': 'missing_function_docstring', 'name': f'function_{i}', 'line': i*2, 'message': f'Function function_{i} missing docstring'}
                    for i in range(100)  # Many missing docstrings
                ]
            },
            'metrics': {
                'code_quality_score': 40,
                'maintainability_index': 30,
                'lines_of_code': 2000
            },
            'suggestions': ['Add docstrings to all functions']
        }
        
        # Mock LLM responses
        mock_responses = [
            Mock(response="Large file with many functions needs documentation"),
            Mock(response="File is very large, consider splitting"),
            Mock(response="1. File Size - Too many functions - Split into modules\n2. Documentation - Missing docstrings - Add documentation"),
            Mock(response="- Split large file - Improve modularity - High\n- Add documentation - Improve readability - Medium"),
            Mock(response="File is too large and poorly documented"),
            Mock(response="1. Modularize code - Split into smaller files - High priority\n2. Add documentation - Use docstring templates - Medium")
        ]
        agent.llm_caller.generate.side_effect = mock_responses
        
        result = agent.analyze_findings_with_llm(static_results, long_code, 'long_file.py')
        
        assert result['filename'] == 'long_file.py'
        assert 'large' in result['summary'].lower() or 'many' in result['summary'].lower()
        assert len(result['priority_issues']) >= 1


class TestPromptFormattingEdgeCases:
    """Test prompt formatting edge cases"""
    
    @pytest.fixture
    def agent(self, mocker):
        """Create agent với mocked LLM caller"""
        mock_llm_caller = Mock()
        mock_llm_caller.model = "codellama"
        
        with patch('deepcode_insight.agents.llm_orchestrator.OllamaLLMCaller', return_value=mock_llm_caller):
            agent = LLMOrchestratorAgent()
        
        return agent
    
    def test_format_prompts_with_missing_metrics(self, agent):
        """Test prompt formatting khi metrics bị thiếu"""
        incomplete_results = {
            'static_issues': {
                'missing_docstrings': [
                    {'type': 'missing_function_docstring', 'name': 'func1', 'line': 5}
                ]
            },
            'metrics': {},  # Empty metrics
            'suggestions': ['Add docstrings']
        }
        
        # Test all prompt formatting methods
        summary_prompt = agent._format_summary_prompt(incomplete_results, 'test.py')
        assert 'test.py' in summary_prompt
        assert 'N/A' in summary_prompt  # Should handle missing metrics gracefully
        
        detailed_prompt = agent._format_detailed_analysis_prompt(incomplete_results, 'def func1(): pass', 'test.py')
        assert 'test.py' in detailed_prompt
        
        priority_prompt = agent._format_priority_issues_prompt(incomplete_results)
        assert 'missing_docstrings' in priority_prompt
        
        recommendations_prompt = agent._format_recommendations_prompt(incomplete_results)
        assert 'N/A' in recommendations_prompt
        
        quality_prompt = agent._format_quality_assessment_prompt(incomplete_results)
        assert 'N/A' in quality_prompt
        
        improvement_prompt = agent._format_improvement_suggestions_prompt(incomplete_results)
        assert 'Quality Score: N/A' in improvement_prompt
    
    def test_format_prompts_with_special_characters(self, agent):
        """Test prompt formatting với special characters"""
        special_results = {
            'static_issues': {
                'missing_docstrings': [
                    {'type': 'missing_function_docstring', 'name': 'func_with_"quotes"', 'line': 5, 'message': 'Function with "quotes" & <tags>'}
                ],
                'code_smells': [
                    {'type': 'special_chars', 'line': 10, 'message': 'Line contains: & < > " \' \\ / special chars'}
                ]
            },
            'metrics': {
                'code_quality_score': 75.5,
                'maintainability_index': 68.2
            },
            'suggestions': ['Fix "quoted" issues & <special> characters']
        }
        
        # Test that special characters don't break prompt formatting
        summary_prompt = agent._format_summary_prompt(special_results, 'special_chars.py')
        assert 'special_chars.py' in summary_prompt
        # Check that prompt is generated without errors (special chars handled)
        assert len(summary_prompt) > 100  # Should have substantial content
        
        detailed_prompt = agent._format_detailed_analysis_prompt(special_results, 'def func(): pass', 'special_chars.py')
        assert 'special_chars.py' in detailed_prompt
        assert 'special' in detailed_prompt
    
    def test_format_prompts_with_empty_issues(self, agent):
        """Test prompt formatting với empty issues lists"""
        empty_issues_results = {
            'static_issues': {
                'missing_docstrings': [],
                'complex_functions': [],
                'code_smells': [],
                'unused_imports': []
            },
            'metrics': {
                'code_quality_score': 95,
                'maintainability_index': 90
            },
            'suggestions': []
        }
        
        summary_prompt = agent._format_summary_prompt(empty_issues_results, 'clean_code.py')
        assert 'clean_code.py' in summary_prompt
        assert 'Total Issues: 0' in summary_prompt
        assert 'Không có issues được phát hiện' in summary_prompt
        
        priority_prompt = agent._format_priority_issues_prompt(empty_issues_results)
        assert 'Không có issues' in priority_prompt or 'No issues' in priority_prompt


class TestResponseParsingEdgeCases:
    """Test response parsing edge cases"""
    
    @pytest.fixture
    def agent(self, mocker):
        """Create agent với mocked LLM caller"""
        mock_llm_caller = Mock()
        mock_llm_caller.model = "codellama"
        
        with patch('deepcode_insight.agents.llm_orchestrator.OllamaLLMCaller', return_value=mock_llm_caller):
            agent = LLMOrchestratorAgent()
        
        return agent
    
    def test_parse_malformed_priority_issues(self, agent):
        """Test parsing malformed priority issues responses"""
        malformed_responses = [
            "No structured format here",
            "1. Missing separator here is the description",
            "- Also missing - proper - format - too - many - parts",
            "Some random text without any structure",
            "1. Good Format - Proper description - Valid reason\nBad line without format",
        ]
        
        for response in malformed_responses:
            issues = agent._parse_priority_issues(response)
            # Should handle gracefully, returning what it can parse
            assert isinstance(issues, list)
            # Should not crash, may return empty list or partial results
    
    def test_parse_mixed_format_responses(self, agent):
        """Test parsing responses với mixed formats"""
        mixed_response = """
1. First Issue - Description one - Reason one
- Second Issue - Description two - Reason two  
3. Third Issue - Description three - Reason three
• Fourth Issue - Description four - Reason four
"""
        
        issues = agent._parse_priority_issues(mixed_response)
        
        # Should parse what it can recognize
        assert isinstance(issues, list)
        assert len(issues) >= 2  # Should get at least some valid entries
        
        # Check that valid entries are parsed correctly
        valid_issues = [issue for issue in issues if issue.get('type') and issue.get('description')]
        assert len(valid_issues) >= 1
    
    def test_parse_unicode_responses(self, agent):
        """Test parsing responses với Unicode characters"""
        unicode_response = """
1. Vấn đề Unicode - Hàm tính_toán thiếu docstring - Cần thêm documentation
2. Performance Issue - Thuật toán chậm 🐌 - Optimize algorithm
3. Code Style - Sử dụng tên biến tiếng Việt - Consider English names
"""
        
        issues = agent._parse_priority_issues(unicode_response)
        
        assert len(issues) == 3
        assert issues[0]['type'] == 'Vấn đề Unicode'
        assert 'tính_toán' in issues[0]['description']
        assert '🐌' in issues[1]['description']
    
    def test_parse_very_long_responses(self, agent):
        """Test parsing very long responses"""
        long_response = "\n".join([
            f"{i}. Issue Type {i} - Very long description that goes on and on with lots of details about the specific problem found in the code - This is a very important issue that needs immediate attention"
            for i in range(1, 21)  # 20 issues
        ])
        
        issues = agent._parse_priority_issues(long_response)
        
        # Should limit to top 5 as designed
        assert len(issues) <= 5
        assert all(issue.get('type') and issue.get('description') for issue in issues)


class TestErrorHandlingScenarios:
    """Test various error handling scenarios"""
    
    @pytest.fixture
    def agent(self, mocker):
        """Create agent với mocked LLM caller"""
        mock_llm_caller = Mock()
        mock_llm_caller.model = "codellama"
        
        with patch('deepcode_insight.agents.llm_orchestrator.OllamaLLMCaller', return_value=mock_llm_caller):
            agent = LLMOrchestratorAgent()
        
        return agent
    
    def test_partial_llm_failure(self, agent):
        """Test khi first LLM call fails - should return error immediately"""
        static_results = {
            'filename': 'test.py',
            'static_issues': {
                'missing_docstrings': [
                    {'type': 'missing_function_docstring', 'name': 'func1', 'line': 5}
                ]
            },
            'metrics': {
                'code_quality_score': 75,
                'maintainability_index': 70
            },
            'suggestions': ['Add docstrings']
        }
        
        # First call fails - should return error immediately
        agent.llm_caller.generate.side_effect = OllamaAPIError("Summary analysis failed", 500)
        
        result = agent.analyze_findings_with_llm(static_results, "def func1(): pass", "test.py")
        
        # Should have error in result
        assert result['filename'] == 'test.py'
        assert 'error' in result
        assert 'LLM API error' in result['error']
        assert 'Summary analysis failed' in result['error']
    
    def test_timeout_handling(self, agent):
        """Test timeout handling"""
        import requests
        
        static_results = {
            'filename': 'test.py',
            'static_issues': {'missing_docstrings': []},
            'metrics': {'code_quality_score': 75},
            'suggestions': []
        }
        
        # Simulate timeout
        agent.llm_caller.generate.side_effect = requests.exceptions.Timeout("Request timed out")
        
        result = agent.analyze_findings_with_llm(static_results)
        
        assert 'error' in result
        assert 'timeout' in result['error'].lower() or 'timed out' in result['error'].lower()
    
    def test_json_decode_error_handling(self, agent):
        """Test JSON decode error handling"""
        static_results = {
            'filename': 'test.py',
            'static_issues': {'missing_docstrings': []},
            'metrics': {'code_quality_score': 75},
            'suggestions': []
        }
        
        # Simulate JSON decode error
        import json
        agent.llm_caller.generate.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)
        
        result = agent.analyze_findings_with_llm(static_results)
        
        assert 'error' in result
        assert 'json' in result['error'].lower() or 'decode' in result['error'].lower()


class TestPerformanceScenarios:
    """Test performance-related scenarios"""
    
    @pytest.fixture
    def agent(self, mocker):
        """Create agent với mocked LLM caller"""
        mock_llm_caller = Mock()
        mock_llm_caller.model = "codellama"
        
        with patch('deepcode_insight.agents.llm_orchestrator.OllamaLLMCaller', return_value=mock_llm_caller):
            agent = LLMOrchestratorAgent()
        
        return agent
    
    def test_large_state_processing(self, agent):
        """Test processing large state objects"""
        # Create a large state with many nested objects
        large_state = {
            'static_analysis_results': {
                'filename': 'large_file.py',
                'static_issues': {
                    f'issue_type_{i}': [
                        {
                            'type': f'specific_issue_{j}',
                            'name': f'item_{j}',
                            'line': j,
                            'message': f'Issue {j} in category {i}' * 10  # Long messages
                        }
                        for j in range(50)  # 50 issues per type
                    ]
                    for i in range(10)  # 10 issue types
                },
                'metrics': {
                    f'metric_{i}': i * 1.5 for i in range(100)  # Many metrics
                },
                'suggestions': [f'Suggestion {i}' * 5 for i in range(100)]  # Many suggestions
            },
            'code_content': 'def func(): pass\n' * 1000,  # Large code content
            'filename': 'large_file.py',
            'additional_data': {f'key_{i}': f'value_{i}' * 100 for i in range(100)}  # Extra data
        }
        
        # Mock quick responses to test processing speed
        mock_responses = [
            Mock(response="Large file processed"),
            Mock(response="Many issues detected"),
            Mock(response="1. Scale Issue - Too many problems - Refactor needed"),
            Mock(response="- Modularize code - Split into smaller files - High"),
            Mock(response="File is too complex"),
            Mock(response="1. Break down file - Use modules - Critical")
        ]
        agent.llm_caller.generate.side_effect = mock_responses
        
        # Should complete without performance issues
        result = agent.process_findings(large_state)
        
        assert result['processing_status'] == 'llm_analysis_completed'
        assert result['llm_analysis']['filename'] == 'large_file.py'
    
    def test_concurrent_processing_simulation(self, agent):
        """Test simulation of concurrent processing"""
        # Simulate multiple rapid calls
        states = [
            {
                'static_analysis_results': {
                    'filename': f'file_{i}.py',
                    'static_issues': {'missing_docstrings': [{'type': 'test', 'line': 1}]},
                    'metrics': {'code_quality_score': 70 + i},
                    'suggestions': [f'Fix file {i}']
                },
                'code_content': f'def func_{i}(): pass',
                'filename': f'file_{i}.py'
            }
            for i in range(5)
        ]
        
        # Mock responses for all calls
        agent.llm_caller.generate.side_effect = [
            Mock(response=f"Analysis for file {i//6}")
            for i in range(30)  # 6 calls per state * 5 states
        ]
        
        results = []
        for state in states:
            result = agent.process_findings(state)
            results.append(result)
        
        # All should complete successfully
        assert len(results) == 5
        assert all(r['processing_status'] == 'llm_analysis_completed' for r in results)
        assert all(f'file_{i}.py' in r['llm_analysis']['filename'] for i, r in enumerate(results))


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 