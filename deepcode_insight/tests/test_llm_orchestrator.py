"""
Test suite cho LLMOrchestratorAgent
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


class TestLLMOrchestratorAgent:
    """Test LLMOrchestratorAgent initialization và basic functionality"""
    
    def test_init_default_values(self, mocker):
        """Test initialization với default values"""
        mock_ollama_caller = mocker.patch('deepcode_insight.agents.llm_orchestrator.OllamaLLMCaller')
        
        agent = LLMOrchestratorAgent()
        
        mock_ollama_caller.assert_called_once_with(
            model=OllamaModel.CODELLAMA,
            base_url="http://localhost:11434",
            timeout=120
        )
    
    def test_init_custom_values(self, mocker):
        """Test initialization với custom values"""
        mock_ollama_caller = mocker.patch('deepcode_insight.agents.llm_orchestrator.OllamaLLMCaller')
        
        agent = LLMOrchestratorAgent(
            model=OllamaModel.LLAMA2,
            base_url="http://custom:8080",
            timeout=60
        )
        
        mock_ollama_caller.assert_called_once_with(
            model=OllamaModel.LLAMA2,
            base_url="http://custom:8080",
            timeout=60
        )
    
    def test_init_string_model(self, mocker):
        """Test initialization với string model"""
        mock_ollama_caller = mocker.patch('deepcode_insight.agents.llm_orchestrator.OllamaLLMCaller')
        
        agent = LLMOrchestratorAgent(model="custom-model")
        
        mock_ollama_caller.assert_called_once_with(
            model="custom-model",
            base_url="http://localhost:11434",
            timeout=120
        )
    
    def test_init_failure(self, mocker):
        """Test initialization failure"""
        mock_ollama_caller = mocker.patch(
            'agents.llm_orchestrator.OllamaLLMCaller',
            side_effect=Exception("Connection failed")
        )
        
        with pytest.raises(Exception) as exc_info:
            LLMOrchestratorAgent()
        
        assert "Connection failed" in str(exc_info.value)


class TestProcessFindings:
    """Test process_findings method"""
    
    @pytest.fixture
    def agent(self, mocker):
        """Create agent với mocked LLM caller"""
        mock_llm_caller = Mock()
        mock_llm_caller.model = "codellama"
        
        with patch('deepcode_insight.agents.llm_orchestrator.OllamaLLMCaller', return_value=mock_llm_caller):
            agent = LLMOrchestratorAgent()
        
        return agent
    
    @pytest.fixture
    def sample_state(self):
        """Sample LangGraph state"""
        return {
            'static_analysis_results': {
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
            },
            'code_content': 'def func1():\n    pass',
            'filename': 'test.py',
            'current_agent': 'static_analyzer'
        }
    
    def test_process_findings_success(self, agent, sample_state, mocker):
        """Test successful processing của findings"""
        # Mock analyze_findings_with_llm method
        mock_analysis = {
            'filename': 'test.py',
            'summary': 'Code needs improvement',
            'priority_issues': [{'type': 'documentation', 'description': 'Missing docstrings'}],
            'recommendations': [{'action': 'Add docstrings', 'effort': 'Low'}]
        }
        
        mocker.patch.object(agent, 'analyze_findings_with_llm', return_value=mock_analysis)
        
        result = agent.process_findings(sample_state)
        
        assert result['llm_analysis'] == mock_analysis
        assert result['current_agent'] == 'llm_orchestrator'
        assert result['processing_status'] == 'llm_analysis_completed'
    
    def test_process_findings_no_static_results(self, agent):
        """Test processing khi không có static analysis results"""
        state = {'filename': 'test.py'}
        
        result = agent.process_findings(state)
        
        assert 'llm_analysis' in result
        assert result['llm_analysis']['status'] == 'failed'
        assert result['processing_status'] == 'llm_analysis_failed'
        assert 'No static analysis results' in result['llm_analysis']['error']
    
    def test_process_findings_exception(self, agent, sample_state, mocker):
        """Test processing khi có exception"""
        mocker.patch.object(
            agent, 
            'analyze_findings_with_llm', 
            side_effect=Exception("LLM error")
        )
        
        result = agent.process_findings(sample_state)
        
        assert 'llm_analysis' in result
        assert result['llm_analysis']['status'] == 'failed'
        assert result['processing_status'] == 'llm_analysis_failed'
        assert 'LLM error' in result['llm_analysis']['error']


class TestAnalyzeFindingsWithLLM:
    """Test analyze_findings_with_llm method"""
    
    @pytest.fixture
    def agent(self, mocker):
        """Create agent với mocked LLM caller"""
        mock_llm_caller = Mock()
        mock_llm_caller.model = "codellama"
        
        with patch('deepcode_insight.agents.llm_orchestrator.OllamaLLMCaller', return_value=mock_llm_caller):
            agent = LLMOrchestratorAgent()
        
        return agent
    
    @pytest.fixture
    def sample_static_results(self):
        """Sample static analysis results"""
        return {
            'filename': 'test.py',
            'static_issues': {
                'missing_docstrings': [
                    {'type': 'missing_function_docstring', 'name': 'func1', 'line': 5, 'message': 'Missing docstring'}
                ],
                'complex_functions': [
                    {'type': 'high_complexity', 'name': 'func2', 'line': 10, 'complexity': 15, 'message': 'High complexity'}
                ]
            },
            'metrics': {
                'code_quality_score': 75.5,
                'maintainability_index': 68.2,
                'cyclomatic_complexity': 12,
                'lines_of_code': 85
            },
            'suggestions': ['Add docstrings', 'Refactor complex functions']
        }
    
    def test_analyze_findings_success(self, agent, sample_static_results, mocker):
        """Test successful analysis"""
        # Mock LLM responses
        mock_responses = [
            Mock(response="Code needs improvement in documentation"),  # summary
            Mock(response="Detailed analysis shows complexity issues"),  # detailed
            Mock(response="1. High Complexity - func2 - Critical issue\n2. Missing Docs - func1 - Important"),  # priority
            Mock(response="- Add docstrings - Improve readability - Low\n- Refactor functions - Reduce complexity - Medium"),  # recommendations
            Mock(response="Code quality is fair, needs improvement"),  # quality
            Mock(response="1. Add type hints - Use typing module - Better IDE support")  # improvements
        ]
        
        agent.llm_caller.generate.side_effect = mock_responses
        
        result = agent.analyze_findings_with_llm(sample_static_results, "def func1(): pass", "test.py")
        
        assert result['filename'] == 'test.py'
        assert result['summary'] == "Code needs improvement in documentation"
        assert result['detailed_analysis'] == "Detailed analysis shows complexity issues"
        assert len(result['priority_issues']) == 2
        assert len(result['recommendations']) == 2
        assert result['code_quality_assessment'] == "Code quality is fair, needs improvement"
        assert len(result['improvement_suggestions']) == 1
        assert result['llm_metadata']['model_used'] == "codellama"
    
    def test_analyze_findings_without_code_content(self, agent, sample_static_results, mocker):
        """Test analysis without code content"""
        mock_responses = [
            Mock(response="Summary response"),
            Mock(response="Priority issues response"),
            Mock(response="Recommendations response"),
            Mock(response="Quality assessment response"),
            Mock(response="Improvement suggestions response")
        ]
        
        agent.llm_caller.generate.side_effect = mock_responses
        
        result = agent.analyze_findings_with_llm(sample_static_results)
        
        # Should not call detailed analysis (no code content)
        assert result['detailed_analysis'] == ''
        assert agent.llm_caller.generate.call_count == 5  # Not 6 (no detailed analysis)
    
    def test_analyze_findings_llm_api_error(self, agent, sample_static_results, mocker):
        """Test analysis khi có LLM API error"""
        agent.llm_caller.generate.side_effect = OllamaAPIError("API Error", 500)
        
        result = agent.analyze_findings_with_llm(sample_static_results)
        
        assert 'error' in result
        assert 'LLM API error' in result['error']
    
    def test_analyze_findings_general_exception(self, agent, sample_static_results, mocker):
        """Test analysis khi có general exception"""
        agent.llm_caller.generate.side_effect = Exception("General error")
        
        result = agent.analyze_findings_with_llm(sample_static_results)
        
        assert 'error' in result
        assert 'Analysis error' in result['error']


class TestPromptFormatting:
    """Test prompt formatting methods"""
    
    @pytest.fixture
    def agent(self, mocker):
        """Create agent với mocked LLM caller"""
        mock_llm_caller = Mock()
        mock_llm_caller.model = "codellama"
        
        with patch('deepcode_insight.agents.llm_orchestrator.OllamaLLMCaller', return_value=mock_llm_caller):
            agent = LLMOrchestratorAgent()
        
        return agent
    
    @pytest.fixture
    def sample_static_results(self):
        """Sample static analysis results"""
        return {
            'static_issues': {
                'missing_docstrings': [
                    {'type': 'missing_function_docstring', 'name': 'func1', 'line': 5, 'message': 'Missing docstring'}
                ],
                'code_smells': [
                    {'type': 'long_line', 'line': 10, 'length': 150, 'message': 'Line too long'}
                ]
            },
            'metrics': {
                'code_quality_score': 75,
                'maintainability_index': 70,
                'cyclomatic_complexity': 8,
                'lines_of_code': 100
            },
            'suggestions': ['Add docstrings', 'Fix long lines']
        }
    
    def test_format_summary_prompt(self, agent, sample_static_results):
        """Test summary prompt formatting"""
        prompt = agent._format_summary_prompt(sample_static_results, 'test.py')
        
        assert 'test.py' in prompt
        assert 'Total Issues: 2' in prompt
        assert 'Missing Docstrings: 1 issues' in prompt
        assert 'Code Smells: 1 issues' in prompt
        assert 'Quality Score: 75/100' in prompt
        assert 'Maintainability Index: 70/100' in prompt
    
    def test_format_summary_prompt_no_issues(self, agent):
        """Test summary prompt với no issues"""
        empty_results = {
            'static_issues': {},
            'metrics': {'code_quality_score': 95},
            'suggestions': []
        }
        
        prompt = agent._format_summary_prompt(empty_results, 'clean.py')
        
        assert 'clean.py' in prompt
        assert 'Total Issues: 0' in prompt
        assert 'Không có issues được phát hiện' in prompt
    
    def test_format_detailed_analysis_prompt(self, agent, sample_static_results):
        """Test detailed analysis prompt formatting"""
        code = "def func1():\n    pass"
        prompt = agent._format_detailed_analysis_prompt(sample_static_results, code, 'test.py')
        
        assert 'test.py' in prompt
        assert 'missing_docstrings' in prompt  # Issue type, not specific type
        assert 'code_smells' in prompt  # Issue type, not specific type
        assert 'Code structure và organization' in prompt
    
    def test_format_priority_issues_prompt(self, agent, sample_static_results):
        """Test priority issues prompt formatting"""
        prompt = agent._format_priority_issues_prompt(sample_static_results)
        
        assert 'tech lead' in prompt
        assert 'missing_docstrings' in prompt  # Issue type, not specific type
        assert 'code_smells' in prompt  # Issue type, not specific type
        assert 'top 5 priority issues' in prompt
    
    def test_format_recommendations_prompt(self, agent, sample_static_results):
        """Test recommendations prompt formatting"""
        prompt = agent._format_recommendations_prompt(sample_static_results)
        
        assert 'software architect' in prompt
        assert 'Quality Score: 75/100' in prompt
        assert 'Add docstrings' in prompt
        assert '5-7 actionable recommendations' in prompt
    
    def test_format_quality_assessment_prompt(self, agent, sample_static_results):
        """Test quality assessment prompt formatting"""
        prompt = agent._format_quality_assessment_prompt(sample_static_results)
        
        assert 'code quality expert' in prompt
        assert 'Overall Score: 75/100' in prompt
        assert 'Total Issues: 2' in prompt
        assert 'Overall code quality level' in prompt
    
    def test_format_improvement_suggestions_prompt(self, agent, sample_static_results):
        """Test improvement suggestions prompt formatting"""
        prompt = agent._format_improvement_suggestions_prompt(sample_static_results)
        
        assert 'senior developer mentor' in prompt
        assert 'Documentation' in prompt  # Problem area
        assert 'Quality Score: 75/100' in prompt
        assert '5-6 specific improvement actions' in prompt


class TestResponseParsing:
    """Test response parsing methods"""
    
    @pytest.fixture
    def agent(self, mocker):
        """Create agent với mocked LLM caller"""
        mock_llm_caller = Mock()
        mock_llm_caller.model = "codellama"
        
        with patch('deepcode_insight.agents.llm_orchestrator.OllamaLLMCaller', return_value=mock_llm_caller):
            agent = LLMOrchestratorAgent()
        
        return agent
    
    def test_parse_priority_issues(self, agent):
        """Test parsing priority issues response"""
        response = """
1. High Complexity - Function 'process_data' có complexity cao - Critical for maintainability
2. Missing Documentation - 2 functions thiếu docstring - Affects readability
3. Code Smell - Long line detected - Style violation
"""
        
        issues = agent._parse_priority_issues(response)
        
        assert len(issues) == 3
        assert issues[0]['type'] == 'High Complexity'
        assert issues[0]['description'] == "Function 'process_data' có complexity cao"
        assert issues[0]['reason'] == 'Critical for maintainability'
        assert issues[1]['type'] == 'Missing Documentation'
        assert issues[2]['type'] == 'Code Smell'
    
    def test_parse_priority_issues_with_dashes(self, agent):
        """Test parsing với dash format"""
        response = """
- Security Issue - SQL injection vulnerability - High risk
- Performance - Inefficient algorithm - Medium impact
"""
        
        issues = agent._parse_priority_issues(response)
        
        assert len(issues) == 2
        assert issues[0]['type'] == 'Security Issue'
        assert issues[1]['type'] == 'Performance'
    
    def test_parse_recommendations(self, agent):
        """Test parsing recommendations response"""
        response = """
- Refactor complex functions - Reduce complexity below 10 - Medium
- Add comprehensive docstrings - Improve documentation - Low
- Implement unit tests - Increase code coverage - High
"""
        
        recommendations = agent._parse_recommendations(response)
        
        assert len(recommendations) == 3
        assert recommendations[0]['action'] == 'Refactor complex functions'
        assert recommendations[0]['impact'] == 'Reduce complexity below 10'
        assert recommendations[0]['effort'] == 'Medium'
        assert recommendations[1]['effort'] == 'Low'
        assert recommendations[2]['effort'] == 'High'
    
    def test_parse_improvement_suggestions(self, agent):
        """Test parsing improvement suggestions response"""
        response = """
1. Add type hints - Use Python typing module - Better IDE support
2. Implement error handling - Add try-catch blocks - Increased robustness
3. Setup pre-commit hooks - Automate quality checks - Consistent standards
"""
        
        suggestions = agent._parse_improvement_suggestions(response)
        
        assert len(suggestions) == 3
        assert suggestions[0]['action'] == 'Add type hints'
        assert suggestions[0]['implementation'] == 'Use Python typing module'
        assert suggestions[0]['benefit'] == 'Better IDE support'
        assert suggestions[1]['action'] == 'Implement error handling'
        assert suggestions[2]['action'] == 'Setup pre-commit hooks'
    
    def test_parse_empty_response(self, agent):
        """Test parsing empty hoặc invalid response"""
        empty_response = ""
        invalid_response = "No structured data here"
        
        assert agent._parse_priority_issues(empty_response) == []
        assert agent._parse_recommendations(invalid_response) == []
        assert agent._parse_improvement_suggestions(empty_response) == []


class TestSeverityEstimation:
    """Test severity estimation logic"""
    
    @pytest.fixture
    def agent(self, mocker):
        """Create agent với mocked LLM caller"""
        mock_llm_caller = Mock()
        
        with patch('deepcode_insight.agents.llm_orchestrator.OllamaLLMCaller', return_value=mock_llm_caller):
            agent = LLMOrchestratorAgent()
        
        return agent
    
    def test_estimate_severity_known_types(self, agent):
        """Test severity estimation cho known issue types"""
        test_cases = [
            ('syntax_error', {}, 10),
            ('security_issue', {}, 9),
            ('high_complexity', {}, 8),
            ('god_class', {}, 7),
            ('missing_docstring', {}, 3),
            ('unused_import', {}, 2),
            ('long_line', {}, 1)
        ]
        
        for issue_type, issue_data, expected_severity in test_cases:
            severity = agent._estimate_severity(issue_type, issue_data)
            assert severity == expected_severity
    
    def test_estimate_severity_unknown_type(self, agent):
        """Test severity estimation cho unknown issue type"""
        severity = agent._estimate_severity('unknown_issue', {})
        assert severity == 4  # Default severity
    
    def test_estimate_severity_with_complexity_adjustment(self, agent):
        """Test severity adjustment based on complexity"""
        issue_with_high_complexity = {'complexity': 20}
        severity = agent._estimate_severity('high_complexity', issue_with_high_complexity)
        assert severity == 10  # 8 + 2 for high complexity
    
    def test_estimate_severity_with_count_adjustment(self, agent):
        """Test severity adjustment based on count"""
        issue_with_high_count = {'count': 15}
        severity = agent._estimate_severity('too_many_parameters', issue_with_high_count)
        assert severity == 7  # 6 + 1 for high count


class TestHealthAndUtilities:
    """Test health check và utility methods"""
    
    @pytest.fixture
    def agent(self, mocker):
        """Create agent với mocked LLM caller"""
        mock_llm_caller = Mock()
        
        with patch('deepcode_insight.agents.llm_orchestrator.OllamaLLMCaller', return_value=mock_llm_caller):
            agent = LLMOrchestratorAgent()
        
        return agent
    
    def test_check_llm_health_success(self, agent):
        """Test successful health check"""
        agent.llm_caller.check_health.return_value = True
        
        assert agent.check_llm_health() is True
        agent.llm_caller.check_health.assert_called_once()
    
    def test_check_llm_health_failure(self, agent):
        """Test failed health check"""
        agent.llm_caller.check_health.return_value = False
        
        assert agent.check_llm_health() is False
    
    def test_check_llm_health_exception(self, agent):
        """Test health check với exception"""
        agent.llm_caller.check_health.side_effect = Exception("Connection error")
        
        assert agent.check_llm_health() is False
    
    def test_get_available_models_success(self, agent):
        """Test successful model listing"""
        expected_models = ['codellama', 'llama2', 'mistral']
        agent.llm_caller.list_models.return_value = expected_models
        
        models = agent.get_available_models()
        
        assert models == expected_models
        agent.llm_caller.list_models.assert_called_once()
    
    def test_get_available_models_exception(self, agent):
        """Test model listing với exception"""
        agent.llm_caller.list_models.side_effect = Exception("API error")
        
        models = agent.get_available_models()
        
        assert models == []


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_create_llm_orchestrator_agent_default(self, mocker):
        """Test create_llm_orchestrator_agent với default parameters"""
        mock_agent_class = mocker.patch('deepcode_insight.agents.llm_orchestrator.LLMOrchestratorAgent')
        
        agent = create_llm_orchestrator_agent()
        
        mock_agent_class.assert_called_once_with(
            model=OllamaModel.CODELLAMA,
            base_url="http://localhost:11434",
            timeout=120
        )
    
    def test_create_llm_orchestrator_agent_custom(self, mocker):
        """Test create_llm_orchestrator_agent với custom parameters"""
        mock_agent_class = mocker.patch('deepcode_insight.agents.llm_orchestrator.LLMOrchestratorAgent')
        
        agent = create_llm_orchestrator_agent(
            model=OllamaModel.LLAMA2,
            base_url="http://custom:8080",
            timeout=60
        )
        
        mock_agent_class.assert_called_once_with(
            model=OllamaModel.LLAMA2,
            base_url="http://custom:8080",
            timeout=60
        )
    
    def test_llm_orchestrator_node(self, mocker):
        """Test llm_orchestrator_node function"""
        mock_agent = Mock()
        mock_agent.process_findings.return_value = {'result': 'success'}
        
        mock_create_agent = mocker.patch(
            'agents.llm_orchestrator.create_llm_orchestrator_agent',
            return_value=mock_agent
        )
        
        state = {'test': 'data'}
        result = llm_orchestrator_node(state)
        
        mock_create_agent.assert_called_once()
        mock_agent.process_findings.assert_called_once_with(state)
        assert result == {'result': 'success'}


class TestIntegration:
    """Integration tests"""
    
    def test_full_workflow_mock(self, mocker):
        """Test full workflow với mocked LLM calls"""
        # Mock LLM caller
        mock_llm_caller = Mock()
        mock_llm_caller.model = "codellama"
        
        # Mock LLM responses
        mock_responses = [
            Mock(response="Code quality is fair"),  # summary
            Mock(response="Detailed analysis shows issues"),  # detailed
            Mock(response="1. High Priority - Critical issue - Must fix"),  # priority
            Mock(response="- Improve code - Better quality - Medium"),  # recommendations
            Mock(response="Code needs improvement"),  # quality
            Mock(response="1. Add tests - Use pytest - Better coverage")  # improvements
        ]
        mock_llm_caller.generate.side_effect = mock_responses
        
        with patch('deepcode_insight.agents.llm_orchestrator.OllamaLLMCaller', return_value=mock_llm_caller):
            agent = LLMOrchestratorAgent()
        
        # Sample state
        state = {
            'static_analysis_results': {
                'filename': 'test.py',
                'static_issues': {
                    'missing_docstrings': [
                        {'type': 'missing_function_docstring', 'name': 'func1', 'line': 5}
                    ]
                },
                'metrics': {
                    'code_quality_score': 70,
                    'maintainability_index': 65
                },
                'suggestions': ['Add docstrings']
            },
            'code_content': 'def func1():\n    pass',
            'filename': 'test.py'
        }
        
        # Process findings
        result = agent.process_findings(state)
        
        # Verify results
        assert result['processing_status'] == 'llm_analysis_completed'
        assert result['current_agent'] == 'llm_orchestrator'
        assert 'llm_analysis' in result
        
        analysis = result['llm_analysis']
        assert analysis['filename'] == 'test.py'
        assert analysis['summary'] == "Code quality is fair"
        assert analysis['detailed_analysis'] == "Detailed analysis shows issues"
        assert len(analysis['priority_issues']) == 1
        assert len(analysis['recommendations']) == 1
        assert len(analysis['improvement_suggestions']) == 1
        
        # Verify LLM was called correctly
        assert mock_llm_caller.generate.call_count == 6 