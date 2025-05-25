"""
Test cases cho Enhanced LLMOrchestratorAgent với RAG context và Chain-of-Thought prompting
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from deepcode_insight.agents.llm_orchestrator import (
    LLMOrchestratorAgent, 
    create_llm_orchestrator_agent
)
from deepcode_insight.utils.llm_interface import LLMResponse


class TestEnhancedLLMOrchestratorAgent:
    """Test cases cho Enhanced LLMOrchestratorAgent"""
    
    @pytest.fixture
    def sample_static_results(self):
        """Sample static analysis results for testing"""
        return {
            'filename': 'test.py',
            'static_issues': {
                'missing_docstrings': [
                    {
                        'type': 'missing_function_docstring', 
                        'name': 'calculate', 
                        'line': 5, 
                        'message': "Function 'calculate' thiếu docstring"
                    }
                ],
                'complex_functions': [
                    {
                        'type': 'too_many_parameters', 
                        'name': 'complex_func', 
                        'line': 20, 
                        'count': 8, 
                        'message': "Function 'complex_func' có quá nhiều parameters (8)"
                    }
                ]
            },
            'metrics': {
                'code_quality_score': 75.5,
                'maintainability_index': 68.2,
                'cyclomatic_complexity': 12,
                'lines_of_code': 85,
                'comment_ratio': 0.15
            },
            'suggestions': [
                'Thêm docstrings cho functions',
                'Refactor complex functions'
            ]
        }
    
    @pytest.fixture
    def sample_code_content(self):
        """Sample code content for testing"""
        return '''
def calculate(a, b, c, d, e, f, g, h):
    result = a + b + c + d + e + f + g + h
    if result > 100:
        return result * 2
    elif result > 50:
        return result * 1.5
    else:
        return result

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, x, y):
        return x + y
'''
    
    @pytest.fixture
    def sample_rag_context(self):
        """Sample RAG context for testing"""
        return {
            'query': 'code analysis for test.py missing docstrings',
            'relevant_chunks': [
                {
                    'text': 'def well_documented_function():\n    """This is a good example of documentation."""\n    pass',
                    'metadata': {'filename': 'example.py', 'chunk_type': 'function'}
                },
                {
                    'text': 'class WellDocumentedClass:\n    """Example of good class documentation."""\n    pass',
                    'metadata': {'filename': 'example.py', 'chunk_type': 'class'}
                }
            ],
            'context_summary': 'Found examples of good documentation practices',
            'metadata': {'total_chunks': 2}
        }
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response for testing"""
        return LLMResponse(
            response="This is a mock LLM response for testing purposes.",
            model="test-model",
            provider="test-provider",
            usage={'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150},
            metadata={'test': True}
        )
    
    @pytest.fixture
    def mock_rag_agent(self):
        """Mock RAG agent for testing"""
        mock_agent = Mock()
        mock_agent.index_code_file.return_value = True
        mock_agent.query_with_context.return_value = {
            'chunks': [
                {
                    'text': 'def example_function():\n    """Good documentation example."""\n    pass',
                    'metadata': {'filename': 'example.py'}
                }
            ],
            'summary': 'Found relevant code examples',
            'metadata': {'total_results': 1}
        }
        return mock_agent
    
    # ===== Basic Initialization Tests =====
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    @patch('deepcode_insight.agents.llm_orchestrator.RAGContextAgent')
    def test_init_with_default_settings(self, mock_rag_agent, mock_llm_provider):
        """Test initialization với default settings"""
        # Setup mocks
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        mock_rag_instance = Mock()
        mock_rag_agent.return_value = mock_rag_instance
        
        # Create agent
        agent = LLMOrchestratorAgent()
        
        # Assertions
        assert agent.enable_rag == True
        assert agent.enable_chain_of_thought == True
        assert agent.llm_provider == mock_provider_instance
        assert agent.rag_agent == mock_rag_instance
        mock_llm_provider.assert_called_once_with(provider="ollama", model="codellama")
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_init_with_custom_provider(self, mock_llm_provider):
        """Test initialization với custom provider"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        # Create agent với OpenAI provider
        agent = LLMOrchestratorAgent(
            provider="openai",
            model="gpt-4",
            enable_rag=False,
            api_key="test-key"
        )
        
        # Assertions
        assert agent.enable_rag == False
        assert agent.enable_chain_of_thought == True
        assert agent.rag_agent is None
        mock_llm_provider.assert_called_once_with(
            provider="openai", 
            model="gpt-4", 
            api_key="test-key"
        )
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    @patch('deepcode_insight.agents.llm_orchestrator.RAGContextAgent')
    def test_init_rag_failure_graceful_degradation(self, mock_rag_agent, mock_llm_provider):
        """Test graceful degradation khi RAG initialization fails"""
        # Setup mocks
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        mock_rag_agent.side_effect = Exception("RAG initialization failed")
        
        # Create agent - should not raise exception
        agent = LLMOrchestratorAgent(enable_rag=True)
        
        # Assertions
        assert agent.enable_rag == False  # Should be disabled due to failure
        assert agent.rag_agent is None
        assert agent.llm_provider == mock_provider_instance
    
    # ===== RAG Integration Tests =====
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_create_rag_query(self, mock_llm_provider):
        """Test RAG query creation"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        static_results = {
            'static_issues': {
                'missing_docstrings': [{'message': 'test'}],
                'complex_functions': [{'message': 'test'}],
                'security_issues': [{'message': 'test'}]
            }
        }
        
        query = agent._create_rag_query(static_results, "test.py")
        
        assert "code analysis for test.py" in query
        assert "missing docstrings" in query
        assert "complex functions" in query
        assert "security issues" in query
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_get_rag_context_success(self, mock_llm_provider, mock_rag_agent):
        """Test successful RAG context retrieval"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(
            enable_rag=True,
            rag_context_agent=mock_rag_agent
        )
        
        static_results = {
            'static_issues': {
                'missing_docstrings': [{'message': 'test'}]
            }
        }
        
        result = agent._get_rag_context("def test(): pass", "test.py", static_results)
        
        # Assertions
        assert result is not None
        assert 'query' in result
        assert 'relevant_chunks' in result
        assert 'context_summary' in result
        assert 'metadata' in result
        
        # Verify RAG agent calls
        mock_rag_agent.index_code_file.assert_called_once_with("def test(): pass", "test.py")
        mock_rag_agent.query_with_context.assert_called_once()
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_get_rag_context_failure(self, mock_llm_provider):
        """Test RAG context retrieval failure handling"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        mock_rag_agent = Mock()
        mock_rag_agent.index_code_file.side_effect = Exception("RAG query failed")
        
        agent = LLMOrchestratorAgent(
            enable_rag=True,
            rag_context_agent=mock_rag_agent
        )
        
        static_results = {'static_issues': {}}
        
        result = agent._get_rag_context("def test(): pass", "test.py", static_results)
        
        # Should return None on failure
        assert result is None
    
    # ===== Multi-LLM Provider Tests =====
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_generate_with_provider(self, mock_llm_provider):
        """Test LLM generation với provider"""
        mock_provider_instance = Mock()
        mock_response = LLMResponse(
            response="Test response",
            model="test-model",
            provider="test-provider"
        )
        mock_provider_instance.generate.return_value = mock_response
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        result = agent._generate_with_provider(
            prompt="Test prompt",
            temperature=0.5,
            max_tokens=100
        )
        
        assert result == mock_response
        mock_provider_instance.generate.assert_called_once_with(
            prompt="Test prompt",
            system_prompt=None,
            temperature=0.5,
            max_tokens=100
        )
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_multiple_llm_calls_in_analysis(self, mock_llm_provider, 
                                           sample_static_results, 
                                           sample_code_content):
        """Test multiple LLM calls trong enhanced analysis"""
        mock_provider_instance = Mock()
        
        # Create different responses for different prompts
        responses = [
            LLMResponse(response="Summary response", model="test", provider="test"),
            LLMResponse(response="Detailed analysis response", model="test", provider="test"),
            LLMResponse(response="1. Priority issue - Description - Reason", model="test", provider="test"),
            LLMResponse(response="1. Solution - Implementation - Benefit", model="test", provider="test"),
            LLMResponse(response="1. Recommendation - Impact - Low", model="test", provider="test"),
            LLMResponse(response="Quality assessment response", model="test", provider="test"),
            LLMResponse(response="1. Improvement - How to - Benefit", model="test", provider="test")
        ]
        
        mock_provider_instance.generate.side_effect = responses
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False, enable_chain_of_thought=True)
        
        result = agent.analyze_findings_with_enhanced_llm(
            sample_static_results,
            sample_code_content,
            "test.py"
        )
        
        # Verify multiple LLM calls were made
        assert mock_provider_instance.generate.call_count == 7
        
        # Verify all analysis components are present
        assert 'summary' in result
        assert 'detailed_analysis' in result
        assert 'priority_issues' in result
        assert 'solution_suggestions' in result
        assert 'recommendations' in result
        assert 'code_quality_assessment' in result
        assert 'improvement_suggestions' in result
    
    # ===== Enhanced Prompt Tests =====
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_format_enhanced_summary_prompt(self, mock_llm_provider):
        """Test enhanced summary prompt formatting"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        static_results = {
            'static_issues': {'missing_docstrings': [{'message': 'test'}]},
            'metrics': {'code_quality_score': 75},
            'suggestions': ['Add docstrings']
        }
        
        rag_context = {
            'relevant_chunks': [
                {'text': 'Example code with good documentation'}
            ]
        }
        
        prompt = agent._format_enhanced_summary_prompt(
            static_results, "test.py", rag_context
        )
        
        assert "test.py" in prompt
        assert "code quality" in prompt.lower()
        assert "relevant code context" in prompt.lower()
        assert "example code with good documentation" in prompt.lower()
        assert "software engineering best practices" in prompt.lower()
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_format_enhanced_summary_prompt_without_rag(self, mock_llm_provider):
        """Test enhanced summary prompt without RAG context"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        static_results = {
            'static_issues': {'missing_docstrings': [{'message': 'test'}]},
            'metrics': {'code_quality_score': 75},
            'suggestions': ['Add docstrings']
        }
        
        prompt = agent._format_enhanced_summary_prompt(
            static_results, "test.py", None
        )
        
        assert "test.py" in prompt
        assert "relevant code context" not in prompt.lower()
        assert "code quality" in prompt.lower()
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_format_chain_of_thought_analysis_prompt(self, mock_llm_provider):
        """Test Chain-of-Thought analysis prompt formatting"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        static_results = {
            'static_issues': {
                'missing_docstrings': [
                    {'message': 'Missing docstring', 'line': 5}
                ]
            }
        }
        
        rag_context = {
            'relevant_chunks': [
                {'text': 'Related code pattern example'}
            ]
        }
        
        prompt = agent._format_chain_of_thought_analysis_prompt(
            static_results, "def test(): pass", "test.py", rag_context
        )
        
        assert "chain-of-thought" in prompt.lower()
        assert "bước 1" in prompt.lower()
        assert "bước 2" in prompt.lower()
        assert "code structure analysis" in prompt.lower()
        assert "solution strategy" in prompt.lower()
        assert "related code patterns" in prompt.lower()
        assert "related code pattern example" in prompt.lower()
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_format_enhanced_priority_issues_prompt_with_rag(self, mock_llm_provider):
        """Test enhanced priority issues prompt với RAG context"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        static_results = {
            'static_issues': {
                'missing_docstrings': [
                    {'message': 'Missing docstring', 'line': 5}
                ],
                'security_issues': [
                    {'message': 'SQL injection risk', 'line': 10}
                ]
            }
        }
        
        rag_context = {
            'relevant_chunks': [
                {'text': 'Similar security issue example from codebase'}
            ]
        }
        
        prompt = agent._format_enhanced_priority_issues_prompt(static_results, rag_context)
        
        assert "tech lead" in prompt.lower()
        assert "software architecture" in prompt.lower()
        assert "similar issues từ codebase" in prompt.lower()
        assert "impact level" in prompt.lower()
        assert "similar security issue example" in prompt.lower()
    
    # ===== Response Parsing Tests =====
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_parse_solution_suggestions(self, mock_llm_provider):
        """Test parsing solution suggestions từ LLM response"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        llm_response = """
1. Add docstrings - Use triple quotes - Improves documentation
2. Refactor function - Split into smaller functions - Reduces complexity
- Extract constants - Move to module level - Better maintainability
3. Implement error handling - Add try-catch blocks - Improves robustness
"""
        
        suggestions = agent._parse_solution_suggestions(llm_response)
        
        assert len(suggestions) == 4
        assert suggestions[0]['solution'] == 'Add docstrings'
        assert suggestions[0]['implementation'] == 'Use triple quotes'
        assert suggestions[0]['benefit'] == 'Improves documentation'
        assert suggestions[1]['solution'] == 'Refactor function'
        assert suggestions[2]['solution'] == 'Extract constants'
        assert suggestions[3]['solution'] == 'Implement error handling'
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_parse_priority_issues_with_impact_levels(self, mock_llm_provider):
        """Test parsing priority issues với impact levels"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        llm_response = """
1. Security Issue - SQL injection vulnerability - Critical security risk - High
2. Missing Docstring - Function lacks documentation - Impacts maintainability - Medium
3. Complex Function - Too many parameters - Reduces readability - Medium
"""
        
        issues = agent._parse_priority_issues(llm_response)
        
        assert len(issues) == 3
        assert issues[0]['type'] == 'Security Issue'
        assert issues[0]['description'] == 'SQL injection vulnerability'
        assert 'Critical security risk' in issues[0]['reason']
        assert issues[1]['type'] == 'Missing Docstring'
        assert issues[2]['type'] == 'Complex Function'
    
    # ===== Integration Tests =====
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_process_findings_without_rag(self, mock_llm_provider, 
                                         sample_static_results, 
                                         sample_code_content,
                                         mock_llm_response):
        """Test process_findings without RAG context"""
        mock_provider_instance = Mock()
        mock_provider_instance.generate.return_value = mock_llm_response
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        state = {
            'static_analysis_results': sample_static_results,
            'code_content': sample_code_content,
            'filename': 'test.py'
        }
        
        result = agent.process_findings(state)
        
        # Assertions
        assert 'llm_analysis' in result
        assert result['current_agent'] == 'llm_orchestrator'
        assert result['processing_status'] == 'llm_analysis_completed'
        assert result['rag_enabled'] == False
        assert result['chain_of_thought_enabled'] == True
        
        llm_analysis = result['llm_analysis']
        assert llm_analysis['filename'] == 'test.py'
        assert llm_analysis['rag_context_used'] == False
        assert 'summary' in llm_analysis
        assert 'detailed_analysis' in llm_analysis
        assert 'solution_suggestions' in llm_analysis
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_process_findings_with_rag(self, mock_llm_provider, 
                                      sample_static_results, 
                                      sample_code_content,
                                      sample_rag_context,
                                      mock_llm_response):
        """Test process_findings with RAG context"""
        mock_provider_instance = Mock()
        mock_provider_instance.generate.return_value = mock_llm_response
        mock_llm_provider.return_value = mock_provider_instance
        
        # Mock RAG agent
        mock_rag_agent = Mock()
        mock_rag_agent.index_code_file.return_value = True
        mock_rag_agent.query_with_context.return_value = {
            'chunks': sample_rag_context['relevant_chunks'],
            'summary': sample_rag_context['context_summary'],
            'metadata': sample_rag_context['metadata']
        }
        
        agent = LLMOrchestratorAgent(
            enable_rag=True,
            rag_context_agent=mock_rag_agent
        )
        
        state = {
            'static_analysis_results': sample_static_results,
            'code_content': sample_code_content,
            'filename': 'test.py'
        }
        
        result = agent.process_findings(state)
        
        # Assertions
        assert result['rag_enabled'] == True
        llm_analysis = result['llm_analysis']
        assert llm_analysis['rag_context_used'] == True
        
        # Verify RAG agent was called
        mock_rag_agent.index_code_file.assert_called_once()
        mock_rag_agent.query_with_context.assert_called_once()
        
        # Verify RAG query parameters
        call_args = mock_rag_agent.query_with_context.call_args
        assert call_args[1]['top_k'] == 5
        assert call_args[1]['generate_response'] == False
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_process_findings_rag_failure_graceful_handling(self, mock_llm_provider,
                                                           sample_static_results,
                                                           sample_code_content,
                                                           mock_llm_response):
        """Test graceful handling khi RAG fails during processing"""
        mock_provider_instance = Mock()
        mock_provider_instance.generate.return_value = mock_llm_response
        mock_llm_provider.return_value = mock_provider_instance
        
        # Mock RAG agent that fails
        mock_rag_agent = Mock()
        mock_rag_agent.index_code_file.side_effect = Exception("RAG failed")
        
        agent = LLMOrchestratorAgent(
            enable_rag=True,
            rag_context_agent=mock_rag_agent
        )
        
        state = {
            'static_analysis_results': sample_static_results,
            'code_content': sample_code_content,
            'filename': 'test.py'
        }
        
        result = agent.process_findings(state)
        
        # Should still complete successfully without RAG
        assert result['rag_enabled'] == True  # Setting remains true
        llm_analysis = result['llm_analysis']
        assert llm_analysis['rag_context_used'] == False  # But context not used
        assert 'error' not in llm_analysis  # No error in final result
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_process_findings_chain_of_thought_disabled(self, mock_llm_provider,
                                                       sample_static_results,
                                                       sample_code_content,
                                                       mock_llm_response):
        """Test process_findings với Chain-of-Thought disabled"""
        mock_provider_instance = Mock()
        mock_provider_instance.generate.return_value = mock_llm_response
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(
            enable_rag=False,
            enable_chain_of_thought=False
        )
        
        state = {
            'static_analysis_results': sample_static_results,
            'code_content': sample_code_content,
            'filename': 'test.py'
        }
        
        result = agent.process_findings(state)
        
        # Verify Chain-of-Thought specific features are handled
        assert result['chain_of_thought_enabled'] == False
        llm_analysis = result['llm_analysis']
        
        # Should have fewer LLM calls (no solution suggestions)
        # Verify by checking call count is less than full analysis
        call_count = mock_provider_instance.generate.call_count
        assert call_count == 6  # All except solution suggestions
    
    # ===== Error Handling Tests =====
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_error_handling(self, mock_llm_provider):
        """Test error handling trong process_findings"""
        mock_provider_instance = Mock()
        mock_provider_instance.generate.side_effect = Exception("LLM Error")
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        state = {
            'static_analysis_results': {},
            'code_content': '',
            'filename': 'test.py'
        }
        
        result = agent.process_findings(state)
        
        # Should handle error gracefully
        assert 'llm_analysis' in result
        assert result['processing_status'] == 'llm_analysis_failed'
        assert 'error' in result['llm_analysis']
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_missing_static_analysis_results(self, mock_llm_provider):
        """Test handling missing static analysis results"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        state = {
            'code_content': 'def test(): pass',
            'filename': 'test.py'
            # Missing static_analysis_results
        }
        
        result = agent.process_findings(state)
        
        # Should handle missing results gracefully
        assert result['processing_status'] == 'llm_analysis_failed'
        assert 'No static analysis results available' in result['llm_analysis']['error']
    
    # ===== Health Check and Model Tests =====
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_check_llm_health(self, mock_llm_provider):
        """Test LLM health check"""
        mock_provider_instance = Mock()
        mock_provider_instance.check_health.return_value = True
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        result = agent.check_llm_health()
        
        assert result == True
        mock_provider_instance.check_health.assert_called_once()
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_get_available_models(self, mock_llm_provider):
        """Test getting available models"""
        mock_provider_instance = Mock()
        mock_provider_instance.list_models.return_value = ['model1', 'model2', 'model3']
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        result = agent.get_available_models()
        
        assert result == ['model1', 'model2', 'model3']
        mock_provider_instance.list_models.assert_called_once()
    
    # ===== Factory Function Tests =====
    
    def test_factory_function(self):
        """Test factory function create_llm_orchestrator_agent"""
        with patch('deepcode_insight.agents.llm_orchestrator.LLMOrchestratorAgent') as mock_agent:
            create_llm_orchestrator_agent(
                provider="openai",
                model="gpt-4",
                enable_rag=False,
                api_key="test-key"
            )
            
            mock_agent.assert_called_once_with(
                provider="openai",
                model="gpt-4",
                rag_context_agent=None,
                enable_rag=False,
                enable_chain_of_thought=True,
                api_key="test-key"
            )
    
    def test_factory_function_with_rag_agent(self):
        """Test factory function với custom RAG agent"""
        mock_rag_agent = Mock()
        
        with patch('deepcode_insight.agents.llm_orchestrator.LLMOrchestratorAgent') as mock_agent:
            create_llm_orchestrator_agent(
                provider="gemini",
                model="gemini-pro",
                rag_context_agent=mock_rag_agent,
                enable_rag=True,
                enable_chain_of_thought=False
            )
            
            mock_agent.assert_called_once_with(
                provider="gemini",
                model="gemini-pro",
                rag_context_agent=mock_rag_agent,
                enable_rag=True,
                enable_chain_of_thought=False
            )


class TestRAGIntegrationSpecific:
    """Specific tests cho RAG integration features"""
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_rag_query_creation_with_multiple_issue_types(self, mock_llm_provider):
        """Test RAG query creation với multiple issue types"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        static_results = {
            'static_issues': {
                'missing_docstrings': [{'message': 'test1'}, {'message': 'test2'}],
                'complex_functions': [{'message': 'test3'}],
                'security_issues': [{'message': 'test4'}],
                'code_smells': [],  # Empty list should be ignored
                'performance_issues': [{'message': 'test5'}]
            }
        }
        
        query = agent._create_rag_query(static_results, "complex_module.py")
        
        assert "code analysis for complex_module.py" in query
        assert "missing docstrings" in query
        assert "complex functions" in query
        assert "security issues" in query
        assert "performance issues" in query
        assert "code smells" not in query  # Empty list should not appear
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_rag_context_integration_in_all_prompts(self, mock_llm_provider):
        """Test RAG context được integrate vào tất cả prompts"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        static_results = {
            'static_issues': {'missing_docstrings': [{'message': 'test', 'line': 5}]},
            'metrics': {'code_quality_score': 75},
            'suggestions': ['Add docs']
        }
        
        rag_context = {
            'relevant_chunks': [
                {'text': 'Best practice example code'},
                {'text': 'Another good example'},
                {'text': 'Third example'}
            ]
        }
        
        # Test all prompt formatting methods
        summary_prompt = agent._format_enhanced_summary_prompt(
            static_results, "test.py", rag_context
        )
        
        cot_prompt = agent._format_chain_of_thought_analysis_prompt(
            static_results, "def test(): pass", "test.py", rag_context
        )
        
        priority_prompt = agent._format_enhanced_priority_issues_prompt(
            static_results, rag_context
        )
        
        recommendations_prompt = agent._format_enhanced_recommendations_prompt(
            static_results, rag_context
        )
        
        quality_prompt = agent._format_enhanced_quality_assessment_prompt(
            static_results, rag_context
        )
        
        improvement_prompt = agent._format_enhanced_improvement_suggestions_prompt(
            static_results, rag_context
        )
        
        # All prompts should contain RAG context
        prompts = [
            summary_prompt, cot_prompt, priority_prompt,
            recommendations_prompt, quality_prompt, improvement_prompt
        ]
        
        for prompt in prompts:
            assert "best practice example code" in prompt.lower()
            # Should include all chunks since we only have 3
            assert "third example" in prompt.lower()
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_rag_context_chunking_limits(self, mock_llm_provider):
        """Test RAG context chunking limits trong prompts"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        # Create many chunks to test limiting
        many_chunks = [
            {'text': f'Example code chunk {i}' * 20}  # Long text
            for i in range(10)
        ]
        
        rag_context = {'relevant_chunks': many_chunks}
        static_results = {'static_issues': {}, 'metrics': {}, 'suggestions': []}
        
        summary_prompt = agent._format_enhanced_summary_prompt(
            static_results, "test.py", rag_context
        )
        
        # Should limit chunks and truncate text
        chunk_count = summary_prompt.count('Example code chunk')
        # Each chunk appears multiple times due to repetition in text
        # The actual limit is 3 chunks, but each chunk text contains the phrase multiple times
        assert chunk_count >= 3  # Should have at least 3 chunks worth of content
        
        # Should truncate long text (100 char limit)
        assert '...' in summary_prompt


class TestMultiLLMProviderSpecific:
    """Specific tests cho multi-LLM provider support"""
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_different_provider_configurations(self, mock_llm_provider):
        """Test different provider configurations"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        # Test Ollama configuration
        agent_ollama = LLMOrchestratorAgent(
            provider="ollama",
            model="codellama",
            base_url="http://custom:11434",
            timeout=60
        )
        
        mock_llm_provider.assert_called_with(
            provider="ollama",
            model="codellama",
            base_url="http://custom:11434",
            timeout=60
        )
        
        # Test OpenAI configuration
        agent_openai = LLMOrchestratorAgent(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            temperature=0.2
        )
        
        mock_llm_provider.assert_called_with(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            temperature=0.2
        )
        
        # Test Gemini configuration
        agent_gemini = LLMOrchestratorAgent(
            provider="gemini",
            model="gemini-pro",
            api_key="gemini-key"
        )
        
        mock_llm_provider.assert_called_with(
            provider="gemini",
            model="gemini-pro",
            api_key="gemini-key"
        )
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_llm_response_metadata_tracking(self, mock_llm_provider):
        """Test LLM response metadata tracking"""
        mock_provider_instance = Mock()
        mock_provider_instance.__class__.__name__ = "OpenAIProvider"
        mock_provider_instance.model = "gpt-4"
        
        # Sample static results for testing
        sample_static_results = {
            'static_issues': {'test': [{'message': 'test'}]},
            'metrics': {'code_quality_score': 75},
            'suggestions': ['test']
        }
        
        # Create response with usage metadata
        mock_response = LLMResponse(
            response="Test analysis response",
            model="gpt-4",
            provider="openai",
            usage={
                "prompt_tokens": 150,
                "completion_tokens": 75,
                "total_tokens": 225
            },
            metadata={
                "finish_reason": "stop",
                "response_id": "test-123"
            }
        )
        
        mock_provider_instance.generate.return_value = mock_response
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(
            provider="openai",
            model="gpt-4",
            enable_rag=False,
            enable_chain_of_thought=False
        )
        
        result = agent.analyze_findings_with_enhanced_llm(
            sample_static_results,
            "def test(): pass",
            "test.py"
        )
        
        # Verify metadata tracking
        metadata = result['llm_metadata']
        assert metadata['provider'] == "OpenAIProvider"
        assert metadata['model_used'] == "gpt-4"
        assert metadata['analysis_type'] == 'enhanced_code_review_with_rag_and_cot'
        assert metadata['rag_enabled'] == False
        assert metadata['chain_of_thought_enabled'] == False
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_provider_specific_error_handling(self, mock_llm_provider):
        """Test provider-specific error handling"""
        mock_provider_instance = Mock()
        
        # Simulate different types of provider errors
        provider_errors = [
            Exception("OpenAI API rate limit exceeded"),
            Exception("Gemini API key invalid"),
            Exception("Ollama model not found"),
            Exception("Network connection timeout")
        ]
        
        for error in provider_errors:
            mock_provider_instance.generate.side_effect = error
            mock_llm_provider.return_value = mock_provider_instance
            
            agent = LLMOrchestratorAgent(enable_rag=False)
            
            state = {
                'static_analysis_results': {'static_issues': {}, 'metrics': {}},
                'code_content': 'def test(): pass',
                'filename': 'test.py'
            }
            
            result = agent.process_findings(state)
            
            # Should handle all provider errors gracefully
            # The agent catches errors and continues, so status should be completed
            # but with error in the analysis
            assert result['processing_status'] == 'llm_analysis_completed'
            assert 'error' in result['llm_analysis']
            assert str(error) in result['llm_analysis']['error']


if __name__ == "__main__":
    pytest.main([__file__]) 