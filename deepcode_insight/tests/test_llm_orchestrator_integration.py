"""
Integration tests cho Enhanced LLMOrchestratorAgent với real RAG và LLM providers
"""

import pytest
import sys
import os
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from deepcode_insight.agents.llm_orchestrator import (
    LLMOrchestratorAgent, 
    create_llm_orchestrator_agent,
    llm_orchestrator_node
)
from deepcode_insight.utils.llm_interface import LLMResponse, LLMProvider
from deepcode_insight.agents.rag_context import RAGContextAgent


class TestLLMOrchestratorIntegration:
    """Integration tests cho LLMOrchestratorAgent với mocked external services"""
    
    @pytest.fixture
    def complex_static_results(self):
        """Complex static analysis results for integration testing"""
        return {
            'filename': 'complex_module.py',
            'static_issues': {
                'missing_docstrings': [
                    {
                        'type': 'missing_function_docstring',
                        'name': 'calculate_metrics',
                        'line': 15,
                        'message': "Function 'calculate_metrics' thiếu docstring"
                    },
                    {
                        'type': 'missing_class_docstring',
                        'name': 'DataProcessor',
                        'line': 45,
                        'message': "Class 'DataProcessor' thiếu docstring"
                    }
                ],
                'complex_functions': [
                    {
                        'type': 'high_cyclomatic_complexity',
                        'name': 'process_data',
                        'line': 67,
                        'complexity': 18,
                        'message': "Function 'process_data' có complexity cao (18)"
                    }
                ],
                'security_issues': [
                    {
                        'type': 'sql_injection_risk',
                        'line': 89,
                        'message': "Potential SQL injection vulnerability"
                    }
                ],
                'performance_issues': [
                    {
                        'type': 'inefficient_loop',
                        'line': 123,
                        'message': "Nested loop có thể optimize"
                    }
                ],
                'code_smells': [
                    {
                        'type': 'long_parameter_list',
                        'name': 'initialize_system',
                        'line': 156,
                        'count': 12,
                        'message': "Function có quá nhiều parameters (12)"
                    }
                ]
            },
            'metrics': {
                'code_quality_score': 62.3,
                'maintainability_index': 45.8,
                'cyclomatic_complexity': 24,
                'lines_of_code': 450,
                'comment_ratio': 0.08,
                'test_coverage': 0.35
            },
            'suggestions': [
                'Thêm docstrings cho 2 functions/classes',
                'Refactor 1 complex function để giảm complexity',
                'Fix 1 security vulnerability',
                'Optimize 1 performance issue',
                'Reduce parameter count trong 1 function'
            ]
        }
    
    @pytest.fixture
    def complex_code_content(self):
        """Complex code content for integration testing"""
        return '''
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple

class DataProcessor:
    def __init__(self, db_path: str, config: Dict[str, Any], 
                 cache_size: int, timeout: int, retry_count: int,
                 batch_size: int, parallel_workers: int, debug_mode: bool,
                 log_level: str, output_format: str, compression: bool,
                 encryption_key: Optional[str]):
        self.db_path = db_path
        self.config = config
        self.cache_size = cache_size
        # ... more initialization
    
    def calculate_metrics(self, data):
        result = 0
        for item in data:
            if item > 0:
                result += item * 2
            elif item < 0:
                result -= item
            else:
                result += 1
        return result
    
    def process_data(self, input_data: List[Dict], filters: Dict, 
                    options: Dict, metadata: Dict) -> Dict[str, Any]:
        results = []
        
        # Complex nested logic
        for item in input_data:
            if item.get('type') == 'A':
                if item.get('status') == 'active':
                    if item.get('priority') > 5:
                        if filters.get('include_high_priority'):
                            for sub_item in item.get('sub_items', []):
                                if sub_item.get('valid'):
                                    processed = self._process_sub_item(sub_item)
                                    if processed:
                                        results.append(processed)
                                    else:
                                        continue
                                else:
                                    continue
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
            else:
                continue
        
        return {'results': results, 'count': len(results)}
    
    def query_database(self, user_input: str) -> List[Dict]:
        # SQL injection vulnerability
        query = f"SELECT * FROM users WHERE name = '{user_input}'"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()
    
    def inefficient_processing(self, data: List[List[int]]) -> List[int]:
        results = []
        # Inefficient nested loops
        for i in range(len(data)):
            for j in range(len(data[i])):
                for k in range(len(data)):
                    if data[i][j] == data[k][0]:
                        results.append(data[i][j])
        return results
'''
    
    @pytest.fixture
    def mock_comprehensive_rag_agent(self):
        """Comprehensive mock RAG agent for integration testing"""
        mock_agent = Mock(spec=RAGContextAgent)
        
        # Mock successful indexing
        mock_agent.index_code_file.return_value = True
        
        # Mock comprehensive query response
        mock_agent.query_with_context.return_value = {
            'chunks': [
                {
                    'text': '''def well_documented_function(param1: str, param2: int) -> str:
    """
    This function demonstrates good documentation practices.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        
    Returns:
        str: Description of return value
        
    Raises:
        ValueError: When invalid input is provided
    """
    return f"{param1}_{param2}"''',
                    'metadata': {
                        'filename': 'best_practices.py',
                        'chunk_type': 'function',
                        'quality_score': 0.95
                    }
                },
                {
                    'text': '''class WellStructuredClass:
    """
    Example of a well-structured class with proper documentation.
    
    This class demonstrates best practices for:
    - Clear documentation
    - Proper type hints
    - Reasonable complexity
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config = config
    
    def process(self, data: List[str]) -> List[str]:
        """Process data with proper error handling."""
        try:
            return [item.strip().lower() for item in data if item]
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise''',
                    'metadata': {
                        'filename': 'best_practices.py',
                        'chunk_type': 'class',
                        'quality_score': 0.92
                    }
                },
                {
                    'text': '''# Security best practice example
def safe_database_query(user_input: str) -> List[Dict]:
    """
    Secure database query using parameterized statements.
    
    Args:
        user_input: User provided input (sanitized)
        
    Returns:
        List of query results
    """
    # Use parameterized queries to prevent SQL injection
    query = "SELECT * FROM users WHERE name = ?"
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(query, (user_input,))
    return cursor.fetchall()''',
                    'metadata': {
                        'filename': 'security_examples.py',
                        'chunk_type': 'function',
                        'security_pattern': 'sql_injection_prevention'
                    }
                },
                {
                    'text': '''# Performance optimization example
def optimized_processing(data: List[List[int]]) -> List[int]:
    """
    Optimized version using set operations and list comprehension.
    
    Time complexity: O(n*m) instead of O(n*m*k)
    """
    # Create set for O(1) lookup
    first_elements = {row[0] for row in data if row}
    
    # Use list comprehension for efficiency
    results = [
        item for row in data 
        for item in row 
        if item in first_elements
    ]
    
    return results''',
                    'metadata': {
                        'filename': 'performance_examples.py',
                        'chunk_type': 'function',
                        'optimization_type': 'algorithmic'
                    }
                }
            ],
            'summary': 'Found relevant examples of documentation, security, and performance best practices',
            'metadata': {
                'total_results': 4,
                'query_time': 0.15,
                'relevance_scores': [0.95, 0.92, 0.88, 0.85]
            }
        }
        
        return mock_agent
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_full_integration_with_rag_and_chain_of_thought(self, 
                                                           mock_llm_provider,
                                                           complex_static_results,
                                                           complex_code_content,
                                                           mock_comprehensive_rag_agent):
        """Test full integration với RAG và Chain-of-Thought enabled"""
        
        # Setup comprehensive LLM responses
        mock_provider_instance = Mock()
        
        llm_responses = [
            # Summary response
            LLMResponse(
                response="""Code quality analysis cho complex_module.py cho thấy several critical issues cần attention. 
                File có quality score thấp (62.3/100) với maintainability index chỉ 45.8/100, indicating significant technical debt. 
                Main concerns include missing documentation, high complexity functions, và security vulnerabilities. 
                Immediate refactoring recommended để improve maintainability và security posture.""",
                model="test-model",
                provider="test"
            ),
            
            # Chain-of-Thought detailed analysis
            LLMResponse(
                response="""**Bước 1: Code Structure Analysis**
                - Architecture: Monolithic class với multiple responsibilities
                - Design patterns: Không có clear separation of concerns
                - Coupling: High coupling với database và external dependencies
                - Cohesion: Low cohesion, class handles too many different tasks
                
                **Bước 2: Issue Impact Assessment**
                - Security vulnerability (SQL injection): CRITICAL - có thể lead to data breach
                - High complexity function: HIGH - impacts maintainability và testing
                - Missing documentation: MEDIUM - affects team productivity
                - Performance issues: MEDIUM - impacts user experience
                
                **Bước 3: Risk Evaluation**
                - Security risk: Immediate threat to data integrity
                - Maintenance risk: Code sẽ become increasingly difficult to modify
                - Performance risk: Scalability issues under load
                - Team productivity risk: New developers sẽ struggle to understand code
                
                **Bước 4: Solution Strategy**
                - Immediate: Fix security vulnerability using parameterized queries
                - Short-term: Break down complex functions, add documentation
                - Long-term: Refactor class architecture, implement proper separation of concerns""",
                model="test-model",
                provider="test"
            ),
            
            # Priority issues
            LLMResponse(
                response="""1. SQL Injection Vulnerability - Critical security flaw in query_database method - Immediate fix required - Critical
                2. High Cyclomatic Complexity - process_data function too complex (18) - Impacts maintainability - High  
                3. Missing Documentation - Class và key functions lack docstrings - Affects team productivity - Medium
                4. Long Parameter List - initialize_system has 12 parameters - Reduces code readability - Medium
                5. Performance Issue - Inefficient nested loops in processing - Impacts user experience - Medium""",
                model="test-model",
                provider="test"
            ),
            
            # Solution suggestions (Chain-of-Thought)
            LLMResponse(
                response="""1. Implement parameterized queries - Use cursor.execute(query, params) - Prevents SQL injection
                2. Extract method pattern - Break process_data into smaller functions - Reduces complexity
                3. Add comprehensive docstrings - Use Google/Sphinx style documentation - Improves maintainability
                4. Create configuration object - Replace long parameter list with config class - Better encapsulation
                5. Optimize algorithm complexity - Use set operations instead of nested loops - Improves performance
                6. Implement dependency injection - Decouple database access from business logic - Better testability
                7. Add input validation - Sanitize all user inputs before processing - Enhanced security""",
                model="test-model",
                provider="test"
            ),
            
            # Recommendations
            LLMResponse(
                response="""1. Security audit và fix - Implement secure coding practices - High - High
                2. Code complexity reduction - Refactor complex methods - Medium - High
                3. Documentation improvement - Add comprehensive docstrings - Low - Medium
                4. Performance optimization - Optimize critical algorithms - Medium - Medium
                5. Architecture refactoring - Implement clean architecture principles - High - Low
                6. Unit testing implementation - Add comprehensive test coverage - Medium - Medium""",
                model="test-model",
                provider="test"
            ),
            
            # Quality assessment
            LLMResponse(
                response="""Overall code quality: POOR (62.3/100). Code exhibits multiple serious issues including security vulnerabilities, 
                high complexity, và poor documentation. Main strengths include use of type hints và structured data handling. 
                Key improvement areas: security hardening, complexity reduction, documentation enhancement. 
                Production readiness: NOT READY - requires significant refactoring before deployment. 
                Comparison với industry standards: Below average, needs substantial improvement to meet enterprise standards.""",
                model="test-model",
                provider="test"
            ),
            
            # Improvement suggestions
            LLMResponse(
                response="""1. Security hardening - Implement input validation và parameterized queries - Prevents vulnerabilities - 1-2 days
                2. Function decomposition - Break complex functions into smaller units - Improves maintainability - 3-5 days
                3. Documentation sprint - Add docstrings và code comments - Enhances team productivity - 2-3 days
                4. Performance profiling - Identify và optimize bottlenecks - Better user experience - 1 week
                5. Architecture review - Design cleaner separation of concerns - Long-term maintainability - 2-3 weeks
                6. Test coverage improvement - Implement comprehensive unit tests - Better reliability - 1-2 weeks""",
                model="test-model",
                provider="test"
            )
        ]
        
        mock_provider_instance.generate.side_effect = llm_responses
        mock_provider_instance.__class__.__name__ = "OllamaProvider"
        mock_provider_instance.model = "codellama"
        mock_llm_provider.return_value = mock_provider_instance
        
        # Create agent với full features enabled
        agent = LLMOrchestratorAgent(
            provider="ollama",
            model="codellama",
            rag_context_agent=mock_comprehensive_rag_agent,
            enable_rag=True,
            enable_chain_of_thought=True
        )
        
        # Create comprehensive state
        state = {
            'static_analysis_results': complex_static_results,
            'code_content': complex_code_content,
            'filename': 'complex_module.py',
            'project_context': {
                'language': 'python',
                'framework': 'django',
                'project_size': 'large'
            }
        }
        
        # Process findings
        result = agent.process_findings(state)
        
        # Comprehensive assertions
        assert result['processing_status'] == 'llm_analysis_completed'
        assert result['rag_enabled'] == True
        assert result['chain_of_thought_enabled'] == True
        
        llm_analysis = result['llm_analysis']
        
        # Verify all analysis components
        assert llm_analysis['filename'] == 'complex_module.py'
        assert llm_analysis['rag_context_used'] == True
        assert 'summary' in llm_analysis
        assert 'detailed_analysis' in llm_analysis
        assert 'priority_issues' in llm_analysis
        assert 'solution_suggestions' in llm_analysis
        assert 'recommendations' in llm_analysis
        assert 'code_quality_assessment' in llm_analysis
        assert 'improvement_suggestions' in llm_analysis
        
        # Verify RAG integration
        mock_comprehensive_rag_agent.index_code_file.assert_called_once_with(
            complex_code_content, 'complex_module.py'
        )
        mock_comprehensive_rag_agent.query_with_context.assert_called_once()
        
        # Verify LLM calls
        assert mock_provider_instance.generate.call_count == 7
        
        # Verify metadata
        metadata = llm_analysis['llm_metadata']
        assert metadata['provider'] == 'OllamaProvider'
        assert metadata['model_used'] == 'codellama'
        assert metadata['rag_enabled'] == True
        assert metadata['chain_of_thought_enabled'] == True
        
        # Verify parsed results structure
        assert len(llm_analysis['priority_issues']) > 0
        assert len(llm_analysis['solution_suggestions']) > 0
        assert len(llm_analysis['recommendations']) > 0
        assert len(llm_analysis['improvement_suggestions']) > 0
        
        # Verify priority issues parsing
        priority_issue = llm_analysis['priority_issues'][0]
        assert 'type' in priority_issue
        assert 'description' in priority_issue
        assert 'reason' in priority_issue
        
        # Verify solution suggestions parsing
        solution = llm_analysis['solution_suggestions'][0]
        assert 'solution' in solution
        assert 'implementation' in solution
        assert 'benefit' in solution
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_langgraph_node_integration(self, mock_llm_provider):
        """Test LangGraph node function integration"""
        
        # Setup mock
        mock_provider_instance = Mock()
        mock_response = LLMResponse(
            response="Node integration test response",
            model="test-model",
            provider="test"
        )
        mock_provider_instance.generate.return_value = mock_response
        mock_llm_provider.return_value = mock_provider_instance
        
        # Test state
        state = {
            'static_analysis_results': {
                'static_issues': {'test_issues': [{'message': 'test'}]},
                'metrics': {'code_quality_score': 80},
                'suggestions': ['test suggestion']
            },
            'code_content': 'def test(): pass',
            'filename': 'test.py'
        }
        
        # Call LangGraph node function
        result = llm_orchestrator_node(state)
        
        # Verify node function works
        assert 'llm_analysis' in result
        assert result['current_agent'] == 'llm_orchestrator'
        assert result['processing_status'] == 'llm_analysis_completed'
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_performance_with_large_codebase(self, mock_llm_provider):
        """Test performance với large codebase simulation"""
        
        # Setup fast mock responses
        mock_provider_instance = Mock()
        mock_response = LLMResponse(
            response="Fast response for performance test",
            model="test-model",
            provider="test"
        )
        mock_provider_instance.generate.return_value = mock_response
        mock_llm_provider.return_value = mock_provider_instance
        
        # Create large static results
        large_static_results = {
            'static_issues': {
                'missing_docstrings': [
                    {'message': f'Missing docstring {i}', 'line': i}
                    for i in range(100)
                ],
                'complex_functions': [
                    {'message': f'Complex function {i}', 'line': i*10}
                    for i in range(50)
                ]
            },
            'metrics': {
                'code_quality_score': 45.2,
                'lines_of_code': 10000
            },
            'suggestions': [f'Suggestion {i}' for i in range(200)]
        }
        
        # Large code content
        large_code_content = "def function_{}(): pass\n" * 1000
        
        agent = LLMOrchestratorAgent(enable_rag=False, enable_chain_of_thought=False)
        
        state = {
            'static_analysis_results': large_static_results,
            'code_content': large_code_content,
            'filename': 'large_module.py'
        }
        
        # Measure performance
        start_time = time.time()
        result = agent.process_findings(state)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Verify performance (should complete within reasonable time)
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert result['processing_status'] == 'llm_analysis_completed'
        
        # Verify handling of large data
        assert 'llm_analysis' in result
        assert result['llm_analysis']['filename'] == 'large_module.py'
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_concurrent_processing_simulation(self, mock_llm_provider):
        """Test concurrent processing simulation"""
        
        # Setup mock với different response times
        mock_provider_instance = Mock()
        
        def slow_generate(*args, **kwargs):
            time.sleep(0.1)  # Simulate network latency
            return LLMResponse(
                response="Concurrent test response",
                model="test-model",
                provider="test"
            )
        
        mock_provider_instance.generate.side_effect = slow_generate
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        # Create multiple states for concurrent processing
        states = [
            {
                'static_analysis_results': {
                    'static_issues': {'test': [{'message': f'test {i}'}]},
                    'metrics': {'code_quality_score': 70 + i},
                    'suggestions': [f'suggestion {i}']
                },
                'code_content': f'def test_{i}(): pass',
                'filename': f'test_{i}.py'
            }
            for i in range(3)
        ]
        
        # Process multiple states
        results = []
        start_time = time.time()
        
        for state in states:
            result = agent.process_findings(state)
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all processed successfully
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result['processing_status'] == 'llm_analysis_completed'
            assert result['llm_analysis']['filename'] == f'test_{i}.py'
        
        # Verify reasonable total processing time
        assert total_time < 10.0  # Should complete within 10 seconds


class TestErrorRecoveryAndResilience:
    """Tests cho error recovery và resilience"""
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_partial_llm_failure_recovery(self, mock_llm_provider):
        """Test recovery từ partial LLM failures"""
        
        mock_provider_instance = Mock()
        
        # Simulate some calls succeeding, some failing
        responses = [
            LLMResponse(response="Success 1", model="test", provider="test"),  # Summary
            Exception("Network timeout"),  # Detailed analysis fails
            LLMResponse(response="Success 3", model="test", provider="test"),  # Priority issues
            LLMResponse(response="Success 4", model="test", provider="test"),  # Solutions
            Exception("Rate limit exceeded"),  # Recommendations fail
            LLMResponse(response="Success 6", model="test", provider="test"),  # Quality
            LLMResponse(response="Success 7", model="test", provider="test"),  # Improvements
        ]
        
        mock_provider_instance.generate.side_effect = responses
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        state = {
            'static_analysis_results': {
                'static_issues': {'test': [{'message': 'test'}]},
                'metrics': {'code_quality_score': 75},
                'suggestions': ['test']
            },
            'code_content': 'def test(): pass',
            'filename': 'test.py'
        }
        
        result = agent.analyze_findings_with_enhanced_llm(
            state['static_analysis_results'],
            state['code_content'],
            state['filename']
        )
        
        # Should have partial results
        assert 'summary' in result
        assert result['summary'] == "Success 1"
        
        # Failed components should have empty/default values
        assert 'detailed_analysis' in result
        assert 'priority_issues' in result
        assert 'solution_suggestions' in result
        assert 'recommendations' in result
        assert 'code_quality_assessment' in result
        assert 'improvement_suggestions' in result
        
        # Should not have error in final result (graceful degradation)
        assert 'error' not in result
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_rag_failure_with_llm_success(self, mock_llm_provider):
        """Test LLM success khi RAG fails"""
        
        mock_provider_instance = Mock()
        mock_provider_instance.generate.return_value = LLMResponse(
            response="LLM success without RAG",
            model="test",
            provider="test"
        )
        mock_llm_provider.return_value = mock_provider_instance
        
        # Mock failing RAG agent
        mock_rag_agent = Mock()
        mock_rag_agent.index_code_file.side_effect = Exception("RAG service unavailable")
        
        agent = LLMOrchestratorAgent(
            enable_rag=True,
            rag_context_agent=mock_rag_agent
        )
        
        state = {
            'static_analysis_results': {
                'static_issues': {'test': [{'message': 'test'}]},
                'metrics': {'code_quality_score': 75},
                'suggestions': ['test']
            },
            'code_content': 'def test(): pass',
            'filename': 'test.py'
        }
        
        result = agent.process_findings(state)
        
        # Should complete successfully without RAG
        assert result['processing_status'] == 'llm_analysis_completed'
        assert result['rag_enabled'] == True  # Setting remains true
        
        llm_analysis = result['llm_analysis']
        assert llm_analysis['rag_context_used'] == False  # But context not used
        assert 'error' not in llm_analysis
        
        # LLM calls should still work
        assert mock_provider_instance.generate.call_count > 0
    
    @patch('deepcode_insight.agents.llm_orchestrator.create_llm_provider')
    def test_malformed_llm_response_handling(self, mock_llm_provider):
        """Test handling của malformed LLM responses"""
        
        mock_provider_instance = Mock()
        
        # Malformed responses that could break parsing
        malformed_responses = [
            LLMResponse(response="", model="test", provider="test"),  # Empty
            LLMResponse(response="No structured format here", model="test", provider="test"),  # No format
            LLMResponse(response="1. Item without separators", model="test", provider="test"),  # Missing separators
            LLMResponse(response="Random text\n\nMore random text", model="test", provider="test"),  # No structure
            LLMResponse(response="1. Good - Format - Works\n2. Bad format", model="test", provider="test"),  # Mixed
            LLMResponse(response="Normal response", model="test", provider="test"),  # Normal
            LLMResponse(response="Another normal response", model="test", provider="test"),  # Normal
        ]
        
        mock_provider_instance.generate.side_effect = malformed_responses
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = LLMOrchestratorAgent(enable_rag=False)
        
        state = {
            'static_analysis_results': {
                'static_issues': {'test': [{'message': 'test'}]},
                'metrics': {'code_quality_score': 75},
                'suggestions': ['test']
            },
            'code_content': 'def test(): pass',
            'filename': 'test.py'
        }
        
        result = agent.analyze_findings_with_enhanced_llm(
            state['static_analysis_results'],
            state['code_content'],
            state['filename']
        )
        
        # Should handle malformed responses gracefully
        assert 'error' not in result
        assert 'priority_issues' in result
        assert 'solution_suggestions' in result
        assert 'recommendations' in result
        
        # Parsing should return empty lists for malformed responses
        assert isinstance(result['priority_issues'], list)
        assert isinstance(result['solution_suggestions'], list)
        assert isinstance(result['recommendations'], list)


if __name__ == "__main__":
    pytest.main([__file__]) 