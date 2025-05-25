"""
Integration tests cho ReportingAgent với các agents khác
Kiểm tra workflow từ StaticAnalysisAgent -> LLMOrchestratorAgent -> ReportingAgent
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ..agents.static_analyzer import StaticAnalysisAgent
from ..agents.llm_orchestrator import LLMOrchestratorAgent
from ..agents.reporter import ReportingAgent


class TestReportingIntegration:
    """Test integration của ReportingAgent với other agents"""
    
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
    
    def test_full_workflow_static_to_report(self, temp_output_dir, sample_python_code):
        """Test complete workflow từ static analysis đến report generation"""
        
        # Step 1: Static Analysis
        static_agent = StaticAnalysisAgent()
        
        # Run static analysis với correct parameters
        static_analysis_results = static_agent.analyze_code(sample_python_code, 'test_integration.py')
        
        # Create state với static analysis results
        static_result = {
            'static_analysis_results': static_analysis_results,
            'code_content': sample_python_code,
            'filename': 'test_integration.py',
            'current_agent': 'static_analyzer',
            'processing_status': 'static_analysis_completed'
        }
        
        # Verify static analysis worked
        assert static_result['processing_status'] == 'static_analysis_completed'
        assert 'static_analysis_results' in static_result
        
        # Step 2: Mock LLM Analysis (since we don't want to call real LLM in tests)
        llm_analysis = {
            'filename': 'test_integration.py',
            'summary': 'Integration test analysis summary',
            'detailed_analysis': 'The code has some documentation issues that need attention.',
            'priority_issues': [
                {
                    'priority': 'High',
                    'description': 'Missing docstrings in functions',
                    'action': 'Add comprehensive documentation'
                }
            ],
            'recommendations': [
                {
                    'description': 'Add docstrings to all functions',
                    'effort': 'Medium'
                },
                {
                    'description': 'Fix line length issues',
                    'effort': 'Low'
                }
            ],
            'code_quality_assessment': 'Code structure is good but needs documentation improvements.',
            'improvement_suggestions': [
                {
                    'title': 'Documentation',
                    'description': 'Add comprehensive docstrings',
                    'effort': 'Medium'
                }
            ],
            'llm_metadata': {
                'model_used': 'test_model',
                'analysis_type': 'integration_test'
            }
        }
        
        # Add LLM analysis to state
        static_result['llm_analysis'] = llm_analysis
        static_result['current_agent'] = 'llm_orchestrator'
        static_result['processing_status'] = 'llm_analysis_completed'
        
        # Step 3: Report Generation
        reporter = ReportingAgent(output_dir=temp_output_dir)
        
        # Generate report
        final_result = reporter.generate_report(static_result)
        
        # Verify report generation
        assert final_result['processing_status'] == 'report_generated'
        assert final_result['current_agent'] == 'reporter'
        assert 'report' in final_result
        
        # Verify report file exists
        report_info = final_result['report']
        assert os.path.exists(report_info['output_path'])
        
        # Verify report content
        content = report_info['content']
        
        # Should contain static analysis results
        assert 'Missing Docstrings' in content
        assert 'Code Metrics' in content
        
        # Should contain LLM analysis
        assert 'Integration test analysis summary' in content
        assert 'Missing docstrings in functions' in content
        assert 'Add comprehensive documentation' in content
        
        # Should contain recommendations
        assert 'Action Items & Recommendations' in content
        assert 'Add docstrings to all functions' in content
        
        # Should contain metadata
        assert 'test_model' in content
        assert 'integration_test' in content
    
    def test_workflow_with_no_issues_found(self, temp_output_dir):
        """Test workflow khi static analysis không tìm thấy issues"""
        
        # Clean code without issues
        clean_code = '''
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the result.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
    """
    return a + b


class Calculator:
    """A simple calculator class."""
    
    def multiply(self, x: int, y: int) -> int:
        """Multiply two numbers.
        
        Args:
            x: First number
            y: Second number
            
        Returns:
            Product of x and y
        """
        return x * y
'''
        
        # Static analysis
        static_agent = StaticAnalysisAgent()
        
        # Run static analysis với correct parameters
        static_analysis_results = static_agent.analyze_code(clean_code, 'clean_code.py')
        
        # Create state với static analysis results
        static_result = {
            'static_analysis_results': static_analysis_results,
            'code_content': clean_code,
            'filename': 'clean_code.py',
            'current_agent': 'static_analyzer',
            'processing_status': 'static_analysis_completed'
        }
        
        # Mock LLM analysis for clean code
        llm_analysis = {
            'filename': 'clean_code.py',
            'summary': 'Excellent code quality with proper documentation.',
            'detailed_analysis': 'The code follows best practices with comprehensive documentation.',
            'priority_issues': [],  # No issues
            'recommendations': [
                {
                    'description': 'Continue maintaining current code quality standards',
                    'effort': 'Low'
                }
            ],
            'code_quality_assessment': 'Excellent code quality.',
            'improvement_suggestions': [],
            'llm_metadata': {
                'model_used': 'test_model',
                'analysis_type': 'clean_code_test'
            }
        }
        
        static_result['llm_analysis'] = llm_analysis
        static_result['processing_status'] = 'llm_analysis_completed'
        
        # Generate report
        reporter = ReportingAgent(output_dir=temp_output_dir)
        final_result = reporter.generate_report(static_result)
        
        # Verify report
        assert final_result['processing_status'] == 'report_generated'
        
        content = final_result['report']['content']
        
        # Should indicate clean code or have static analysis section
        assert ('No Issues Found' in content or 
                'Static Analysis Results' in content or 
                'static_issues' in content)
        assert 'Excellent code quality' in content
        assert 'Continue maintaining current code quality' in content
    
    def test_workflow_with_static_only(self, temp_output_dir, sample_python_code):
        """Test workflow với chỉ static analysis (không có LLM)"""
        
        # Static analysis only
        static_agent = StaticAnalysisAgent()
        
        # Run static analysis với correct parameters
        static_analysis_results = static_agent.analyze_code(sample_python_code, 'static_only.py')
        
        # Create state với static analysis results
        static_result = {
            'static_analysis_results': static_analysis_results,
            'code_content': sample_python_code,
            'filename': 'static_only.py',
            'current_agent': 'static_analyzer',
            'processing_status': 'static_analysis_completed'
        }
        
        # Generate report without LLM analysis
        reporter = ReportingAgent(output_dir=temp_output_dir)
        final_result = reporter.generate_report(static_result)
        
        # Verify report
        assert final_result['processing_status'] == 'report_generated'
        
        content = final_result['report']['content']
        
        # Should contain static analysis
        assert 'Static Analysis Results' in content
        assert 'Code Metrics' in content
        
        # Should NOT contain LLM sections
        assert 'Executive Summary' not in content
        assert 'AI-Powered Analysis' not in content
        assert 'Priority Issues' not in content
    
    def test_workflow_error_handling(self, temp_output_dir):
        """Test error handling trong workflow"""
        
        # Empty state (should cause error)
        empty_state = {
            'filename': 'error_test.py'
        }
        
        reporter = ReportingAgent(output_dir=temp_output_dir)
        result = reporter.generate_report(empty_state)
        
        # Should handle error gracefully
        assert result['processing_status'] == 'report_error'
        assert 'error' in result
        assert 'No analysis results available' in result['error']
    
    def test_workflow_with_large_analysis_results(self, temp_output_dir):
        """Test workflow với large analysis results"""
        
        # Create large static analysis results
        large_static_results = {
            'filename': 'large_file.py',
            'static_issues': {
                'missing_docstrings': [
                    {
                        'type': 'missing_function_docstring',
                        'name': f'function_{i}',
                        'line': i * 10,
                        'message': f'Function {i} lacks docstring'
                    }
                    for i in range(50)  # 50 issues
                ],
                'code_style': [
                    {
                        'type': 'line_too_long',
                        'line': i * 5,
                        'message': f'Line {i * 5} too long'
                    }
                    for i in range(30)  # 30 style issues
                ]
            },
            'metrics': {
                'code_quality_score': 45.5,
                'maintainability_index': 35.2,
                'complexity_score': 8.7,
                'lines_of_code': 1500,
                'cyclomatic_complexity': 25
            },
            'suggestions': [
                'Add comprehensive documentation',
                'Refactor complex functions',
                'Improve code formatting',
                'Add unit tests',
                'Reduce cyclomatic complexity'
            ]
        }
        
        # Large LLM analysis
        large_llm_analysis = {
            'filename': 'large_file.py',
            'summary': 'Large file with significant quality issues requiring comprehensive refactoring.',
            'detailed_analysis': 'This file contains numerous issues across documentation, style, and complexity. Immediate attention required.',
            'priority_issues': [
                {
                    'priority': 'High',
                    'description': f'Critical issue {i}',
                    'action': f'Fix critical issue {i} immediately'
                }
                for i in range(10)
            ],
            'recommendations': [
                {
                    'description': f'Recommendation {i}',
                    'effort': 'High' if i % 3 == 0 else 'Medium' if i % 3 == 1 else 'Low'
                }
                for i in range(20)
            ],
            'code_quality_assessment': 'Code quality is below acceptable standards and requires immediate improvement.',
            'improvement_suggestions': [
                {
                    'title': f'Improvement {i}',
                    'description': f'Detailed improvement suggestion {i}',
                    'effort': 'High' if i % 2 == 0 else 'Medium'
                }
                for i in range(15)
            ],
            'llm_metadata': {
                'model_used': 'test_model',
                'analysis_type': 'large_file_analysis'
            }
        }
        
        # Create state
        large_state = {
            'static_analysis_results': large_static_results,
            'llm_analysis': large_llm_analysis,
            'filename': 'large_file.py',
            'processing_status': 'llm_analysis_completed'
        }
        
        # Generate report
        reporter = ReportingAgent(output_dir=temp_output_dir)
        result = reporter.generate_report(large_state)
        
        # Verify report generation succeeded
        assert result['processing_status'] == 'report_generated'
        
        content = result['report']['content']
        
        # Verify large content is handled properly
        assert 'large_file.py' in content
        assert 'function_0' in content  # First function
        assert 'function_49' in content  # Last function
        assert 'Critical issue 0' in content  # First priority issue
        assert 'Critical issue 9' in content  # Last priority issue
        
        # Verify metrics table
        assert '45.50' in content  # Code quality score
        assert '1500' in content  # Lines of code
        
        # Verify recommendations are grouped properly
        assert 'High Priority Actions' in content
        assert 'Medium Priority Actions' in content
        assert 'Low Priority Actions' in content
        
        # Verify file size is reasonable (not too large)
        assert len(content) < 50000  # Should be under 50KB


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 