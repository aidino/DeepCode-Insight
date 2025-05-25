"""
Extended test cases cho SolutionSuggestionAgent
Focus on raw LLM outputs và expected refined outputs structure
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from deepcode_insight.agents.solution_suggester import (
    SolutionSuggestionAgent,
    RefinedSolution
)
from deepcode_insight.utils.llm_interface import LLMResponse


class TestSolutionSuggesterWithRealOutputs:
    """Test cases với realistic raw LLM outputs và expected refined structures"""
    
    @pytest.fixture
    def sample_raw_llm_outputs(self):
        """Sample raw LLM outputs từ different scenarios"""
        return {
            'simple_suggestion': {
                'solution': 'Add type hints to function parameters',
                'reason': 'Improves code readability and IDE support',
                'priority': 'medium'
            },
            
            'complex_refactoring': {
                'suggestion': 'Break down the large calculate_metrics function into smaller, focused functions',
                'implementation': 'Use Extract Method pattern to separate concerns',
                'benefits': ['Improved testability', 'Better maintainability', 'Reduced complexity'],
                'estimated_time': '4-6 hours'
            },
            
            'security_fix': {
                'text': 'Replace string concatenation in SQL queries with parameterized queries to prevent SQL injection attacks',
                'severity': 'critical',
                'impact': 'Prevents potential data breaches'
            },
            
            'performance_optimization': {
                'recommendation': 'Optimize the nested loop structure by using dictionary lookups instead of linear searches',
                'details': 'Current O(n²) complexity can be reduced to O(n) with proper data structures',
                'expected_improvement': '80% performance gain for large datasets'
            },
            
            'minimal_suggestion': {
                'solution': 'Add docstring'
            },
            
            'verbose_suggestion': {
                'title': 'Comprehensive Error Handling Implementation',
                'description': 'The current function lacks proper error handling which can lead to unexpected crashes and poor user experience',
                'solution': 'Implement try-catch blocks with specific exception handling',
                'steps': [
                    'Identify potential failure points',
                    'Add appropriate exception handling',
                    'Implement logging for debugging',
                    'Add user-friendly error messages'
                ],
                'tools_needed': ['logging library', 'custom exception classes'],
                'time_estimate': '2-3 days',
                'risk_level': 'low'
            }
        }
    
    @pytest.fixture
    def sample_llm_responses(self):
        """Sample LLM responses trong different formats và quality levels"""
        return {
            'well_formatted': """**REFINED_TITLE:** Implement Type Hints for Enhanced Code Quality

**DESCRIPTION:** Adding type hints to function parameters will significantly improve code readability, enable better IDE support with autocomplete and error detection, and help catch type-related bugs during development rather than runtime.

**IMPLEMENTATION_STEPS:**
1. Analyze function signatures and identify parameter types
2. Add type hints using Python typing module
3. Include return type annotations
4. Update docstrings to reflect type information
5. Run mypy for type checking validation

**PREREQUISITES:**
- Understanding of Python typing module
- Knowledge of mypy static type checker
- Familiarity with type annotation syntax

**ESTIMATED_EFFORT:** 2-3 hours for basic type hints implementation

**IMPACT_LEVEL:** Medium - Improves developer experience and code quality

**RISK_ASSESSMENT:** Very low risk - Type hints are optional and don't affect runtime behavior

**CODE_EXAMPLES:**
```python
# Before: No type hints
def calculate_score(data, multiplier, include_bonus):
    return sum(data) * multiplier + (10 if include_bonus else 0)

# After: With type hints
def calculate_score(data: List[float], multiplier: float, include_bonus: bool) -> float:
    return sum(data) * multiplier + (10 if include_bonus else 0)
```

**RELATED_PATTERNS:** Static typing, Code documentation, IDE integration

**SUCCESS_METRICS:**
- All function parameters have type hints
- Mypy type checking passes without errors
- IDE provides better autocomplete suggestions

**CONFIDENCE_SCORE:** 0.90""",

            'partial_format': """**REFINED_TITLE:** Extract Method Refactoring for Complex Function

**DESCRIPTION:** The large function violates Single Responsibility Principle and is difficult to test and maintain.

**IMPLEMENTATION_STEPS:**
1. Identify logical groups of functionality
2. Extract each group into separate methods
3. Update tests for new methods

**ESTIMATED_EFFORT:** 4-6 hours

**CODE_EXAMPLES:**
```python
# Before: Large monolithic function
def process_data(data):
    # validation logic
    # processing logic  
    # formatting logic
    pass

# After: Extracted methods
def validate_data(data): pass
def process_data_core(data): pass
def format_results(data): pass
```

**CONFIDENCE_SCORE:** 0.85""",

            'minimal_format': """**REFINED_TITLE:** Add SQL Injection Protection

**DESCRIPTION:** Use parameterized queries instead of string formatting.

**IMPLEMENTATION_STEPS:**
1. Replace f-strings with parameterized queries
2. Update database calls

**CONFIDENCE_SCORE:** 0.95""",

            'malformed': """This is not a properly formatted response.
It doesn't follow the expected structure at all.
Just some random text about adding error handling.
Maybe mention try-catch blocks somewhere.
No proper sections or formatting.""",

            'mixed_quality': """**REFINED_TITLE:** Performance Optimization Using Data Structures

**DESCRIPTION:** Current nested loops create O(n²) complexity causing performance issues.

**IMPLEMENTATION_STEPS:**
1. Replace nested loops with dictionary lookups
2. Use sets for membership testing
3. Implement caching for expensive operations

Some unstructured text here that doesn't follow format.

**CODE_EXAMPLES:**
```python
# Optimized version
lookup_dict = {item.id: item for item in items}
```

**CONFIDENCE_SCORE:** 0.80"""
        }
    
    # ===== Test Raw Output Processing =====
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_process_simple_suggestion(self, mock_llm_provider, sample_raw_llm_outputs, sample_llm_responses):
        """Test processing simple suggestion với minimal information"""
        mock_provider_instance = Mock()
        mock_provider_instance.generate.return_value = LLMResponse(
            response=sample_llm_responses['well_formatted'],
            model="test-model",
            provider="test"
        )
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        result = agent._refine_single_solution(
            raw_solution=sample_raw_llm_outputs['simple_suggestion'],
            code_context="def calculate(x, y): return x + y",
            filename="simple.py"
        )
        
        # Verify structure
        assert result is not None
        assert isinstance(result, RefinedSolution)
        assert result.refined_title == "Implement Type Hints for Enhanced Code Quality"
        assert "type hints" in result.description.lower()
        assert len(result.implementation_steps) == 5
        assert "Analyze function signatures" in result.implementation_steps[0]
        assert result.confidence_score == 0.90
        assert len(result.code_examples) == 1
        assert "List[float]" in result.code_examples[0]
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_process_complex_refactoring(self, mock_llm_provider, sample_raw_llm_outputs, sample_llm_responses):
        """Test processing complex refactoring suggestion"""
        mock_provider_instance = Mock()
        mock_provider_instance.generate.return_value = LLMResponse(
            response=sample_llm_responses['partial_format'],
            model="test-model",
            provider="test"
        )
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        result = agent._refine_single_solution(
            raw_solution=sample_raw_llm_outputs['complex_refactoring'],
            code_context="def large_function(): pass",
            filename="complex.py"
        )
        
        # Verify structure và content
        assert result is not None
        assert "Extract Method" in result.refined_title
        assert "Single Responsibility" in result.description
        assert len(result.implementation_steps) == 3
        assert "4-6 hours" in result.estimated_effort
        assert result.confidence_score == 0.85
        
        # Verify code examples extracted
        assert len(result.code_examples) == 1
        assert "monolithic function" in result.code_examples[0]
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_process_security_fix(self, mock_llm_provider, sample_raw_llm_outputs, sample_llm_responses):
        """Test processing critical security fix"""
        mock_provider_instance = Mock()
        mock_provider_instance.generate.return_value = LLMResponse(
            response=sample_llm_responses['minimal_format'],
            model="test-model",
            provider="test"
        )
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        result = agent._refine_single_solution(
            raw_solution=sample_raw_llm_outputs['security_fix'],
            code_context="query = f'SELECT * FROM users WHERE id = {user_id}'",
            filename="security.py"
        )
        
        # Verify security-focused structure
        assert result is not None
        assert "SQL Injection" in result.refined_title
        assert "parameterized queries" in result.description
        assert len(result.implementation_steps) == 2
        assert result.confidence_score == 0.95
        
        # Should have high confidence for security fixes
        assert result.confidence_score >= 0.9
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_process_malformed_response(self, mock_llm_provider, sample_raw_llm_outputs, sample_llm_responses):
        """Test processing malformed LLM response"""
        mock_provider_instance = Mock()
        mock_provider_instance.generate.return_value = LLMResponse(
            response=sample_llm_responses['malformed'],
            model="test-model",
            provider="test"
        )
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        result = agent._refine_single_solution(
            raw_solution=sample_raw_llm_outputs['verbose_suggestion'],
            code_context="def error_prone(): pass",
            filename="error.py"
        )
        
        # Should still create result với default values
        assert result is not None
        assert result.refined_title == "Improved Solution"  # Default title
        assert result.description == ""  # No description found
        assert result.implementation_steps == []  # No steps found
        assert result.confidence_score == 0.5  # Default confidence
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_process_mixed_quality_response(self, mock_llm_provider, sample_raw_llm_outputs, sample_llm_responses):
        """Test processing response với mixed quality formatting"""
        mock_provider_instance = Mock()
        mock_provider_instance.generate.return_value = LLMResponse(
            response=sample_llm_responses['mixed_quality'],
            model="test-model",
            provider="test"
        )
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        result = agent._refine_single_solution(
            raw_solution=sample_raw_llm_outputs['performance_optimization'],
            code_context="for i in items: for j in other_items: pass",
            filename="performance.py"
        )
        
        # Should extract what it can
        assert result is not None
        assert "Performance Optimization" in result.refined_title
        assert "O(n²) complexity" in result.description
        assert len(result.implementation_steps) == 3
        assert result.confidence_score == 0.80
        assert len(result.code_examples) == 1
    
    # ===== Test Different Input Formats =====
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_extract_from_different_key_formats(self, mock_llm_provider, sample_raw_llm_outputs):
        """Test extraction từ different key formats trong raw solutions"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        test_cases = [
            (sample_raw_llm_outputs['simple_suggestion'], "Add type hints to function parameters"),
            (sample_raw_llm_outputs['complex_refactoring'], "Break down the large calculate_metrics function"),
            (sample_raw_llm_outputs['security_fix'], "Replace string concatenation in SQL queries"),
            (sample_raw_llm_outputs['performance_optimization'], "Optimize the nested loop structure"),
            (sample_raw_llm_outputs['minimal_suggestion'], "Add docstring"),
        ]
        
        for raw_solution, expected_content in test_cases:
            extracted = agent._extract_solution_text(raw_solution)
            assert expected_content in extracted
    
    # ===== Test Batch Processing =====
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_batch_processing_mixed_quality(self, mock_llm_provider, sample_raw_llm_outputs, sample_llm_responses):
        """Test batch processing với mixed quality responses"""
        mock_provider_instance = Mock()
        
        # Different responses for each call
        responses = [
            LLMResponse(sample_llm_responses['well_formatted'], "test", "test"),
            LLMResponse(sample_llm_responses['malformed'], "test", "test"),
            LLMResponse(sample_llm_responses['partial_format'], "test", "test"),
            LLMResponse(sample_llm_responses['minimal_format'], "test", "test"),
        ]
        mock_provider_instance.generate.side_effect = responses
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        raw_solutions = [
            sample_raw_llm_outputs['simple_suggestion'],
            sample_raw_llm_outputs['verbose_suggestion'],  # Will get malformed response
            sample_raw_llm_outputs['complex_refactoring'],
            sample_raw_llm_outputs['security_fix'],
        ]
        
        results = agent.refine_solutions(
            raw_solutions=raw_solutions,
            code_context="def sample(): pass",
            filename="batch.py"
        )
        
        # Should have 4 results (malformed still creates result với defaults)
        assert len(results) == 4
        
        # First result should be well-formatted
        assert results[0].refined_title == "Implement Type Hints for Enhanced Code Quality"
        assert results[0].confidence_score == 0.90
        
        # Second result should have defaults (malformed response)
        assert results[1].refined_title == "Improved Solution"
        assert results[1].confidence_score == 0.5
        
        # Third result should be partial
        assert "Extract Method" in results[2].refined_title
        assert results[2].confidence_score == 0.85
        
        # Fourth result should be minimal
        assert "SQL Injection" in results[3].refined_title
        assert results[3].confidence_score == 0.95
    
    # ===== Test Edge Cases =====
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_very_long_response_handling(self, mock_llm_provider):
        """Test handling của very long LLM responses"""
        mock_provider_instance = Mock()
        
        # Create very long response
        long_response = """**REFINED_TITLE:** Very Long Solution Title That Goes On And On

**DESCRIPTION:** """ + "This is a very long description. " * 100 + """

**IMPLEMENTATION_STEPS:**
""" + "\n".join([f"{i}. Step {i} with detailed explanation that goes on for a while" for i in range(1, 21)]) + """

**CODE_EXAMPLES:**
```python
# Very long code example
""" + "\n".join([f"def function_{i}(): pass" for i in range(50)]) + """
```

**CONFIDENCE_SCORE:** 0.75"""
        
        mock_provider_instance.generate.return_value = LLMResponse(
            response=long_response,
            model="test-model",
            provider="test"
        )
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        result = agent._refine_single_solution(
            raw_solution={'solution': 'Complex solution'},
            code_context="def test(): pass",
            filename="long.py"
        )
        
        # Should handle long content gracefully
        assert result is not None
        assert len(result.implementation_steps) == 20
        assert len(result.code_examples) == 1
        assert "function_0" in result.code_examples[0]
        assert result.confidence_score == 0.75
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_unicode_and_special_characters(self, mock_llm_provider):
        """Test handling của unicode và special characters"""
        mock_provider_instance = Mock()
        
        unicode_response = """**REFINED_TITLE:** Thêm Unicode Support và Xử lý Ký tự Đặc biệt

**DESCRIPTION:** Cải thiện hỗ trợ Unicode và xử lý các ký tự đặc biệt như: áéíóú, 中文, 🚀, ñ, ß

**IMPLEMENTATION_STEPS:**
1. Sử dụng UTF-8 encoding
2. Thêm validation cho special characters: @#$%^&*()
3. Test với emoji: 😀🎉🔥

**CODE_EXAMPLES:**
```python
# Unicode string handling
text = "Xin chào! 你好! Hello! 🌍"
processed = handle_unicode(text)
```

**CONFIDENCE_SCORE:** 0.88"""
        
        mock_provider_instance.generate.return_value = LLMResponse(
            response=unicode_response,
            model="test-model",
            provider="test"
        )
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        result = agent._refine_single_solution(
            raw_solution={'solution': 'Add unicode support'},
            code_context="def process_text(): pass",
            filename="unicode.py"
        )
        
        # Should handle unicode correctly
        assert result is not None
        assert "Unicode Support" in result.refined_title
        assert "áéíóú" in result.description
        assert "UTF-8" in result.implementation_steps[0]
        assert "🌍" in result.code_examples[0]
        assert result.confidence_score == 0.88
    
    # ===== Test Prompt Quality Impact =====
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_prompt_context_impact(self, mock_llm_provider, sample_llm_responses):
        """Test how different context affects prompt quality"""
        mock_provider_instance = Mock()
        mock_provider_instance.generate.return_value = LLMResponse(
            response=sample_llm_responses['well_formatted'],
            model="test-model",
            provider="test"
        )
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        # Test với rich context
        rich_context = {
            'static_issues': {
                'complexity': [{'message': 'High complexity', 'value': 15}],
                'security': [{'message': 'SQL injection risk'}]
            },
            'metrics': {
                'code_quality_score': 45.2,
                'cyclomatic_complexity': 18,
                'lines_of_code': 250
            }
        }
        
        result_with_context = agent._refine_single_solution(
            raw_solution={'solution': 'Improve code quality'},
            code_context="def complex_function(): pass",
            filename="complex.py",
            static_analysis_context=rich_context
        )
        
        # Test without context
        result_without_context = agent._refine_single_solution(
            raw_solution={'solution': 'Improve code quality'},
            code_context="def simple_function(): pass",
            filename="simple.py",
            static_analysis_context=None
        )
        
        # Both should succeed but prompts should be different
        assert result_with_context is not None
        assert result_without_context is not None
        
        # Verify LLM was called twice với different prompts
        assert mock_provider_instance.generate.call_count == 2
        
        # Check that prompts were different (rich context vs minimal)
        call_args = mock_provider_instance.generate.call_args_list
        prompt_with_context = call_args[0][1]['prompt']
        prompt_without_context = call_args[1][1]['prompt']
        
        assert "Code Quality Score: 45.2" in prompt_with_context
        assert "Total Issues Found: 2" in prompt_with_context
        assert "Code Quality Score: N/A" in prompt_without_context


if __name__ == "__main__":
    pytest.main([__file__]) 