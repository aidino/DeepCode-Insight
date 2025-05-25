"""
Test cases cho SolutionSuggestionAgent
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from deepcode_insight.agents.solution_suggester import (
    SolutionSuggestionAgent,
    RefinedSolution,
    create_solution_suggester_agent,
    solution_suggester_node
)
from deepcode_insight.utils.llm_interface import LLMResponse


class TestSolutionSuggestionAgent:
    """Test cases cho SolutionSuggestionAgent"""
    
    @pytest.fixture
    def sample_raw_solutions(self):
        """Sample raw solutions for testing"""
        return [
            {
                'solution': 'Add docstrings',
                'implementation': 'Use triple quotes',
                'benefit': 'Better documentation'
            },
            {
                'solution': 'Refactor complex function',
                'implementation': 'Break into smaller functions',
                'benefit': 'Reduced complexity'
            },
            {
                'solution': 'Fix security issue',
                'implementation': 'Use parameterized queries',
                'benefit': 'Prevent SQL injection'
            }
        ]
    
    @pytest.fixture
    def sample_code_content(self):
        """Sample code content for testing"""
        return '''
def calculate_metrics(data):
    # Missing docstring
    result = 0
    for item in data:
        if item > 0:
            result += item * 2
        elif item < 0:
            result -= item
        else:
            result += 1
    return result

def query_database(user_input):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    return execute_query(query)
'''
    
    @pytest.fixture
    def sample_static_analysis_context(self):
        """Sample static analysis context"""
        return {
            'static_issues': {
                'missing_docstrings': [
                    {'message': 'Function missing docstring', 'line': 1}
                ],
                'security_issues': [
                    {'message': 'SQL injection risk', 'line': 12}
                ]
            },
            'metrics': {
                'code_quality_score': 65.5,
                'cyclomatic_complexity': 8,
                'lines_of_code': 15
            }
        }
    
    @pytest.fixture
    def mock_refined_llm_response(self):
        """Mock refined LLM response"""
        return LLMResponse(
            response="""**REFINED_TITLE:** Implement Comprehensive Function Documentation

**DESCRIPTION:** The function calculate_metrics lacks proper documentation, making it difficult for other developers to understand its purpose, parameters, and return values. Adding comprehensive docstrings will improve code maintainability and team productivity.

**IMPLEMENTATION_STEPS:**
1. Add function docstring using Google/Sphinx style format
2. Document all parameters with types and descriptions
3. Document return value with type and description
4. Add usage examples if complex logic is involved
5. Include any exceptions that might be raised

**PREREQUISITES:**
- Understanding of Python docstring conventions
- Knowledge of Google or Sphinx documentation style
- Access to function implementation details

**ESTIMATED_EFFORT:** 2-4 hours - Simple documentation task that can be completed quickly

**IMPACT_LEVEL:** Medium - Improves maintainability and team productivity without affecting functionality

**RISK_ASSESSMENT:** Very low risk - Documentation changes don't affect code execution. Only risk is potential merge conflicts if multiple developers modify same functions.

**CODE_EXAMPLES:**
```python
# Before: Missing docstring
def calculate_metrics(data):
    result = 0
    for item in data:
        if item > 0:
            result += item * 2
    return result

# After: Comprehensive documentation
def calculate_metrics(data: List[Union[int, float]]) -> Union[int, float]:
    \"\"\"
    Calculate weighted metrics from input data.
    
    Args:
        data: List of numeric values to process
        
    Returns:
        Calculated metric value based on weighted algorithm
        
    Raises:
        TypeError: If data contains non-numeric values
    \"\"\"
    result = 0
    for item in data:
        if item > 0:
            result += item * 2
    return result
```

**RELATED_PATTERNS:** Documentation patterns, Type hints, Clean code principles

**SUCCESS_METRICS:**
- All functions have comprehensive docstrings
- Documentation follows consistent style guide
- Code review feedback on documentation quality improves

**CONFIDENCE_SCORE:** 0.95""",
            model="test-model",
            provider="test"
        )
    
    # ===== Basic Initialization Tests =====
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_init_with_default_settings(self, mock_llm_provider):
        """Test initialization với default settings"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        assert agent.provider == "ollama"
        assert agent.model == "codellama"
        assert agent.temperature == 0.3
        assert agent.max_tokens == 2000
        assert agent.llm_provider == mock_provider_instance
        
        mock_llm_provider.assert_called_once_with(
            provider="ollama",
            model="codellama",
            api_key=None,
            base_url=None
        )
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_init_with_custom_settings(self, mock_llm_provider):
        """Test initialization với custom settings"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            temperature=0.5,
            max_tokens=1500
        )
        
        assert agent.provider == "openai"
        assert agent.model == "gpt-4"
        assert agent.temperature == 0.5
        assert agent.max_tokens == 1500
        
        mock_llm_provider.assert_called_once_with(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
            base_url=None
        )
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_init_failure_handling(self, mock_llm_provider):
        """Test handling của initialization failure"""
        mock_llm_provider.side_effect = Exception("Provider initialization failed")
        
        with pytest.raises(Exception) as exc_info:
            SolutionSuggestionAgent()
        
        assert "Provider initialization failed" in str(exc_info.value)
    
    # ===== Solution Text Extraction Tests =====
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_extract_solution_text_standard_keys(self, mock_llm_provider):
        """Test extraction với standard keys"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        # Test different standard keys
        test_cases = [
            ({'solution': 'Add docstrings'}, 'Add docstrings'),
            ({'suggestion': 'Refactor code'}, 'Refactor code'),
            ({'recommendation': 'Use type hints'}, 'Use type hints'),
            ({'text': 'Fix security issue'}, 'Fix security issue'),
            ({'description': 'Optimize performance'}, 'Optimize performance')
        ]
        
        for raw_solution, expected in test_cases:
            result = agent._extract_solution_text(raw_solution)
            assert result == expected
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_extract_solution_text_concatenation(self, mock_llm_provider):
        """Test extraction với concatenation fallback"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        raw_solution = {
            'title': 'Fix Issue',
            'details': 'Add proper validation',
            'priority': 'High'
        }
        
        result = agent._extract_solution_text(raw_solution)
        assert 'title: Fix Issue' in result
        assert 'details: Add proper validation' in result
        assert 'priority: High' in result
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_extract_solution_text_string_fallback(self, mock_llm_provider):
        """Test extraction với string fallback"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        # Test với non-dict input
        result = agent._extract_solution_text("Simple string solution")
        assert result == "Simple string solution"
    
    # ===== Prompt Creation Tests =====
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_create_refinement_prompt_basic(self, mock_llm_provider, 
                                           sample_code_content,
                                           sample_static_analysis_context):
        """Test basic refinement prompt creation"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        prompt = agent._create_refinement_prompt(
            solution_text="Add docstrings to functions",
            code_context=sample_code_content,
            filename="test.py",
            static_analysis_context=sample_static_analysis_context
        )
        
        # Verify prompt contains required elements
        assert "test.py" in prompt
        assert "Add docstrings to functions" in prompt
        assert "Code Quality Score: 65.5" in prompt
        assert "Complexity: 8" in prompt
        assert "Total Issues Found: 2" in prompt
        assert "REFINED_TITLE" in prompt
        assert "IMPLEMENTATION_STEPS" in prompt
        assert "CONFIDENCE_SCORE" in prompt
        assert "senior software engineer" in prompt
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_create_refinement_prompt_without_context(self, mock_llm_provider):
        """Test prompt creation without static analysis context"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        prompt = agent._create_refinement_prompt(
            solution_text="Fix security issue",
            code_context="def test(): pass",
            filename="simple.py",
            static_analysis_context=None
        )
        
        assert "simple.py" in prompt
        assert "Fix security issue" in prompt
        assert "Code Quality Score: N/A" in prompt
        assert "Complexity: N/A" in prompt
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_create_refinement_prompt_long_code_truncation(self, mock_llm_provider):
        """Test code truncation trong prompt"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        # Create long code content
        long_code = "def function():\n    pass\n" * 100  # > 500 chars
        
        prompt = agent._create_refinement_prompt(
            solution_text="Optimize code",
            code_context=long_code,
            filename="long.py"
        )
        
        # Should be truncated với "..."
        assert "..." in prompt
        assert len(prompt) < len(long_code) + 1000  # Much shorter than original
    
    # ===== Response Parsing Tests =====
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_parse_refined_solution_complete(self, mock_llm_provider):
        """Test parsing complete refined solution"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        response = """**REFINED_TITLE:** Add Comprehensive Documentation

**DESCRIPTION:** This solution addresses the lack of documentation in the codebase.

**IMPLEMENTATION_STEPS:**
1. Add function docstrings using Google style
2. Include parameter descriptions
3. Document return values

**PREREQUISITES:**
- Knowledge of Python docstring conventions
- Understanding of Google style guide

**ESTIMATED_EFFORT:** 2-4 hours for basic documentation

**IMPACT_LEVEL:** Medium - improves maintainability

**RISK_ASSESSMENT:** Low risk - documentation only changes

**CODE_EXAMPLES:**
```python
def example():
    pass
```

**RELATED_PATTERNS:** Clean code, Documentation patterns

**SUCCESS_METRICS:**
- All functions documented
- Consistent style applied

**CONFIDENCE_SCORE:** 0.85"""
        
        result = agent._parse_refined_solution(response, "Original solution")
        
        assert result is not None
        assert result.refined_title == "Add Comprehensive Documentation"
        assert "lack of documentation" in result.description
        assert len(result.implementation_steps) == 3
        assert "Add function docstrings using Google style" in result.implementation_steps[0]
        assert len(result.prerequisites) == 2
        assert "Knowledge of Python docstring conventions" in result.prerequisites[0]
        assert result.estimated_effort == "2-4 hours for basic documentation"
        assert result.impact_level == "Medium - improves maintainability"
        assert "Low risk" in result.risk_assessment
        assert len(result.code_examples) == 1
        assert "def example():" in result.code_examples[0]
        assert len(result.related_patterns) == 2
        assert "Clean code" in result.related_patterns[0]
        assert len(result.success_metrics) == 2
        assert result.confidence_score == 0.85
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_parse_refined_solution_partial(self, mock_llm_provider):
        """Test parsing partial refined solution"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        # Incomplete response
        response = """**REFINED_TITLE:** Partial Solution

**DESCRIPTION:** This is a partial solution.

**IMPLEMENTATION_STEPS:**
1. Step one
2. Step two"""
        
        result = agent._parse_refined_solution(response, "Original")
        
        assert result is not None
        assert result.refined_title == "Partial Solution"
        assert result.description == "This is a partial solution."
        assert len(result.implementation_steps) == 2
        # Default values for missing sections
        assert result.estimated_effort == "Unknown"
        assert result.impact_level == "Medium"
        assert result.confidence_score == 0.5
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_parse_refined_solution_malformed(self, mock_llm_provider):
        """Test parsing malformed response"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        # Malformed response
        response = "This is not a properly formatted response at all."
        
        result = agent._parse_refined_solution(response, "Original")
        
        # Should still create object với default values
        assert result is not None
        assert result.refined_title == "Improved Solution"
        assert result.description == ""
        assert result.implementation_steps == []
        assert result.confidence_score == 0.5
    
    # ===== Parsing Helper Tests =====
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_parse_numbered_list(self, mock_llm_provider):
        """Test numbered list parsing"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        text = """1. First step with details
2. Second step with more info
3. Third step
- Bullet point item
• Another bullet item"""
        
        result = agent._parse_numbered_list(text)
        
        assert len(result) == 5
        assert "First step with details" in result[0]
        assert "Second step with more info" in result[1]
        assert "Third step" in result[2]
        assert "Bullet point item" in result[3]
        assert "Another bullet item" in result[4]
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_parse_bullet_list(self, mock_llm_provider):
        """Test bullet list parsing"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        text = """- First requirement
• Second requirement
* Third requirement
Normal text line"""
        
        result = agent._parse_bullet_list(text)
        
        assert len(result) == 3
        assert "First requirement" in result[0]
        assert "Second requirement" in result[1]
        assert "Third requirement" in result[2]
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_extract_code_blocks(self, mock_llm_provider):
        """Test code block extraction"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        text = """Some text before
```python
def example():
    pass
```
Text between blocks
```javascript
function test() {
    return true;
}
```
Text after"""
        
        result = agent._extract_code_blocks(text)
        
        assert len(result) == 2
        assert "def example():" in result[0]
        assert "function test()" in result[1]
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_parse_confidence_score(self, mock_llm_provider):
        """Test confidence score parsing"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        test_cases = [
            ("0.85", 0.85),
            ("85%", 0.85),
            ("95", 0.95),
            ("1.0", 1.0),
            ("0.0", 0.0),
            ("invalid", 0.5),  # Default
            ("", 0.5)  # Default
        ]
        
        for input_text, expected in test_cases:
            result = agent._parse_confidence_score(input_text)
            assert result == expected
    
    # ===== Single Solution Refinement Tests =====
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_refine_single_solution_success(self, mock_llm_provider, 
                                           sample_code_content,
                                           mock_refined_llm_response):
        """Test successful single solution refinement"""
        mock_provider_instance = Mock()
        mock_provider_instance.generate.return_value = mock_refined_llm_response
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        raw_solution = {
            'solution': 'Add docstrings',
            'implementation': 'Use triple quotes',
            'benefit': 'Better documentation'
        }
        
        result = agent._refine_single_solution(
            raw_solution=raw_solution,
            code_context=sample_code_content,
            filename="test.py",
            static_analysis_context=None
        )
        
        assert result is not None
        assert isinstance(result, RefinedSolution)
        assert result.refined_title == "Implement Comprehensive Function Documentation"
        assert "calculate_metrics lacks proper documentation" in result.description
        assert len(result.implementation_steps) > 0
        assert result.confidence_score == 0.95
        
        # Verify LLM was called
        mock_provider_instance.generate.assert_called_once()
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_refine_single_solution_empty_text(self, mock_llm_provider):
        """Test refinement với empty solution text"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        raw_solution = {}  # Completely empty dict
        
        result = agent._refine_single_solution(
            raw_solution=raw_solution,
            code_context="def test(): pass",
            filename="test.py"
        )
        
        assert result is None
        # LLM should not be called for empty solution
        mock_provider_instance.generate.assert_not_called()
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_refine_single_solution_with_non_standard_keys(self, mock_llm_provider):
        """Test refinement với non-standard keys that get concatenated"""
        mock_provider_instance = Mock()
        mock_provider_instance.generate.side_effect = Exception("Parsing error")
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        raw_solution = {'other_field': 'value'}  # Non-standard key
        
        result = agent._refine_single_solution(
            raw_solution=raw_solution,
            code_context="def test(): pass",
            filename="test.py"
        )
        
        # The extraction will create "other_field: value" which is not empty
        # So LLM will be called, but parsing will fail and return None
        assert result is None
        # LLM should be called since text is extracted
        mock_provider_instance.generate.assert_called_once()
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_refine_single_solution_llm_error(self, mock_llm_provider):
        """Test refinement với LLM error"""
        mock_provider_instance = Mock()
        mock_provider_instance.generate.side_effect = Exception("LLM error")
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        raw_solution = {'solution': 'Test solution'}
        
        result = agent._refine_single_solution(
            raw_solution=raw_solution,
            code_context="def test(): pass",
            filename="test.py"
        )
        
        assert result is None
    
    # ===== Multiple Solutions Refinement Tests =====
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_refine_solutions_success(self, mock_llm_provider, 
                                     sample_raw_solutions,
                                     sample_code_content,
                                     mock_refined_llm_response):
        """Test successful refinement của multiple solutions"""
        mock_provider_instance = Mock()
        mock_provider_instance.generate.return_value = mock_refined_llm_response
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        results = agent.refine_solutions(
            raw_solutions=sample_raw_solutions,
            code_context=sample_code_content,
            filename="test.py"
        )
        
        assert len(results) == 3  # All solutions refined successfully
        for result in results:
            assert isinstance(result, RefinedSolution)
            assert result.refined_title != ""
            assert result.confidence_score > 0
        
        # Verify LLM called for each solution
        assert mock_provider_instance.generate.call_count == 3
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_refine_solutions_partial_failure(self, mock_llm_provider, 
                                             sample_raw_solutions,
                                             sample_code_content,
                                             mock_refined_llm_response):
        """Test refinement với partial failures"""
        mock_provider_instance = Mock()
        
        # First call succeeds, second fails, third succeeds
        mock_provider_instance.generate.side_effect = [
            mock_refined_llm_response,
            Exception("Network error"),
            mock_refined_llm_response
        ]
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        results = agent.refine_solutions(
            raw_solutions=sample_raw_solutions,
            code_context=sample_code_content,
            filename="test.py"
        )
        
        # Should have 2 successful results (1st and 3rd)
        assert len(results) == 2
        assert mock_provider_instance.generate.call_count == 3
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_refine_solutions_empty_input(self, mock_llm_provider):
        """Test refinement với empty input"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        results = agent.refine_solutions(
            raw_solutions=[],
            code_context="def test(): pass",
            filename="test.py"
        )
        
        assert len(results) == 0
        mock_provider_instance.generate.assert_not_called()
    
    # ===== LangGraph Integration Tests =====
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_process_solutions_success(self, mock_llm_provider,
                                      sample_raw_solutions,
                                      sample_code_content,
                                      mock_refined_llm_response):
        """Test successful LangGraph node processing"""
        mock_provider_instance = Mock()
        mock_provider_instance.generate.return_value = mock_refined_llm_response
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        state = {
            'llm_analysis': {
                'solution_suggestions': sample_raw_solutions
            },
            'code_content': sample_code_content,
            'filename': 'test.py',
            'static_analysis_results': {
                'static_issues': {},
                'metrics': {'code_quality_score': 75}
            }
        }
        
        result = agent.process_solutions(state)
        
        # Verify state updates
        assert result['current_agent'] == 'solution_suggester'
        assert result['processing_status'] == 'solution_refinement_completed'
        assert 'refined_solutions' in result
        assert len(result['refined_solutions']) == 3
        
        # Verify metadata
        metadata = result['refinement_metadata']
        assert metadata['total_raw_solutions'] == 3
        assert metadata['successfully_refined'] == 3
        assert metadata['refinement_success_rate'] == 1.0
        assert metadata['provider'] == 'ollama'
        assert metadata['model'] == 'codellama'
        
        # Verify refined solution structure
        refined_solution = result['refined_solutions'][0]
        assert 'original_solution' in refined_solution
        assert 'refined_title' in refined_solution
        assert 'implementation_steps' in refined_solution
        assert 'confidence_score' in refined_solution
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_process_solutions_no_raw_solutions(self, mock_llm_provider):
        """Test processing khi không có raw solutions"""
        mock_provider_instance = Mock()
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        state = {
            'llm_analysis': {},  # No solution_suggestions
            'code_content': 'def test(): pass',
            'filename': 'test.py'
        }
        
        result = agent.process_solutions(state)
        
        assert result['processing_status'] == 'no_solutions_to_refine'
        assert result['refined_solutions'] == []
        mock_provider_instance.generate.assert_not_called()
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_process_solutions_error_handling(self, mock_llm_provider):
        """Test error handling trong process_solutions"""
        mock_provider_instance = Mock()
        mock_provider_instance.generate.side_effect = Exception("Processing error")
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        state = {
            'llm_analysis': {
                'solution_suggestions': [{'solution': 'test'}]
            },
            'code_content': 'def test(): pass',
            'filename': 'test.py'
        }
        
        result = agent.process_solutions(state)
        
        # Agent handles individual solution errors gracefully
        # So overall processing completes but with 0 successful refinements
        assert result['processing_status'] == 'solution_refinement_completed'
        assert result['refined_solutions'] == []
        
        # Check metadata shows 0 success rate
        metadata = result['refinement_metadata']
        assert metadata['successfully_refined'] == 0
        assert metadata['refinement_success_rate'] == 0.0
    
    # ===== Health Check Tests =====
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_check_health_success(self, mock_llm_provider):
        """Test successful health check"""
        mock_provider_instance = Mock()
        mock_provider_instance.check_health.return_value = True
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        result = agent.check_health()
        
        assert result == True
        mock_provider_instance.check_health.assert_called_once()
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_check_health_failure(self, mock_llm_provider):
        """Test health check failure"""
        mock_provider_instance = Mock()
        mock_provider_instance.check_health.side_effect = Exception("Health check failed")
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        result = agent.check_health()
        
        assert result == False
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_get_available_models_success(self, mock_llm_provider):
        """Test getting available models"""
        mock_provider_instance = Mock()
        mock_provider_instance.list_models.return_value = ['model1', 'model2', 'model3']
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        result = agent.get_available_models()
        
        assert result == ['model1', 'model2', 'model3']
        mock_provider_instance.list_models.assert_called_once()
    
    @patch('deepcode_insight.agents.solution_suggester.create_llm_provider')
    def test_get_available_models_failure(self, mock_llm_provider):
        """Test get available models failure"""
        mock_provider_instance = Mock()
        mock_provider_instance.list_models.side_effect = Exception("List models failed")
        mock_llm_provider.return_value = mock_provider_instance
        
        agent = SolutionSuggestionAgent()
        
        result = agent.get_available_models()
        
        assert result == []
    
    # ===== Factory Function Tests =====
    
    def test_create_solution_suggester_agent_default(self):
        """Test factory function với default parameters"""
        with patch('deepcode_insight.agents.solution_suggester.SolutionSuggestionAgent') as mock_agent:
            create_solution_suggester_agent()
            
            mock_agent.assert_called_once_with(
                provider="ollama",
                model="codellama"
            )
    
    def test_create_solution_suggester_agent_custom(self):
        """Test factory function với custom parameters"""
        with patch('deepcode_insight.agents.solution_suggester.SolutionSuggestionAgent') as mock_agent:
            create_solution_suggester_agent(
                provider="openai",
                model="gpt-4",
                api_key="test-key",
                temperature=0.5
            )
            
            mock_agent.assert_called_once_with(
                provider="openai",
                model="gpt-4",
                api_key="test-key",
                temperature=0.5
            )
    
    def test_solution_suggester_node(self):
        """Test LangGraph node function"""
        test_state = {
            'llm_analysis': {
                'solution_suggestions': [{'solution': 'test'}]
            },
            'code_content': 'def test(): pass',
            'filename': 'test.py'
        }
        
        with patch('deepcode_insight.agents.solution_suggester.create_solution_suggester_agent') as mock_create:
            mock_agent = Mock()
            mock_agent.process_solutions.return_value = {**test_state, 'processed': True}
            mock_create.return_value = mock_agent
            
            result = solution_suggester_node(test_state)
            
            assert result['processed'] == True
            mock_create.assert_called_once()
            mock_agent.process_solutions.assert_called_once_with(test_state)


class TestRefinedSolutionDataclass:
    """Test cases cho RefinedSolution dataclass"""
    
    def test_refined_solution_creation(self):
        """Test RefinedSolution creation"""
        solution = RefinedSolution(
            original_solution="Add docstrings",
            refined_title="Implement Comprehensive Documentation",
            description="Detailed description",
            implementation_steps=["Step 1", "Step 2"],
            prerequisites=["Req 1", "Req 2"],
            estimated_effort="2-4 hours",
            impact_level="Medium",
            risk_assessment="Low risk",
            code_examples=["example code"],
            related_patterns=["Pattern 1"],
            success_metrics=["Metric 1"],
            confidence_score=0.85
        )
        
        assert solution.original_solution == "Add docstrings"
        assert solution.refined_title == "Implement Comprehensive Documentation"
        assert len(solution.implementation_steps) == 2
        assert solution.confidence_score == 0.85
    
    def test_refined_solution_serialization(self):
        """Test RefinedSolution có thể serialize"""
        solution = RefinedSolution(
            original_solution="Test",
            refined_title="Test Title",
            description="Test Description",
            implementation_steps=["Step 1"],
            prerequisites=["Req 1"],
            estimated_effort="1 hour",
            impact_level="Low",
            risk_assessment="No risk",
            code_examples=["code"],
            related_patterns=["pattern"],
            success_metrics=["metric"],
            confidence_score=0.9
        )
        
        # Should be able to convert to dict
        solution_dict = {
            'original_solution': solution.original_solution,
            'refined_title': solution.refined_title,
            'description': solution.description,
            'implementation_steps': solution.implementation_steps,
            'prerequisites': solution.prerequisites,
            'estimated_effort': solution.estimated_effort,
            'impact_level': solution.impact_level,
            'risk_assessment': solution.risk_assessment,
            'code_examples': solution.code_examples,
            'related_patterns': solution.related_patterns,
            'success_metrics': solution.success_metrics,
            'confidence_score': solution.confidence_score
        }
        
        assert solution_dict['refined_title'] == "Test Title"
        assert solution_dict['confidence_score'] == 0.9


if __name__ == "__main__":
    pytest.main([__file__]) 