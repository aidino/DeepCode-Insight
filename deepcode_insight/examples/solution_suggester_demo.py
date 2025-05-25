"""
Demo script cho SolutionSuggestionAgent
Demonstrates how to refine raw LLM solutions into actionable suggestions
"""

import sys
import os
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from deepcode_insight.agents.solution_suggester import (
    SolutionSuggestionAgent,
    create_solution_suggester_agent,
    RefinedSolution
)


def create_sample_raw_solutions() -> List[Dict[str, Any]]:
    """Create sample raw solutions for demonstration"""
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
            'solution': 'Fix security vulnerability',
            'implementation': 'Use parameterized queries',
            'benefit': 'Prevent SQL injection'
        },
        {
            'solution': 'Optimize performance',
            'implementation': 'Use caching and indexing',
            'benefit': 'Faster response times'
        }
    ]


def create_sample_code_content() -> str:
    """Create sample code content for demonstration"""
    return '''
def calculate_user_metrics(user_id, start_date, end_date, include_deleted, 
                          filter_type, sort_order, limit, offset, 
                          include_metadata, format_output):
    # Missing docstring - complex function with many parameters
    
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    
    # Inefficient nested loops
    results = []
    for user in get_all_users():
        for metric in get_all_metrics():
            for date in get_date_range(start_date, end_date):
                if user.id == user_id and metric.date == date:
                    # Expensive calculation in loop
                    score = calculate_complex_score(user, metric, date)
                    results.append(score)
    
    return results

class DataProcessor:
    def __init__(self):
        self.cache = {}
    
    def process_data(self, data):
        # Missing error handling
        result = data.split(',')
        return [int(x) for x in result]
'''


def create_sample_static_analysis_context() -> Dict[str, Any]:
    """Create sample static analysis context"""
    return {
        'static_issues': {
            'missing_docstrings': [
                {
                    'type': 'missing_function_docstring',
                    'name': 'calculate_user_metrics',
                    'line': 1,
                    'message': 'Function missing docstring'
                }
            ],
            'security_issues': [
                {
                    'type': 'sql_injection_risk',
                    'line': 6,
                    'message': 'SQL injection vulnerability detected'
                }
            ],
            'performance_issues': [
                {
                    'type': 'inefficient_nested_loops',
                    'line': 9,
                    'message': 'Inefficient nested loop structure'
                }
            ],
            'complex_functions': [
                {
                    'type': 'too_many_parameters',
                    'name': 'calculate_user_metrics',
                    'line': 1,
                    'count': 10,
                    'message': 'Function has too many parameters'
                }
            ]
        },
        'metrics': {
            'code_quality_score': 45.2,
            'cyclomatic_complexity': 18,
            'lines_of_code': 25,
            'maintainability_index': 32.1,
            'comment_ratio': 0.05
        },
        'suggestions': [
            'Add comprehensive docstrings',
            'Fix security vulnerabilities',
            'Optimize performance bottlenecks',
            'Reduce function complexity'
        ]
    }


class MockLLMProvider:
    """Mock LLM provider for demonstration"""
    
    def __init__(self):
        self.model = "demo-model"
        self.call_count = 0
    
    def generate(self, prompt: str, **kwargs) -> 'MockLLMResponse':
        """Generate mock refined solution response"""
        self.call_count += 1
        
        # Create different responses based on call count
        if self.call_count == 1:
            response = self._create_docstring_response()
        elif self.call_count == 2:
            response = self._create_refactor_response()
        elif self.call_count == 3:
            response = self._create_security_response()
        else:
            response = self._create_performance_response()
        
        return MockLLMResponse(response)
    
    def _create_docstring_response(self) -> str:
        return """**REFINED_TITLE:** Implement Comprehensive Function Documentation

**DESCRIPTION:** The function calculate_user_metrics lacks proper documentation, making it difficult for developers to understand its complex parameter requirements and return values. Adding comprehensive docstrings will significantly improve code maintainability and reduce onboarding time for new team members.

**IMPLEMENTATION_STEPS:**
1. Add function docstring using Google/Sphinx style format
2. Document all 10 parameters with their types and purposes
3. Document return value structure and data types
4. Add usage examples for common scenarios
5. Include information about performance considerations
6. Document any exceptions that might be raised

**PREREQUISITES:**
- Understanding of Python docstring conventions
- Knowledge of Google or Sphinx documentation style
- Access to function requirements and specifications
- Understanding of the business logic behind user metrics

**ESTIMATED_EFFORT:** 4-6 hours - Comprehensive documentation for complex function with research time

**IMPACT_LEVEL:** High - Significantly improves maintainability and team productivity

**RISK_ASSESSMENT:** Very low risk - Documentation changes don't affect functionality. Only risk is potential merge conflicts if multiple developers modify the same function simultaneously.

**CODE_EXAMPLES:**
```python
# Before: Missing docstring
def calculate_user_metrics(user_id, start_date, end_date, include_deleted, 
                          filter_type, sort_order, limit, offset, 
                          include_metadata, format_output):
    query = f"SELECT * FROM users WHERE id = '{user_id}'"

# After: Comprehensive documentation
def calculate_user_metrics(
    user_id: str, 
    start_date: datetime, 
    end_date: datetime,
    include_deleted: bool = False,
    filter_type: str = 'active',
    sort_order: str = 'asc',
    limit: int = 100,
    offset: int = 0,
    include_metadata: bool = True,
    format_output: str = 'json'
) -> List[Dict[str, Any]]:
    \"\"\"
    Calculate comprehensive user metrics for specified time period.
    
    This function retrieves and processes user activity data to generate
    various metrics including engagement scores, activity patterns, and
    performance indicators.
    
    Args:
        user_id: Unique identifier for the user
        start_date: Beginning of analysis period
        end_date: End of analysis period
        include_deleted: Whether to include deleted records
        filter_type: Type of filtering ('active', 'all', 'premium')
        sort_order: Sort direction ('asc', 'desc')
        limit: Maximum number of results to return
        offset: Number of results to skip for pagination
        include_metadata: Whether to include additional metadata
        format_output: Output format ('json', 'csv', 'xml')
        
    Returns:
        List of dictionaries containing user metrics data
        
    Raises:
        ValueError: If date range is invalid or user_id is malformed
        DatabaseError: If database connection fails
        
    Example:
        >>> metrics = calculate_user_metrics(
        ...     user_id="user123",
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 1, 31)
        ... )
        >>> print(len(metrics))
        25
    \"\"\"
```

**RELATED_PATTERNS:** Documentation patterns, Type hints, Clean code principles, API documentation standards

**SUCCESS_METRICS:**
- All function parameters documented with types and descriptions
- Documentation follows consistent style guide
- Code review feedback on documentation quality improves
- New team member onboarding time reduces by 30%
- Function usage errors decrease due to clear documentation

**CONFIDENCE_SCORE:** 0.95"""

    def _create_refactor_response(self) -> str:
        return """**REFINED_TITLE:** Decompose Complex Function Using Extract Method Pattern

**DESCRIPTION:** The calculate_user_metrics function violates the Single Responsibility Principle by handling too many concerns: parameter validation, database querying, data processing, and result formatting. This creates maintenance challenges and makes testing difficult. Applying the Extract Method refactoring pattern will improve code organization and testability.

**IMPLEMENTATION_STEPS:**
1. Extract parameter validation into separate validate_user_metrics_params() function
2. Create dedicated get_user_data() function for database operations
3. Extract data processing logic into process_user_metrics() function
4. Create format_metrics_output() function for result formatting
5. Refactor main function to orchestrate these smaller functions
6. Add comprehensive unit tests for each extracted function
7. Update integration tests to verify end-to-end functionality

**PREREQUISITES:**
- Understanding of Extract Method refactoring pattern
- Knowledge of SOLID principles, especially Single Responsibility
- Access to existing test suite for regression testing
- Understanding of the current function's business logic

**ESTIMATED_EFFORT:** 2-3 days - Includes refactoring, testing, and documentation updates

**IMPACT_LEVEL:** High - Significantly improves code maintainability, testability, and readability

**RISK_ASSESSMENT:** Medium risk - Refactoring complex logic requires careful testing. Mitigation: comprehensive test coverage before and after refactoring, gradual implementation with feature flags.

**CODE_EXAMPLES:**
```python
# Before: Monolithic function
def calculate_user_metrics(user_id, start_date, end_date, include_deleted, 
                          filter_type, sort_order, limit, offset, 
                          include_metadata, format_output):
    # All logic in one place - hard to test and maintain

# After: Decomposed functions
def validate_user_metrics_params(user_id: str, start_date: datetime, 
                                end_date: datetime) -> None:
    \"\"\"Validate input parameters for user metrics calculation.\"\"\"
    if not user_id or not isinstance(user_id, str):
        raise ValueError("Invalid user_id")
    if start_date >= end_date:
        raise ValueError("start_date must be before end_date")

def get_user_data(user_id: str, start_date: datetime, end_date: datetime,
                 include_deleted: bool = False) -> List[Dict]:
    \"\"\"Retrieve user data from database with proper parameterization.\"\"\"
    query = "SELECT * FROM users WHERE id = ? AND date BETWEEN ? AND ?"
    params = [user_id, start_date, end_date]
    if not include_deleted:
        query += " AND deleted = FALSE"
    return execute_query(query, params)

def calculate_user_metrics(user_id: str, start_date: datetime, 
                          end_date: datetime, **options) -> List[Dict]:
    \"\"\"Main orchestration function - now much cleaner.\"\"\"
    validate_user_metrics_params(user_id, start_date, end_date)
    raw_data = get_user_data(user_id, start_date, end_date, 
                            options.get('include_deleted', False))
    processed_data = process_user_metrics(raw_data, options)
    return format_metrics_output(processed_data, options.get('format_output', 'json'))
```

**RELATED_PATTERNS:** Extract Method, Single Responsibility Principle, Command Query Separation, Strategy Pattern

**SUCCESS_METRICS:**
- Function complexity reduced from 18 to under 5 per function
- Test coverage increases to over 90%
- Code review time decreases by 40%
- Bug reports related to this functionality decrease by 60%
- New feature development in this area becomes 50% faster

**CONFIDENCE_SCORE:** 0.88"""

    def _create_security_response(self) -> str:
        return """**REFINED_TITLE:** Eliminate SQL Injection Vulnerability Using Parameterized Queries

**DESCRIPTION:** The current implementation uses string formatting to construct SQL queries, creating a critical security vulnerability that could allow attackers to execute arbitrary SQL commands. This poses severe risks including data theft, data corruption, and unauthorized access. Implementing parameterized queries will completely eliminate this attack vector.

**IMPLEMENTATION_STEPS:**
1. Replace all string formatting in SQL queries with parameterized placeholders
2. Update database connection code to use prepared statements
3. Implement input sanitization and validation as defense-in-depth
4. Add SQL injection detection to automated security testing
5. Conduct security code review of all database interaction code
6. Update logging to detect potential injection attempts
7. Create security documentation and training materials

**PREREQUISITES:**
- Understanding of SQL injection attack vectors and prevention
- Knowledge of database-specific parameterized query syntax
- Access to security testing tools and frameworks
- Understanding of secure coding practices

**ESTIMATED_EFFORT:** 1-2 days - Critical security fix requiring immediate attention

**IMPACT_LEVEL:** Critical - Prevents potential data breaches and system compromise

**RISK_ASSESSMENT:** Low implementation risk, extremely high risk if not fixed. Mitigation: thorough testing in staging environment, gradual rollout with monitoring, immediate security testing validation.

**CODE_EXAMPLES:**
```python
# Before: Vulnerable to SQL injection
def get_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    return execute_query(query)

# Attack example:
# user_id = "1'; DROP TABLE users; --"
# Results in: SELECT * FROM users WHERE id = '1'; DROP TABLE users; --'

# After: Secure parameterized queries
def get_user_data(user_id: str) -> List[Dict]:
    \"\"\"Securely retrieve user data using parameterized queries.\"\"\"
    # Input validation
    if not user_id or not isinstance(user_id, str):
        raise ValueError("Invalid user_id")
    
    # Parameterized query - safe from injection
    query = "SELECT * FROM users WHERE id = ?"
    return execute_query(query, [user_id])

# Alternative with named parameters (depending on database)
def get_user_data_named(user_id: str) -> List[Dict]:
    query = "SELECT * FROM users WHERE id = :user_id"
    return execute_query(query, {"user_id": user_id})
```

**RELATED_PATTERNS:** Parameterized Queries, Input Validation, Defense in Depth, Secure Coding Practices

**SUCCESS_METRICS:**
- Zero SQL injection vulnerabilities in security scans
- All database queries use parameterized statements
- Security test suite passes 100%
- No injection attempts succeed in penetration testing
- Code passes OWASP security guidelines compliance

**CONFIDENCE_SCORE:** 0.98"""

    def _create_performance_response(self) -> str:
        return """**REFINED_TITLE:** Optimize Nested Loop Performance Using Efficient Data Structures

**DESCRIPTION:** The current implementation uses inefficient nested loops with O(n³) time complexity, causing severe performance degradation as data size increases. The algorithm performs redundant database calls and expensive calculations within loops. Optimizing with proper data structures and algorithms will improve performance by orders of magnitude.

**IMPLEMENTATION_STEPS:**
1. Replace nested loops with hash-based lookups using dictionaries/sets
2. Implement database query optimization with JOIN operations
3. Add caching layer for expensive calculations
4. Use list comprehensions and generator expressions where appropriate
5. Implement batch processing for large datasets
6. Add performance monitoring and profiling
7. Create performance benchmarks and regression tests

**PREREQUISITES:**
- Understanding of algorithm complexity and Big O notation
- Knowledge of Python data structures and their performance characteristics
- Database query optimization skills
- Profiling and performance testing tools

**ESTIMATED_EFFORT:** 3-5 days - Includes optimization, testing, and performance validation

**IMPACT_LEVEL:** High - Dramatically improves application performance and user experience

**RISK_ASSESSMENT:** Medium risk - Performance optimizations can introduce bugs. Mitigation: comprehensive performance testing, gradual rollout with monitoring, rollback plan ready.

**CODE_EXAMPLES:**
```python
# Before: Inefficient O(n³) nested loops
def calculate_metrics_slow(user_id, start_date, end_date):
    results = []
    for user in get_all_users():  # O(n)
        for metric in get_all_metrics():  # O(m)
            for date in get_date_range(start_date, end_date):  # O(d)
                if user.id == user_id and metric.date == date:
                    score = calculate_complex_score(user, metric, date)
                    results.append(score)
    return results

# After: Optimized O(n) with proper data structures
def calculate_metrics_optimized(user_id: str, start_date: datetime, 
                               end_date: datetime) -> List[Dict]:
    \"\"\"Optimized metrics calculation with O(n) complexity.\"\"\"
    
    # Single database query with JOIN - much more efficient
    query = \"\"\"
    SELECT u.*, m.*, d.date 
    FROM users u
    JOIN metrics m ON u.id = m.user_id
    JOIN date_range d ON m.date = d.date
    WHERE u.id = ? AND d.date BETWEEN ? AND ?
    \"\"\"
    
    raw_data = execute_query(query, [user_id, start_date, end_date])
    
    # Use dictionary for O(1) lookups instead of nested loops
    metrics_by_date = {}
    for row in raw_data:
        date_key = row['date']
        if date_key not in metrics_by_date:
            metrics_by_date[date_key] = []
        metrics_by_date[date_key].append(row)
    
    # Cache expensive calculations
    score_cache = {}
    results = []
    
    for date_key, metrics in metrics_by_date.items():
        for metric in metrics:
            cache_key = f"{metric['user_id']}_{metric['metric_id']}_{date_key}"
            
            if cache_key not in score_cache:
                score_cache[cache_key] = calculate_complex_score(
                    metric['user_data'], 
                    metric['metric_data'], 
                    date_key
                )
            
            results.append({
                'date': date_key,
                'score': score_cache[cache_key],
                'metric_type': metric['type']
            })
    
    return results
```

**RELATED_PATTERNS:** Algorithm Optimization, Caching Patterns, Database Query Optimization, Lazy Loading

**SUCCESS_METRICS:**
- Response time improves from 30+ seconds to under 2 seconds
- Database query count reduces by 90%
- Memory usage decreases by 60%
- System can handle 10x more concurrent users
- Performance benchmarks show consistent sub-second response times

**CONFIDENCE_SCORE:** 0.92"""

    def check_health(self) -> bool:
        return True
    
    def list_models(self) -> List[str]:
        return ["demo-model"]


class MockLLMResponse:
    """Mock LLM response for demonstration"""
    
    def __init__(self, response: str):
        self.response = response
        self.model = "demo-model"
        self.provider = "demo"


def demonstrate_solution_refinement():
    """Demonstrate the SolutionSuggestionAgent functionality"""
    
    print("🔧 SolutionSuggestionAgent Demo")
    print("=" * 50)
    print()
    
    # Create sample data
    raw_solutions = create_sample_raw_solutions()
    code_content = create_sample_code_content()
    static_analysis_context = create_sample_static_analysis_context()
    
    print("📝 Sample Raw Solutions:")
    for i, solution in enumerate(raw_solutions, 1):
        print(f"{i}. {solution['solution']} - {solution['implementation']}")
    print()
    
    print("📊 Code Quality Metrics:")
    metrics = static_analysis_context['metrics']
    print(f"- Quality Score: {metrics['code_quality_score']}/100")
    print(f"- Complexity: {metrics['cyclomatic_complexity']}")
    print(f"- Lines of Code: {metrics['lines_of_code']}")
    print(f"- Comment Ratio: {metrics['comment_ratio']:.1%}")
    print()
    
    # Create agent with mock LLM provider
    print("🤖 Creating SolutionSuggestionAgent...")
    agent = SolutionSuggestionAgent.__new__(SolutionSuggestionAgent)
    agent.provider = "demo"
    agent.model = "demo-model"
    agent.temperature = 0.3
    agent.max_tokens = 2000
    agent.llm_provider = MockLLMProvider()
    print("✅ Agent created successfully!")
    print()
    
    # Refine solutions
    print("🔄 Refining raw solutions...")
    refined_solutions = agent.refine_solutions(
        raw_solutions=raw_solutions,
        code_context=code_content,
        filename="sample_code.py",
        static_analysis_context=static_analysis_context
    )
    
    print(f"✅ Successfully refined {len(refined_solutions)} solutions!")
    print()
    
    # Display refined solutions
    for i, solution in enumerate(refined_solutions, 1):
        print(f"🎯 Refined Solution {i}: {solution.refined_title}")
        print("-" * 60)
        print(f"📋 Description: {solution.description[:200]}...")
        print()
        print(f"🔧 Implementation Steps ({len(solution.implementation_steps)} steps):")
        for j, step in enumerate(solution.implementation_steps[:3], 1):
            print(f"   {j}. {step}")
        if len(solution.implementation_steps) > 3:
            print(f"   ... and {len(solution.implementation_steps) - 3} more steps")
        print()
        print(f"📋 Prerequisites: {len(solution.prerequisites)} items")
        print(f"⏱️  Estimated Effort: {solution.estimated_effort}")
        print(f"📈 Impact Level: {solution.impact_level}")
        print(f"🎯 Confidence Score: {solution.confidence_score:.2f}")
        print()
        print(f"💻 Code Examples: {len(solution.code_examples)} provided")
        print(f"🔗 Related Patterns: {', '.join(solution.related_patterns[:3])}")
        print(f"📊 Success Metrics: {len(solution.success_metrics)} defined")
        print()
        print("=" * 60)
        print()
    
    # Demonstrate LangGraph integration
    print("🔗 LangGraph Integration Demo")
    print("-" * 30)
    
    state = {
        'llm_analysis': {
            'solution_suggestions': raw_solutions
        },
        'code_content': code_content,
        'filename': 'sample_code.py',
        'static_analysis_results': static_analysis_context
    }
    
    result = agent.process_solutions(state)
    
    print(f"📊 Processing Results:")
    print(f"- Status: {result['processing_status']}")
    print(f"- Current Agent: {result['current_agent']}")
    print(f"- Refined Solutions: {len(result['refined_solutions'])}")
    
    metadata = result['refinement_metadata']
    print(f"- Success Rate: {metadata['refinement_success_rate']:.1%}")
    print(f"- Provider: {metadata['provider']}")
    print(f"- Model: {metadata['model']}")
    print()
    
    print("🎉 Demo completed successfully!")
    print("The SolutionSuggestionAgent has transformed raw solutions into")
    print("comprehensive, actionable recommendations with detailed implementation")
    print("guidance, risk assessments, and success metrics.")


if __name__ == "__main__":
    demonstrate_solution_refinement() 