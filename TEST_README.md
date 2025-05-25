# StaticAnalysisAgent Test Suite

Comprehensive test suite cho `StaticAnalysisAgent` với sample ASTs và detailed assertions.

## 📋 Tổng quan

Test suite này bao gồm:

1. **Basic Functionality Tests** - Test các tính năng cơ bản
2. **Comprehensive Test Suite** - Test toàn diện với pytest
3. **Tree-sitter Query Tests** - Test chuyên biệt cho Tree-sitter queries
4. **AST Integration Tests** - Test tích hợp với ASTParsingAgent
5. **Edge Case Tests** - Test các trường hợp đặc biệt

## 🚀 Chạy Tests

### Quick Start

```bash
# Chạy tất cả tests
python run_tests.py
```

### Chạy từng test riêng lẻ

```bash
# Basic functionality test
python test_static_analyzer.py

# Comprehensive test suite
python -m pytest test_static_analysis_comprehensive.py -v

# Tree-sitter query tests
python -m pytest test_tree_sitter_queries.py -v

# Demo script
python demo_static_analyzer.py
```

### Chạy với pytest options

```bash
# Verbose output
pytest test_static_analysis_comprehensive.py -v

# Stop on first failure
pytest test_static_analysis_comprehensive.py -x

# Run specific test
pytest test_static_analysis_comprehensive.py::TestStaticAnalysisAgent::test_missing_docstrings_detection -v

# Generate coverage report
pytest test_static_analysis_comprehensive.py --cov=agents.static_analyzer
```

## 📁 Test Files

### `test_static_analysis_comprehensive.py`
- **TestStaticAnalysisAgent**: Main test class
  - `test_initialization()` - Test khởi tạo agent
  - `test_missing_docstrings_detection()` - Test detection missing docstrings
  - `test_unused_imports_detection()` - Test detection unused imports
  - `test_complex_functions_detection()` - Test detection complex functions
  - `test_code_smells_detection()` - Test detection code smells
  - `test_metrics_calculation()` - Test calculation metrics
  - `test_suggestions_generation()` - Test generation suggestions
  - `test_empty_code()` - Test với empty code
  - `test_syntax_error_code()` - Test với syntax errors
  - `test_file_analysis()` - Test analyze file
  - `test_comprehensive_analysis()` - Test với realistic code

### `test_tree_sitter_queries.py`
- **TestTreeSitterQueries**: Test Tree-sitter functionality
  - `test_function_query_basic()` - Test function detection
  - `test_class_query_basic()` - Test class detection
  - `test_import_query_basic()` - Test import detection
  - `test_string_query_for_docstrings()` - Test string detection
  - `test_if_query_for_complexity()` - Test if statement detection
  - Helper method tests: `_has_docstring`, `_count_parameters`, etc.

- **TestASTIntegration**: Test AST integration
  - `test_ast_integration_basic()` - Test basic AST integration
  - `test_ast_metrics_integration()` - Test metrics integration
  - `test_god_class_detection_integration()` - Test god class detection

- **TestQueryEdgeCases**: Test edge cases
  - `test_empty_functions_and_classes()` - Test empty constructs
  - `test_complex_inheritance()` - Test inheritance patterns
  - `test_decorators_and_async()` - Test decorators và async
  - `test_lambda_and_comprehensions()` - Test lambda và comprehensions

### `test_static_analyzer.py`
- Simple test script không cần pytest
- Test basic functionality
- Test Tree-sitter queries
- Bypass dependency issues

## 🧪 Test Cases

### Missing Docstrings Detection

```python
def test_missing_docstrings_detection(self, analyzer):
    code_with_missing_docs = '''
def function_without_docstring():
    return "no docs"

def function_with_docstring():
    """This function has a docstring."""
    return "has docs"

class ClassWithoutDocstring:
    pass

class ClassWithDocstring:
    """This class has a docstring."""
    pass
'''
    
    result = analyzer.analyze_code(code_with_missing_docs, "test.py")
    missing_docs = result['static_issues']['missing_docstrings']
    
    # Assertions
    assert len(missing_docs) == 2  # function_without_docstring và ClassWithoutDocstring
    assert missing_docs[0]['name'] == 'function_without_docstring'
    assert missing_docs[1]['name'] == 'ClassWithoutDocstring'
```

### Complex Functions Detection

```python
def test_complex_functions_detection(self, analyzer):
    code_with_complex_functions = '''
def function_with_many_params(a, b, c, d, e, f, g, h):
    """Function with too many parameters."""
    return a + b + c + d + e + f + g + h

def function_with_nested_functions():
    """Function with nested functions."""
    def nested1():
        def nested2():
            def nested3():
                return "deeply nested"
            return nested3()
        return nested2()
    return nested1()
'''
    
    result = analyzer.analyze_code(code_with_complex_functions, "test.py")
    complex_functions = result['static_issues']['complex_functions']
    
    # Check for too many parameters
    param_issues = [issue for issue in complex_functions if issue['type'] == 'too_many_parameters']
    assert len(param_issues) >= 1
    assert param_issues[0]['count'] == 8
    
    # Check for too many nested functions
    nested_issues = [issue for issue in complex_functions if issue['type'] == 'too_many_nested_functions']
    assert len(nested_issues) >= 1
    assert nested_issues[0]['count'] > 2
```

### Tree-sitter Query Tests

```python
def test_function_query_basic(self, analyzer):
    code = '''
def simple_function():
    pass

def function_with_params(a, b, c):
    return a + b + c
'''
    
    tree = analyzer.parser.parse(bytes(code, 'utf8'))
    captures = analyzer.function_query.captures(tree.root_node)
    
    # Should capture functions
    assert 'function' in captures
    assert 'func_name' in captures
    assert len(captures['function']) == 2
    
    # Check function names
    func_names = [analyzer._get_node_text(node, code) for node in captures['func_name']]
    assert 'simple_function' in func_names
    assert 'function_with_params' in func_names
```

## 📊 Test Coverage

Tests cover các areas sau:

### Core Functionality
- ✅ StaticAnalysisAgent initialization
- ✅ Tree-sitter query execution
- ✅ AST parsing integration
- ✅ Code analysis pipeline

### Detection Features
- ✅ Missing docstrings (functions, classes)
- ✅ Unused imports
- ✅ Complex functions (parameters, nesting, complexity)
- ✅ Code smells (long lines, god classes, globals)

### Metrics Calculation
- ✅ Cyclomatic complexity
- ✅ Maintainability index
- ✅ Code quality score
- ✅ Comment ratio
- ✅ Function to class ratio

### Edge Cases
- ✅ Empty code
- ✅ Syntax errors
- ✅ File not found
- ✅ Complex inheritance
- ✅ Decorators và async functions
- ✅ Lambda functions
- ✅ Comprehensions

### Integration
- ✅ ASTParsingAgent integration
- ✅ File analysis
- ✅ Repository analysis (structure)
- ✅ Error handling

## 🔧 Dependencies

```bash
pip install pytest tree-sitter tree-sitter-python
```

## 📈 Expected Results

### Perfect Code
```python
# Should have:
# - Quality score > 80
# - No missing docstrings
# - No unused imports
# - No complex functions
# - Minimal suggestions
```

### Problematic Code
```python
# Should detect:
# - Missing docstrings
# - Unused imports
# - Too many parameters
# - High complexity
# - Code smells
# - Generate suggestions
```

## 🐛 Debugging Tests

### Verbose Output
```bash
pytest test_static_analysis_comprehensive.py -v -s
```

### Debug Specific Test
```bash
pytest test_static_analysis_comprehensive.py::TestStaticAnalysisAgent::test_missing_docstrings_detection -v -s --pdb
```

### Print Debug Info
```python
def test_debug_example(self, analyzer):
    result = analyzer.analyze_code(code, "test.py")
    
    # Debug prints
    print(f"Result: {result}")
    print(f"Issues: {result['static_issues']}")
    print(f"Metrics: {result['metrics']}")
    
    # Assertions
    assert len(result['static_issues']['missing_docstrings']) == 2
```

## 📝 Adding New Tests

### Test Template
```python
def test_new_feature(self, analyzer):
    """Test description"""
    
    # Arrange
    code = '''
    # Sample code here
    '''
    
    # Act
    result = analyzer.analyze_code(code, "test.py")
    
    # Assert
    assert result['some_field'] == expected_value
    assert len(result['static_issues']['some_category']) == expected_count
```

### Best Practices
1. **Descriptive test names** - Clearly indicate what is being tested
2. **Sample ASTs** - Use realistic code samples
3. **Specific assertions** - Test exact values, not just existence
4. **Edge cases** - Test boundary conditions
5. **Error cases** - Test error handling
6. **Integration** - Test component interactions

## 🎯 Test Results Interpretation

### Success Indicators
- ✅ All tests pass
- ✅ High code coverage
- ✅ Correct issue detection
- ✅ Accurate metrics calculation
- ✅ Appropriate suggestions

### Failure Investigation
1. Check test output for specific assertion failures
2. Verify sample code syntax
3. Check Tree-sitter query syntax
4. Validate expected vs actual results
5. Debug with print statements or pdb

## 📚 References

- [pytest Documentation](https://docs.pytest.org/)
- [Tree-sitter Documentation](https://tree-sitter.github.io/tree-sitter/)
- [Tree-sitter Python Grammar](https://github.com/tree-sitter/tree-sitter-python)
- [Python AST Documentation](https://docs.python.org/3/library/ast.html) 