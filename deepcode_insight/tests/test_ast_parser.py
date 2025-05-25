"""
Test cases cho ASTParsingAgent
"""

import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch

from ..parsers.ast_parser import ASTParsingAgent, analyze_repository_code


class TestASTParsingAgent:
    """Test ASTParsingAgent functionality"""
    
    def setup_method(self):
        """Setup cho mỗi test method"""
        self.parser = ASTParsingAgent()
    
    def test_initialization(self):
        """Test ASTParsingAgent initialization"""
        assert self.parser is not None
        assert self.parser.python_language is not None
        assert self.parser.parser is not None
    
    def test_parse_simple_function(self):
        """Test parsing simple function"""
        code = '''
def hello(name: str) -> str:
    """Say hello to someone"""
    return f"Hello, {name}!"
'''
        result = self.parser.parse_code(code, "test.py")
        
        assert result['filename'] == "test.py"
        assert result['stats']['total_functions'] == 1
        assert len(result['functions']) == 1
        
        func = result['functions'][0]
        assert func['name'] == 'hello'
        assert func['class_name'] is None
        assert func['is_method'] is False
        assert func['docstring'] == "Say hello to someone"
        assert func['return_type'] == "str"
        assert len(func['parameters']) == 1
        assert func['parameters'][0]['name'] == 'name'
        assert func['parameters'][0]['type'] == 'str'
    
    def test_parse_class_with_methods(self):
        """Test parsing class với methods"""
        code = '''
class Calculator:
    """A simple calculator"""
    
    def __init__(self, value: int = 0):
        """Initialize calculator"""
        self.value = value
    
    def add(self, x: int) -> int:
        """Add x to current value"""
        self.value += x
        return self.value
    
    def _private_method(self):
        """Private method"""
        pass
    
    def __str__(self) -> str:
        """String representation"""
        return str(self.value)
'''
        result = self.parser.parse_code(code, "calculator.py")
        
        assert result['stats']['total_classes'] == 1
        assert result['stats']['total_functions'] == 4
        
        # Check class
        cls = result['classes'][0]
        assert cls['name'] == 'Calculator'
        assert cls['docstring'] == "A simple calculator"
        assert cls['method_count'] == 4
        assert cls['is_private'] is False
        
        # Check methods
        methods = {f['name']: f for f in result['functions']}
        
        # __init__ method
        init_method = methods['__init__']
        assert init_method['class_name'] == 'Calculator'
        assert init_method['is_method'] is True
        assert init_method['is_dunder'] is True
        assert init_method['docstring'] == "Initialize calculator"
        
        # add method
        add_method = methods['add']
        assert add_method['class_name'] == 'Calculator'
        assert add_method['return_type'] == "int"
        assert len(add_method['parameters']) == 2  # self, x
        
        # private method
        private_method = methods['_private_method']
        assert private_method['is_private'] is True
        assert private_method['is_dunder'] is False
        
        # dunder method
        str_method = methods['__str__']
        assert str_method['is_dunder'] is True
    
    def test_parse_imports(self):
        """Test parsing import statements"""
        code = '''
import os
import sys
from typing import List, Dict, Optional
from pathlib import Path
from collections import defaultdict, Counter
'''
        result = self.parser.parse_code(code, "imports.py")
        
        assert result['stats']['total_imports'] == 5
        
        imports = {imp['text']: imp for imp in result['imports']}
        
        # Simple imports
        assert 'import os' in imports
        assert 'import sys' in imports
        
        # From imports
        typing_import = None
        for imp in result['imports']:
            if imp['text'].startswith('from typing'):
                typing_import = imp
                break
        
        assert typing_import is not None
        assert typing_import['type'] == 'from_import'
        assert typing_import['module'] == 'typing'
        assert 'List' in typing_import['names']
        assert 'Dict' in typing_import['names']
        assert 'Optional' in typing_import['names']
    
    def test_parse_global_variables(self):
        """Test parsing global variables"""
        code = '''
DEBUG = True
VERSION = "1.0.0"
MAX_ITEMS = 100
CONFIG = {"host": "localhost", "port": 8080}
'''
        result = self.parser.parse_code(code, "config.py")
        
        assert result['stats']['total_variables'] == 4
        
        variables = {var['variables'][0]: var for var in result['variables']}
        
        assert 'DEBUG' in variables
        assert variables['DEBUG']['value'] == 'True'
        
        assert 'VERSION' in variables
        assert variables['VERSION']['value'] == '"1.0.0"'
        
        assert 'MAX_ITEMS' in variables
        assert variables['MAX_ITEMS']['value'] == '100'
    
    def test_parse_decorators(self):
        """Test parsing decorators"""
        code = '''
from dataclasses import dataclass
from functools import wraps

@dataclass
class Person:
    name: str
    age: int

@wraps(func)
def decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@property
def status(self):
    return "active"
'''
        result = self.parser.parse_code(code, "decorators.py")
        
        # Check class decorators
        cls = result['classes'][0]
        assert '@dataclass' in cls['decorators']
        
        # Check function decorators
        functions = {f['name']: f for f in result['functions']}
        
        decorator_func = functions['decorator']
        assert '@wraps(func)' in decorator_func['decorators']
        
        status_func = functions['status']
        assert '@property' in status_func['decorators']
    
    def test_parse_syntax_error(self):
        """Test parsing code với syntax errors"""
        code = '''
def broken_function(
    # Missing closing parenthesis
    return "broken"
'''
        result = self.parser.parse_code(code, "broken.py")
        
        assert len(result['errors']) > 0
        assert "Syntax errors detected" in result['errors'][0]
    
    def test_parse_empty_code(self):
        """Test parsing empty code"""
        result = self.parser.parse_code("", "empty.py")
        
        assert result['filename'] == "empty.py"
        assert result['stats']['total_functions'] == 0
        assert result['stats']['total_classes'] == 0
        assert result['stats']['total_imports'] == 0
        assert result['stats']['total_variables'] == 0
        assert len(result['errors']) == 0
    
    def test_parse_complex_parameters(self):
        """Test parsing complex function parameters"""
        code = '''
def complex_func(
    a: int,
    b: str = "default",
    c: Optional[List[str]] = None,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """Complex function với nhiều parameter types"""
    pass
'''
        result = self.parser.parse_code(code, "complex.py")
        
        func = result['functions'][0]
        assert func['name'] == 'complex_func'
        
        # Check parameters (tree-sitter có thể không parse hết *args, **kwargs)
        params = func['parameters']
        assert len(params) >= 3  # a, b, c ít nhất
        
        # Check typed parameters
        param_names = [p['name'] for p in params]
        assert 'a' in param_names
        assert 'b' in param_names
        assert 'c' in param_names
    
    def test_parse_nested_classes(self):
        """Test parsing nested classes"""
        code = '''
class Outer:
    """Outer class"""
    
    class Inner:
        """Inner class"""
        
        def inner_method(self):
            pass
    
    def outer_method(self):
        pass
'''
        result = self.parser.parse_code(code, "nested.py")
        
        # Should find both classes
        assert result['stats']['total_classes'] == 2
        
        class_names = [cls['name'] for cls in result['classes']]
        assert 'Outer' in class_names
        assert 'Inner' in class_names
    
    def test_error_handling(self):
        """Test error handling trong parsing"""
        # Test với None input
        result = self.parser.parse_code(None, "invalid.py")
        assert result['stats']['total_lines'] == 0
        assert len(result['errors']) == 0  # None input is handled gracefully
        
        # Test với empty string
        result = self.parser.parse_code("", "empty.py")
        assert result['stats']['total_lines'] == 0
        assert len(result['errors']) == 0
        
        # Test với very large code
        large_code = "# comment\n" * 10000 + "def test(): pass"
        result = self.parser.parse_code(large_code, "large.py")
        assert result['stats']['total_lines'] == 10001


class TestAnalyzeRepositoryCode:
    """Test analyze_repository_code function"""
    
    def test_analyze_repository_code_structure(self):
        """Test structure của analyze_repository_code result"""
        # Mock CodeFetcherAgent
        mock_agent = MagicMock()
        mock_agent.list_repository_files.return_value = ['test.py', 'README.md']
        mock_agent.get_file_content.return_value = '''
def hello():
    """Hello function"""
    return "Hello, World!"
'''
        
        result = analyze_repository_code(mock_agent, "https://github.com/test/repo")
        
        # Check result structure
        assert 'repository' in result
        assert 'files_analyzed' in result
        assert 'summary' in result
        assert 'errors' in result
        assert 'analysis_timestamp' in result
        
        # Check summary structure
        summary = result['summary']
        assert 'total_files' in summary
        assert 'total_functions' in summary
        assert 'total_classes' in summary
        assert 'total_imports' in summary
        assert 'total_variables' in summary
        assert 'total_lines' in summary
    
    def test_analyze_repository_code_with_errors(self):
        """Test analyze_repository_code với errors"""
        # Mock CodeFetcherAgent với errors
        mock_agent = MagicMock()
        mock_agent.list_repository_files.side_effect = Exception("Network error")
        
        result = analyze_repository_code(mock_agent, "https://github.com/test/repo")
        
        assert len(result['errors']) > 0
        assert "Error analyzing repository" in result['errors'][0]
    
    def test_analyze_repository_code_no_python_files(self):
        """Test analyze_repository_code với no Python files"""
        mock_agent = MagicMock()
        mock_agent.list_repository_files.return_value = ['README.md', 'package.json']
        
        result = analyze_repository_code(mock_agent, "https://github.com/test/repo")
        
        assert result['summary']['total_files'] == 0
        assert len(result['files_analyzed']) == 0
    
    def test_analyze_repository_code_file_read_error(self):
        """Test analyze_repository_code với file read errors"""
        mock_agent = MagicMock()
        mock_agent.list_repository_files.return_value = ['test.py']
        mock_agent.get_file_content.return_value = None  # Simulate read error
        
        result = analyze_repository_code(mock_agent, "https://github.com/test/repo")
        
        assert len(result['errors']) > 0
        assert "Could not read content" in result['errors'][0]


class TestASTParsingAgentEdgeCases:
    """Test edge cases và error scenarios"""
    
    def setup_method(self):
        self.parser = ASTParsingAgent()
    
    def test_unicode_code(self):
        """Test parsing code với Unicode characters"""
        # Test function với Unicode
        func_code = '''
def greet(name: str) -> str:
    """Chào mừng người dùng 👋"""
    return f"Xin chào, {name}! 🎉"
'''
        result = self.parser.parse_code(func_code, "unicode_func.py")
        
        assert result['stats']['total_functions'] == 1
        func = result['functions'][0]
        assert func['name'] == 'greet'
        assert "Chào mừng" in func['docstring']
        
        # Test class với Unicode
        class_code = '''
class Người:
    """Class đại diện cho một người"""
    pass
'''
        result = self.parser.parse_code(class_code, "unicode_class.py")
        
        assert result['stats']['total_classes'] == 1
        cls = result['classes'][0]
        assert cls['name'] == 'Người'
    
    def test_very_long_lines(self):
        """Test parsing code với very long lines"""
        long_string = "a" * 1000
        code = f'''
def long_function():
    """Function với very long string"""
    return "{long_string}"
'''
        result = self.parser.parse_code(code, "long.py")
        
        assert result['stats']['total_functions'] == 1
        func = result['functions'][0]
        assert func['name'] == 'long_function'
    
    def test_mixed_indentation(self):
        """Test parsing code với mixed indentation"""
        code = '''
def mixed_indent():
    """Function với mixed indentation"""
    if True:
        print("spaces")
\tif True:
\t\tprint("tabs")
'''
        result = self.parser.parse_code(code, "mixed.py")
        
        # Should still parse successfully
        assert result['stats']['total_functions'] == 1
    
    def test_special_characters_in_strings(self):
        """Test parsing code với special characters trong strings"""
        code = '''
def special_chars():
    """Function với special characters"""
    return "String with \\"quotes\\" and \\n newlines"
'''
        result = self.parser.parse_code(code, "special.py")
        
        assert result['stats']['total_functions'] == 1
        func = result['functions'][0]
        assert func['name'] == 'special_chars'


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 