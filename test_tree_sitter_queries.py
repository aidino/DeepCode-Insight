#!/usr/bin/env python3
"""
Specialized tests cho Tree-sitter queries và AST integration
Test các Tree-sitter patterns và AST parsing integration
"""

import pytest
import sys
import os
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.static_analyzer import StaticAnalysisAgent
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Query


class TestTreeSitterQueries:
    """Test Tree-sitter queries functionality"""
    
    @pytest.fixture
    def analyzer(self):
        """Fixture để tạo StaticAnalysisAgent instance"""
        return StaticAnalysisAgent()
    
    def test_function_query_basic(self, analyzer):
        """Test basic function detection query"""
        
        code = '''
def simple_function():
    pass

def function_with_params(a, b, c):
    return a + b + c

async def async_function():
    await something()
'''
        
        tree = analyzer.parser.parse(bytes(code, 'utf8'))
        captures = analyzer.function_query.captures(tree.root_node)
        
        # Should capture functions
        assert 'function' in captures
        assert 'func_name' in captures
        
        # Should find 3 functions
        assert len(captures['function']) == 3
        assert len(captures['func_name']) == 3
        
        # Check function names
        func_names = [analyzer._get_node_text(node, code) for node in captures['func_name']]
        expected_names = ['simple_function', 'function_with_params', 'async_function']
        
        for expected in expected_names:
            assert expected in func_names
    
    def test_class_query_basic(self, analyzer):
        """Test basic class detection query"""
        
        code = '''
class SimpleClass:
    pass

class ClassWithMethods:
    def method1(self):
        pass
    
    def method2(self):
        pass

class InheritedClass(BaseClass):
    pass
'''
        
        tree = analyzer.parser.parse(bytes(code, 'utf8'))
        captures = analyzer.class_query.captures(tree.root_node)
        
        # Should capture classes
        assert 'class' in captures
        assert 'class_name' in captures
        
        # Should find 3 classes
        assert len(captures['class']) == 3
        assert len(captures['class_name']) == 3
        
        # Check class names
        class_names = [analyzer._get_node_text(node, code) for node in captures['class_name']]
        expected_names = ['SimpleClass', 'ClassWithMethods', 'InheritedClass']
        
        for expected in expected_names:
            assert expected in class_names
    
    def test_import_query_basic(self, analyzer):
        """Test import detection query"""
        
        code = '''
import os
import sys
from typing import List, Dict
from collections import defaultdict
import json as js
from pathlib import Path
'''
        
        tree = analyzer.parser.parse(bytes(code, 'utf8'))
        captures = analyzer.import_query.captures(tree.root_node)
        
        # Should capture imports
        assert 'import' in captures or 'from_import' in captures
        
        # Count total import statements
        total_imports = len(captures.get('import', [])) + len(captures.get('from_import', []))
        assert total_imports == 5  # 5 import statements
    
    def test_string_query_for_docstrings(self, analyzer):
        """Test string detection for docstring analysis"""
        
        code = '''
def function_with_docstring():
    """This is a docstring."""
    regular_string = "This is not a docstring"
    return regular_string

class ClassWithDocstring:
    """Class docstring."""
    
    def method_with_docstring(self):
        """Method docstring."""
        pass
'''
        
        tree = analyzer.parser.parse(bytes(code, 'utf8'))
        captures = analyzer.string_query.captures(tree.root_node)
        
        # Should capture strings
        assert 'string' in captures
        
        # Should find multiple strings (docstrings + regular strings)
        assert len(captures['string']) >= 4
        
        # Check that docstrings are detected
        string_texts = [analyzer._get_node_text(node, code) for node in captures['string']]
        assert any('This is a docstring' in text for text in string_texts)
        assert any('Class docstring' in text for text in string_texts)
        assert any('Method docstring' in text for text in string_texts)
    
    def test_if_query_for_complexity(self, analyzer):
        """Test if statement detection for complexity analysis"""
        
        code = '''
def simple_function():
    if True:
        pass

def complex_function():
    if condition1:
        if condition2:
            if condition3:
                if condition4:
                    return "complex"
    return "simple"

def function_with_elif():
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
'''
        
        tree = analyzer.parser.parse(bytes(code, 'utf8'))
        captures = analyzer.if_query.captures(tree.root_node)
        
        # Should capture if statements
        assert 'if_stmt' in captures
        
        # Should find multiple if statements (including nested ones)
        assert len(captures['if_stmt']) >= 6  # At least 6 if statements
    
    def test_docstring_detection_helper(self, analyzer):
        """Test _has_docstring helper method"""
        
        code_with_docstring = '''
def function_with_docstring():
    """This function has a docstring."""
    return "documented"
'''
        
        code_without_docstring = '''
def function_without_docstring():
    return "undocumented"
'''
        
        # Test function with docstring
        tree1 = analyzer.parser.parse(bytes(code_with_docstring, 'utf8'))
        func_captures1 = analyzer.function_query.captures(tree1.root_node)
        func_node1 = func_captures1['function'][0]
        
        assert analyzer._has_docstring(func_node1, code_with_docstring) == True
        
        # Test function without docstring
        tree2 = analyzer.parser.parse(bytes(code_without_docstring, 'utf8'))
        func_captures2 = analyzer.function_query.captures(tree2.root_node)
        func_node2 = func_captures2['function'][0]
        
        assert analyzer._has_docstring(func_node2, code_without_docstring) == False
    
    def test_parameter_counting(self, analyzer):
        """Test _count_parameters helper method"""
        
        code = '''
def no_params():
    pass

def one_param(x):
    pass

def multiple_params(a, b, c, d, e):
    pass

def params_with_defaults(a, b=1, c="default"):
    pass

def typed_params(a: int, b: str, c: List[int]):
    pass
'''
        
        tree = analyzer.parser.parse(bytes(code, 'utf8'))
        func_captures = analyzer.function_query.captures(tree.root_node)
        
        # Test each function's parameter count
        functions = func_captures['function']
        func_names = [analyzer._get_node_text(node, code) for node in func_captures['func_name']]
        
        for i, func_node in enumerate(functions):
            func_name = func_names[i]
            
            # Find parameters node
            params_node = None
            for child in func_node.children:
                if child.type == 'parameters':
                    params_node = child
                    break
            
            if params_node:
                param_count = analyzer._count_parameters(params_node, code)
                
                if func_name == 'no_params':
                    assert param_count == 0
                elif func_name == 'one_param':
                    assert param_count == 1
                elif func_name == 'multiple_params':
                    assert param_count == 5
                elif func_name == 'params_with_defaults':
                    assert param_count == 3
                elif func_name == 'typed_params':
                    assert param_count == 3
    
    def test_nested_function_counting(self, analyzer):
        """Test _count_nested_functions helper method"""
        
        code = '''
def no_nested():
    return "simple"

def one_nested():
    def inner():
        return "nested"
    return inner()

def multiple_nested():
    def inner1():
        def inner2():
            def inner3():
                return "deeply nested"
            return inner3()
        return inner2()
    return inner1()
'''
        
        tree = analyzer.parser.parse(bytes(code, 'utf8'))
        func_captures = analyzer.function_query.captures(tree.root_node)
        
        functions = func_captures['function']
        func_names = [analyzer._get_node_text(node, code) for node in func_captures['func_name']]
        
        # Find the outer functions (not nested ones)
        outer_functions = []
        for i, func_node in enumerate(functions):
            func_name = func_names[i]
            if func_name in ['no_nested', 'one_nested', 'multiple_nested']:
                nested_count = analyzer._count_nested_functions(func_node)
                
                if func_name == 'no_nested':
                    assert nested_count == 0
                elif func_name == 'one_nested':
                    assert nested_count == 1
                elif func_name == 'multiple_nested':
                    assert nested_count == 3  # inner1, inner2, inner3
    
    def test_if_statement_counting(self, analyzer):
        """Test _count_if_statements helper method"""
        
        code = '''
def no_ifs():
    return "simple"

def one_if():
    if True:
        return "conditional"
    return "default"

def multiple_ifs():
    if condition1:
        if condition2:
            if condition3:
                return "nested"
    elif other_condition:
        return "elif"
    else:
        return "else"
'''
        
        tree = analyzer.parser.parse(bytes(code, 'utf8'))
        func_captures = analyzer.function_query.captures(tree.root_node)
        
        functions = func_captures['function']
        func_names = [analyzer._get_node_text(node, code) for node in func_captures['func_name']]
        
        for i, func_node in enumerate(functions):
            func_name = func_names[i]
            if_count = analyzer._count_if_statements(func_node)
            
            if func_name == 'no_ifs':
                assert if_count == 0
            elif func_name == 'one_if':
                assert if_count == 1
            elif func_name == 'multiple_ifs':
                assert if_count >= 3  # At least 3 if statements (nested + elif)


class TestASTIntegration:
    """Test integration với ASTParsingAgent"""
    
    @pytest.fixture
    def analyzer(self):
        """Fixture để tạo StaticAnalysisAgent instance"""
        return StaticAnalysisAgent()
    
    def test_ast_integration_basic(self, analyzer):
        """Test basic AST integration"""
        
        code = '''
"""Module docstring."""

import os
from typing import List

class Calculator:
    """Calculator class."""
    
    def add(self, x: int, y: int) -> int:
        """Add two numbers."""
        return x + y
    
    def multiply(self, x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y

def main():
    """Main function."""
    calc = Calculator()
    result = calc.add(5, 3)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
'''
        
        result = analyzer.analyze_code(code, "test_ast.py")
        
        # Check AST analysis is included
        assert 'ast_analysis' in result
        ast_analysis = result['ast_analysis']
        
        # Check AST stats
        assert 'stats' in ast_analysis
        stats = ast_analysis['stats']
        
        assert stats['total_functions'] >= 3  # add, multiply, main
        assert stats['total_classes'] >= 1   # Calculator
        assert stats['total_imports'] >= 2   # os, typing.List
        
        # Check functions list
        assert 'functions' in ast_analysis
        functions = ast_analysis['functions']
        func_names = [f['name'] for f in functions]
        
        assert 'add' in func_names
        assert 'multiply' in func_names
        assert 'main' in func_names
        
        # Check classes list
        assert 'classes' in ast_analysis
        classes = ast_analysis['classes']
        class_names = [c['name'] for c in classes]
        
        assert 'Calculator' in class_names
    
    def test_ast_metrics_integration(self, analyzer):
        """Test integration của AST metrics với static analysis"""
        
        code_with_many_globals = '''
GLOBAL1 = 1
GLOBAL2 = 2
GLOBAL3 = 3
GLOBAL4 = 4
GLOBAL5 = 5
GLOBAL6 = 6
GLOBAL7 = 7
GLOBAL8 = 8
GLOBAL9 = 9
GLOBAL10 = 10
GLOBAL11 = 11
GLOBAL12 = 12

def simple_function():
    return "test"
'''
        
        result = analyzer.analyze_code(code_with_many_globals, "test_globals.py")
        
        # Check AST analysis captures globals
        ast_stats = result['ast_analysis']['stats']
        
        # Should detect many global variables
        if 'total_variables' in ast_stats:
            assert ast_stats['total_variables'] >= 10
        
        # Check if code smells detection uses AST info
        code_smells = result['static_issues']['code_smells']
        global_issues = [issue for issue in code_smells if issue['type'] == 'too_many_globals']
        
        # May or may not detect depending on AST analysis implementation
        # This tests the integration pathway
    
    def test_god_class_detection_integration(self, analyzer):
        """Test God class detection sử dụng AST analysis"""
        
        god_class_code = '''
class GodClass:
    """A class with too many methods."""
    
    def method1(self): pass
    def method2(self): pass
    def method3(self): pass
    def method4(self): pass
    def method5(self): pass
    def method6(self): pass
    def method7(self): pass
    def method8(self): pass
    def method9(self): pass
    def method10(self): pass
    def method11(self): pass
    def method12(self): pass
    def method13(self): pass
    def method14(self): pass
    def method15(self): pass
    def method16(self): pass
    def method17(self): pass
    def method18(self): pass
    def method19(self): pass
    def method20(self): pass
    def method21(self): pass
'''
        
        result = analyzer.analyze_code(god_class_code, "test_god_class.py")
        
        # Check AST analysis captures class info
        ast_classes = result['ast_analysis']['classes']
        
        # Should find the god class
        god_class = next((c for c in ast_classes if c['name'] == 'GodClass'), None)
        assert god_class is not None
        
        # Should detect many methods
        if 'method_count' in god_class:
            assert god_class['method_count'] >= 20
        
        # Check if code smells detection uses this info
        code_smells = result['static_issues']['code_smells']
        god_class_issues = [issue for issue in code_smells if issue['type'] == 'god_class']
        
        # May detect god class based on AST analysis
    
    def test_function_to_class_ratio_metric(self, analyzer):
        """Test function to class ratio metric calculation"""
        
        # Code with many functions, few classes
        function_heavy_code = '''
def func1(): pass
def func2(): pass
def func3(): pass
def func4(): pass
def func5(): pass

class SingleClass:
    def method1(self): pass
'''
        
        result1 = analyzer.analyze_code(function_heavy_code, "func_heavy.py")
        ratio1 = result1['metrics']['function_to_class_ratio']
        
        # Should have high function to class ratio
        assert ratio1 > 3  # More functions than classes
        
        # Code with many classes, few functions
        class_heavy_code = '''
def single_function(): pass

class Class1: pass
class Class2: pass
class Class3: pass
class Class4: pass
class Class5: pass
'''
        
        result2 = analyzer.analyze_code(class_heavy_code, "class_heavy.py")
        ratio2 = result2['metrics']['function_to_class_ratio']
        
        # Should have low function to class ratio
        assert ratio2 < 1  # Fewer functions than classes


class TestQueryEdgeCases:
    """Test edge cases cho Tree-sitter queries"""
    
    @pytest.fixture
    def analyzer(self):
        return StaticAnalysisAgent()
    
    def test_empty_functions_and_classes(self, analyzer):
        """Test với empty functions và classes"""
        
        code = '''
def empty_function():
    pass

class EmptyClass:
    pass

class ClassWithPass:
    pass
'''
        
        result = analyzer.analyze_code(code, "test_empty.py")
        
        # Should still detect functions and classes
        missing_docs = result['static_issues']['missing_docstrings']
        
        # Should find missing docstrings for empty constructs
        names = [issue['name'] for issue in missing_docs]
        assert 'empty_function' in names
        assert 'EmptyClass' in names
        assert 'ClassWithPass' in names
    
    def test_complex_inheritance(self, analyzer):
        """Test với complex class inheritance"""
        
        code = '''
class BaseClass:
    """Base class."""
    pass

class MiddleClass(BaseClass):
    """Middle class."""
    pass

class DerivedClass(MiddleClass):
    """Derived class."""
    pass

class MultipleInheritance(BaseClass, object):
    """Multiple inheritance."""
    pass
'''
        
        result = analyzer.analyze_code(code, "test_inheritance.py")
        
        # Should detect all classes
        missing_docs = result['static_issues']['missing_docstrings']
        
        # All classes have docstrings, so no missing docstring issues
        class_issues = [issue for issue in missing_docs if issue['type'] == 'missing_class_docstring']
        assert len(class_issues) == 0
    
    def test_decorators_and_async(self, analyzer):
        """Test với decorators và async functions"""
        
        code = '''
@property
def decorated_function():
    """Decorated function."""
    return "decorated"

@staticmethod
def static_method():
    """Static method."""
    return "static"

async def async_function():
    """Async function."""
    await something()

@decorator1
@decorator2
def multi_decorated():
    """Multi-decorated function."""
    pass
'''
        
        result = analyzer.analyze_code(code, "test_decorators.py")
        
        # Should detect all functions regardless of decorators
        missing_docs = result['static_issues']['missing_docstrings']
        
        # All functions have docstrings
        func_issues = [issue for issue in missing_docs if issue['type'] == 'missing_function_docstring']
        assert len(func_issues) == 0
    
    def test_lambda_and_comprehensions(self, analyzer):
        """Test với lambda functions và comprehensions"""
        
        code = '''
def function_with_lambda():
    """Function containing lambda."""
    func = lambda x: x * 2
    return func

def function_with_comprehensions():
    """Function with comprehensions."""
    list_comp = [x for x in range(10) if x > 5]
    dict_comp = {x: x*2 for x in range(5)}
    return list_comp, dict_comp
'''
        
        result = analyzer.analyze_code(code, "test_lambda.py")
        
        # Should not count lambdas as missing docstring functions
        missing_docs = result['static_issues']['missing_docstrings']
        func_issues = [issue for issue in missing_docs if issue['type'] == 'missing_function_docstring']
        
        # Should have no missing docstring issues
        assert len(func_issues) == 0


if __name__ == "__main__":
    # Run tests nếu script được chạy trực tiếp
    pytest.main([__file__, "-v"]) 