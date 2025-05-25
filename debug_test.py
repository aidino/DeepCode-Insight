#!/usr/bin/env python3
"""
Debug script để kiểm tra missing docstrings detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.static_analyzer import StaticAnalysisAgent

def debug_missing_docstrings():
    """Debug missing docstrings detection"""
    
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

def _private_function():
    return "private"

def __dunder_method__(self):
    return "dunder"
'''
    
    print("🔍 Debug Missing Docstrings Detection")
    print("="*60)
    
    analyzer = StaticAnalysisAgent()
    result = analyzer.analyze_code(code_with_missing_docs, "test_missing_docs.py")
    
    print(f"📁 File: {result['filename']}")
    print(f"🎯 Quality Score: {result['metrics']['code_quality_score']:.1f}/100")
    
    # Debug missing docstrings
    missing_docs = result['static_issues']['missing_docstrings']
    print(f"\n📝 Missing Docstrings Found: {len(missing_docs)}")
    
    for issue in missing_docs:
        print(f"  - {issue['type']}: {issue['name']} (line {issue['line']})")
    
    # Debug Tree-sitter queries directly
    print(f"\n🌳 Tree-sitter Query Debug:")
    tree = analyzer.parser.parse(bytes(code_with_missing_docs, 'utf8'))
    
    # Function query
    func_captures = analyzer.function_query.captures(tree.root_node)
    print(f"Function captures: {list(func_captures.keys())}")
    if 'function' in func_captures:
        print(f"Functions found: {len(func_captures['function'])}")
    if 'func_name' in func_captures:
        print(f"Function names found: {len(func_captures['func_name'])}")
        for node in func_captures['func_name']:
            name = analyzer._get_node_text(node, code_with_missing_docs)
            print(f"  - Function name: {name}")
    
    # Class query
    class_captures = analyzer.class_query.captures(tree.root_node)
    print(f"Class captures: {list(class_captures.keys())}")
    if 'class' in class_captures:
        print(f"Classes found: {len(class_captures['class'])}")
    if 'class_name' in class_captures:
        print(f"Class names found: {len(class_captures['class_name'])}")
        for node in class_captures['class_name']:
            name = analyzer._get_node_text(node, code_with_missing_docs)
            print(f"  - Class name: {name}")
    
    # Test docstring detection for each function
    print(f"\n📋 Docstring Detection Test:")
    if 'function' in func_captures and 'func_name' in func_captures:
        for i, func_node in enumerate(func_captures['function']):
            if i < len(func_captures['func_name']):
                name_node = func_captures['func_name'][i]
                name = analyzer._get_node_text(name_node, code_with_missing_docs)
                has_docstring = analyzer._has_docstring(func_node, code_with_missing_docs)
                print(f"  - Function '{name}': has_docstring = {has_docstring}")
    
    # Test docstring detection for each class
    if 'class' in class_captures and 'class_name' in class_captures:
        for i, class_node in enumerate(class_captures['class']):
            if i < len(class_captures['class_name']):
                name_node = class_captures['class_name'][i]
                name = analyzer._get_node_text(name_node, code_with_missing_docs)
                has_docstring = analyzer._has_docstring(class_node, code_with_missing_docs)
                print(f"  - Class '{name}': has_docstring = {has_docstring}")

if __name__ == "__main__":
    debug_missing_docstrings() 