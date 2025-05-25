#!/usr/bin/env python3
"""
Debug script để xem cấu trúc AST của function với docstring
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.static_analyzer import StaticAnalysisAgent

def print_ast_structure(node, code, indent=0):
    """Print AST structure recursively"""
    prefix = "  " * indent
    node_text = code[node.start_byte:node.end_byte]
    # Truncate long text
    if len(node_text) > 50:
        node_text = node_text[:47] + "..."
    node_text = repr(node_text)
    
    print(f"{prefix}{node.type}: {node_text}")
    
    for child in node.children:
        print_ast_structure(child, code, indent + 1)

def debug_function_ast():
    """Debug AST structure của function với và không có docstring"""
    
    code_with_docstring = '''
def function_with_docstring():
    """This function has a docstring."""
    return "has docs"
'''
    
    code_without_docstring = '''
def function_without_docstring():
    return "no docs"
'''
    
    analyzer = StaticAnalysisAgent()
    
    print("🔍 Function WITH Docstring AST Structure:")
    print("="*60)
    tree1 = analyzer.parser.parse(bytes(code_with_docstring, 'utf8'))
    func_captures1 = analyzer.function_query.captures(tree1.root_node)
    if 'function' in func_captures1:
        func_node1 = func_captures1['function'][0]
        print_ast_structure(func_node1, code_with_docstring)
        
        print(f"\n_has_docstring result: {analyzer._has_docstring(func_node1, code_with_docstring)}")
    
    print("\n" + "="*60)
    print("🔍 Function WITHOUT Docstring AST Structure:")
    print("="*60)
    tree2 = analyzer.parser.parse(bytes(code_without_docstring, 'utf8'))
    func_captures2 = analyzer.function_query.captures(tree2.root_node)
    if 'function' in func_captures2:
        func_node2 = func_captures2['function'][0]
        print_ast_structure(func_node2, code_without_docstring)
        
        print(f"\n_has_docstring result: {analyzer._has_docstring(func_node2, code_without_docstring)}")

if __name__ == "__main__":
    debug_function_ast() 