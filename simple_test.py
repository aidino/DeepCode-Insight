#!/usr/bin/env python3
"""
Simple test để debug Tree-sitter captures
"""

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Query

def test_tree_sitter():
    """Test Tree-sitter captures format"""
    
    # Initialize
    python_language = Language(tspython.language())
    parser = Parser(python_language)
    
    # Simple query
    query = Query(
        python_language,
        """
        (function_definition
            name: (identifier) @func_name
        ) @function
        """
    )
    
    # Sample code
    code = '''
def test_function():
    pass

def another_function():
    return 42
'''
    
    # Parse
    tree = parser.parse(bytes(code, 'utf8'))
    
    # Get captures
    captures = query.captures(tree.root_node)
    
    print(f"Captures type: {type(captures)}")
    print(f"Captures: {captures}")
    
    if isinstance(captures, dict):
        print("Captures is a dict:")
        for capture_name, nodes in captures.items():
            print(f"  {capture_name}: {len(nodes)} nodes")
            for node in nodes:
                print(f"    - {node.type} at line {node.start_point[0] + 1}")
    else:
        print("Captures is not a dict, trying to iterate:")
        for i, capture in enumerate(captures):
            print(f"Capture {i}: {capture}")
            try:
                node, capture_name = capture
                print(f"  Unpacked: node={node.type}, name={capture_name}")
            except Exception as e:
                print(f"  Error unpacking: {e}")

if __name__ == "__main__":
    test_tree_sitter() 