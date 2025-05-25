#!/usr/bin/env python3
"""Debug script cho Python bare except detection"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'deepcode_insight'))

from deepcode_insight.agents.static_analyzer import StaticAnalysisAgent

def debug_bare_except():
    """Debug Python bare except detection"""
    
    # Test case từ comprehensive test
    python_code = '''
# Bad: Bare except clauses
try:
    risky_operation()
except:
    pass

try:
    another_risky_operation()
except:
    print("Something went wrong")

# Good: Specific exception types
try:
    safe_operation()
except ValueError as e:
    print(f"Value error: {e}")
except Exception as e:
    print(f"General error: {e}")
'''
    
    analyzer = StaticAnalysisAgent()
    
    print("🔍 Debug Python Bare Except Detection")
    print("=" * 50)
    
    result = analyzer.analyze_code(python_code, "test_exceptions.py")
    
    print(f"Language: {result['language']}")
    print(f"Google style violations: {len(result['static_issues']['google_style_violations'])}")
    
    for violation in result['static_issues']['google_style_violations']:
        print(f"  - {violation['type']}: {violation['message']} (Line {violation['line']})")
    
    # Debug tree-sitter parsing
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser, Query
    
    python_language = Language(tspython.language())
    python_parser = Parser(python_language)
    
    print("\n3. Tree-sitter analysis:")
    tree = python_parser.parse(bytes(python_code, 'utf8'))
    
    python_exception_query = Query(
        python_language,
        """
        (try_statement) @try_stmt
        (except_clause) @except_clause
        (raise_statement) @raise_stmt
        """
    )
    
    captures = python_exception_query.captures(tree.root_node)
    print(f"   Found {len(captures.get('except_clause', []))} except clauses")
    
    if 'except_clause' in captures:
        for i, node in enumerate(captures['except_clause']):
            except_text = python_code[node.start_byte:node.end_byte]
            print(f"\n   Except clause {i+1}:")
            print(f"     Text: {repr(except_text)}")
            print(f"     Lines: {node.start_point[0]+1}-{node.end_point[0]+1}")
            print(f"     Stripped: {repr(except_text.strip())}")
            print(f"     Is bare except: {except_text.strip() == 'except:'}")

if __name__ == "__main__":
    debug_bare_except() 