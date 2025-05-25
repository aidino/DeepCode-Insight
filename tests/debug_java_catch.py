#!/usr/bin/env python3
"""Debug script cho Java empty catch block detection"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'deepcode_insight'))

from deepcode_insight.agents.static_analyzer import StaticAnalysisAgent

def debug_java_catch():
    """Debug Java catch block detection"""
    
    # Test case từ test_java_rules.py
    java_code_from_test = '''
import java.util.*;
import java.io.*;

public class TestClass {
    private int value;
    
    // Rule 1: Missing Javadoc for public methods
    public void publicMethodWithoutJavadoc() {
        System.out.println("This method should have Javadoc");
    }
    
    /**
     * This method has proper Javadoc
     */
    public void properlyDocumentedMethod() {
        System.out.println("This method has Javadoc");
    }
    
    // Rule 2: Empty catch blocks
    public void methodWithEmptyCatch() {
        try {
            int result = 10 / 0;
        } catch (Exception e) {
            // Empty catch block - this is bad practice
        }
    }
    
    public void methodWithProperCatch() {
        try {
            int result = 10 / 0;
        } catch (Exception e) {
            System.err.println("Error occurred: " + e.getMessage());
            // Proper exception handling
        }
    }
}
'''
    
    # Test case đơn giản hơn
    simple_java_code = '''
public class SimpleTest {
    public void emptycatch1() {
        try {
            int x = 10 / 0;
        } catch (Exception e) {
        }
    }
    
    public void emptyWithComment() {
        try {
            int x = 10 / 0;
        } catch (Exception e) {
            // Just a comment
        }
    }
    
    public void notEmpty() {
        try {
            int x = 10 / 0;
        } catch (Exception e) {
            System.out.println("Error");
        }
    }
}
'''
    
    analyzer = StaticAnalysisAgent()
    
    print("🔍 Debug Java Catch Block Detection")
    print("=" * 50)
    
    # Test với code từ test file
    print("\n1. Testing code from test_java_rules.py:")
    result1 = analyzer.analyze_code(java_code_from_test, "TestClass.java")
    code_smells1 = result1['static_issues']['code_smells']
    empty_catches1 = [s for s in code_smells1 if s['type'] == 'empty_catch_block']
    
    print(f"   Found {len(empty_catches1)} empty catch blocks:")
    for catch in empty_catches1:
        print(f"     - Line {catch['line']}: {catch['message']}")
    
    # Test với simple code
    print("\n2. Testing simple code:")
    result2 = analyzer.analyze_code(simple_java_code, "SimpleTest.java")
    code_smells2 = result2['static_issues']['code_smells']
    empty_catches2 = [s for s in code_smells2 if s['type'] == 'empty_catch_block']
    
    print(f"   Found {len(empty_catches2)} empty catch blocks:")
    for catch in empty_catches2:
        print(f"     - Line {catch['line']}: {catch['message']}")
    
    # Debug tree-sitter parsing
    import tree_sitter_java as tsjava
    from tree_sitter import Language, Parser, Query
    
    java_language = Language(tsjava.language())
    java_parser = Parser(java_language)
    
    print("\n3. Tree-sitter analysis of test code:")
    tree = java_parser.parse(bytes(java_code_from_test, 'utf8'))
    
    java_exception_query = Query(
        java_language,
        """
        (try_statement) @try_stmt
        (catch_clause) @catch_clause
        (throw_statement) @throw_stmt
        """
    )
    
    captures = java_exception_query.captures(tree.root_node)
    print(f"   Found {len(captures.get('catch_clause', []))} catch clauses")
    
    if 'catch_clause' in captures:
        for i, node in enumerate(captures['catch_clause']):
            catch_text = java_code_from_test[node.start_byte:node.end_byte]
            print(f"\n   Catch clause {i+1}:")
            print(f"     Text: {repr(catch_text)}")
            print(f"     Lines: {node.start_point[0]+1}-{node.end_point[0]+1}")
            
            # Analyze body content
            brace_start = catch_text.find('{')
            if brace_start != -1:
                after_brace = catch_text[brace_start + 1:]
                brace_count = 1
                body_end = -1
                for j, char in enumerate(after_brace):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            body_end = j
                            break
                
                if body_end != -1:
                    body = after_brace[:body_end].strip()
                    print(f"     Body: {repr(body)}")
                    
                    # Check for executable code
                    lines = body.split('\n')
                    has_executable_code = False
                    
                    for line in lines:
                        stripped = line.strip()
                        print(f"       Line: {repr(stripped)}")
                        if stripped and not stripped.startswith('//') and not stripped.startswith('/*') and not stripped.startswith('*'):
                            if not stripped.endswith('*/'):
                                has_executable_code = True
                                print(f"         -> Has executable code: {stripped}")
                                break
                    
                    print(f"     Has executable code: {has_executable_code}")

if __name__ == "__main__":
    debug_java_catch() 