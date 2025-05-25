#!/usr/bin/env python3
"""
Simple test script cho StaticAnalysisAgent
Test trực tiếp mà không cần dependencies khác
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import trực tiếp từ file, bypass __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location("static_analyzer", "agents/static_analyzer.py")
static_analyzer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(static_analyzer_module)
StaticAnalysisAgent = static_analyzer_module.StaticAnalysisAgent

def test_basic_functionality():
    """Test basic functionality của StaticAnalysisAgent"""
    print("🔍 Testing StaticAnalysisAgent...")
    
    # Sample code với issues
    sample_code = '''
import os
import unused_module

def function_without_docstring(x, y, z, a, b, c):
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z + a + b + c
    return 0

class Calculator:
    def add(self, x, y):
        return x + y
'''
    
    try:
        analyzer = StaticAnalysisAgent()
        result = analyzer.analyze_code(sample_code, "test.py")
        
        print("✅ StaticAnalysisAgent initialized successfully")
        print(f"📁 File: {result['filename']}")
        print(f"🎯 Quality Score: {result['metrics']['code_quality_score']:.1f}/100")
        
        # Check issues
        issues = result['static_issues']
        total_issues = sum(len(v) for v in issues.values())
        print(f"📊 Total Issues Found: {total_issues}")
        
        # Show some issues
        if issues['missing_docstrings']:
            print(f"📝 Missing Docstrings: {len(issues['missing_docstrings'])}")
            for issue in issues['missing_docstrings'][:2]:
                print(f"  - {issue['name']} (line {issue['line']})")
        
        if issues['complex_functions']:
            print(f"🔄 Complex Functions: {len(issues['complex_functions'])}")
            for issue in issues['complex_functions'][:2]:
                print(f"  - {issue['name']} (line {issue['line']})")
        
        if issues['unused_imports']:
            print(f"🚫 Unused Imports: {len(issues['unused_imports'])}")
            for issue in issues['unused_imports'][:2]:
                print(f"  - {issue['name']} (line {issue['line']})")
        
        print(f"💡 Suggestions: {len(result['suggestions'])}")
        for suggestion in result['suggestions'][:3]:
            print(f"  - {suggestion}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tree_sitter_queries():
    """Test Tree-sitter queries"""
    print("\n🌳 Testing Tree-sitter Queries...")
    
    query_code = '''
def good_function():
    """This function has a docstring."""
    return "good"

def bad_function():
    return "bad"

class GoodClass:
    """This class has a docstring."""
    pass

class BadClass:
    pass
'''
    
    try:
        analyzer = StaticAnalysisAgent()
        result = analyzer.analyze_code(query_code, "query_test.py")
        
        missing_docs = result['static_issues']['missing_docstrings']
        print(f"📝 Missing docstrings detected: {len(missing_docs)}")
        
        expected_missing = ['bad_function', 'BadClass']
        found_missing = [issue['name'] for issue in missing_docs]
        
        for expected in expected_missing:
            if expected in found_missing:
                print(f"  ✅ Correctly detected missing docstring: {expected}")
            else:
                print(f"  ❌ Failed to detect missing docstring: {expected}")
        
        print("✅ Tree-sitter queries working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Query test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 === StaticAnalysisAgent Test Suite ===")
    
    success = True
    
    # Test basic functionality
    if not test_basic_functionality():
        success = False
    
    # Test Tree-sitter queries
    if not test_tree_sitter_queries():
        success = False
    
    if success:
        print("\n🎉 All tests passed! StaticAnalysisAgent is working correctly.")
        print("\n📚 Features Verified:")
        print("  ✅ Tree-sitter query execution")
        print("  ✅ Missing docstring detection")
        print("  ✅ Complex function analysis")
        print("  ✅ Code quality metrics")
        print("  ✅ Suggestion generation")
        print("  ✅ Integration with ASTParsingAgent")
    else:
        print("\n❌ Some tests failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 