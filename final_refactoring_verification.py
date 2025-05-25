#!/usr/bin/env python3
"""
Final verification script for DeepCode-Insight refactoring

Kiểm tra toàn diện các thành phần đã được refactor
"""

import sys
from pathlib import Path

def test_core_interfaces():
    """Test core interfaces and utilities"""
    print("🔍 Testing Core Interfaces...")
    
    try:
        from deepcode_insight.core import (
            AnalysisLanguage, AnalysisResult, BaseAgent, CodeAnalyzer,
            detect_language_from_filename, calculate_complexity_score,
            setup_logging
        )
        
        # Test AnalysisLanguage enum
        assert AnalysisLanguage.PYTHON.value == "python"
        assert AnalysisLanguage.JAVA.value == "java"
        
        # Test AnalysisResult
        result = AnalysisResult("test.py", AnalysisLanguage.PYTHON)
        result.add_issue("test_issue", "Test message", 1, "warning")
        result.add_metric("test_metric", 42)
        result.add_suggestion("Test suggestion")
        
        assert len(result.issues) == 1
        assert result.metrics["test_metric"] == 42
        assert len(result.suggestions) == 1
        
        # Test utility functions
        lang = detect_language_from_filename("test.py")
        assert lang == AnalysisLanguage.PYTHON
        
        lang = detect_language_from_filename("Test.java")
        assert lang == AnalysisLanguage.JAVA
        
        score = calculate_complexity_score({
            'cyclomatic_complexity': 5,
            'lines_of_code': 100,
            'parameter_count': 3
        })
        assert 0 <= score <= 1
        
        print("   ✅ Core interfaces working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Core interfaces failed: {e}")
        return False

def test_tree_sitter_queries():
    """Test Tree-sitter query manager"""
    print("🔍 Testing Tree-sitter Query Manager...")
    
    try:
        from deepcode_insight.parsers.tree_sitter_queries import get_query_manager
        from deepcode_insight.core import AnalysisLanguage
        
        query_manager = get_query_manager()
        
        # Test language support
        assert query_manager.supports_language(AnalysisLanguage.PYTHON)
        assert query_manager.supports_language(AnalysisLanguage.JAVA)
        
        # Test query retrieval
        py_functions_query = query_manager.get_query(AnalysisLanguage.PYTHON, 'functions')
        java_functions_query = query_manager.get_query(AnalysisLanguage.JAVA, 'functions')
        
        assert py_functions_query is not None
        assert java_functions_query is not None
        
        # Test language objects
        py_lang = query_manager.get_language(AnalysisLanguage.PYTHON)
        java_lang = query_manager.get_language(AnalysisLanguage.JAVA)
        
        assert py_lang is not None
        assert java_lang is not None
        
        print("   ✅ Tree-sitter query manager working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Tree-sitter query manager failed: {e}")
        return False

def test_analyzers():
    """Test refactored analyzers"""
    print("🔍 Testing Refactored Analyzers...")
    
    try:
        from deepcode_insight.analyzers import PythonAnalyzer, JavaAnalyzer, BaseCodeAnalyzer
        from deepcode_insight.core import AnalysisLanguage
        
        # Test Python analyzer
        py_analyzer = PythonAnalyzer()
        assert py_analyzer.supports_language(AnalysisLanguage.PYTHON)
        assert not py_analyzer.supports_language(AnalysisLanguage.JAVA)
        
        # Test with simple Python code
        python_code = '''
def hello_world():
    """A simple function"""
    print("Hello, World!")

class MyClass:
    """A simple class"""
    pass
'''
        
        py_result = py_analyzer.analyze(python_code, "test.py")
        assert py_result.success
        assert py_result.language == AnalysisLanguage.PYTHON
        assert 'total_functions' in py_result.metrics
        
        # Test Java analyzer
        java_analyzer = JavaAnalyzer()
        assert java_analyzer.supports_language(AnalysisLanguage.JAVA)
        assert not java_analyzer.supports_language(AnalysisLanguage.PYTHON)
        
        # Test with simple Java code
        java_code = '''
/**
 * A simple class
 */
public class HelloWorld {
    /**
     * A simple method
     */
    public void sayHello() {
        System.out.println("Hello, World!");
    }
}
'''
        
        java_result = java_analyzer.analyze(java_code, "HelloWorld.java")
        assert java_result.success
        assert java_result.language == AnalysisLanguage.JAVA
        assert 'total_methods' in java_result.metrics
        
        print("   ✅ Analyzers working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Analyzers failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between components"""
    print("🔍 Testing Component Integration...")
    
    try:
        from deepcode_insight.analyzers import PythonAnalyzer
        from deepcode_insight.core import AnalysisLanguage
        
        # Test complex Python code with various issues
        complex_python_code = '''
def badFunction():  # Missing docstring, bad naming
    pass

class badclass:  # Missing docstring, bad naming
    def __init__(self):
        pass

def complex_function(a, b, c, d, e, f):  # Too many parameters
    if a:
        if b:
            if c:
                if d:  # High complexity
                    return e + f
    return 0

import unused_module  # Unused import
'''
        
        analyzer = PythonAnalyzer()
        result = analyzer.analyze(complex_python_code, "complex_test.py")
        
        assert result.success
        assert len(result.issues) > 0  # Should find issues
        assert len(result.suggestions) > 0  # Should have suggestions
        assert result.metrics['total_functions'] >= 2  # Should find functions
        
        # Check for specific issue types
        issue_types = [issue['type'] for issue in result.issues]
        assert 'missing_docstring' in issue_types
        
        print("   ✅ Integration working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test that existing functionality still works"""
    print("🔍 Testing Backward Compatibility...")
    
    try:
        # Test that we can still import from old locations if needed
        from deepcode_insight.core.interfaces import AnalysisLanguage, AnalysisResult
        from deepcode_insight.core.utils import detect_language_from_filename
        
        # Test basic functionality
        lang = detect_language_from_filename("example.py")
        assert lang == AnalysisLanguage.PYTHON
        
        result = AnalysisResult("test.py", AnalysisLanguage.PYTHON)
        result.add_issue("test", "Test issue", 1)
        assert len(result.issues) == 1
        
        print("   ✅ Backward compatibility maintained")
        return True
        
    except Exception as e:
        print(f"   ❌ Backward compatibility failed: {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 Final Refactoring Verification")
    print("=" * 60)
    
    tests = [
        ("Core Interfaces", test_core_interfaces),
        ("Tree-sitter Queries", test_tree_sitter_queries),
        ("Analyzers", test_analyzers),
        ("Integration", test_integration),
        ("Backward Compatibility", test_backward_compatibility),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n📊 Verification Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12} {test_name}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All verification tests passed!")
        print("✨ Refactoring completed successfully!")
        print("\n📈 Improvements achieved:")
        print("   • Standardized interfaces and abstract base classes")
        print("   • Centralized Tree-sitter query management")
        print("   • Improved modularity and code organization")
        print("   • Better error handling and logging")
        print("   • Enhanced maintainability and extensibility")
        return True
    else:
        print("❌ Some verification tests failed!")
        print("🔧 Please review and fix the failing components.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 