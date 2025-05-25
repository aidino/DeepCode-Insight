#!/usr/bin/env python3
"""
Test script for refactored analyzers
"""

from deepcode_insight.analyzers import PythonAnalyzer, JavaAnalyzer
from deepcode_insight.core import AnalysisLanguage

def test_python_analyzer():
    """Test Python analyzer"""
    print("🧪 Testing Python Analyzer")
    
    analyzer = PythonAnalyzer()
    
    # Test Python code with various issues
    python_code = '''
def badFunction():  # Should be snake_case
    pass  # Missing docstring

class badclass:  # Should be PascalCase
    pass  # Missing docstring

def complex_function(a, b, c, d, e, f, g):  # Too many parameters
    if a:
        if b:
            if c:
                if d:
                    if e:  # High complexity
                        return f + g
    return 0

lambda x: x + 1 if x > 0 else x - 1 if x < 0 else 0  # Complex lambda

import unused_module  # Unused import
'''
    
    result = analyzer.analyze(python_code, "test.py")
    
    print(f"   ✅ Analysis completed successfully")
    print(f"   📊 Found {len(result.issues)} issues")
    print(f"   📈 Metrics: {result.metrics}")
    print(f"   💡 Suggestions: {len(result.suggestions)}")
    
    # Print some issues
    for issue in result.issues[:3]:  # Show first 3 issues
        print(f"   🔍 {issue['type']}: {issue['message']}")
    
    return len(result.issues) > 0

def test_java_analyzer():
    """Test Java analyzer"""
    print("\n🧪 Testing Java Analyzer")
    
    analyzer = JavaAnalyzer()
    
    # Test Java code with various issues
    java_code = '''
public class badclass {  // Should be PascalCase
    
    public void BadMethod() {  // Should be camelCase
        int magicNumber = 42;  // Magic number
        
        try {
            // Some risky operation
        } catch (Exception e) {
            // Empty catch block
        }
    }
    
    public void methodWithTooManyParams(int a, int b, int c, int d, int e, int f, int g) {
        // Too many parameters
        if (a > 0) {
            if (b > 0) {
                if (c > 0) {
                    if (d > 0) {  // High complexity
                        System.out.println("Complex logic");
                    }
                }
            }
        }
    }
}
'''
    
    result = analyzer.analyze(java_code, "Test.java")
    
    print(f"   ✅ Analysis completed successfully")
    print(f"   📊 Found {len(result.issues)} issues")
    print(f"   📈 Metrics: {result.metrics}")
    print(f"   💡 Suggestions: {len(result.suggestions)}")
    
    # Print some issues
    for issue in result.issues[:3]:  # Show first 3 issues
        print(f"   🔍 {issue['type']}: {issue['message']}")
    
    return len(result.issues) > 0

def test_analyzer_interfaces():
    """Test analyzer interfaces"""
    print("\n🧪 Testing Analyzer Interfaces")
    
    py_analyzer = PythonAnalyzer()
    java_analyzer = JavaAnalyzer()
    
    # Test language support
    assert py_analyzer.supports_language(AnalysisLanguage.PYTHON)
    assert not py_analyzer.supports_language(AnalysisLanguage.JAVA)
    
    assert java_analyzer.supports_language(AnalysisLanguage.JAVA)
    assert not java_analyzer.supports_language(AnalysisLanguage.PYTHON)
    
    print("   ✅ Language support checks passed")
    
    # Test empty code handling
    empty_result = py_analyzer.analyze("", "empty.py")
    assert empty_result.success
    assert len(empty_result.issues) == 0
    
    print("   ✅ Empty code handling passed")
    
    return True

def main():
    """Main test function"""
    print("🚀 Testing Refactored Analyzers")
    print("=" * 50)
    
    try:
        # Test Python analyzer
        py_success = test_python_analyzer()
        
        # Test Java analyzer
        java_success = test_java_analyzer()
        
        # Test interfaces
        interface_success = test_analyzer_interfaces()
        
        print("\n📊 Test Summary")
        print("=" * 50)
        print(f"✅ Python Analyzer: {'PASSED' if py_success else 'FAILED'}")
        print(f"✅ Java Analyzer: {'PASSED' if java_success else 'FAILED'}")
        print(f"✅ Interfaces: {'PASSED' if interface_success else 'FAILED'}")
        
        if py_success and java_success and interface_success:
            print("\n🎉 All tests passed! Refactored analyzers are working correctly.")
            return True
        else:
            print("\n❌ Some tests failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 