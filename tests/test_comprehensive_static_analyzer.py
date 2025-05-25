#!/usr/bin/env python3
"""Comprehensive test suite cho Enhanced StaticAnalysisAgent với Python và Java rules"""

import sys
import os
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Node, Query
import re

# Setup logging
logging.basicConfig(level=logging.INFO)

# Mock ASTParsingAgent để test độc lập
class MockASTParsingAgent:
    def parse_code(self, code: str, filename: str) -> Dict:
        # Simulate AST analysis results
        lines = code.split('\n')
        return {
            'stats': {
                'total_functions': code.count('def ') + code.count('public ') + code.count('private '),
                'total_classes': code.count('class ') + code.count('public class '),
                'total_variables': len([line for line in lines if '=' in line and not line.strip().startswith('#')])
            },
            'classes': [
                {
                    'name': 'TestClass',
                    'start_line': 1,
                    'method_count': 5
                }
            ]
        }

# Import StaticAnalysisAgent class (simplified version for testing)
sys.path.append(os.path.join(os.path.dirname(__file__), 'deepcode_insight'))

from deepcode_insight.agents.static_analyzer import StaticAnalysisAgent


class TestStaticAnalysisAgent:
    """Comprehensive test class cho StaticAnalysisAgent"""
    
    def __init__(self):
        self.analyzer = StaticAnalysisAgent()
        self.test_results = []
    
    def run_all_tests(self):
        """Chạy tất cả test cases"""
        print("🧪 === Comprehensive StaticAnalysisAgent Test Suite ===\n")
        
        # Python tests
        self.test_python_google_style_guide()
        self.test_python_lambda_assignments()
        self.test_python_exception_handling()
        self.test_python_string_formatting()
        self.test_python_line_length()
        self.test_python_comprehensions()
        
        # Java tests
        self.test_java_missing_javadoc()
        self.test_java_empty_catch_blocks()
        self.test_java_naming_conventions()
        self.test_java_long_lines()
        self.test_java_metrics()
        
        # Multi-language tests
        self.test_language_detection()
        self.test_unsupported_language()
        
        # Summary
        self.print_test_summary()
    
    def test_python_google_style_guide(self):
        """Test Google Python Style Guide compliance"""
        print("🐍 Testing Python Google Style Guide Rules")
        
        python_code = '''
import os
import unused_import

# Bad class naming
class bad_class_name:
    def __init__(self):
        pass
    
    # Bad method naming
    def BadMethodName(self):
        pass

# Bad variable naming
BadVariable = "should be snake_case"
another_Bad_Variable = "mixed case"

# Good examples
class GoodClassName:
    def good_method_name(self):
        good_variable = "snake_case"
        return good_variable

def function_without_docstring():
    return "missing docstring"
'''
        
        result = self.analyzer.analyze_code(python_code, "test_google_style.py")
        
        # Check naming violations
        naming_violations = result['static_issues']['naming_violations']
        class_violations = [v for v in naming_violations if v['type'] == 'class_naming_violation']
        function_violations = [v for v in naming_violations if v['type'] == 'function_naming_violation']
        variable_violations = [v for v in naming_violations if v['type'] == 'variable_naming_violation']
        
        self.assert_test(
            len(class_violations) >= 1,
            "Should detect bad class naming (bad_class_name)",
            f"Found {len(class_violations)} class naming violations"
        )
        
        self.assert_test(
            len(function_violations) >= 1,
            "Should detect bad method naming (BadMethodName)",
            f"Found {len(function_violations)} function naming violations"
        )
        
        self.assert_test(
            len(variable_violations) >= 1,
            "Should detect bad variable naming",
            f"Found {len(variable_violations)} variable naming violations"
        )
        
        print(f"  ✓ Naming violations: {len(naming_violations)} total\n")
    
    def test_python_lambda_assignments(self):
        """Test lambda assignment detection"""
        print("🐍 Testing Python Lambda Assignment Detection")
        
        python_code = '''
# Bad: Lambda assignments (discouraged by Google Style Guide)
my_lambda = lambda x: x * 2
another_lambda = lambda a, b: a + b

# Good: Use def instead
def my_function(x):
    return x * 2

# Lambda in other contexts (should not trigger)
numbers = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, numbers))
'''
        
        result = self.analyzer.analyze_code(python_code, "test_lambda.py")
        google_violations = result['static_issues']['google_style_violations']
        lambda_violations = [v for v in google_violations if v['type'] == 'lambda_assignment']
        
        self.assert_test(
            len(lambda_violations) >= 2,
            "Should detect lambda assignments",
            f"Found {len(lambda_violations)} lambda assignment violations"
        )
        
        print(f"  ✓ Lambda assignments detected: {len(lambda_violations)}\n")
    
    def test_python_exception_handling(self):
        """Test exception handling patterns"""
        print("🐍 Testing Python Exception Handling")
        
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
        
        result = self.analyzer.analyze_code(python_code, "test_exceptions.py")
        google_violations = result['static_issues']['google_style_violations']
        bare_except_violations = [v for v in google_violations if v['type'] == 'bare_except']
        
        self.assert_test(
            len(bare_except_violations) >= 2,
            "Should detect bare except clauses",
            f"Found {len(bare_except_violations)} bare except violations"
        )
        
        print(f"  ✓ Bare except clauses detected: {len(bare_except_violations)}\n")
    
    def test_python_string_formatting(self):
        """Test string formatting detection"""
        print("🐍 Testing Python String Formatting")
        
        python_code = '''
name = "Alice"
count = 5

# Bad: Old-style % formatting
message1 = "Hello %s, you have %d messages" % (name, count)
message2 = "User %(name)s has %(count)d items" % {"name": name, "count": count}

# Good: Modern formatting
message3 = f"Hello {name}, you have {count} messages"
message4 = "Hello {}, you have {} messages".format(name, count)
'''
        
        result = self.analyzer.analyze_code(python_code, "test_formatting.py")
        google_violations = result['static_issues']['google_style_violations']
        formatting_violations = [v for v in google_violations if v['type'] == 'old_string_formatting']
        
        self.assert_test(
            len(formatting_violations) >= 1,
            "Should detect old-style string formatting",
            f"Found {len(formatting_violations)} old formatting violations"
        )
        
        print(f"  ✓ Old string formatting detected: {len(formatting_violations)}\n")
    
    def test_python_line_length(self):
        """Test line length detection"""
        print("🐍 Testing Python Line Length")
        
        python_code = '''
# Short line
x = 1

# Long line exceeding 79 characters (Google Style Guide limit)
very_long_variable_name = "This is a very long string that definitely exceeds the 79 character limit recommended by Google Python Style Guide"

# Another long line
result = some_function_with_very_long_name(parameter1, parameter2, parameter3, parameter4, parameter5)
'''
        
        result = self.analyzer.analyze_code(python_code, "test_line_length.py")
        google_violations = result['static_issues']['google_style_violations']
        line_length_violations = [v for v in google_violations if v['type'] == 'line_too_long']
        
        self.assert_test(
            len(line_length_violations) >= 2,
            "Should detect lines exceeding 79 characters",
            f"Found {len(line_length_violations)} long line violations"
        )
        
        print(f"  ✓ Long lines detected: {len(line_length_violations)}\n")
    
    def test_python_comprehensions(self):
        """Test complex comprehension detection"""
        print("🐍 Testing Python Complex Comprehensions")
        
        python_code = '''
# Simple comprehension (should be OK)
numbers = [x for x in range(10)]

# Complex comprehension (should trigger warning)
complex_comp = [
    x * y + z
    for x in range(10)
    for y in range(5)
    for z in range(3)
    if x > y and y > z
]

# Very long comprehension
long_comp = [very_long_function_name(x, y, z) for x in very_long_iterable_name for y in another_long_iterable if some_complex_condition(x, y)]
'''
        
        result = self.analyzer.analyze_code(python_code, "test_comprehensions.py")
        google_violations = result['static_issues']['google_style_violations']
        comp_violations = [v for v in google_violations if v['type'] == 'complex_comprehension']
        
        self.assert_test(
            len(comp_violations) >= 1,
            "Should detect complex comprehensions",
            f"Found {len(comp_violations)} complex comprehension violations"
        )
        
        print(f"  ✓ Complex comprehensions detected: {len(comp_violations)}\n")
    
    def test_java_missing_javadoc(self):
        """Test Java Javadoc detection"""
        print("☕ Testing Java Missing Javadoc Detection")
        
        java_code = '''
import java.util.*;

// Missing Javadoc for class
public class TestClass {
    private int value;
    
    // Missing Javadoc for public method
    public void publicMethod() {
        System.out.println("No Javadoc");
    }
    
    /**
     * This method has proper Javadoc
     * @param param the parameter
     * @return the result
     */
    public String documentedMethod(String param) {
        return param.toUpperCase();
    }
    
    // Private method (should not require Javadoc)
    private void privateMethod() {
        System.out.println("Private method");
    }
}

/**
 * This class has proper Javadoc
 */
public class DocumentedClass {
    /**
     * Constructor with Javadoc
     */
    public DocumentedClass() {
    }
}
'''
        
        result = self.analyzer.analyze_code(java_code, "TestClass.java")
        missing_javadoc = result['static_issues']['missing_docstrings']
        
        class_javadoc_missing = [j for j in missing_javadoc if j['type'] == 'missing_class_javadoc']
        method_javadoc_missing = [j for j in missing_javadoc if j['type'] == 'missing_method_javadoc']
        
        self.assert_test(
            len(class_javadoc_missing) >= 1,
            "Should detect missing class Javadoc",
            f"Found {len(class_javadoc_missing)} missing class Javadoc"
        )
        
        self.assert_test(
            len(method_javadoc_missing) >= 1,
            "Should detect missing method Javadoc",
            f"Found {len(method_javadoc_missing)} missing method Javadoc"
        )
        
        print(f"  ✓ Missing Javadoc detected: {len(missing_javadoc)} total\n")
    
    def test_java_empty_catch_blocks(self):
        """Test Java empty catch block detection"""
        print("☕ Testing Java Empty Catch Block Detection")
        
        java_code = '''
public class ExceptionTest {
    
    public void methodWithEmptyCatch() {
        try {
            int result = 10 / 0;
        } catch (Exception e) {
            // Empty catch block - bad practice
        }
    }
    
    public void anotherEmptyCatch() {
        try {
            riskyOperation();
        } catch (RuntimeException e) {}
    }
    
    public void properCatchHandling() {
        try {
            riskyOperation();
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            // Proper exception handling
        }
    }
}
'''
        
        result = self.analyzer.analyze_code(java_code, "ExceptionTest.java")
        code_smells = result['static_issues']['code_smells']
        empty_catches = [s for s in code_smells if s['type'] == 'empty_catch_block']
        
        self.assert_test(
            len(empty_catches) >= 1,
            "Should detect empty catch blocks",
            f"Found {len(empty_catches)} empty catch blocks"
        )
        
        print(f"  ✓ Empty catch blocks detected: {len(empty_catches)}\n")
    
    def test_java_naming_conventions(self):
        """Test Java naming conventions"""
        print("☕ Testing Java Naming Conventions")
        
        java_code = '''
// Bad class name (should be PascalCase)
public class badClassName {
    
    // Bad field names
    private int BadFieldName;
    private String bad_field_name;
    
    // Bad method names
    public void BadMethodName() {}
    public void bad_method_name() {}
    
    // Bad variable names
    public void testMethod() {
        int BadVariable = 5;
        String bad_variable = "test";
    }
}

// Good naming examples
public class GoodClassName {
    private int goodFieldName;
    private String anotherGoodField;
    
    public void goodMethodName() {
        int goodVariable = 5;
        String anotherGoodVariable = "test";
    }
}
'''
        
        result = self.analyzer.analyze_code(java_code, "NamingTest.java")
        naming_violations = result['static_issues']['naming_violations']
        
        class_violations = [v for v in naming_violations if v['type'] == 'java_class_naming_violation']
        method_violations = [v for v in naming_violations if v['type'] == 'java_method_naming_violation']
        variable_violations = [v for v in naming_violations if v['type'] == 'java_variable_naming_violation']
        
        self.assert_test(
            len(class_violations) >= 1,
            "Should detect bad Java class naming",
            f"Found {len(class_violations)} class naming violations"
        )
        
        self.assert_test(
            len(method_violations) >= 1,
            "Should detect bad Java method naming",
            f"Found {len(method_violations)} method naming violations"
        )
        
        self.assert_test(
            len(variable_violations) >= 1,
            "Should detect bad Java variable naming",
            f"Found {len(variable_violations)} variable naming violations"
        )
        
        print(f"  ✓ Java naming violations: {len(naming_violations)} total\n")
    
    def test_java_long_lines(self):
        """Test Java long line detection"""
        print("☕ Testing Java Long Line Detection")
        
        java_code = '''
public class LongLineTest {
    
    // Short line
    private int x = 1;
    
    // Long line exceeding 120 characters (Java convention)
    private String veryLongVariableName = "This is a very long string that definitely exceeds the 120 character limit commonly used in Java projects for better readability";
    
    // Another long line
    public void methodWithVeryLongSignature(String parameterOne, String parameterTwo, String parameterThree, String parameterFour, String parameterFive) {
        System.out.println("Method with long signature");
    }
}
'''
        
        result = self.analyzer.analyze_code(java_code, "LongLineTest.java")
        code_smells = result['static_issues']['code_smells']
        long_lines = [s for s in code_smells if s['type'] == 'long_line']
        
        self.assert_test(
            len(long_lines) >= 2,
            "Should detect lines exceeding 120 characters",
            f"Found {len(long_lines)} long line violations"
        )
        
        print(f"  ✓ Long lines detected: {len(long_lines)}\n")
    
    def test_java_metrics(self):
        """Test Java metrics calculation"""
        print("☕ Testing Java Metrics Calculation")
        
        java_code = '''
/**
 * Test class for metrics calculation
 */
public class MetricsTest {
    
    /**
     * Method with some complexity
     */
    public void complexMethod(int param) {
        if (param > 0) {
            for (int i = 0; i < param; i++) {
                if (i % 2 == 0) {
                    System.out.println("Even: " + i);
                } else {
                    System.out.println("Odd: " + i);
                }
            }
        }
        
        try {
            riskyOperation();
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}
'''
        
        result = self.analyzer.analyze_code(java_code, "MetricsTest.java")
        metrics = result['metrics']
        
        self.assert_test(
            metrics['cyclomatic_complexity'] > 0,
            "Should calculate cyclomatic complexity",
            f"Complexity: {metrics['cyclomatic_complexity']}"
        )
        
        self.assert_test(
            metrics['comment_ratio'] > 0,
            "Should calculate comment ratio",
            f"Comment ratio: {metrics['comment_ratio']:.2f}"
        )
        
        self.assert_test(
            0 <= metrics['code_quality_score'] <= 100,
            "Should calculate quality score between 0-100",
            f"Quality score: {metrics['code_quality_score']:.1f}"
        )
        
        print(f"  ✓ Java metrics calculated successfully\n")
    
    def test_language_detection(self):
        """Test language detection from file extensions"""
        print("🔍 Testing Language Detection")
        
        python_result = self.analyzer.analyze_code("print('hello')", "test.py")
        java_result = self.analyzer.analyze_code("System.out.println(\"hello\");", "Test.java")
        
        self.assert_test(
            python_result['language'] == 'python',
            "Should detect Python from .py extension",
            f"Detected: {python_result['language']}"
        )
        
        self.assert_test(
            java_result['language'] == 'java',
            "Should detect Java from .java extension",
            f"Detected: {java_result['language']}"
        )
        
        print(f"  ✓ Language detection working correctly\n")
    
    def test_unsupported_language(self):
        """Test handling of unsupported languages"""
        print("🔍 Testing Unsupported Language Handling")
        
        result = self.analyzer.analyze_code("console.log('hello');", "test.js")
        
        self.assert_test(
            result['language'] == 'unknown',
            "Should detect unknown language",
            f"Detected: {result['language']}"
        )
        
        code_smells = result['static_issues']['code_smells']
        unsupported_errors = [s for s in code_smells if s['type'] == 'unsupported_language']
        
        self.assert_test(
            len(unsupported_errors) > 0,
            "Should report unsupported language error",
            f"Found {len(unsupported_errors)} unsupported language errors"
        )
        
        print(f"  ✓ Unsupported language handling working\n")
    
    def assert_test(self, condition: bool, description: str, details: str = ""):
        """Helper method để assert test results"""
        if condition:
            self.test_results.append(("PASS", description, details))
            print(f"    ✅ {description} - {details}")
        else:
            self.test_results.append(("FAIL", description, details))
            print(f"    ❌ {description} - {details}")
    
    def print_test_summary(self):
        """In tóm tắt kết quả test"""
        print("=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = len([r for r in self.test_results if r[0] == "PASS"])
        failed = len([r for r in self.test_results if r[0] == "FAIL"])
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if failed > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.test_results:
                if result[0] == "FAIL":
                    print(f"  - {result[1]}")
        
        print(f"\n✨ Enhanced Features Tested:")
        print(f"  ✓ Google Python Style Guide compliance")
        print(f"  ✓ Python naming conventions (PascalCase, snake_case)")
        print(f"  ✓ Lambda assignment detection")
        print(f"  ✓ Exception handling patterns")
        print(f"  ✓ String formatting modernization")
        print(f"  ✓ Line length validation")
        print(f"  ✓ Complex comprehension detection")
        print(f"  ✓ Java Javadoc requirements")
        print(f"  ✓ Java empty catch block detection")
        print(f"  ✓ Java naming conventions (PascalCase, camelCase)")
        print(f"  ✓ Java long line detection")
        print(f"  ✓ Java metrics calculation")
        print(f"  ✓ Multi-language support")
        print(f"  ✓ Language detection and error handling")
        
        return failed == 0


def main():
    """Main test runner"""
    try:
        tester = TestStaticAnalysisAgent()
        success = tester.run_all_tests()
        
        if success:
            print("\n🎉 All tests passed! Enhanced StaticAnalysisAgent is working correctly.")
            return 0
        else:
            print("\n💥 Some tests failed! Please check the implementation.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 