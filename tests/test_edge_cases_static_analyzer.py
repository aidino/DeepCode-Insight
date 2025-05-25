#!/usr/bin/env python3
"""Edge cases và advanced test scenarios cho StaticAnalysisAgent"""

import sys
import os
import logging
from typing import Dict, List, Optional, Any

# Setup logging
logging.basicConfig(level=logging.INFO)

# Import test infrastructure
from deepcode_insight.agents.static_analyzer import StaticAnalysisAgent

class MockASTParsingAgent:
    def parse_code(self, code: str, filename: str) -> Dict:
        return {'stats': {'total_functions': 1, 'total_classes': 1, 'total_variables': 1}, 'classes': []}


class EdgeCaseTestSuite:
    """Test suite cho edge cases và advanced scenarios"""
    
    def __init__(self):
        self.analyzer = StaticAnalysisAgent()
        self.test_results = []
    
    def run_edge_case_tests(self):
        """Chạy tất cả edge case tests"""
        print("🔬 === Edge Cases & Advanced Test Scenarios ===\n")
        
        # Python edge cases
        self.test_python_edge_cases()
        self.test_python_complex_scenarios()
        self.test_python_unicode_and_encoding()
        
        # Java edge cases
        self.test_java_edge_cases()
        self.test_java_complex_scenarios()
        self.test_java_generics_and_annotations()
        
        # Error handling
        self.test_error_handling()
        self.test_malformed_code()
        
        # Performance scenarios
        self.test_large_files()
        self.test_deeply_nested_code()
        
        # Summary
        self.print_test_summary()
    
    def test_python_edge_cases(self):
        """Test Python edge cases"""
        print("🐍 Testing Python Edge Cases")
        
        # Test với dunder methods (should not trigger naming violations)
        python_dunder = '''
class MyClass:
    def __init__(self):
        pass
    
    def __str__(self):
        return "MyClass"
    
    def __len__(self):
        return 0
    
    def __getitem__(self, key):
        return None
    
    # This should trigger violation
    def BadMethodName(self):
        pass
'''
        
        result = self.analyzer.analyze_code(python_dunder, "test_dunder.py")
        naming_violations = result['static_issues']['naming_violations']
        function_violations = [v for v in naming_violations if v['type'] == 'function_naming_violation']
        
        # Should only find BadMethodName, not dunder methods
        bad_method_violations = [v for v in function_violations if 'BadMethodName' in v.get('name', '')]
        dunder_violations = [v for v in function_violations if v.get('name', '').startswith('__')]
        
        self.assert_test(
            len(bad_method_violations) >= 1,
            "Should detect BadMethodName violation",
            f"Found {len(bad_method_violations)} bad method names"
        )
        
        self.assert_test(
            len(dunder_violations) == 0,
            "Should NOT flag dunder methods as violations",
            f"Incorrectly flagged {len(dunder_violations)} dunder methods"
        )
        
        # Test private methods (should not trigger docstring requirements)
        python_private = '''
class TestClass:
    def _private_method(self):
        return "private"
    
    def __private_method(self):
        return "very private"
    
    def public_method(self):
        return "public"
'''
        
        result = self.analyzer.analyze_code(python_private, "test_private.py")
        missing_docstrings = result['static_issues']['missing_docstrings']
        
        # Should only require docstring for public_method and class
        public_method_missing = [d for d in missing_docstrings if 'public_method' in d.get('name', '')]
        private_method_missing = [d for d in missing_docstrings if '_private_method' in d.get('name', '') or '__private_method' in d.get('name', '')]
        
        self.assert_test(
            len(public_method_missing) >= 1,
            "Should require docstring for public methods",
            f"Found {len(public_method_missing)} public methods missing docstrings"
        )
        
        self.assert_test(
            len(private_method_missing) == 0,
            "Should NOT require docstrings for private methods",
            f"Incorrectly required docstrings for {len(private_method_missing)} private methods"
        )
        
        print("  ✓ Python edge cases handled correctly\n")
    
    def test_python_complex_scenarios(self):
        """Test complex Python scenarios"""
        print("🐍 Testing Python Complex Scenarios")
        
        # Test constants (should not trigger variable naming violations)
        python_constants = '''
# Constants (should be OK)
MAX_SIZE = 100
DEFAULT_TIMEOUT = 30
API_VERSION = "v1"

# Regular variables (should follow snake_case)
BadVariableName = "should be snake_case"
good_variable_name = "correct"

# Class variables
class Config:
    MAX_RETRIES = 5  # Constant, should be OK
    BadClassVar = "should be snake_case"  # Should trigger violation
'''
        
        result = self.analyzer.analyze_code(python_constants, "test_constants.py")
        naming_violations = result['static_issues']['naming_violations']
        variable_violations = [v for v in naming_violations if v['type'] == 'variable_naming_violation']
        
        # Should find BadVariableName and BadClassVar, but not constants
        bad_vars = [v for v in variable_violations if v.get('name') in ['BadVariableName', 'BadClassVar']]
        constant_violations = [v for v in variable_violations if v.get('name') in ['MAX_SIZE', 'DEFAULT_TIMEOUT', 'API_VERSION', 'MAX_RETRIES']]
        
        self.assert_test(
            len(bad_vars) >= 1,
            "Should detect bad variable naming",
            f"Found {len(bad_vars)} bad variable names"
        )
        
        self.assert_test(
            len(constant_violations) == 0,
            "Should NOT flag constants as violations",
            f"Incorrectly flagged {len(constant_violations)} constants"
        )
        
        # Test complex lambda scenarios
        python_complex_lambda = '''
# Lambda in function call (should not trigger)
numbers = list(filter(lambda x: x > 0, [-1, 0, 1, 2]))

# Lambda in comprehension (should not trigger)
squared = [func(x) for func in [lambda y: y**2] for x in range(5)]

# Lambda assignment (should trigger)
my_filter = lambda x: x > 0
my_mapper = lambda x: x * 2

# Nested lambda assignment (should trigger)
def create_lambda():
    nested_lambda = lambda x: x + 1
    return nested_lambda
'''
        
        result = self.analyzer.analyze_code(python_complex_lambda, "test_complex_lambda.py")
        google_violations = result['static_issues']['google_style_violations']
        lambda_violations = [v for v in google_violations if v['type'] == 'lambda_assignment']
        
        self.assert_test(
            len(lambda_violations) >= 2,
            "Should detect lambda assignments but not lambda usage",
            f"Found {len(lambda_violations)} lambda assignment violations"
        )
        
        print("  ✓ Python complex scenarios handled correctly\n")
    
    def test_python_unicode_and_encoding(self):
        """Test Python với Unicode và encoding issues"""
        print("🐍 Testing Python Unicode & Encoding")
        
        python_unicode = '''
# Unicode trong variable names
tên_biến = "Vietnamese variable name"
变量名 = "Chinese variable name"

# Unicode trong strings
message = "Xin chào! 你好! こんにちは!"

# Unicode trong comments
# Đây là comment tiếng Việt
# 这是中文注释

class VietnameseClass:
    """Class với tên tiếng Việt"""
    def phương_thức(self):
        """Method với tên tiếng Việt"""
        return "OK"
'''
        
        try:
            result = self.analyzer.analyze_code(python_unicode, "test_unicode.py")
            
            self.assert_test(
                result['language'] == 'python',
                "Should handle Unicode code correctly",
                f"Language detected: {result['language']}"
            )
            
            self.assert_test(
                len(result['static_issues']['code_smells']) == 0 or 
                not any(s['type'] == 'analysis_error' for s in result['static_issues']['code_smells']),
                "Should not have analysis errors with Unicode",
                "Unicode handling successful"
            )
            
        except Exception as e:
            self.assert_test(
                False,
                "Should handle Unicode without exceptions",
                f"Exception: {str(e)}"
            )
        
        print("  ✓ Unicode handling tested\n")
    
    def test_java_edge_cases(self):
        """Test Java edge cases"""
        print("☕ Testing Java Edge Cases")
        
        # Test với interfaces và abstract classes
        java_interfaces = '''
/**
 * Documented interface
 */
public interface DocumentedInterface {
    /**
     * Documented method
     */
    void documentedMethod();
    
    // Missing Javadoc
    void undocumentedMethod();
}

// Missing Javadoc for interface
public interface UndocumentedInterface {
    void someMethod();
}

/**
 * Abstract class
 */
public abstract class AbstractClass {
    /**
     * Documented abstract method
     */
    public abstract void documentedAbstractMethod();
    
    // Missing Javadoc
    public abstract void undocumentedAbstractMethod();
}
'''
        
        result = self.analyzer.analyze_code(java_interfaces, "TestInterfaces.java")
        missing_javadoc = result['static_issues']['missing_docstrings']
        
        self.assert_test(
            len(missing_javadoc) >= 2,
            "Should detect missing Javadoc in interfaces and abstract classes",
            f"Found {len(missing_javadoc)} missing Javadoc issues"
        )
        
        # Test với nested classes
        java_nested = '''
public class OuterClass {
    
    // Missing Javadoc for nested class
    public static class NestedClass {
        public void nestedMethod() {
            System.out.println("Nested");
        }
    }
    
    /**
     * Documented inner class
     */
    public class InnerClass {
        /**
         * Documented inner method
         */
        public void innerMethod() {
            System.out.println("Inner");
        }
    }
}
'''
        
        result = self.analyzer.analyze_code(java_nested, "NestedClasses.java")
        missing_javadoc = result['static_issues']['missing_docstrings']
        
        self.assert_test(
            len(missing_javadoc) >= 1,
            "Should detect missing Javadoc in nested classes",
            f"Found {len(missing_javadoc)} missing Javadoc in nested structures"
        )
        
        print("  ✓ Java edge cases handled correctly\n")
    
    def test_java_complex_scenarios(self):
        """Test complex Java scenarios"""
        print("☕ Testing Java Complex Scenarios")
        
        # Test với exception handling patterns
        java_exceptions = '''
public class ExceptionPatterns {
    
    public void multiCatchBlock() {
        try {
            riskyOperation();
        } catch (IOException | SQLException e) {
            System.err.println("Multiple exception types: " + e.getMessage());
        }
    }
    
    public void tryWithResources() {
        try (FileInputStream fis = new FileInputStream("file.txt")) {
            // Process file
        } catch (IOException e) {
            // Proper handling
            System.err.println("File error: " + e.getMessage());
        }
    }
    
    public void emptyFinally() {
        try {
            riskyOperation();
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
        } finally {
            // Empty finally block - might be OK
        }
    }
    
    public void nestedTryCatch() {
        try {
            try {
                veryRiskyOperation();
            } catch (SpecificException e) {
                // Handle specific case
                handleSpecific(e);
            }
        } catch (Exception e) {
            // Empty outer catch - bad practice
        }
    }
}
'''
        
        result = self.analyzer.analyze_code(java_exceptions, "ExceptionPatterns.java")
        code_smells = result['static_issues']['code_smells']
        empty_catches = [s for s in code_smells if s['type'] == 'empty_catch_block']
        
        self.assert_test(
            len(empty_catches) >= 1,
            "Should detect empty catch blocks in complex scenarios",
            f"Found {len(empty_catches)} empty catch blocks"
        )
        
        # Test với method overloading
        java_overloading = '''
public class MethodOverloading {
    
    // Good naming - all camelCase
    public void processData(String data) {}
    public void processData(String data, int flags) {}
    public void processData(List<String> dataList) {}
    
    // Bad naming - should be camelCase
    public void ProcessData(byte[] data) {}
    public void process_data(Map<String, Object> dataMap) {}
}
'''
        
        result = self.analyzer.analyze_code(java_overloading, "MethodOverloading.java")
        naming_violations = result['static_issues']['naming_violations']
        method_violations = [v for v in naming_violations if v['type'] == 'java_method_naming_violation']
        
        self.assert_test(
            len(method_violations) >= 2,
            "Should detect bad method naming in overloaded methods",
            f"Found {len(method_violations)} method naming violations"
        )
        
        print("  ✓ Java complex scenarios handled correctly\n")
    
    def test_java_generics_and_annotations(self):
        """Test Java generics và annotations"""
        print("☕ Testing Java Generics & Annotations")
        
        java_generics = '''
import java.util.*;

/**
 * Generic class example
 */
public class GenericClass<T extends Comparable<T>> {
    
    private List<T> items;
    
    /**
     * Generic method
     */
    public <U> void genericMethod(U item, List<? extends U> list) {
        // Implementation
    }
    
    // Missing Javadoc for generic method
    public <K, V> Map<K, V> createMap(K key, V value) {
        Map<K, V> map = new HashMap<>();
        map.put(key, value);
        return map;
    }
}

@Deprecated
@SuppressWarnings("unchecked")
public class AnnotatedClass {
    
    @Override
    public String toString() {
        return "AnnotatedClass";
    }
    
    // Missing Javadoc despite annotations
    @CustomAnnotation(value = "test")
    public void annotatedMethod() {
        System.out.println("Annotated");
    }
}
'''
        
        result = self.analyzer.analyze_code(java_generics, "Generics.java")
        missing_javadoc = result['static_issues']['missing_docstrings']
        
        self.assert_test(
            len(missing_javadoc) >= 1,
            "Should detect missing Javadoc even with generics and annotations",
            f"Found {len(missing_javadoc)} missing Javadoc issues"
        )
        
        print("  ✓ Java generics & annotations handled correctly\n")
    
    def test_error_handling(self):
        """Test error handling scenarios"""
        print("🚨 Testing Error Handling")
        
        # Test với empty code
        result = self.analyzer.analyze_code("", "empty.py")
        self.assert_test(
            result['language'] == 'python',
            "Should handle empty code gracefully",
            f"Language: {result['language']}"
        )
        
        # Test với whitespace only
        result = self.analyzer.analyze_code("   \n\n   \t  \n", "whitespace.java")
        self.assert_test(
            result['language'] == 'java',
            "Should handle whitespace-only code",
            f"Language: {result['language']}"
        )
        
        # Test với syntax errors
        python_syntax_error = '''
def broken_function(
    # Missing closing parenthesis
    print("This will cause syntax error"
'''
        
        result = self.analyzer.analyze_code(python_syntax_error, "syntax_error.py")
        code_smells = result['static_issues']['code_smells']
        syntax_errors = [s for s in code_smells if s['type'] == 'syntax_error']
        
        self.assert_test(
            len(syntax_errors) >= 1,
            "Should detect and handle syntax errors",
            f"Found {len(syntax_errors)} syntax errors"
        )
        
        print("  ✓ Error handling working correctly\n")
    
    def test_malformed_code(self):
        """Test với malformed code patterns"""
        print("🚨 Testing Malformed Code Handling")
        
        # Test với incomplete Java class
        java_incomplete = '''
public class IncompleteClass {
    public void method1() {
        System.out.println("Method 1");
    
    // Missing closing brace for method
    
    public void method2() {
        System.out.println("Method 2");
    }
// Missing closing brace for class
'''
        
        result = self.analyzer.analyze_code(java_incomplete, "Incomplete.java")
        
        self.assert_test(
            'error' not in result or result.get('language') == 'java',
            "Should handle incomplete Java code without crashing",
            f"Analysis completed for malformed Java"
        )
        
        # Test với mixed indentation Python
        python_mixed_indent = '''
def function_with_mixed_indent():
    if True:
        print("Tab indented")
	print("Space indented")  # This might cause issues
    	print("Mixed indentation")
'''
        
        result = self.analyzer.analyze_code(python_mixed_indent, "mixed_indent.py")
        
        self.assert_test(
            'error' not in result or result.get('language') == 'python',
            "Should handle mixed indentation without crashing",
            f"Analysis completed for mixed indentation"
        )
        
        print("  ✓ Malformed code handling working\n")
    
    def test_large_files(self):
        """Test performance với large files"""
        print("⚡ Testing Large File Performance")
        
        # Generate large Python file
        large_python = "# Large Python file\n"
        for i in range(100):
            large_python += f'''
class LargeClass{i}:
    """Class number {i}"""
    
    def method_{i}_1(self):
        """Method 1 of class {i}"""
        return {i} * 1
    
    def method_{i}_2(self):
        """Method 2 of class {i}"""
        return {i} * 2
    
    def BadMethodName{i}(self):
        return {i} * 3

def function_{i}():
    """Function number {i}"""
    return {i}

BadVariableName{i} = "should be snake_case"
'''
        
        try:
            result = self.analyzer.analyze_code(large_python, "large_file.py")
            
            self.assert_test(
                result['language'] == 'python',
                "Should handle large files without timeout",
                f"Analyzed {len(large_python.split())} lines successfully"
            )
            
            # Should find many violations
            total_issues = sum(len(issues) for issues in result['static_issues'].values())
            
            self.assert_test(
                total_issues > 50,
                "Should find many issues in large file",
                f"Found {total_issues} total issues"
            )
            
        except Exception as e:
            self.assert_test(
                False,
                "Should handle large files without exceptions",
                f"Exception: {str(e)}"
            )
        
        print("  ✓ Large file performance acceptable\n")
    
    def test_deeply_nested_code(self):
        """Test với deeply nested code structures"""
        print("🔄 Testing Deeply Nested Code")
        
        # Generate deeply nested Python code
        nested_python = '''
def deeply_nested_function():
    """Function with deep nesting"""
    if True:
        if True:
            if True:
                if True:
                    if True:
                        if True:
                            if True:
                                if True:
                                    if True:
                                        if True:
                                            print("Very deeply nested")
                                            return "deep"
    return "shallow"

class DeeplyNestedClass:
    """Class with nested structures"""
    
    def method_with_nested_functions(self):
        """Method containing nested functions"""
        
        def level1():
            def level2():
                def level3():
                    def level4():
                        def level5():
                            return "very nested function"
                        return level5()
                    return level4()
                return level3()
            return level2()
        
        return level1()
'''
        
        try:
            result = self.analyzer.analyze_code(nested_python, "nested.py")
            
            self.assert_test(
                result['language'] == 'python',
                "Should handle deeply nested code",
                f"Analysis completed successfully"
            )
            
            # Check if complexity is detected
            complex_functions = result['static_issues']['complex_functions']
            high_complexity = [f for f in complex_functions if f.get('type') == 'high_complexity']
            
            self.assert_test(
                len(high_complexity) >= 1,
                "Should detect high complexity in nested code",
                f"Found {len(high_complexity)} high complexity functions"
            )
            
        except Exception as e:
            self.assert_test(
                False,
                "Should handle deeply nested code without exceptions",
                f"Exception: {str(e)}"
            )
        
        print("  ✓ Deeply nested code handled correctly\n")
    
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
        print("📊 EDGE CASE TEST SUMMARY")
        print("=" * 60)
        
        passed = len([r for r in self.test_results if r[0] == "PASS"])
        failed = len([r for r in self.test_results if r[0] == "FAIL"])
        total = len(self.test_results)
        
        print(f"Total Edge Case Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if failed > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.test_results:
                if result[0] == "FAIL":
                    print(f"  - {result[1]}")
        
        print(f"\n🔬 Edge Cases Tested:")
        print(f"  ✓ Python dunder methods handling")
        print(f"  ✓ Private method docstring requirements")
        print(f"  ✓ Constants vs variables naming")
        print(f"  ✓ Complex lambda scenarios")
        print(f"  ✓ Unicode and encoding support")
        print(f"  ✓ Java interfaces and abstract classes")
        print(f"  ✓ Nested classes and inner classes")
        print(f"  ✓ Complex exception handling patterns")
        print(f"  ✓ Method overloading scenarios")
        print(f"  ✓ Generics and annotations")
        print(f"  ✓ Error handling and recovery")
        print(f"  ✓ Malformed code resilience")
        print(f"  ✓ Large file performance")
        print(f"  ✓ Deeply nested code structures")
        
        return failed == 0


def main():
    """Main test runner cho edge cases"""
    try:
        tester = EdgeCaseTestSuite()
        success = tester.run_edge_case_tests()
        
        if success:
            print("\n🎉 All edge case tests passed! StaticAnalysisAgent is robust.")
            return 0
        else:
            print("\n💥 Some edge case tests failed! Check robustness.")
            return 1
            
    except Exception as e:
        print(f"\n💥 Edge case test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 