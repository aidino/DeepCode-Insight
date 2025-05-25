#!/usr/bin/env python3
"""
Demo script cho StaticAnalysisAgent
Showcase các tính năng phân tích tĩnh code Python
"""

import logging
from agents.static_analyzer import StaticAnalysisAgent
from agents.code_fetcher import CodeFetcherAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def demo_basic_static_analysis():
    """Demo basic static analysis functionality"""
    print("🔍 === Demo Basic Static Analysis ===")
    
    # Sample code với nhiều issues
    sample_code = '''
import os
import sys
import unused_module
from typing import List, Dict, Optional
from collections import defaultdict, Counter

class Calculator:
    def __init__(self, initial_value=0):
        self.value = initial_value
    
    def add(self, x, y, z, a, b, c, d):  # Too many parameters
        """Add multiple numbers"""
        if x > 0:
            if y > 0:
                if z > 0:
                    if a > 0:
                        if b > 0:
                            if c > 0:
                                if d > 0:
                                    result = x + y + z + a + b + c + d
                                    self.value += result
                                    return self.value
        return 0
    
    def complex_method(self, param1, param2, param3, param4, param5, param6):
        # This is a very long line that exceeds the recommended line length limit and should be flagged by the static analyzer as a code smell
        for i in range(100):
            for j in range(100):
                for k in range(100):
                    if i > j:
                        if j > k:
                            if k > 0:
                                print(f"Complex calculation: {i * j * k}")
                                
                                def nested_function():
                                    def deeply_nested():
                                        def very_deeply_nested():
                                            return "too much nesting"
                                        return very_deeply_nested()
                                    return deeply_nested()
                                
                                result = nested_function()
        return result

def function_without_docstring(x, y):
    return x + y

def another_function():
    pass

class DataProcessor:
    def method1(self): pass
    def method2(self): pass
    def method3(self): pass
    def method4(self): pass
    def method5(self): pass
    def method6(self): pass
    def method7(self): pass
    def method8(self): pass
    def method9(self): pass
    def method10(self): pass
    def method11(self): pass
    def method12(self): pass
    def method13(self): pass
    def method14(self): pass
    def method15(self): pass
    def method16(self): pass
    def method17(self): pass
    def method18(self): pass
    def method19(self): pass
    def method20(self): pass
    def method21(self): pass  # God class

# Too many global variables
GLOBAL1 = 1
GLOBAL2 = 2
GLOBAL3 = 3
GLOBAL4 = 4
GLOBAL5 = 5
GLOBAL6 = 6
GLOBAL7 = 7
GLOBAL8 = 8
GLOBAL9 = 9
GLOBAL10 = 10
GLOBAL11 = 11
GLOBAL12 = 12
'''
    
    analyzer = StaticAnalysisAgent()
    result = analyzer.analyze_code(sample_code, "demo.py")
    
    print(f"📁 File: {result['filename']}")
    print(f"🎯 Quality Score: {result['metrics']['code_quality_score']:.1f}/100")
    print(f"🔧 Maintainability Index: {result['metrics']['maintainability_index']:.1f}/100")
    print(f"🔄 Cyclomatic Complexity: {result['metrics']['cyclomatic_complexity']}")
    print()
    
    print("📋 Issues Found:")
    total_issues = 0
    for category, issues in result['static_issues'].items():
        if issues:
            print(f"\n  📌 {category.replace('_', ' ').title()} ({len(issues)} issues):")
            total_issues += len(issues)
            for issue in issues[:3]:  # Show first 3 issues
                print(f"    - Line {issue.get('line', '?')}: {issue['message']}")
            if len(issues) > 3:
                print(f"    ... và {len(issues) - 3} issues khác")
    
    print(f"\n📊 Total Issues: {total_issues}")
    
    print(f"\n💡 Suggestions ({len(result['suggestions'])}):")
    for i, suggestion in enumerate(result['suggestions'], 1):
        print(f"  {i}. {suggestion}")
    
    print(f"\n📈 Detailed Metrics:")
    for metric, value in result['metrics'].items():
        if isinstance(value, float):
            print(f"  - {metric.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"  - {metric.replace('_', ' ').title()}: {value}")

def demo_good_code_analysis():
    """Demo với good quality code"""
    print("\n🌟 === Demo Good Code Analysis ===")
    
    good_code = '''
"""
A well-documented calculator module.
This module provides a Calculator class for basic arithmetic operations.
"""

from typing import Union, Optional


class Calculator:
    """
    A simple calculator class for basic arithmetic operations.
    
    This class maintains an internal value and provides methods
    to perform arithmetic operations on it.
    """
    
    def __init__(self, initial_value: float = 0.0) -> None:
        """
        Initialize the calculator with an initial value.
        
        Args:
            initial_value: The starting value for calculations
        """
        self.value = initial_value
    
    def add(self, number: Union[int, float]) -> float:
        """
        Add a number to the current value.
        
        Args:
            number: The number to add
            
        Returns:
            The new current value after addition
        """
        self.value += number
        return self.value
    
    def subtract(self, number: Union[int, float]) -> float:
        """
        Subtract a number from the current value.
        
        Args:
            number: The number to subtract
            
        Returns:
            The new current value after subtraction
        """
        self.value -= number
        return self.value
    
    def multiply(self, number: Union[int, float]) -> float:
        """
        Multiply the current value by a number.
        
        Args:
            number: The number to multiply by
            
        Returns:
            The new current value after multiplication
        """
        self.value *= number
        return self.value
    
    def divide(self, number: Union[int, float]) -> Optional[float]:
        """
        Divide the current value by a number.
        
        Args:
            number: The number to divide by
            
        Returns:
            The new current value after division, or None if division by zero
        """
        if number == 0:
            return None
        self.value /= number
        return self.value
    
    def reset(self) -> None:
        """Reset the calculator value to zero."""
        self.value = 0.0
    
    def get_value(self) -> float:
        """
        Get the current calculator value.
        
        Returns:
            The current value
        """
        return self.value


def create_calculator(initial: float = 0.0) -> Calculator:
    """
    Factory function to create a new Calculator instance.
    
    Args:
        initial: Initial value for the calculator
        
    Returns:
        A new Calculator instance
    """
    return Calculator(initial)


def main() -> None:
    """Main function to demonstrate calculator usage."""
    calc = create_calculator(10.0)
    
    # Perform some calculations
    calc.add(5)
    calc.multiply(2)
    calc.subtract(3)
    
    print(f"Final result: {calc.get_value()}")


if __name__ == "__main__":
    main()
'''
    
    analyzer = StaticAnalysisAgent()
    result = analyzer.analyze_code(good_code, "good_calculator.py")
    
    print(f"📁 File: {result['filename']}")
    print(f"🎯 Quality Score: {result['metrics']['code_quality_score']:.1f}/100")
    print(f"🔧 Maintainability Index: {result['metrics']['maintainability_index']:.1f}/100")
    print(f"💬 Comment Ratio: {result['metrics']['comment_ratio']:.2f}")
    print()
    
    total_issues = sum(len(issues) for issues in result['static_issues'].values())
    print(f"📊 Total Issues: {total_issues}")
    
    if total_issues == 0:
        print("✅ No issues found! This is well-written code.")
    else:
        print("📋 Issues Found:")
        for category, issues in result['static_issues'].items():
            if issues:
                print(f"  📌 {category.replace('_', ' ').title()}: {len(issues)} issues")
    
    if result['suggestions']:
        print(f"\n💡 Suggestions:")
        for suggestion in result['suggestions']:
            print(f"  - {suggestion}")
    else:
        print("\n💡 No suggestions needed - code quality is excellent!")

def demo_tree_sitter_queries():
    """Demo Tree-sitter queries trực tiếp"""
    print("\n🌳 === Demo Tree-sitter Queries ===")
    
    query_demo_code = '''
def function_with_docstring():
    """This function has a docstring."""
    pass

def function_without_docstring():
    pass

class ClassWithDocstring:
    """This class has a docstring."""
    pass

class ClassWithoutDocstring:
    pass

def complex_function(a, b, c, d, e, f, g):  # 7 parameters
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        if f > 0:
                            if g > 0:
                                def nested():
                                    def deeply_nested():
                                        return "nested"
                                    return deeply_nested()
                                return nested()
    return None
'''
    
    analyzer = StaticAnalysisAgent()
    result = analyzer.analyze_code(query_demo_code, "query_demo.py")
    
    print("🔍 Tree-sitter Query Results:")
    
    # Missing docstrings
    missing_docs = result['static_issues']['missing_docstrings']
    print(f"\n📝 Missing Docstrings ({len(missing_docs)}):")
    for issue in missing_docs:
        print(f"  - {issue['type']}: {issue['name']} (line {issue['line']})")
    
    # Complex functions
    complex_funcs = result['static_issues']['complex_functions']
    print(f"\n🔄 Complex Functions ({len(complex_funcs)}):")
    for issue in complex_funcs:
        print(f"  - {issue['type']}: {issue['name']} (line {issue['line']})")
        if 'count' in issue:
            print(f"    Count: {issue['count']}")
        if 'complexity' in issue:
            print(f"    Complexity: {issue['complexity']}")

def demo_file_analysis():
    """Demo phân tích file thực tế"""
    print("\n📄 === Demo File Analysis ===")
    
    # Analyze the static analyzer itself
    analyzer = StaticAnalysisAgent()
    
    try:
        result = analyzer.analyze_file("agents/static_analyzer.py")
        
        print(f"📁 Analyzing: {result['filename']}")
        print(f"🎯 Quality Score: {result['metrics']['code_quality_score']:.1f}/100")
        print(f"📏 Lines of Code: {result['metrics']['lines_of_code']}")
        print(f"🔄 Cyclomatic Complexity: {result['metrics']['cyclomatic_complexity']}")
        
        total_issues = sum(len(issues) for issues in result['static_issues'].values())
        print(f"📊 Total Issues: {total_issues}")
        
        if result['suggestions']:
            print(f"\n💡 Top Suggestions:")
            for i, suggestion in enumerate(result['suggestions'][:3], 1):
                print(f"  {i}. {suggestion}")
        
        # Show AST analysis summary
        ast_stats = result['ast_analysis']['stats']
        print(f"\n🌳 AST Analysis:")
        print(f"  - Functions: {ast_stats['total_functions']}")
        print(f"  - Classes: {ast_stats['total_classes']}")
        print(f"  - Imports: {ast_stats['total_imports']}")
        
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")

def demo_repository_analysis():
    """Demo phân tích repository (nếu có CodeFetcherAgent)"""
    print("\n🗂️ === Demo Repository Analysis ===")
    
    try:
        # Tạo sample repository analysis
        analyzer = StaticAnalysisAgent()
        code_fetcher = CodeFetcherAgent()
        
        # Thay vì analyze repository thực, ta sẽ simulate
        print("📋 Repository Analysis Features:")
        print("  ✅ Analyze multiple Python files")
        print("  ✅ Aggregate quality metrics")
        print("  ✅ Generate repository-level suggestions")
        print("  ✅ Issue breakdown by category")
        print("  ✅ Average quality score calculation")
        print()
        print("💡 To analyze a real repository:")
        print("  analyzer = StaticAnalysisAgent()")
        print("  code_fetcher = CodeFetcherAgent()")
        print("  result = analyzer.analyze_repository(code_fetcher, 'repo_url')")
        
    except Exception as e:
        print(f"ℹ️ Repository analysis demo skipped: {e}")

def main():
    """Main demo function"""
    print("🚀 === StaticAnalysisAgent Demo ===")
    print("Demonstrating Python static code analysis với Tree-sitter queries")
    print()
    
    try:
        demo_basic_static_analysis()
        print("\n" + "="*60)
        
        demo_good_code_analysis()
        print("\n" + "="*60)
        
        demo_tree_sitter_queries()
        print("\n" + "="*60)
        
        demo_file_analysis()
        print("\n" + "="*60)
        
        demo_repository_analysis()
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 StaticAnalysisAgent demo completed!")
    print("\n📚 Key Features Demonstrated:")
    print("  🔍 Tree-sitter queries for pattern detection")
    print("  📝 Missing docstring detection")
    print("  🚫 Unused import detection")
    print("  🔄 Complex function analysis")
    print("  📊 Code quality metrics")
    print("  💡 Automated suggestions")
    print("  🌳 Integration with ASTParsingAgent")


if __name__ == "__main__":
    main() 