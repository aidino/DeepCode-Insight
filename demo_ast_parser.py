#!/usr/bin/env python3
"""
Demo script cho ASTParsingAgent
Showcase các tính năng chính của parser
"""

import logging
from parsers.ast_parser import ASTParsingAgent, analyze_repository_code
from agents.code_fetcher import CodeFetcherAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def demo_basic_parsing():
    """Demo basic parsing functionality"""
    print("🔍 === Demo Basic Parsing ===")
    
    sample_code = '''
import os
import sys
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from functools import wraps

@dataclass
class User:
    """Represents a user in the system"""
    name: str
    email: str
    age: int = 0
    
    def __post_init__(self):
        """Validate user data after initialization"""
        if not self.email or '@' not in self.email:
            raise ValueError("Invalid email address")
    
    @property
    def is_adult(self) -> bool:
        """Check if user is an adult"""
        return self.age >= 18
    
    def greet(self, formal: bool = False) -> str:
        """Generate greeting message"""
        if formal:
            return f"Hello, {self.name}"
        return f"Hi {self.name}!"

@wraps(func)
def log_calls(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log_calls
def process_users(users: List[User]) -> Dict[str, int]:
    """Process a list of users and return statistics"""
    stats = {"total": len(users), "adults": 0, "minors": 0}
    
    for user in users:
        if user.is_adult:
            stats["adults"] += 1
        else:
            stats["minors"] += 1
    
    return stats

def main():
    """Main function to demonstrate usage"""
    users = [
        User("Alice", "alice@example.com", 25),
        User("Bob", "bob@example.com", 17),
        User("Charlie", "charlie@example.com", 30)
    ]
    
    stats = process_users(users)
    print(f"User statistics: {stats}")

# Global configuration
DEBUG_MODE = True
MAX_USERS = 1000
CONFIG = {
    "database_url": "sqlite:///users.db",
    "log_level": "INFO"
}
'''
    
    parser = ASTParsingAgent()
    result = parser.parse_code(sample_code, "user_management.py")
    
    print(f"📊 **Parsing Results for {result['filename']}**")
    print(f"   📏 Total lines: {result['stats']['total_lines']}")
    print(f"   🔧 Functions: {result['stats']['total_functions']}")
    print(f"   📦 Classes: {result['stats']['total_classes']}")
    print(f"   📥 Imports: {result['stats']['total_imports']}")
    print(f"   🔧 Variables: {result['stats']['total_variables']}")
    print()
    
    # Show detailed function analysis
    print("🔧 **Functions Analysis:**")
    for func in result['functions']:
        class_info = f" (in {func['class_name']})" if func['class_name'] else ""
        print(f"   • {func['name']}{class_info}")
        print(f"     📍 Lines: {func['start_line']}-{func['end_line']}")
        print(f"     📝 Parameters: {len(func['parameters'])}")
        if func['parameters']:
            param_names = [p['name'] for p in func['parameters']]
            print(f"       └─ {', '.join(param_names)}")
        if func['return_type']:
            print(f"     ↩️  Returns: {func['return_type']}")
        if func['docstring']:
            print(f"     📖 Doc: {func['docstring'][:60]}...")
        if func['decorators']:
            print(f"     🎨 Decorators: {', '.join(func['decorators'])}")
        print(f"     🏷️  Type: {'Method' if func['is_method'] else 'Function'}")
        if func['is_private']:
            print(f"     🔒 Private: Yes")
        if func['is_dunder']:
            print(f"     ⚡ Dunder: Yes")
        print()
    
    # Show class analysis
    print("📦 **Classes Analysis:**")
    for cls in result['classes']:
        print(f"   • {cls['name']}")
        print(f"     📍 Lines: {cls['start_line']}-{cls['end_line']}")
        print(f"     🔧 Methods: {cls['method_count']}")
        if cls['base_classes']:
            print(f"     👨‍👩‍👧‍👦 Inherits: {', '.join(cls['base_classes'])}")
        if cls['docstring']:
            print(f"     📖 Doc: {cls['docstring'][:60]}...")
        if cls['decorators']:
            print(f"     🎨 Decorators: {', '.join(cls['decorators'])}")
        if cls['is_private']:
            print(f"     🔒 Private: Yes")
        print()
    
    # Show imports
    print("📥 **Imports Analysis:**")
    for imp in result['imports']:
        if imp['type'] == 'import':
            print(f"   • import {', '.join(imp['modules'])}")
        else:
            print(f"   • from {imp['module']} import {', '.join(imp['names'])}")
    print()
    
    # Show variables
    print("🔧 **Global Variables:**")
    for var in result['variables']:
        for var_name in var['variables']:
            print(f"   • {var_name} = {var['value']}")
    print()

def demo_repository_analysis():
    """Demo repository analysis với CodeFetcherAgent"""
    print("🌐 === Demo Repository Analysis ===")
    
    # Test với một repository nhỏ
    test_repo = "https://github.com/octocat/Hello-World"
    
    try:
        code_fetcher = CodeFetcherAgent()
        
        print(f"🔍 Analyzing repository: {test_repo}")
        
        # Perform analysis
        analysis_result = analyze_repository_code(code_fetcher, test_repo)
        
        print(f"📊 **Analysis Summary:**")
        summary = analysis_result['summary']
        print(f"   📁 Files analyzed: {summary['total_files']}")
        print(f"   📏 Total lines: {summary['total_lines']}")
        print(f"   🔧 Total functions: {summary['total_functions']}")
        print(f"   📦 Total classes: {summary['total_classes']}")
        print(f"   📥 Total imports: {summary['total_imports']}")
        print(f"   🔧 Total variables: {summary['total_variables']}")
        print(f"   ⏰ Analysis time: {analysis_result['analysis_timestamp']}")
        print()
        
        if analysis_result['files_analyzed']:
            print("📁 **Files Analyzed:**")
            for file_analysis in analysis_result['files_analyzed'][:3]:  # Show first 3 files
                file_path = file_analysis['file_path']
                parse_result = file_analysis['parse_result']
                print(f"   • {file_path}")
                print(f"     📏 Lines: {parse_result['stats']['total_lines']}")
                print(f"     🔧 Functions: {parse_result['stats']['total_functions']}")
                print(f"     📦 Classes: {parse_result['stats']['total_classes']}")
                
                if parse_result['functions']:
                    func_names = [f['name'] for f in parse_result['functions'][:3]]
                    print(f"     └─ Functions: {', '.join(func_names)}")
                
                if parse_result['classes']:
                    class_names = [c['name'] for c in parse_result['classes'][:3]]
                    print(f"     └─ Classes: {', '.join(class_names)}")
                print()
        
        if analysis_result['errors']:
            print("⚠️  **Errors encountered:**")
            for error in analysis_result['errors'][:3]:  # Show first 3 errors
                print(f"   • {error}")
            print()
        
        print("✅ Repository analysis completed!")
        
    except Exception as e:
        print(f"❌ Repository analysis failed: {e}")
    
    finally:
        if 'code_fetcher' in locals():
            code_fetcher.cleanup()

def demo_edge_cases():
    """Demo edge cases và error handling"""
    print("🧪 === Demo Edge Cases ===")
    
    parser = ASTParsingAgent()
    
    # Test với Unicode
    unicode_code = '''
def chào_mừng(tên: str) -> str:
    """Chào mừng người dùng"""
    return f"Xin chào, {tên}!"

class Người:
    """Đại diện cho một người"""
    def __init__(self, tên: str, tuổi: int):
        self.tên = tên
        self.tuổi = tuổi
'''
    
    print("🌍 Testing Unicode support...")
    result = parser.parse_code(unicode_code, "unicode_test.py")
    print(f"   ✅ Functions found: {result['stats']['total_functions']}")
    print(f"   ✅ Classes found: {result['stats']['total_classes']}")
    
    # Test với syntax errors
    broken_code = '''
def broken_function(
    # Missing closing parenthesis
    return "This will cause a syntax error"
'''
    
    print("🔧 Testing syntax error handling...")
    result = parser.parse_code(broken_code, "broken.py")
    print(f"   ⚠️  Errors detected: {len(result['errors'])}")
    if result['errors']:
        print(f"   └─ {result['errors'][0]}")
    
    # Test với empty code
    print("📄 Testing empty code...")
    result = parser.parse_code("", "empty.py")
    print(f"   ✅ Handled gracefully: {result['stats']['total_lines']} lines")
    
    # Test với None input
    print("🚫 Testing None input...")
    result = parser.parse_code(None, "none.py")
    print(f"   ✅ Handled gracefully: {result['stats']['total_lines']} lines")
    
    print("✅ Edge cases testing completed!")

def main():
    """Main demo function"""
    print("🚀 === ASTParsingAgent Demo ===")
    print("Demonstrating Python code parsing capabilities using tree-sitter")
    print()
    
    try:
        demo_basic_parsing()
        print("\n" + "="*60 + "\n")
        
        demo_edge_cases()
        print("\n" + "="*60 + "\n")
        
        demo_repository_analysis()
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Demo completed! ASTParsingAgent is ready for use.")

if __name__ == "__main__":
    main() 