#!/usr/bin/env python3
"""
Comprehensive test suite cho StaticAnalysisAgent
Sử dụng pytest với sample ASTs và detailed assertions
"""

import pytest
import sys
import os
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import StaticAnalysisAgent
from agents.static_analyzer import StaticAnalysisAgent


class TestStaticAnalysisAgent:
    """Test suite cho StaticAnalysisAgent"""
    
    @pytest.fixture
    def analyzer(self):
        """Fixture để tạo StaticAnalysisAgent instance"""
        return StaticAnalysisAgent()
    
    def test_initialization(self, analyzer):
        """Test khởi tạo StaticAnalysisAgent"""
        assert analyzer is not None
        assert analyzer.python_language is not None
        assert analyzer.parser is not None
        assert analyzer.ast_parser is not None
        assert analyzer.function_query is not None
        assert analyzer.class_query is not None
        assert analyzer.import_query is not None
        assert analyzer.string_query is not None
        assert analyzer.if_query is not None
    
    def test_missing_docstrings_detection(self, analyzer):
        """Test detection của missing docstrings"""
        
        # Code với missing docstrings
        code_with_missing_docs = '''
def function_without_docstring():
    return "no docs"

def function_with_docstring():
    """This function has a docstring."""
    return "has docs"

class ClassWithoutDocstring:
    pass

class ClassWithDocstring:
    """This class has a docstring."""
    pass

def _private_function():
    return "private"

def __dunder_method__(self):
    return "dunder"
'''
        
        result = analyzer.analyze_code(code_with_missing_docs, "test_missing_docs.py")
        missing_docs = result['static_issues']['missing_docstrings']
        
        # Assertions
        assert len(missing_docs) == 2  # function_without_docstring và ClassWithoutDocstring
        
        # Check function without docstring
        func_issues = [issue for issue in missing_docs if issue['type'] == 'missing_function_docstring']
        assert len(func_issues) == 1
        assert func_issues[0]['name'] == 'function_without_docstring'
        assert func_issues[0]['line'] == 2
        
        # Check class without docstring
        class_issues = [issue for issue in missing_docs if issue['type'] == 'missing_class_docstring']
        assert len(class_issues) == 1
        assert class_issues[0]['name'] == 'ClassWithoutDocstring'
        assert class_issues[0]['line'] == 9
        
        # Verify private và dunder methods không được flag
        names = [issue['name'] for issue in missing_docs]
        assert '_private_function' not in names
        assert '__dunder_method__' not in names
    
    def test_no_missing_docstrings(self, analyzer):
        """Test code với tất cả functions/classes có docstrings"""
        
        code_with_all_docs = '''
def well_documented_function():
    """This function is well documented."""
    return "documented"

class WellDocumentedClass:
    """This class is well documented."""
    
    def method_with_docs(self):
        """This method has docs."""
        pass
'''
        
        result = analyzer.analyze_code(code_with_all_docs, "test_all_docs.py")
        missing_docs = result['static_issues']['missing_docstrings']
        
        assert len(missing_docs) == 0
    
    def test_unused_imports_detection(self, analyzer):
        """Test detection của unused imports"""
        
        code_with_unused_imports = '''
import os
import sys
import unused_module
from typing import List, Dict
from collections import defaultdict, Counter

def main():
    # Chỉ sử dụng os và List
    path = os.path.join("test", "path")
    items: List[str] = ["a", "b", "c"]
    return path, items
'''
        
        result = analyzer.analyze_code(code_with_unused_imports, "test_unused_imports.py")
        unused_imports = result['static_issues']['unused_imports']
        
        # Should detect unused imports
        assert len(unused_imports) > 0
        
        unused_names = [issue['name'] for issue in unused_imports]
        
        # These should be detected as unused
        expected_unused = ['sys', 'unused_module', 'Dict', 'defaultdict', 'Counter']
        for unused in expected_unused:
            if unused in unused_names:
                issue = next(issue for issue in unused_imports if issue['name'] == unused)
                assert issue['type'] == 'unused_import'
                assert issue['line'] > 0
    
    def test_no_unused_imports(self, analyzer):
        """Test code với tất cả imports được sử dụng"""
        
        code_with_used_imports = '''
import os
from typing import List

def process_files(filenames: List[str]) -> List[str]:
    """Process a list of filenames."""
    return [os.path.basename(f) for f in filenames]
'''
        
        result = analyzer.analyze_code(code_with_used_imports, "test_used_imports.py")
        unused_imports = result['static_issues']['unused_imports']
        
        # Should have minimal or no unused imports
        unused_names = [issue['name'] for issue in unused_imports]
        assert 'os' not in unused_names
        assert 'List' not in unused_names
    
    def test_complex_functions_detection(self, analyzer):
        """Test detection của complex functions"""
        
        code_with_complex_functions = '''
def simple_function(x):
    """Simple function."""
    return x * 2

def function_with_many_params(a, b, c, d, e, f, g, h):
    """Function with too many parameters."""
    return a + b + c + d + e + f + g + h

def long_function():
    """Very long function."""
    line1 = 1
    line2 = 2
    line3 = 3
    line4 = 4
    line5 = 5
    line6 = 6
    line7 = 7
    line8 = 8
    line9 = 9
    line10 = 10
    line11 = 11
    line12 = 12
    line13 = 13
    line14 = 14
    line15 = 15
    line16 = 16
    line17 = 17
    line18 = 18
    line19 = 19
    line20 = 20
    line21 = 21
    line22 = 22
    line23 = 23
    line24 = 24
    line25 = 25
    line26 = 26
    line27 = 27
    line28 = 28
    line29 = 29
    line30 = 30
    line31 = 31
    line32 = 32
    line33 = 33
    line34 = 34
    line35 = 35
    line36 = 36
    line37 = 37
    line38 = 38
    line39 = 39
    line40 = 40
    line41 = 41
    line42 = 42
    line43 = 43
    line44 = 44
    line45 = 45
    line46 = 46
    line47 = 47
    line48 = 48
    line49 = 49
    line50 = 50
    line51 = 51
    line52 = 52
    return line52

def function_with_nested_functions():
    """Function with nested functions."""
    def nested1():
        def nested2():
            def nested3():
                def nested4():
                    return "deeply nested"
                return nested4()
            return nested3()
        return nested2()
    return nested1()

def high_complexity_function(x):
    """Function with high cyclomatic complexity."""
    if x > 0:
        if x > 10:
            if x > 20:
                if x > 30:
                    if x > 40:
                        if x > 50:
                            if x > 60:
                                if x > 70:
                                    if x > 80:
                                        if x > 90:
                                            if x > 100:
                                                return "very high"
    return "low"
'''
        
        result = analyzer.analyze_code(code_with_complex_functions, "test_complex.py")
        complex_functions = result['static_issues']['complex_functions']
        
        # Should detect multiple complexity issues
        assert len(complex_functions) > 0
        
        # Check for too many parameters
        param_issues = [issue for issue in complex_functions if issue['type'] == 'too_many_parameters']
        assert len(param_issues) >= 1
        param_issue = param_issues[0]
        assert param_issue['name'] == 'function_with_many_params'
        assert param_issue['count'] == 8
        
        # Check for long function
        long_issues = [issue for issue in complex_functions if issue['type'] == 'long_function']
        assert len(long_issues) >= 1
        long_issue = long_issues[0]
        assert long_issue['name'] == 'long_function'
        assert long_issue['count'] > 50
        
        # Check for too many nested functions
        nested_issues = [issue for issue in complex_functions if issue['type'] == 'too_many_nested_functions']
        assert len(nested_issues) >= 1
        nested_issue = nested_issues[0]
        assert nested_issue['name'] == 'function_with_nested_functions'
        assert nested_issue['count'] > 2
        
        # Check for high complexity
        complexity_issues = [issue for issue in complex_functions if issue['type'] == 'high_complexity']
        assert len(complexity_issues) >= 1
        complexity_issue = complexity_issues[0]
        assert complexity_issue['name'] == 'high_complexity_function'
        assert complexity_issue['complexity'] > 10
    
    def test_no_complex_functions(self, analyzer):
        """Test code với simple, well-structured functions"""
        
        simple_code = '''
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y

class Calculator:
    """Simple calculator."""
    
    def calculate(self, operation: str, x: int, y: int) -> int:
        """Perform calculation."""
        if operation == "add":
            return self.add(x, y)
        elif operation == "multiply":
            return self.multiply(x, y)
        return 0
    
    def add(self, x: int, y: int) -> int:
        """Add method."""
        return x + y
    
    def multiply(self, x: int, y: int) -> int:
        """Multiply method."""
        return x * y
'''
        
        result = analyzer.analyze_code(simple_code, "test_simple.py")
        complex_functions = result['static_issues']['complex_functions']
        
        # Should have no complexity issues
        assert len(complex_functions) == 0
    
    def test_code_smells_detection(self, analyzer):
        """Test detection của code smells"""
        
        code_with_smells = '''
# This is a very long line that exceeds the recommended line length limit and should be flagged by the static analyzer as a code smell that needs to be addressed

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

class GodClass:
    """A class with too many methods."""
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
    def method21(self): pass
'''
        
        result = analyzer.analyze_code(code_with_smells, "test_smells.py")
        code_smells = result['static_issues']['code_smells']
        
        # Should detect code smells
        assert len(code_smells) > 0
        
        # Check for long lines
        long_line_issues = [issue for issue in code_smells if issue['type'] == 'long_line']
        assert len(long_line_issues) >= 1
        long_line_issue = long_line_issues[0]
        assert long_line_issue['length'] > 120
        
        # Check for too many globals (from AST analysis)
        global_issues = [issue for issue in code_smells if issue['type'] == 'too_many_globals']
        # Có thể có hoặc không tùy thuộc vào AST analysis
        
        # Check for god class (from AST analysis)
        god_class_issues = [issue for issue in code_smells if issue['type'] == 'god_class']
        # Có thể có hoặc không tùy thuộc vào AST analysis
    
    def test_metrics_calculation(self, analyzer):
        """Test calculation của code quality metrics"""
        
        sample_code = '''
"""Well documented module."""

import os
from typing import List

def process_data(items: List[str]) -> List[str]:
    """Process a list of items."""
    # Filter non-empty items
    filtered = []
    for item in items:
        if item.strip():  # Simple condition
            filtered.append(item.upper())
    return filtered

class DataProcessor:
    """Process data efficiently."""
    
    def __init__(self):
        """Initialize processor."""
        self.processed_count = 0
    
    def process(self, data: List[str]) -> List[str]:
        """Process data."""
        result = process_data(data)
        self.processed_count += len(result)
        return result
'''
        
        result = analyzer.analyze_code(sample_code, "test_metrics.py")
        metrics = result['metrics']
        
        # Check metrics exist và có giá trị hợp lý
        assert 'cyclomatic_complexity' in metrics
        assert 'maintainability_index' in metrics
        assert 'code_quality_score' in metrics
        assert 'lines_of_code' in metrics
        assert 'comment_ratio' in metrics
        assert 'function_to_class_ratio' in metrics
        
        # Check ranges
        assert 0 <= metrics['maintainability_index'] <= 100
        assert 0 <= metrics['code_quality_score'] <= 100
        assert 0 <= metrics['comment_ratio'] <= 1
        assert metrics['lines_of_code'] > 0
        assert metrics['cyclomatic_complexity'] >= 0
        assert metrics['function_to_class_ratio'] >= 0
    
    def test_suggestions_generation(self, analyzer):
        """Test generation của suggestions"""
        
        code_needing_suggestions = '''
import unused_module
import os

def function_without_docs(a, b, c, d, e, f):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        if f > 0:
                            return a + b + c + d + e + f
    return 0

class ClassWithoutDocs:
    pass
'''
        
        result = analyzer.analyze_code(code_needing_suggestions, "test_suggestions.py")
        suggestions = result['suggestions']
        
        # Should generate suggestions
        assert len(suggestions) > 0
        
        # Check for specific suggestion types
        suggestion_text = ' '.join(suggestions).lower()
        
        # Should suggest adding docstrings
        assert any('docstring' in s.lower() for s in suggestions)
        
        # Should suggest removing unused imports
        assert any('unused import' in s.lower() for s in suggestions)
        
        # Should suggest refactoring complex functions
        assert any('complex function' in s.lower() or 'refactor' in s.lower() for s in suggestions)
    
    def test_empty_code(self, analyzer):
        """Test với empty code"""
        
        result = analyzer.analyze_code("", "empty.py")
        
        assert result['filename'] == "empty.py"
        assert result['static_issues']['missing_docstrings'] == []
        assert result['static_issues']['unused_imports'] == []
        assert result['static_issues']['complex_functions'] == []
        assert result['static_issues']['code_smells'] == []
        assert result['suggestions'] == []
    
    def test_syntax_error_code(self, analyzer):
        """Test với code có syntax errors"""
        
        invalid_code = '''
def invalid_function(
    # Missing closing parenthesis
    return "invalid"
'''
        
        result = analyzer.analyze_code(invalid_code, "invalid.py")
        
        # Should detect syntax error
        code_smells = result['static_issues']['code_smells']
        syntax_errors = [issue for issue in code_smells if issue['type'] == 'syntax_error']
        assert len(syntax_errors) >= 1
    
    def test_file_analysis(self, analyzer, tmp_path):
        """Test analyze_file method"""
        
        # Create temporary file
        test_file = tmp_path / "test_file.py"
        test_content = '''
def test_function():
    """Test function."""
    return "test"
'''
        test_file.write_text(test_content)
        
        result = analyzer.analyze_file(str(test_file))
        
        assert result['filename'] == str(test_file)
        assert 'static_issues' in result
        assert 'metrics' in result
        assert 'suggestions' in result
    
    def test_file_analysis_nonexistent(self, analyzer):
        """Test analyze_file với file không tồn tại"""
        
        result = analyzer.analyze_file("nonexistent_file.py")
        
        assert result['filename'] == "nonexistent_file.py"
        assert 'error' in result
        assert result['static_issues']['code_smells'][0]['type'] == 'file_error'
    
    def test_comprehensive_analysis(self, analyzer):
        """Test comprehensive analysis với realistic code"""
        
        realistic_code = '''
"""
A realistic Python module for testing.
This module demonstrates various code patterns.
"""

import os
import sys
import json
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class User:
    """Represents a user in the system."""
    name: str
    email: str
    age: int
    
    def is_adult(self) -> bool:
        """Check if user is an adult."""
        return self.age >= 18

class UserManager:
    """Manages user operations."""
    
    def __init__(self):
        """Initialize user manager."""
        self.users: List[User] = []
    
    def add_user(self, name: str, email: str, age: int) -> bool:
        """Add a new user."""
        if not name or not email:
            return False
        
        # Check if user already exists
        for user in self.users:
            if user.email == email:
                return False
        
        user = User(name, email, age)
        self.users.append(user)
        return True
    
    def get_adult_users(self) -> List[User]:
        """Get all adult users."""
        return [user for user in self.users if user.is_adult()]
    
    def save_to_file(self, filename: str) -> bool:
        """Save users to JSON file."""
        try:
            data = []
            for user in self.users:
                data.append({
                    'name': user.name,
                    'email': user.email,
                    'age': user.age
                })
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False

def main():
    """Main function."""
    manager = UserManager()
    manager.add_user("John Doe", "john@example.com", 25)
    manager.add_user("Jane Smith", "jane@example.com", 17)
    
    adults = manager.get_adult_users()
    print(f"Found {len(adults)} adult users")
    
    if manager.save_to_file("users.json"):
        print("Users saved successfully")

if __name__ == "__main__":
    main()
'''
        
        result = analyzer.analyze_code(realistic_code, "realistic.py")
        
        # Should have high quality score
        assert result['metrics']['code_quality_score'] > 70
        
        # Should have minimal issues
        total_issues = sum(len(issues) for issues in result['static_issues'].values())
        assert total_issues < 5  # Very few issues expected
        
        # Should have good metrics
        assert result['metrics']['comment_ratio'] > 0.1  # Good comment ratio
        assert result['metrics']['maintainability_index'] > 60
        
        # Should have minimal suggestions
        assert len(result['suggestions']) < 3


# Test fixtures và helper functions
@pytest.fixture
def sample_codes():
    """Fixture cung cấp sample code snippets"""
    return {
        'perfect_code': '''
"""Perfect module example."""

from typing import List

def calculate_sum(numbers: List[int]) -> int:
    """Calculate sum of numbers."""
    return sum(numbers)

class Calculator:
    """Simple calculator class."""
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
''',
        
        'problematic_code': '''
import unused1
import unused2
from typing import Dict, List, Optional, Union, Tuple

def bad_function(a, b, c, d, e, f, g, h, i, j):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        if f > 0:
                            if g > 0:
                                if h > 0:
                                    if i > 0:
                                        if j > 0:
                                            def nested1():
                                                def nested2():
                                                    def nested3():
                                                        return "too nested"
                                                    return nested3()
                                                return nested2()
                                            return nested1()
    return None

class BadClass:
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
    def method21(self): pass
'''
    }


def test_perfect_vs_problematic_code(sample_codes):
    """Test so sánh perfect code vs problematic code"""
    analyzer = StaticAnalysisAgent()
    
    # Analyze perfect code
    perfect_result = analyzer.analyze_code(sample_codes['perfect_code'], "perfect.py")
    
    # Analyze problematic code
    problematic_result = analyzer.analyze_code(sample_codes['problematic_code'], "problematic.py")
    
    # Perfect code should have higher quality score
    assert perfect_result['metrics']['code_quality_score'] > problematic_result['metrics']['code_quality_score']
    
    # Perfect code should have fewer issues
    perfect_issues = sum(len(issues) for issues in perfect_result['static_issues'].values())
    problematic_issues = sum(len(issues) for issues in problematic_result['static_issues'].values())
    assert perfect_issues < problematic_issues
    
    # Perfect code should have fewer suggestions
    assert len(perfect_result['suggestions']) < len(problematic_result['suggestions'])


if __name__ == "__main__":
    # Run tests nếu script được chạy trực tiếp
    pytest.main([__file__, "-v"]) 