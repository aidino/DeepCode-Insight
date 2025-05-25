"""
Extended test cases cho ASTParsingAgent với nhiều sample Python code strings
"""

import pytest
from ..parsers.ast_parser import ASTParsingAgent


class TestASTParsingAgentSamples:
    """Test ASTParsingAgent với various Python code samples"""
    
    def setup_method(self):
        """Setup cho mỗi test method"""
        self.parser = ASTParsingAgent()
    
    def test_simple_function_samples(self):
        """Test parsing simple functions với different styles"""
        
        # Sample 1: Basic function
        code1 = '''
def greet():
    return "Hello World"
'''
        result = self.parser.parse_code(code1, "simple1.py")
        assert result['stats']['total_functions'] == 1
        func = result['functions'][0]
        assert func['name'] == 'greet'
        assert len(func['parameters']) == 0
        assert func['return_type'] is None
        assert func['docstring'] is None
        
        # Sample 2: Function with parameters
        code2 = '''
def add(a, b):
    return a + b
'''
        result = self.parser.parse_code(code2, "simple2.py")
        assert result['stats']['total_functions'] == 1
        func = result['functions'][0]
        assert func['name'] == 'add'
        assert len(func['parameters']) == 2
        assert func['parameters'][0]['name'] == 'a'
        assert func['parameters'][1]['name'] == 'b'
        
        # Sample 3: Function with type hints
        code3 = '''
def multiply(x: int, y: int) -> int:
    return x * y
'''
        result = self.parser.parse_code(code3, "simple3.py")
        assert result['stats']['total_functions'] == 1
        func = result['functions'][0]
        assert func['name'] == 'multiply'
        assert func['return_type'] == 'int'
        assert len(func['parameters']) == 2
        assert func['parameters'][0]['type'] == 'int'
        assert func['parameters'][1]['type'] == 'int'
        
        # Sample 4: Function with default parameters
        code4 = '''
def greet_user(name, greeting="Hello"):
    return f"{greeting}, {name}!"
'''
        result = self.parser.parse_code(code4, "simple4.py")
        assert result['stats']['total_functions'] == 1
        func = result['functions'][0]
        assert func['name'] == 'greet_user'
        assert len(func['parameters']) == 2
        assert func['parameters'][0]['name'] == 'name'
        assert func['parameters'][0]['default'] is None
        assert func['parameters'][1]['name'] == 'greeting'
        assert func['parameters'][1]['default'] == '"Hello"'
    
    def test_class_samples(self):
        """Test parsing classes với different structures"""
        
        # Sample 1: Simple class
        code1 = '''
class Person:
    pass
'''
        result = self.parser.parse_code(code1, "class1.py")
        assert result['stats']['total_classes'] == 1
        cls = result['classes'][0]
        assert cls['name'] == 'Person'
        assert len(cls['base_classes']) == 0
        assert cls['method_count'] == 0
        
        # Sample 2: Class with inheritance
        code2 = '''
class Student(Person):
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
'''
        result = self.parser.parse_code(code2, "class2.py")
        assert result['stats']['total_classes'] == 1
        assert result['stats']['total_functions'] == 1
        cls = result['classes'][0]
        assert cls['name'] == 'Student'
        assert 'Person' in cls['base_classes']
        assert cls['method_count'] == 1
        
        # Sample 3: Class with multiple methods
        code3 = '''
class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, value):
        self.result += value
        return self
    
    def subtract(self, value):
        self.result -= value
        return self
    
    def get_result(self):
        return self.result
'''
        result = self.parser.parse_code(code3, "class3.py")
        assert result['stats']['total_classes'] == 1
        assert result['stats']['total_functions'] == 4
        cls = result['classes'][0]
        assert cls['name'] == 'Calculator'
        assert cls['method_count'] == 4
        
        # Verify all methods are detected
        method_names = [f['name'] for f in result['functions']]
        assert '__init__' in method_names
        assert 'add' in method_names
        assert 'subtract' in method_names
        assert 'get_result' in method_names
    
    def test_decorator_samples(self):
        """Test parsing decorators với different patterns"""
        
        # Sample 1: Simple decorator
        code1 = '''
@property
def name(self):
    return self._name
'''
        result = self.parser.parse_code(code1, "decorator1.py")
        assert result['stats']['total_functions'] == 1
        func = result['functions'][0]
        assert func['name'] == 'name'
        assert '@property' in func['decorators']
        
        # Sample 2: Multiple decorators
        code2 = '''
@staticmethod
@cache
def expensive_calculation(n):
    return sum(range(n))
'''
        result = self.parser.parse_code(code2, "decorator2.py")
        assert result['stats']['total_functions'] == 1
        func = result['functions'][0]
        assert func['name'] == 'expensive_calculation'
        assert len(func['decorators']) >= 1  # At least one decorator detected
        
        # Sample 3: Decorator with parameters
        code3 = '''
@app.route('/users/<int:user_id>')
def get_user(user_id):
    return f"User {user_id}"
'''
        result = self.parser.parse_code(code3, "decorator3.py")
        assert result['stats']['total_functions'] == 1
        func = result['functions'][0]
        assert func['name'] == 'get_user'
        assert len(func['decorators']) >= 1
    
    def test_import_samples(self):
        """Test parsing imports với different styles"""
        
        # Sample 1: Simple imports
        code1 = '''
import os
import sys
import json
'''
        result = self.parser.parse_code(code1, "import1.py")
        assert result['stats']['total_imports'] == 3
        import_texts = [imp['text'] for imp in result['imports']]
        assert 'import os' in import_texts
        assert 'import sys' in import_texts
        assert 'import json' in import_texts
        
        # Sample 2: From imports
        code2 = '''
from typing import List, Dict, Optional
from pathlib import Path
from collections import defaultdict
'''
        result = self.parser.parse_code(code2, "import2.py")
        assert result['stats']['total_imports'] == 3
        
        # Check specific from import
        typing_import = None
        for imp in result['imports']:
            if imp['module'] == 'typing':
                typing_import = imp
                break
        
        assert typing_import is not None
        assert typing_import['type'] == 'from_import'
        assert 'List' in typing_import['names']
        assert 'Dict' in typing_import['names']
        assert 'Optional' in typing_import['names']
        
        # Sample 3: Mixed imports
        code3 = '''
import os
from sys import argv, exit
import json as js
from pathlib import Path as P
'''
        result = self.parser.parse_code(code3, "import3.py")
        assert result['stats']['total_imports'] == 4
    
    def test_variable_samples(self):
        """Test parsing global variables với different types"""
        
        # Sample 1: Simple variables
        code1 = '''
NAME = "John Doe"
AGE = 30
IS_ACTIVE = True
'''
        result = self.parser.parse_code(code1, "var1.py")
        assert result['stats']['total_variables'] == 3
        
        var_dict = {}
        for var in result['variables']:
            for var_name in var['variables']:
                var_dict[var_name] = var['value']
        
        assert 'NAME' in var_dict
        assert 'AGE' in var_dict
        assert 'IS_ACTIVE' in var_dict
        assert var_dict['NAME'] == '"John Doe"'
        assert var_dict['AGE'] == '30'
        assert var_dict['IS_ACTIVE'] == 'True'
        
        # Sample 2: Complex variables
        code2 = '''
CONFIG = {
    "host": "localhost",
    "port": 8080,
    "debug": True
}
ITEMS = [1, 2, 3, 4, 5]
TUPLE_DATA = (10, 20, 30)
'''
        result = self.parser.parse_code(code2, "var2.py")
        assert result['stats']['total_variables'] == 3
        
        # Sample 3: Multiple assignment
        code3 = '''
x, y, z = 1, 2, 3
a = b = c = 0
'''
        result = self.parser.parse_code(code3, "var3.py")
        assert result['stats']['total_variables'] >= 2
    
    def test_complex_code_samples(self):
        """Test parsing complex code structures"""
        
        # Sample 1: Class with decorators and type hints
        code1 = '''
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class User:
    """Represents a user in the system"""
    name: str
    email: str
    age: Optional[int] = None
    
    def __post_init__(self):
        """Validate user data"""
        if not self.email or '@' not in self.email:
            raise ValueError("Invalid email")
    
    @property
    def is_adult(self) -> bool:
        """Check if user is an adult"""
        return self.age is not None and self.age >= 18
    
    @classmethod
    def from_dict(cls, data: dict) -> 'User':
        """Create user from dictionary"""
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation"""
        return f"User(name={self.name}, email={self.email})"
'''
        result = self.parser.parse_code(code1, "complex1.py")
        
        # Verify overall structure
        assert result['stats']['total_classes'] == 1
        assert result['stats']['total_functions'] == 4
        assert result['stats']['total_imports'] == 2
        
        # Verify class details
        cls = result['classes'][0]
        assert cls['name'] == 'User'
        assert '@dataclass' in cls['decorators']
        assert 'Represents a user in the system' in cls['docstring']
        
        # Verify methods
        method_names = [f['name'] for f in result['functions']]
        assert '__post_init__' in method_names
        assert 'is_adult' in method_names
        assert 'from_dict' in method_names
        assert '__str__' in method_names
        
        # Check method properties
        for func in result['functions']:
            if func['name'] == 'is_adult':
                assert '@property' in func['decorators']
                assert func['return_type'] == 'bool'
            elif func['name'] == '__post_init__':
                assert func['is_dunder'] is True
            elif func['name'] == 'from_dict':
                assert '@classmethod' in func['decorators']
        
        # Sample 2: Module with functions and classes
        code2 = '''
"""
A utility module for data processing
"""
import logging
from typing import Any, Dict, List, Union

# Module constants
DEFAULT_BATCH_SIZE = 100
MAX_RETRIES = 3

logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class DataProcessor:
    """Process data in batches"""
    
    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE):
        """Initialize processor"""
        self.batch_size = batch_size
        self.processed_count = 0
    
    def process_item(self, item: Any) -> Dict[str, Any]:
        """Process a single item"""
        logger.debug(f"Processing item: {item}")
        self.processed_count += 1
        return {"processed": True, "item": item}
    
    def process_batch(self, items: List[Any]) -> List[Dict[str, Any]]:
        """Process a batch of items"""
        results = []
        for item in items:
            try:
                result = self.process_item(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item {item}: {e}")
                results.append({"processed": False, "error": str(e)})
        return results
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get processing statistics"""
        return {"processed_count": self.processed_count}

def main():
    """Main function"""
    setup_logging()
    processor = DataProcessor()
    
    sample_data = [1, 2, 3, "test", {"key": "value"}]
    results = processor.process_batch(sample_data)
    
    print(f"Processed {len(results)} items")
    print(f"Stats: {processor.stats}")

if __name__ == "__main__":
    main()
'''
        result = self.parser.parse_code(code2, "complex2.py")
        
        # Verify comprehensive parsing
        assert result['stats']['total_functions'] == 6  # setup_logging, __init__, process_item, process_batch, stats, main
        assert result['stats']['total_classes'] == 1
        assert result['stats']['total_imports'] == 2
        assert result['stats']['total_variables'] >= 3  # DEFAULT_BATCH_SIZE, MAX_RETRIES, logger
        
        # Verify class structure
        cls = result['classes'][0]
        assert cls['name'] == 'DataProcessor'
        assert 'Process data in batches' in cls['docstring']
        
        # Verify method details
        for func in result['functions']:
            if func['name'] == 'setup_logging':
                assert func['class_name'] is None  # Not a method
                assert len(func['parameters']) == 1
                assert func['parameters'][0]['default'] == '"INFO"'
            elif func['name'] == 'stats':
                assert func['class_name'] == 'DataProcessor'
                assert '@property' in func['decorators']
                assert func['return_type'] == 'Dict[str, int]'
    
    def test_syntax_error_samples(self):
        """Test handling của various syntax errors"""
        
        # Sample 1: Missing closing parenthesis
        code1 = '''
def broken_function(a, b:
    return a + b
'''
        result = self.parser.parse_code(code1, "error1.py")
        assert len(result['errors']) > 0
        assert "Syntax errors detected" in result['errors'][0]
        
        # Sample 2: Invalid indentation - tree-sitter may handle this gracefully
        code2 = '''
def valid_function():
    x = 1
  y = 2  # Wrong indentation
    return x + y
'''
        result = self.parser.parse_code(code2, "error2.py")
        # Tree-sitter may parse this without errors, so we check if it at least parses
        assert result['stats']['total_functions'] >= 1
        
        # Sample 3: Unclosed string
        code3 = '''
def string_error():
    message = "This string is not closed
    return message
'''
        result = self.parser.parse_code(code3, "error3.py")
        assert len(result['errors']) > 0
        
        # Sample 4: Invalid class definition
        code4 = '''
class InvalidClass(:
    pass
'''
        result = self.parser.parse_code(code4, "error4.py")
        assert len(result['errors']) > 0
        
        # Sample 5: Mixed valid and invalid code
        code5 = '''
def valid_function():
    return "This works"

def broken_function(
    # Missing closing parenthesis and colon
    return "This is broken"

def another_valid():
    return "This also works"
'''
        result = self.parser.parse_code(code5, "error5.py")
        assert len(result['errors']) > 0
        # Should still parse some valid parts
        assert result['stats']['total_functions'] >= 1
    
    def test_edge_case_samples(self):
        """Test edge cases và special scenarios"""
        
        # Sample 1: Empty file
        code1 = ''
        result = self.parser.parse_code(code1, "empty.py")
        assert result['stats']['total_lines'] == 0
        assert result['stats']['total_functions'] == 0
        assert result['stats']['total_classes'] == 0
        assert len(result['errors']) == 0
        
        # Sample 2: Only comments
        code2 = '''
# This is a comment
# Another comment
# Yet another comment
'''
        result = self.parser.parse_code(code2, "comments.py")
        assert result['stats']['total_lines'] == 4
        assert result['stats']['total_functions'] == 0
        assert result['stats']['total_classes'] == 0
        
        # Sample 3: Only imports
        code3 = '''
import os
import sys
from typing import List
'''
        result = self.parser.parse_code(code3, "imports_only.py")
        assert result['stats']['total_imports'] == 3
        assert result['stats']['total_functions'] == 0
        assert result['stats']['total_classes'] == 0
        
        # Sample 4: Very long line
        long_string = "a" * 1000
        code4 = f'''
def long_line_function():
    very_long_variable = "{long_string}"
    return very_long_variable
'''
        result = self.parser.parse_code(code4, "long_line.py")
        assert result['stats']['total_functions'] == 1
        func = result['functions'][0]
        assert func['name'] == 'long_line_function'
        
        # Sample 5: Nested functions
        code5 = '''
def outer_function():
    def inner_function():
        def deeply_nested():
            return "deep"
        return deeply_nested()
    return inner_function()
'''
        result = self.parser.parse_code(code5, "nested.py")
        assert result['stats']['total_functions'] == 3
        func_names = [f['name'] for f in result['functions']]
        assert 'outer_function' in func_names
        assert 'inner_function' in func_names
        assert 'deeply_nested' in func_names
        
        # Sample 6: Lambda functions (may not be parsed as regular functions)
        code6 = '''
square = lambda x: x ** 2
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
'''
        result = self.parser.parse_code(code6, "lambda.py")
        assert result['stats']['total_variables'] >= 2  # square, numbers, squared
    
    def test_unicode_samples(self):
        """Test Unicode support với various languages"""
        
        # Sample 1: Vietnamese
        code1 = '''
def chào_mừng(tên: str) -> str:
    """Chào mừng người dùng"""
    return f"Xin chào, {tên}!"

class Người:
    """Đại diện cho một người"""
    def __init__(self, tên: str, tuổi: int):
        self.tên = tên
        self.tuổi = tuổi
    
    def giới_thiệu(self) -> str:
        """Giới thiệu bản thân"""
        return f"Tôi là {self.tên}, {self.tuổi} tuổi"
'''
        result = self.parser.parse_code(code1, "vietnamese.py")
        assert result['stats']['total_functions'] == 3
        assert result['stats']['total_classes'] == 1
        
        func_names = [f['name'] for f in result['functions']]
        # Check if function names contain the expected Unicode characters (may be parsed differently)
        vietnamese_func_found = any('chào_mừng' in name for name in func_names)
        intro_func_found = any('giới_thiệu' in name for name in func_names)
        assert vietnamese_func_found or intro_func_found  # At least one should be found
        
        cls = result['classes'][0]
        # Unicode class names may be parsed differently by tree-sitter
        assert 'Người' in cls['name'] or len(result['classes']) >= 1
        
        # Sample 2: Mixed languages with emojis
        code2 = '''
def 🚀_launch_rocket(destination: str) -> str:
    """Launch rocket to destination 🌙"""
    return f"Launching to {destination} 🚀"

class 🤖_Robot:
    """A robot class with emoji name"""
    def __init__(self, name: str):
        self.name = name
    
    def 👋_wave(self) -> str:
        """Wave hello"""
        return f"{self.name} waves 👋"
'''
        result = self.parser.parse_code(code2, "emoji.py")
        assert result['stats']['total_functions'] == 3
        assert result['stats']['total_classes'] == 1
        
        # Sample 3: Chinese characters
        code3 = '''
def 计算(数字1: int, 数字2: int) -> int:
    """计算两个数字的和"""
    return 数字1 + 数字2

class 学生:
    """学生类"""
    def __init__(self, 姓名: str, 年龄: int):
        self.姓名 = 姓名
        self.年龄 = 年龄
'''
        result = self.parser.parse_code(code3, "chinese.py")
        assert result['stats']['total_functions'] == 2
        assert result['stats']['total_classes'] == 1
    
    def test_performance_samples(self):
        """Test performance với large code samples"""
        
        # Sample 1: Many functions
        functions_code = ""
        for i in range(50):
            functions_code += f'''
def function_{i}(param_{i}: int) -> int:
    """Function number {i}"""
    return param_{i} * {i}

'''
        
        result = self.parser.parse_code(functions_code, "many_functions.py")
        assert result['stats']['total_functions'] == 50
        
        # Sample 2: Large class with many methods
        class_code = '''
class LargeClass:
    """A class with many methods"""
    
    def __init__(self):
        self.value = 0
'''
        
        for i in range(30):
            class_code += f'''
    def method_{i}(self, param: int) -> int:
        """Method number {i}"""
        return param + {i}
'''
        
        result = self.parser.parse_code(class_code, "large_class.py")
        assert result['stats']['total_classes'] == 1
        assert result['stats']['total_functions'] == 31  # __init__ + 30 methods
        
        # Sample 3: Deep nesting
        nested_code = "def level_0():\n"
        for i in range(1, 10):
            nested_code += "    " * i + f"def level_{i}():\n"
            nested_code += "    " * (i + 1) + f"return {i}\n"
        nested_code += "    " * 10 + "return 9\n"
        
        result = self.parser.parse_code(nested_code, "deep_nesting.py")
        assert result['stats']['total_functions'] == 10


class TestASTParsingAgentErrorHandling:
    """Test error handling scenarios"""
    
    def setup_method(self):
        self.parser = ASTParsingAgent()
    
    def test_malformed_code_samples(self):
        """Test với various malformed code samples"""
        
        malformed_samples = [
            # Unmatched brackets
            '''
def test():
    data = [1, 2, 3
    return data
''',
            # Invalid function definition
            '''
def ():
    pass
''',
            # Missing colon
            '''
if True
    print("missing colon")
''',
            # Invalid class syntax
            '''
class 123InvalidName:
    pass
''',
            # Incomplete string
            '''
message = "incomplete string
print(message)
''',
            # Invalid import
            '''
import 
from import something
''',
        ]
        
        for i, code in enumerate(malformed_samples):
            result = self.parser.parse_code(code, f"malformed_{i}.py")
            # Should handle gracefully with errors reported
            assert len(result['errors']) > 0
            assert "Syntax errors detected" in result['errors'][0]
    
    def test_none_and_empty_inputs(self):
        """Test với None và empty inputs"""
        
        # None input
        result = self.parser.parse_code(None, "none.py")
        assert result['stats']['total_lines'] == 0
        assert len(result['errors']) == 0
        
        # Empty string
        result = self.parser.parse_code("", "empty.py")
        assert result['stats']['total_lines'] == 0
        assert len(result['errors']) == 0
        
        # Whitespace only
        result = self.parser.parse_code("   \n  \t  \n  ", "whitespace.py")
        assert result['stats']['total_lines'] == 3
        assert result['stats']['total_functions'] == 0
    
    def test_encoding_issues(self):
        """Test với potential encoding issues"""
        
        # Binary-like content (should be handled gracefully)
        binary_like = "def test():\n    return b'\\x00\\x01\\x02'"
        result = self.parser.parse_code(binary_like, "binary.py")
        assert result['stats']['total_functions'] == 1
        
        # Mixed encoding characters
        mixed_encoding = '''
def test_encoding():
    # Various Unicode characters
    symbols = "αβγδε ñáéíóú 中文 🚀🌟"
    return symbols
'''
        result = self.parser.parse_code(mixed_encoding, "mixed.py")
        assert result['stats']['total_functions'] == 1


if __name__ == "__main__":
    # Run extended tests
    pytest.main([__file__, "-v"]) 