# ASTParsingAgent Test Suite

Comprehensive test suite cho `ASTParsingAgent` với nhiều sample Python code strings để verify AST generation và error handling.

## Test Files Overview

### 1. `test_ast_parser.py` - Core Tests (19 tests)
**Mục đích**: Test các tính năng cơ bản của ASTParsingAgent

**Test Cases**:
- ✅ `test_initialization` - Kiểm tra khởi tạo parser
- ✅ `test_parse_simple_function` - Parse function đơn giản với type hints
- ✅ `test_parse_class_with_methods` - Parse class với nhiều methods
- ✅ `test_parse_imports` - Parse import statements (import và from import)
- ✅ `test_parse_global_variables` - Parse global variables
- ✅ `test_parse_decorators` - Parse decorators (@property, @dataclass, etc.)
- ✅ `test_parse_syntax_error` - Handle syntax errors gracefully
- ✅ `test_parse_empty_code` - Handle empty code
- ✅ `test_parse_complex_parameters` - Parse complex function parameters
- ✅ `test_parse_nested_classes` - Parse nested classes
- ✅ `test_error_handling` - Error handling với invalid inputs
- ✅ `test_analyze_repository_code_*` - Integration với CodeFetcherAgent
- ✅ `test_unicode_code` - Unicode support
- ✅ `test_very_long_lines` - Performance với long lines
- ✅ `test_mixed_indentation` - Handle mixed indentation
- ✅ `test_special_characters_in_strings` - Special characters trong strings

### 2. `test_ast_parser_extended.py` - Extended Tests (13 tests)
**Mục đích**: Test với nhiều sample Python code patterns khác nhau

#### TestASTParsingAgentSamples (10 tests)
- ✅ `test_simple_function_samples` - 4 function styles khác nhau
- ✅ `test_class_samples` - 3 class structures khác nhau
- ✅ `test_decorator_samples` - 3 decorator patterns
- ✅ `test_import_samples` - 3 import styles
- ✅ `test_variable_samples` - 3 variable types
- ✅ `test_complex_code_samples` - 2 complex code structures
- ✅ `test_syntax_error_samples` - 5 syntax error scenarios
- ✅ `test_edge_case_samples` - 6 edge cases
- ✅ `test_unicode_samples` - 3 Unicode language samples
- ✅ `test_performance_samples` - 3 performance test scenarios

#### TestASTParsingAgentErrorHandling (3 tests)
- ✅ `test_malformed_code_samples` - 6 malformed code patterns
- ✅ `test_none_and_empty_inputs` - None và empty input handling
- ✅ `test_encoding_issues` - Encoding và binary content handling

### 3. `test_ast_parser_real_world.py` - Real-World Tests (5 tests)
**Mục đích**: Test với actual Python code patterns từ real projects

- ✅ `test_flask_app_sample` - Flask web application code
- ✅ `test_django_model_sample` - Django models với inheritance
- ✅ `test_data_science_sample` - Data science/ML pipeline code
- ✅ `test_async_web_scraper_sample` - Async/await patterns
- ✅ `test_testing_framework_sample` - Pytest và unittest patterns

## Sample Code Categories

### 1. Valid Python Code Samples

#### Basic Functions
```python
# Simple function
def greet():
    return "Hello World"

# Function with parameters
def add(a, b):
    return a + b

# Function with type hints
def multiply(x: int, y: int) -> int:
    return x * y

# Function with defaults
def greet_user(name, greeting="Hello"):
    return f"{greeting}, {name}!"
```

#### Classes
```python
# Simple class
class Person:
    pass

# Class with inheritance
class Student(Person):
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

# Class with multiple methods
class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, value):
        self.result += value
        return self
```

#### Decorators
```python
# Property decorator
@property
def name(self):
    return self._name

# Multiple decorators
@staticmethod
@cache
def expensive_calculation(n):
    return sum(range(n))

# Decorator with parameters
@app.route('/users/<int:user_id>')
def get_user(user_id):
    return f"User {user_id}"
```

#### Imports
```python
# Simple imports
import os
import sys
import json

# From imports
from typing import List, Dict, Optional
from pathlib import Path
from collections import defaultdict

# Mixed imports
import os
from sys import argv, exit
import json as js
from pathlib import Path as P
```

#### Variables
```python
# Simple variables
NAME = "John Doe"
AGE = 30
IS_ACTIVE = True

# Complex variables
CONFIG = {
    "host": "localhost",
    "port": 8080,
    "debug": True
}
ITEMS = [1, 2, 3, 4, 5]
TUPLE_DATA = (10, 20, 30)

# Multiple assignment
x, y, z = 1, 2, 3
a = b = c = 0
```

#### Unicode Support
```python
# Vietnamese
def chào_mừng(tên: str) -> str:
    """Chào mừng người dùng"""
    return f"Xin chào, {tên}!"

class Người:
    """Đại diện cho một người"""
    def __init__(self, tên: str, tuổi: int):
        self.tên = tên
        self.tuổi = tuổi

# Emojis
def 🚀_launch_rocket(destination: str) -> str:
    """Launch rocket to destination 🌙"""
    return f"Launching to {destination} 🚀"

# Chinese
def 计算(数字1: int, 数字2: int) -> int:
    """计算两个数字的和"""
    return 数字1 + 数字2
```

### 2. Invalid Python Code Samples

#### Syntax Errors
```python
# Missing closing parenthesis
def broken_function(a, b:
    return a + b

# Invalid function definition
def ():
    pass

# Missing colon
if True
    print("missing colon")

# Invalid class syntax
class 123InvalidName:
    pass

# Incomplete string
message = "incomplete string
print(message)

# Invalid import
import 
from import something
```

#### Malformed Code
```python
# Unmatched brackets
def test():
    data = [1, 2, 3
    return data

# Invalid indentation (may be handled gracefully by tree-sitter)
def valid_function():
    x = 1
  y = 2  # Wrong indentation
    return x + y
```

### 3. Real-World Code Samples

#### Flask Application
```python
from flask import Flask, request, jsonify
from functools import wraps
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'No authorization header'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/api/users', methods=['GET', 'POST'])
@require_auth
def users():
    """Handle user operations"""
    if request.method == 'GET':
        return jsonify({'users': []})
    elif request.method == 'POST':
        data = request.get_json()
        return jsonify({'created': data}), 201
```

#### Django Models
```python
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone
from typing import Optional

class User(AbstractUser):
    """Custom user model"""
    
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']
    
    class Meta:
        db_table = 'users'
        verbose_name = 'User'
        verbose_name_plural = 'Users'
    
    def __str__(self) -> str:
        """String representation"""
        return self.email
    
    @property
    def is_profile_complete(self) -> bool:
        """Check if user profile is complete"""
        return bool(self.first_name and self.last_name and self.phone)
```

#### Async/Await Patterns
```python
import asyncio
import aiohttp
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class ScrapingResult:
    """Result of a scraping operation"""
    url: str
    status_code: int
    content: Optional[str] = None
    error: Optional[str] = None

class AsyncWebScraper:
    """Asynchronous web scraper"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_url(self, url: str) -> ScrapingResult:
        """Fetch a single URL"""
        async with self.semaphore:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        content = await response.text()
                        return ScrapingResult(
                            url=url,
                            status_code=response.status,
                            content=content
                        )
            except Exception as e:
                return ScrapingResult(
                    url=url,
                    status_code=0,
                    error=str(e)
                )
```

## Test Assertions

### AST Generation Tests
- ✅ **Function parsing**: name, parameters, return types, docstrings, decorators
- ✅ **Class parsing**: name, base classes, methods, docstrings, decorators
- ✅ **Import parsing**: import statements, from imports, module names
- ✅ **Variable parsing**: global variables, assignments, values
- ✅ **Decorator parsing**: function decorators, class decorators
- ✅ **Type hint parsing**: parameter types, return types
- ✅ **Docstring parsing**: function docstrings, class docstrings
- ✅ **Line number tracking**: start/end lines for functions và classes

### Error Handling Tests
- ✅ **Syntax errors**: Graceful handling với error reporting
- ✅ **Malformed code**: Robust parsing với partial success
- ✅ **Empty inputs**: Handle None, empty strings, whitespace
- ✅ **Unicode support**: Full Unicode character support
- ✅ **Encoding issues**: Handle binary content và mixed encoding
- ✅ **Large files**: Performance với large code samples
- ✅ **Deep nesting**: Handle deeply nested structures

### Integration Tests
- ✅ **Repository analysis**: Integration với CodeFetcherAgent
- ✅ **File processing**: Batch processing của multiple files
- ✅ **Error aggregation**: Collect và report errors across files
- ✅ **Statistics generation**: Comprehensive code statistics

## Running Tests

```bash
# Run all AST parser tests
poetry run pytest tests/test_ast_parser*.py -v

# Run specific test file
poetry run pytest tests/test_ast_parser.py -v
poetry run pytest tests/test_ast_parser_extended.py -v
poetry run pytest tests/test_ast_parser_real_world.py -v

# Run with coverage
poetry run pytest tests/test_ast_parser*.py --cov=parsers.ast_parser

# Run specific test
poetry run pytest tests/test_ast_parser.py::TestASTParsingAgent::test_parse_simple_function -v
```

## Test Statistics

- **Total Tests**: 37 tests
- **Core Tests**: 19 tests
- **Extended Tests**: 13 tests  
- **Real-World Tests**: 5 tests
- **Sample Code Patterns**: 50+ different Python code samples
- **Error Scenarios**: 15+ different error/edge cases
- **Unicode Tests**: 3 different languages (Vietnamese, Chinese, Emojis)
- **Real-World Frameworks**: Flask, Django, AsyncIO, Pytest, Data Science

## Coverage Areas

✅ **Basic Python Constructs**
- Functions, classes, imports, variables
- Type hints, decorators, docstrings
- Inheritance, nested structures

✅ **Advanced Python Features**  
- Async/await patterns
- Dataclasses, properties
- Context managers, generators
- Complex parameter types

✅ **Error Handling**
- Syntax errors, malformed code
- Unicode support, encoding issues
- Empty/None inputs, edge cases

✅ **Real-World Patterns**
- Web frameworks (Flask, Django)
- Data science libraries
- Testing frameworks (Pytest, unittest)
- Async programming patterns

✅ **Performance & Scale**
- Large files, many functions/classes
- Deep nesting, long lines
- Batch processing, repository analysis

Tất cả tests đều pass và verify rằng `ASTParsingAgent` có thể handle một wide range của Python code patterns một cách robust và accurate. 