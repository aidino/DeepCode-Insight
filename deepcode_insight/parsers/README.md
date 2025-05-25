# ASTParsingAgent

ASTParsingAgent là một Python code parser mạnh mẽ sử dụng `tree-sitter` để phân tích cú pháp và extract thông tin chi tiết từ Python source code.

## Tính năng chính

### 🔍 Phân tích toàn diện
- **Functions**: Extract tên, parameters, return types, docstrings, decorators
- **Classes**: Extract tên, base classes, methods, docstrings, decorators  
- **Imports**: Phân tích import statements và from imports
- **Variables**: Extract global variables và assignments
- **Decorators**: Identify và extract decorators

### 🌍 Hỗ trợ Unicode
- Xử lý tên biến, function, class với ký tự Unicode
- Hỗ trợ docstrings với ký tự đặc biệt

### 🛡️ Error Handling
- Graceful handling của syntax errors
- Robust parsing với malformed code
- Detailed error reporting

### 📊 Thống kê chi tiết
- Line counts, function counts, class counts
- Method analysis (private, dunder methods)
- Parameter type analysis

## Cài đặt

Dependencies đã được cài đặt trong `pyproject.toml`:

```toml
tree-sitter = "^0.24.0"
tree-sitter-python = "^0.23.6"
```

## Sử dụng cơ bản

### Parse một file Python

```python
from parsers.ast_parser import ASTParsingAgent

# Initialize parser
parser = ASTParsingAgent()

# Parse code
code = '''
def hello(name: str) -> str:
    """Say hello to someone"""
    return f"Hello, {name}!"

class Calculator:
    """A simple calculator"""
    
    def add(self, x: int, y: int) -> int:
        """Add two numbers"""
        return x + y
'''

result = parser.parse_code(code, "example.py")

# Access results
print(f"Functions: {result['stats']['total_functions']}")
print(f"Classes: {result['stats']['total_classes']}")

for func in result['functions']:
    print(f"Function: {func['name']}")
    print(f"  Parameters: {[p['name'] for p in func['parameters']]}")
    print(f"  Return type: {func['return_type']}")
    print(f"  Docstring: {func['docstring']}")
```

### Analyze repository với CodeFetcherAgent

```python
from parsers.ast_parser import analyze_repository_code
from agents.code_fetcher import CodeFetcherAgent

# Initialize agents
code_fetcher = CodeFetcherAgent()

# Analyze repository
result = analyze_repository_code(
    code_fetcher, 
    "https://github.com/user/repo"
)

print(f"Files analyzed: {result['summary']['total_files']}")
print(f"Total functions: {result['summary']['total_functions']}")
print(f"Total classes: {result['summary']['total_classes']}")

# Cleanup
code_fetcher.cleanup()
```

## Kết quả Parse

### Function Information
```python
{
    'name': 'function_name',
    'class_name': 'ClassName',  # None nếu không phải method
    'parameters': [
        {
            'name': 'param_name',
            'type': 'param_type',  # None nếu không có type annotation
            'default': 'default_value'  # None nếu không có default
        }
    ],
    'decorators': ['@decorator1', '@decorator2'],
    'docstring': 'Function documentation',
    'return_type': 'return_type',
    'start_line': 10,
    'end_line': 15,
    'is_method': True,
    'is_private': False,
    'is_dunder': False
}
```

### Class Information
```python
{
    'name': 'ClassName',
    'base_classes': ['BaseClass1', 'BaseClass2'],
    'decorators': ['@dataclass'],
    'docstring': 'Class documentation',
    'methods': [...],  # List of method dictionaries
    'start_line': 5,
    'end_line': 20,
    'is_private': False,
    'method_count': 3
}
```

### Import Information
```python
{
    'type': 'from_import',  # hoặc 'import'
    'module': 'typing',
    'names': ['List', 'Dict', 'Optional'],
    'text': 'from typing import List, Dict, Optional',
    'line': 2
}
```

### Statistics
```python
{
    'total_lines': 50,
    'total_functions': 5,
    'total_classes': 2,
    'total_imports': 3,
    'total_variables': 1
}
```

## Tích hợp với CodeFetcherAgent

ASTParsingAgent được thiết kế để hoạt động seamlessly với `CodeFetcherAgent`:

```python
from parsers.ast_parser import analyze_repository_code
from agents.code_fetcher import CodeFetcherAgent

def analyze_python_repository(repo_url: str):
    """Analyze toàn bộ Python repository"""
    code_fetcher = CodeFetcherAgent()
    
    try:
        # Perform comprehensive analysis
        analysis = analyze_repository_code(code_fetcher, repo_url)
        
        # Process results
        for file_analysis in analysis['files_analyzed']:
            file_path = file_analysis['file_path']
            parse_result = file_analysis['parse_result']
            
            print(f"\n📁 {file_path}")
            print(f"   Functions: {len(parse_result['functions'])}")
            print(f"   Classes: {len(parse_result['classes'])}")
            
            # Show function details
            for func in parse_result['functions']:
                class_info = f" (in {func['class_name']})" if func['class_name'] else ""
                print(f"   🔧 {func['name']}{class_info}")
        
        return analysis
        
    finally:
        code_fetcher.cleanup()
```

## Testing

Comprehensive test suite trong `tests/test_ast_parser.py`:

```bash
# Run all tests
poetry run pytest tests/test_ast_parser.py -v

# Run specific test
poetry run pytest tests/test_ast_parser.py::TestASTParsingAgent::test_parse_simple_function -v
```

## Demo

Chạy demo để xem các tính năng:

```bash
poetry run python demo_ast_parser.py
```

Demo bao gồm:
- Basic parsing functionality
- Edge cases và error handling  
- Repository analysis với CodeFetcherAgent
- Unicode support demonstration

## Limitations

1. **Docstring Parsing**: Tree-sitter có thể có issues với complex docstring formats
2. **Complex Expressions**: Chỉ parse basic parameter types và return types
3. **Dynamic Code**: Không thể analyze dynamically generated code

## Architecture

```
ASTParsingAgent
├── Core Parsing
│   ├── _extract_functions()
│   ├── _extract_classes()
│   ├── _extract_imports()
│   ├── _extract_variables()
│   └── _extract_decorators()
├── Helper Methods
│   ├── _get_function_name()
│   ├── _get_class_name()
│   ├── _get_function_parameters()
│   ├── _get_function_docstring()
│   └── ...
└── Integration
    └── analyze_repository_code()
```

## Contributing

1. Thêm test cases cho edge cases mới
2. Improve docstring parsing accuracy
3. Extend support cho Python syntax mới
4. Optimize performance cho large files

## License

Part of the LangGraph Demo project. 