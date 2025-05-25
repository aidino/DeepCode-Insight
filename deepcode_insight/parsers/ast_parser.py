"""ASTParsingAgent - Parse Python code với tree-sitter"""

import logging
from typing import Dict, List, Optional, Any, Union
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node


class ASTParsingAgent:
    """Agent để parse Python code thành AST và extract thông tin chi tiết"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.python_language = Language(tspython.language())
            self.parser = Parser(self.python_language)
            self.logger.info("ASTParsingAgent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ASTParsingAgent: {e}")
            raise
    
    def parse_code(self, code: str, filename: str = "<string>") -> Dict[str, Any]:
        """
        Parse Python code string và extract thông tin chi tiết
        
        Args:
            code: Python source code
            filename: Tên file (để tracking)
            
        Returns:
            Dict chứa parsed information:
            {
                'filename': str,
                'functions': List[Dict],
                'classes': List[Dict], 
                'imports': List[Dict],
                'variables': List[Dict],
                'decorators': List[Dict],
                'errors': List[str],
                'stats': Dict[str, int]
            }
        """
        result = {
            'filename': filename,
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'decorators': [],
            'errors': [],
            'stats': {
                'total_lines': len(code.splitlines()) if code else 0,
                'total_functions': 0,
                'total_classes': 0,
                'total_imports': 0,
                'total_variables': 0
            }
        }
        
        try:
            # Validate input
            if not code:
                return result
            
            # Parse code
            tree = self.parser.parse(bytes(code, 'utf8'))
            
            if tree.root_node.has_error:
                result['errors'].append("Syntax errors detected in code")
                self.logger.warning(f"Syntax errors in {filename}")
            
            # Extract information
            result['functions'] = self._extract_functions(tree.root_node, code)
            result['classes'] = self._extract_classes(tree.root_node, code)
            result['imports'] = self._extract_imports(tree.root_node, code)
            result['variables'] = self._extract_variables(tree.root_node, code)
            result['decorators'] = self._extract_decorators(tree.root_node, code)
            
            # Update stats
            result['stats'].update({
                'total_functions': len(result['functions']),
                'total_classes': len(result['classes']),
                'total_imports': len(result['imports']),
                'total_variables': len(result['variables'])
            })
            
            self.logger.debug(f"Parsed {filename}: {result['stats']}")
            
        except Exception as e:
            error_msg = f"Parse error in {filename}: {e}"
            result['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        return result
    
    def _extract_functions(self, root_node: Node, code: str) -> List[Dict]:
        """Extract functions với thông tin chi tiết"""
        functions = []
        
        def traverse(node: Node, class_name: Optional[str] = None):
            if node.type == 'function_definition':
                func_info = self._parse_function_definition(node, code, class_name)
                if func_info:
                    functions.append(func_info)
            
            # Recursively traverse children
            for child in node.children:
                current_class = class_name
                if child.type == 'class_definition':
                    current_class = self._get_class_name(child, code)
                traverse(child, current_class)
        
        traverse(root_node)
        return functions
    
    def _extract_classes(self, root_node: Node, code: str) -> List[Dict]:
        """Extract classes với thông tin chi tiết"""
        classes = []
        
        def traverse(node: Node):
            if node.type == 'class_definition':
                class_info = self._parse_class_definition(node, code)
                if class_info:
                    classes.append(class_info)
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return classes
    
    def _extract_imports(self, root_node: Node, code: str) -> List[Dict]:
        """Extract imports với thông tin chi tiết"""
        imports = []
        
        def traverse(node: Node):
            if node.type in ['import_statement', 'import_from_statement']:
                import_info = self._parse_import_statement(node, code)
                if import_info:
                    imports.append(import_info)
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return imports
    
    def _extract_variables(self, root_node: Node, code: str) -> List[Dict]:
        """Extract global variables"""
        variables = []
        
        def traverse(node: Node, depth: int = 0):
            # Only get top-level assignments (global variables)
            if depth == 0 and node.type == 'assignment':
                var_info = self._parse_assignment(node, code)
                if var_info:
                    variables.append(var_info)
            
            for child in node.children:
                new_depth = depth + 1 if node.type in ['function_definition', 'class_definition'] else depth
                traverse(child, new_depth)
        
        traverse(root_node)
        return variables
    
    def _extract_decorators(self, root_node: Node, code: str) -> List[Dict]:
        """Extract decorators"""
        decorators = []
        
        def traverse(node: Node):
            if node.type == 'decorator':
                decorator_info = self._parse_decorator(node, code)
                if decorator_info:
                    decorators.append(decorator_info)
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return decorators
    
    def _parse_function_definition(self, node: Node, code: str, class_name: Optional[str] = None) -> Optional[Dict]:
        """Parse function definition với chi tiết"""
        try:
            name = self._get_function_name(node, code)
            if not name:
                return None
            
            # Get parameters
            parameters = self._get_function_parameters(node, code)
            
            # Get docstring
            docstring = self._get_function_docstring(node, code)
            
            # Get decorators
            decorators = self._get_function_decorators(node, code)
            
            # Get return type annotation if exists
            return_type = self._get_return_type_annotation(node, code)
            
            return {
                'name': name,
                'class_name': class_name,
                'parameters': parameters,
                'decorators': decorators,
                'docstring': docstring,
                'return_type': return_type,
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1,
                'is_method': class_name is not None,
                'is_private': name.startswith('_'),
                'is_dunder': name.startswith('__') and name.endswith('__')
            }
        except Exception as e:
            self.logger.warning(f"Error parsing function: {e}")
            return None
    
    def _parse_class_definition(self, node: Node, code: str) -> Optional[Dict]:
        """Parse class definition với chi tiết"""
        try:
            name = self._get_class_name(node, code)
            if not name:
                return None
            
            # Get base classes
            base_classes = self._get_base_classes(node, code)
            
            # Get docstring
            docstring = self._get_class_docstring(node, code)
            
            # Get decorators
            decorators = self._get_class_decorators(node, code)
            
            # Get methods
            methods = []
            for child in node.children:
                if child.type == 'block':
                    for grandchild in child.children:
                        if grandchild.type == 'function_definition':
                            method_info = self._parse_function_definition(grandchild, code, name)
                            if method_info:
                                methods.append(method_info)
            
            return {
                'name': name,
                'base_classes': base_classes,
                'decorators': decorators,
                'docstring': docstring,
                'methods': methods,
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1,
                'is_private': name.startswith('_'),
                'method_count': len(methods)
            }
        except Exception as e:
            self.logger.warning(f"Error parsing class: {e}")
            return None
    
    def _parse_import_statement(self, node: Node, code: str) -> Optional[Dict]:
        """Parse import statement với chi tiết"""
        try:
            import_text = self._get_node_text(node, code).strip()
            
            if node.type == 'import_statement':
                # import module1, module2
                modules = []
                for child in node.children:
                    if child.type == 'dotted_as_names' or child.type == 'dotted_name':
                        modules.extend(self._extract_module_names(child, code))
                
                return {
                    'type': 'import',
                    'modules': modules,
                    'text': import_text,
                    'line': node.start_point[0] + 1
                }
            
            elif node.type == 'import_from_statement':
                # from module import name1, name2
                module = None
                names = []
                
                # Look for module name after 'from' keyword
                found_from = False
                for child in node.children:
                    if child.type == 'from':
                        found_from = True
                    elif found_from and child.type == 'dotted_name':
                        module = self._get_node_text(child, code)
                        break
                
                # Look for imported names
                found_import = False
                for child in node.children:
                    if child.type == 'import':
                        found_import = True
                    elif found_import and child.type == 'dotted_name':
                        # Individual import names after 'import' keyword
                        names.append(self._get_node_text(child, code))
                    elif child.type in ['import_list', 'wildcard_import']:
                        names.extend(self._extract_import_names(child, code))
                
                return {
                    'type': 'from_import',
                    'module': module,
                    'names': names,
                    'text': import_text,
                    'line': node.start_point[0] + 1
                }
        except Exception as e:
            self.logger.warning(f"Error parsing import: {e}")
            return None
    
    def _parse_assignment(self, node: Node, code: str) -> Optional[Dict]:
        """Parse assignment statement"""
        try:
            assignment_text = self._get_node_text(node, code).strip()
            
            # Simple parsing: split by '='
            if '=' in assignment_text:
                parts = assignment_text.split('=', 1)
                if len(parts) == 2:
                    var_part = parts[0].strip()
                    value_part = parts[1].strip()
                    
                    # Handle multiple variables: a, b = values
                    variables = [v.strip() for v in var_part.split(',')]
                    
                    return {
                        'variables': variables,
                        'value': value_part,
                        'line': node.start_point[0] + 1,
                        'text': assignment_text
                    }
        except Exception as e:
            self.logger.warning(f"Error parsing assignment: {e}")
            return None
    
    def _parse_decorator(self, node: Node, code: str) -> Optional[Dict]:
        """Parse decorator"""
        try:
            decorator_text = self._get_node_text(node, code).strip()
            
            # Extract decorator name
            name = None
            for child in node.children:
                if child.type == 'identifier':
                    name = self._get_node_text(child, code)
                    break
                elif child.type == 'attribute':
                    name = self._get_node_text(child, code)
                    break
            
            return {
                'name': name,
                'text': decorator_text,
                'line': node.start_point[0] + 1
            }
        except Exception as e:
            self.logger.warning(f"Error parsing decorator: {e}")
            return None
    
    # Helper methods
    def _get_node_text(self, node: Node, code: str) -> str:
        """Get text content của node"""
        return code[node.start_byte:node.end_byte]
    
    def _get_function_name(self, node: Node, code: str) -> Optional[str]:
        """Get function name"""
        # For function_definition, the second child should be the identifier (after 'def')
        if len(node.children) >= 2 and node.children[1].type == 'identifier':
            return self._get_node_text(node.children[1], code)
        
        # Fallback: look for identifier that comes after 'def' keyword
        found_def = False
        for child in node.children:
            if child.type == 'def':
                found_def = True
            elif found_def and child.type == 'identifier':
                return self._get_node_text(child, code)
        
        return None
    
    def _get_class_name(self, node: Node, code: str) -> Optional[str]:
        """Get class name"""
        # For class_definition, the second child should be the identifier (after 'class')
        if len(node.children) >= 2 and node.children[1].type == 'identifier':
            name = self._get_node_text(node.children[1], code)
            # Clean up name (remove trailing colon and whitespace)
            return name.split(':')[0].strip()
        
        # Fallback: look for identifier that comes after 'class' keyword
        found_class = False
        for child in node.children:
            if child.type == 'class':
                found_class = True
            elif found_class and child.type == 'identifier':
                name = self._get_node_text(child, code)
                return name.split(':')[0].strip()
        
        return None
    
    def _get_function_parameters(self, node: Node, code: str) -> List[Dict]:
        """Get function parameters"""
        parameters = []
        
        for child in node.children:
            if child.type == 'parameters':
                for param_child in child.children:
                    if param_child.type == 'identifier':
                        param_name = self._get_node_text(param_child, code)
                        parameters.append({
                            'name': param_name,
                            'type': None,
                            'default': None
                        })
                    elif param_child.type == 'default_parameter':
                        # Handle default parameters
                        param_name = None
                        default_value = None
                        for default_child in param_child.children:
                            if default_child.type == 'identifier':
                                param_name = self._get_node_text(default_child, code)
                            elif default_child.type not in ['=', 'identifier']:
                                default_value = self._get_node_text(default_child, code)
                        
                        if param_name:
                            parameters.append({
                                'name': param_name,
                                'type': None,
                                'default': default_value
                            })
                    elif param_child.type == 'typed_parameter':
                        # Handle typed parameters like x: int
                        param_name = None
                        param_type = None
                        for typed_child in param_child.children:
                            if typed_child.type == 'identifier':
                                param_name = self._get_node_text(typed_child, code)
                            elif typed_child.type == 'type':
                                param_type = self._get_node_text(typed_child, code)
                        
                        if param_name:
                            parameters.append({
                                'name': param_name,
                                'type': param_type,
                                'default': None
                            })
                    elif param_child.type == 'typed_default_parameter':
                        # Handle typed default parameters like x: int = 5
                        param_name = None
                        param_type = None
                        default_value = None
                        for typed_default_child in param_child.children:
                            if typed_default_child.type == 'identifier':
                                param_name = self._get_node_text(typed_default_child, code)
                            elif typed_default_child.type == 'type':
                                param_type = self._get_node_text(typed_default_child, code)
                            elif typed_default_child.type not in ['=', 'identifier', 'type', ':']:
                                default_value = self._get_node_text(typed_default_child, code)
                        
                        if param_name:
                            parameters.append({
                                'name': param_name,
                                'type': param_type,
                                'default': default_value
                            })
        
        return parameters
    
    def _get_function_docstring(self, node: Node, code: str) -> Optional[str]:
        """Get function docstring"""
        # Look for first string literal in function body
        for child in node.children:
            if child.type == 'block':
                for stmt in child.children:
                    if stmt.type == 'expression_statement':
                        for expr_child in stmt.children:
                            if expr_child.type == 'string':
                                # Get the actual string content, not the whole node
                                docstring_text = ""
                                for string_part in expr_child.children:
                                    if string_part.type == 'string_content':
                                        docstring_text += self._get_node_text(string_part, code)
                                
                                if docstring_text:
                                    return docstring_text.strip()
                                
                                # Fallback: use full string and clean it
                                full_string = self._get_node_text(expr_child, code)
                                if full_string.startswith('"""') or full_string.startswith("'''"):
                                    # Find the end of the docstring
                                    end_quote = full_string.find('"""', 3)
                                    if end_quote != -1:
                                        return full_string[3:end_quote].strip()
                                elif full_string.startswith('"') or full_string.startswith("'"):
                                    quote_char = full_string[0]
                                    end_quote = full_string.find(quote_char, 1)
                                    if end_quote != -1:
                                        return full_string[1:end_quote].strip()
                                
                                return full_string.strip()
                        break
                break
        return None
    
    def _get_class_docstring(self, node: Node, code: str) -> Optional[str]:
        """Get class docstring"""
        return self._get_function_docstring(node, code)  # Same logic
    
    def _get_function_decorators(self, node: Node, code: str) -> List[str]:
        """Get function decorators"""
        decorators = []
        
        # Look for decorators before function definition
        parent = node.parent
        if parent:
            for child in parent.children:
                if child == node:
                    break
                if child.type == 'decorator':
                    decorators.append(self._get_node_text(child, code).strip())
        
        return decorators
    
    def _get_class_decorators(self, node: Node, code: str) -> List[str]:
        """Get class decorators"""
        return self._get_function_decorators(node, code)  # Same logic
    
    def _get_return_type_annotation(self, node: Node, code: str) -> Optional[str]:
        """Get return type annotation"""
        for child in node.children:
            if child.type == 'type':
                return self._get_node_text(child, code)
        return None
    
    def _get_base_classes(self, node: Node, code: str) -> List[str]:
        """Get base classes"""
        base_classes = []
        
        for child in node.children:
            if child.type == 'argument_list':
                for arg_child in child.children:
                    if arg_child.type == 'identifier':
                        base_classes.append(self._get_node_text(arg_child, code))
        
        return base_classes
    
    def _extract_module_names(self, node: Node, code: str) -> List[str]:
        """Extract module names từ import statement"""
        modules = []
        
        if node.type == 'dotted_name':
            modules.append(self._get_node_text(node, code))
        elif node.type == 'dotted_as_names':
            for child in node.children:
                if child.type == 'dotted_name':
                    modules.append(self._get_node_text(child, code))
        
        return modules
    
    def _extract_import_names(self, node: Node, code: str) -> List[str]:
        """Extract import names từ from import statement"""
        names = []
        
        if node.type == 'wildcard_import':
            names.append('*')
        elif node.type == 'import_list':
            for child in node.children:
                if child.type == 'identifier':
                    names.append(self._get_node_text(child, code))
        
        return names


def analyze_repository_code(code_fetcher_agent, repo_url: str) -> Dict[str, Any]:
    """
    Analyze Python code trong repository sử dụng CodeFetcherAgent và ASTParsingAgent
    
    Args:
        code_fetcher_agent: Instance của CodeFetcherAgent
        repo_url: Repository URL
        
    Returns:
        Dict chứa analysis results
    """
    ast_parser = ASTParsingAgent()
    results = {
        'repository': repo_url,
        'files_analyzed': [],
        'summary': {
            'total_files': 0,
            'total_functions': 0,
            'total_classes': 0,
            'total_imports': 0,
            'total_variables': 0,
            'total_lines': 0
        },
        'errors': [],
        'analysis_timestamp': None
    }
    
    try:
        from datetime import datetime
        results['analysis_timestamp'] = datetime.now().isoformat()
        
        # Get Python files từ repository
        ast_parser.logger.info(f"Starting analysis of repository: {repo_url}")
        files = code_fetcher_agent.list_repository_files(repo_url)
        python_files = [f for f in files if f.endswith('.py')]
        
        ast_parser.logger.info(f"Found {len(python_files)} Python files to analyze")
        
        for file_path in python_files:
            try:
                content = code_fetcher_agent.get_file_content(repo_url, file_path)
                if content:
                    parse_result = ast_parser.parse_code(content, file_path)
                    
                    results['files_analyzed'].append({
                        'file_path': file_path,
                        'parse_result': parse_result
                    })
                    
                    # Update summary
                    results['summary']['total_files'] += 1
                    results['summary']['total_functions'] += parse_result['stats']['total_functions']
                    results['summary']['total_classes'] += parse_result['stats']['total_classes']
                    results['summary']['total_imports'] += parse_result['stats']['total_imports']
                    results['summary']['total_variables'] += parse_result['stats']['total_variables']
                    results['summary']['total_lines'] += parse_result['stats']['total_lines']
                    
                    if parse_result['errors']:
                        results['errors'].extend([
                            f"{file_path}: {error}" for error in parse_result['errors']
                        ])
                else:
                    results['errors'].append(f"Could not read content of {file_path}")
            
            except Exception as e:
                error_msg = f"Error analyzing {file_path}: {e}"
                results['errors'].append(error_msg)
                ast_parser.logger.error(error_msg)
        
        ast_parser.logger.info(f"Analysis completed: {results['summary']}")
        
    except Exception as e:
        error_msg = f"Error analyzing repository: {e}"
        results['errors'].append(error_msg)
        ast_parser.logger.error(error_msg)
    
    return results


if __name__ == "__main__":
    # Test với sample code
    sample_code = '''
import os
from typing import List, Optional

@dataclass
class Calculator:
    """A simple calculator class"""
    
    def __init__(self, initial_value: int = 0):
        """Initialize calculator với initial value"""
        self.value = initial_value
    
    @property
    def current_value(self) -> int:
        """Get current value"""
        return self.value
    
    def add(self, x: int) -> int:
        """Add x to current value"""
        self.value += x
        return self.value

def main() -> None:
    """Main function"""
    calc = Calculator(initial_value=10)
    print(calc.add(5))

# Global variable
DEBUG_MODE = True
'''
    
    # Test parsing
    parser = ASTParsingAgent()
    result = parser.parse_code(sample_code, "test.py")
    
    print("=== AST Parsing Results ===")
    print(f"File: {result['filename']}")
    print(f"Stats: {result['stats']}")
    print()
    
    print("Functions:")
    for func in result['functions']:
        print(f"  - {func['name']} (lines {func['start_line']}-{func['end_line']})")
        if func['parameters']:
            print(f"    Parameters: {[p['name'] for p in func['parameters']]}")
        if func['docstring']:
            print(f"    Docstring: {func['docstring'][:50]}...")
        if func['decorators']:
            print(f"    Decorators: {func['decorators']}")
    print()
    
    print("Classes:")
    for cls in result['classes']:
        print(f"  - {cls['name']} (lines {cls['start_line']}-{cls['end_line']})")
        if cls['base_classes']:
            print(f"    Base classes: {cls['base_classes']}")
        if cls['docstring']:
            print(f"    Docstring: {cls['docstring'][:50]}...")
        if cls['decorators']:
            print(f"    Decorators: {cls['decorators']}")
        print(f"    Methods: {cls['method_count']}")
    print()
    
    print("Imports:")
    for imp in result['imports']:
        print(f"  - {imp['text']} (line {imp['line']})")
    print()
    
    print("Variables:")
    for var in result['variables']:
        print(f"  - {var['variables']} = {var['value']} (line {var['line']})")
    
    if result['errors']:
        print("\nErrors:")
        for error in result['errors']:
            print(f"  - {error}") 