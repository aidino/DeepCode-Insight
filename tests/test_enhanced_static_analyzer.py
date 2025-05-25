#!/usr/bin/env python3
"""Test script cho Enhanced StaticAnalysisAgent với Python và Java support"""

import sys
import os
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Node, Query
import re

# Mock ASTParsingAgent
class MockASTParsingAgent:
    def parse_code(self, code: str, filename: str) -> Dict:
        return {
            'stats': {
                'total_functions': 2,
                'total_classes': 1,
                'total_variables': 3
            },
            'classes': [
                {
                    'name': 'badClassName',
                    'start_line': 8,
                    'method_count': 2
                }
            ]
        }

# Copy StaticAnalysisAgent class với mock
class StaticAnalysisAgent:
    """Agent để thực hiện phân tích tĩnh code Python và Java sử dụng Tree-sitter queries"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            # Initialize languages
            self.python_language = Language(tspython.language())
            self.java_language = Language(tsjava.language())
            
            # Initialize parsers
            self.python_parser = Parser(self.python_language)
            self.java_parser = Parser(self.java_language)
            
            self.ast_parser = MockASTParsingAgent()
            
            # Initialize Tree-sitter queries
            self._init_python_queries()
            self._init_java_queries()
            
            self.logger.info("StaticAnalysisAgent initialized successfully with Python and Java support")
        except Exception as e:
            self.logger.error(f"Failed to initialize StaticAnalysisAgent: {e}")
            raise
    
    def _init_python_queries(self):
        """Initialize Tree-sitter queries cho Python patterns"""
        
        # Existing queries
        self.python_function_query = Query(
            self.python_language,
            """
            (function_definition
                name: (identifier) @func_name
            ) @function
            """
        )
        
        self.python_class_query = Query(
            self.python_language,
            """
            (class_definition
                name: (identifier) @class_name
            ) @class
            """
        )
        
        self.python_import_query = Query(
            self.python_language,
            """
            (import_statement) @import
            (import_from_statement) @from_import
            """
        )
        
        # New Google Style Guide queries
        self.python_naming_query = Query(
            self.python_language,
            """
            (class_definition name: (identifier) @class_name)
            (function_definition name: (identifier) @function_name)
            (assignment left: (identifier) @variable_name)
            (assignment left: (pattern_list (identifier) @variable_name))
            """
        )
        
        self.python_lambda_query = Query(
            self.python_language,
            """
            (lambda) @lambda_expr
            """
        )
        
        self.python_comprehension_query = Query(
            self.python_language,
            """
            (list_comprehension) @list_comp
            (dictionary_comprehension) @dict_comp
            (set_comprehension) @set_comp
            """
        )
        
        self.python_exception_query = Query(
            self.python_language,
            """
            (try_statement) @try_stmt
            (except_clause) @except_clause
            (raise_statement) @raise_stmt
            """
        )
        
        self.python_string_query = Query(
            self.python_language,
            """
            (string) @string_literal
            """
        )
    
    def _init_java_queries(self):
        """Initialize Tree-sitter queries cho Java patterns"""
        
        self.java_class_query = Query(
            self.java_language,
            """
            (class_declaration
                name: (identifier) @class_name
            ) @class
            """
        )
        
        self.java_method_query = Query(
            self.java_language,
            """
            (method_declaration
                name: (identifier) @method_name
            ) @method
            """
        )
        
        self.java_import_query = Query(
            self.java_language,
            """
            (import_declaration) @import
            """
        )
        
        self.java_naming_query = Query(
            self.java_language,
            """
            (class_declaration name: (identifier) @class_name)
            (method_declaration name: (identifier) @method_name)
            (variable_declarator name: (identifier) @variable_name)
            (field_declaration declarator: (variable_declarator name: (identifier) @field_name))
            """
        )
        
        self.java_exception_query = Query(
            self.java_language,
            """
            (try_statement) @try_stmt
            (catch_clause) @catch_clause
            (throw_statement) @throw_stmt
            """
        )

    def analyze_code(self, code: str, filename: str = "<string>") -> Dict[str, Any]:
        """
        Thực hiện phân tích tĩnh toàn diện cho Python hoặc Java code
        
        Args:
            code: Source code (Python hoặc Java)
            filename: Tên file (để tracking và xác định ngôn ngữ)
            
        Returns:
            Dict chứa kết quả phân tích
        """
        # Determine language from file extension
        language = self._detect_language(filename)
        
        result = {
            'filename': filename,
            'language': language,
            'ast_analysis': {},
            'static_issues': {
                'missing_docstrings': [],
                'unused_imports': [],
                'complex_functions': [],
                'code_smells': [],
                'naming_violations': [],
                'google_style_violations': []
            },
            'metrics': {
                'cyclomatic_complexity': 0,
                'maintainability_index': 0,
                'code_quality_score': 0
            },
            'suggestions': []
        }
        
        try:
            if not code:
                return result
            
            if language == 'python':
                return self._analyze_python_code(code, filename, result)
            elif language == 'java':
                return self._analyze_java_code(code, filename, result)
            else:
                result['static_issues']['code_smells'].append({
                    'type': 'unsupported_language',
                    'message': f'Ngôn ngữ không được hỗ trợ: {language}',
                    'line': 1
                })
                return result
                
        except Exception as e:
            error_msg = f"Static analysis error in {filename}: {e}"
            result['static_issues']['code_smells'].append({
                'type': 'analysis_error',
                'message': error_msg,
                'line': 1
            })
            self.logger.error(error_msg)
        
        return result
    
    def _detect_language(self, filename: str) -> str:
        """Xác định ngôn ngữ từ file extension"""
        if filename.endswith('.py'):
            return 'python'
        elif filename.endswith('.java'):
            return 'java'
        else:
            return 'unknown'
    
    def _analyze_python_code(self, code: str, filename: str, result: Dict) -> Dict:
        """Phân tích Python code với các quy tắc mở rộng"""
        
        # Parse code với ASTParsingAgent
        result['ast_analysis'] = self.ast_parser.parse_code(code, filename)
        
        # Parse với Tree-sitter
        tree = self.python_parser.parse(bytes(code, 'utf8'))
        
        if tree.root_node.has_error:
            result['static_issues']['code_smells'].append({
                'type': 'syntax_error',
                'message': 'Code có syntax errors',
                'line': 1
            })
            self.logger.warning(f"Syntax errors in {filename}")
            return result
        
        # Existing checks
        result['static_issues']['missing_docstrings'] = self._check_missing_docstrings_simple(
            tree.root_node, code
        )
        result['static_issues']['unused_imports'] = self._check_unused_imports_simple(
            tree.root_node, code
        )
        result['static_issues']['complex_functions'] = self._check_complex_functions_simple(
            tree.root_node, code
        )
        result['static_issues']['code_smells'] = self._check_code_smells(
            tree.root_node, code, result['ast_analysis']
        )
        
        # New Google Style Guide checks
        result['static_issues']['naming_violations'] = self._check_python_naming_conventions(
            tree.root_node, code
        )
        result['static_issues']['google_style_violations'] = self._check_python_google_style(
            tree.root_node, code
        )
        
        # Calculate metrics
        result['metrics'] = self._calculate_metrics(
            tree.root_node, code, result['ast_analysis']
        )
        
        # Generate suggestions
        result['suggestions'] = self._generate_suggestions(result)
        
        self.logger.debug(f"Python static analysis completed for {filename}")
        return result
    
    def _analyze_java_code(self, code: str, filename: str, result: Dict) -> Dict:
        """Phân tích Java code"""
        
        # Parse với Tree-sitter
        tree = self.java_parser.parse(bytes(code, 'utf8'))
        
        if tree.root_node.has_error:
            result['static_issues']['code_smells'].append({
                'type': 'syntax_error',
                'message': 'Code có syntax errors',
                'line': 1
            })
            self.logger.warning(f"Syntax errors in {filename}")
            return result
        
        # Java-specific checks
        result['static_issues']['missing_docstrings'] = self._check_java_missing_javadoc(
            tree.root_node, code
        )
        result['static_issues']['naming_violations'] = self._check_java_naming_conventions(
            tree.root_node, code
        )
        result['static_issues']['code_smells'] = self._check_java_code_smells(
            tree.root_node, code
        )
        
        # Calculate basic metrics for Java
        result['metrics'] = self._calculate_java_metrics(tree.root_node, code)
        
        # Generate suggestions
        result['suggestions'] = self._generate_java_suggestions(result)
        
        self.logger.debug(f"Java static analysis completed for {filename}")
        return result
    
    def _check_python_naming_conventions(self, root_node: Node, code: str) -> List[Dict]:
        """Kiểm tra naming conventions theo Google Python Style Guide"""
        violations = []
        
        captures = self.python_naming_query.captures(root_node)
        
        # Check class names (should be CapWords/PascalCase)
        if 'class_name' in captures:
            for node in captures['class_name']:
                name = self._get_node_text(node, code)
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', name):
                    violations.append({
                        'type': 'class_naming_violation',
                        'name': name,
                        'line': node.start_point[0] + 1,
                        'message': f"Class '{name}' should use CapWords convention (e.g., MyClass)"
                    })
        
        # Check function names (should be snake_case)
        if 'function_name' in captures:
            for node in captures['function_name']:
                name = self._get_node_text(node, code)
                # Skip dunder methods
                if not (name.startswith('__') and name.endswith('__')):
                    if not re.match(r'^[a-z_][a-z0-9_]*$', name):
                        violations.append({
                            'type': 'function_naming_violation',
                            'name': name,
                            'line': node.start_point[0] + 1,
                            'message': f"Function '{name}' should use snake_case convention (e.g., my_function)"
                        })
        
        # Check variable names (should be snake_case)
        if 'variable_name' in captures:
            for node in captures['variable_name']:
                name = self._get_node_text(node, code)
                # Skip constants (ALL_CAPS) and private variables (_name)
                if not (name.isupper() or name.startswith('_')):
                    if not re.match(r'^[a-z_][a-z0-9_]*$', name):
                        violations.append({
                            'type': 'variable_naming_violation',
                            'name': name,
                            'line': node.start_point[0] + 1,
                            'message': f"Variable '{name}' should use snake_case convention (e.g., my_variable)"
                        })
        
        return violations
    
    def _check_python_google_style(self, root_node: Node, code: str) -> List[Dict]:
        """Kiểm tra các quy tắc Google Python Style Guide"""
        violations = []
        
        # Check for lambda assignments (discouraged)
        lambda_captures = self.python_lambda_query.captures(root_node)
        if 'lambda_expr' in lambda_captures:
            for node in lambda_captures['lambda_expr']:
                # Check if lambda is assigned to a variable
                parent = node.parent
                if parent and parent.type == 'assignment':
                    violations.append({
                        'type': 'lambda_assignment',
                        'line': node.start_point[0] + 1,
                        'message': 'Avoid assigning lambda to variables, use def instead'
                    })
        
        # Check exception handling patterns
        exception_captures = self.python_exception_query.captures(root_node)
        if 'except_clause' in exception_captures:
            for node in exception_captures['except_clause']:
                except_text = self._get_node_text(node, code)
                # Check for bare except
                if except_text.strip() == 'except:':
                    violations.append({
                        'type': 'bare_except',
                        'line': node.start_point[0] + 1,
                        'message': 'Avoid bare except clauses, specify exception types'
                    })
        
        # Check for string formatting
        string_captures = self.python_string_query.captures(root_node)
        if 'string_literal' in string_captures:
            for node in string_captures['string_literal']:
                string_text = self._get_node_text(node, code)
                # Check for old-style % formatting
                if '%' in string_text and ('%(') in string_text:
                    violations.append({
                        'type': 'old_string_formatting',
                        'line': node.start_point[0] + 1,
                        'message': 'Use f-strings or .format() instead of % formatting'
                    })
        
        # Check line length (already in code_smells but adding here for completeness)
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 79:  # Google Style Guide recommends 79 characters
                violations.append({
                    'type': 'line_too_long',
                    'line': i + 1,
                    'length': len(line),
                    'message': f'Line exceeds 79 characters ({len(line)} chars)'
                })
        
        return violations
    
    def _check_java_missing_javadoc(self, root_node: Node, code: str) -> List[Dict]:
        """Kiểm tra missing Javadoc cho Java classes và methods"""
        issues = []
        
        # Check classes
        class_captures = self.java_class_query.captures(root_node)
        if 'class' in class_captures:
            for class_node in class_captures['class']:
                class_name = None
                if 'class_name' in class_captures:
                    for name_node in class_captures['class_name']:
                        if self._is_child_of(name_node, class_node):
                            class_name = self._get_node_text(name_node, code)
                            break
                
                if class_name and not self._has_javadoc(class_node, code):
                    issues.append({
                        'type': 'missing_class_javadoc',
                        'name': class_name,
                        'line': class_node.start_point[0] + 1,
                        'message': f"Class '{class_name}' thiếu Javadoc"
                    })
        
        # Check methods
        method_captures = self.java_method_query.captures(root_node)
        if 'method' in method_captures:
            for method_node in method_captures['method']:
                method_name = None
                if 'method_name' in method_captures:
                    for name_node in method_captures['method_name']:
                        if self._is_child_of(name_node, method_node):
                            method_name = self._get_node_text(name_node, code)
                            break
                
                if method_name and not method_name.startswith('_') and not self._has_javadoc(method_node, code):
                    issues.append({
                        'type': 'missing_method_javadoc',
                        'name': method_name,
                        'line': method_node.start_point[0] + 1,
                        'message': f"Method '{method_name}' thiếu Javadoc"
                    })
        
        return issues
    
    def _check_java_naming_conventions(self, root_node: Node, code: str) -> List[Dict]:
        """Kiểm tra Java naming conventions"""
        violations = []
        
        captures = self.java_naming_query.captures(root_node)
        
        # Check class names (should be PascalCase)
        if 'class_name' in captures:
            for node in captures['class_name']:
                name = self._get_node_text(node, code)
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', name):
                    violations.append({
                        'type': 'java_class_naming_violation',
                        'name': name,
                        'line': node.start_point[0] + 1,
                        'message': f"Java class '{name}' should use PascalCase (e.g., MyClass)"
                    })
        
        # Check method names (should be camelCase)
        if 'method_name' in captures:
            for node in captures['method_name']:
                name = self._get_node_text(node, code)
                if not re.match(r'^[a-z][a-zA-Z0-9]*$', name):
                    violations.append({
                        'type': 'java_method_naming_violation',
                        'name': name,
                        'line': node.start_point[0] + 1,
                        'message': f"Java method '{name}' should use camelCase (e.g., myMethod)"
                    })
        
        # Check variable names (should be camelCase)
        if 'variable_name' in captures:
            for node in captures['variable_name']:
                name = self._get_node_text(node, code)
                if not re.match(r'^[a-z][a-zA-Z0-9]*$', name):
                    violations.append({
                        'type': 'java_variable_naming_violation',
                        'name': name,
                        'line': node.start_point[0] + 1,
                        'message': f"Java variable '{name}' should use camelCase (e.g., myVariable)"
                    })
        
        return violations
    
    def _check_java_code_smells(self, root_node: Node, code: str) -> List[Dict]:
        """Kiểm tra Java code smells"""
        issues = []
        
        # Check for very long lines
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 120:  # Java convention is often 120 characters
                issues.append({
                    'type': 'long_line',
                    'line': i + 1,
                    'length': len(line),
                    'message': f"Line quá dài ({len(line)} characters)"
                })
        
        # Check for empty catch blocks
        exception_captures = self.java_exception_query.captures(root_node)
        if 'catch_clause' in exception_captures:
            for node in exception_captures['catch_clause']:
                catch_text = self._get_node_text(node, code)
                # Simple check for empty catch block
                if '{}' in catch_text or catch_text.count('{') == catch_text.count('}') == 1:
                    body_start = catch_text.find('{')
                    body_end = catch_text.rfind('}')
                    if body_start != -1 and body_end != -1:
                        body = catch_text[body_start+1:body_end].strip()
                        if not body:
                            issues.append({
                                'type': 'empty_catch_block',
                                'line': node.start_point[0] + 1,
                                'message': 'Empty catch block - consider logging or handling the exception'
                            })
        
        return issues
    
    def _calculate_java_metrics(self, root_node: Node, code: str) -> Dict:
        """Tính toán metrics cho Java code"""
        metrics = {
            'cyclomatic_complexity': 0,
            'maintainability_index': 0,
            'code_quality_score': 0,
            'lines_of_code': len(code.split('\n')),
            'comment_ratio': 0
        }
        
        try:
            # Calculate cyclomatic complexity for Java
            complexity_keywords = ['if', 'else', 'for', 'while', 'switch', 'case', 'catch', 'try']
            total_complexity = 0
            
            for line in code.split('\n'):
                stripped = line.strip()
                for keyword in complexity_keywords:
                    if keyword in stripped:
                        total_complexity += 1
            
            metrics['cyclomatic_complexity'] = total_complexity
            
            # Calculate comment ratio (including Javadoc)
            comment_lines = sum(1 for line in code.split('\n') 
                              if line.strip().startswith('//') or line.strip().startswith('/*') or line.strip().startswith('*'))
            total_lines = len(code.split('\n'))
            metrics['comment_ratio'] = comment_lines / total_lines if total_lines > 0 else 0
            
            # Simple maintainability index for Java
            loc_factor = max(0, 100 - (total_lines / 10))
            complexity_factor = max(0, 100 - (total_complexity * 2))
            comment_factor = min(100, metrics['comment_ratio'] * 200)
            
            metrics['maintainability_index'] = (loc_factor + complexity_factor + comment_factor) / 3
            metrics['code_quality_score'] = max(0, min(100, metrics['maintainability_index']))
            
        except Exception as e:
            self.logger.warning(f"Error calculating Java metrics: {e}")
        
        return metrics
    
    def _generate_java_suggestions(self, analysis_result: Dict) -> List[str]:
        """Generate suggestions cho Java code"""
        suggestions = []
        
        issues = analysis_result['static_issues']
        metrics = analysis_result['metrics']
        
        if issues['missing_docstrings']:
            suggestions.append(
                f"Thêm Javadoc cho {len(issues['missing_docstrings'])} classes/methods"
            )
        
        if issues['naming_violations']:
            suggestions.append(
                f"Sửa {len(issues['naming_violations'])} naming convention violations"
            )
        
        if issues['code_smells']:
            empty_catches = [s for s in issues['code_smells'] if s['type'] == 'empty_catch_block']
            if empty_catches:
                suggestions.append("Xử lý exceptions trong empty catch blocks")
        
        if metrics['comment_ratio'] < 0.1:
            suggestions.append("Thêm Javadoc và comments để cải thiện documentation")
        
        return suggestions
    
    def _has_javadoc(self, node: Node, code: str) -> bool:
        """Check if Java class/method has Javadoc"""
        # Look for /** comment before the node
        lines = code.split('\n')
        node_line = node.start_point[0]
        
        # Check a few lines before the node for Javadoc
        for i in range(max(0, node_line - 5), node_line):
            if i < len(lines):
                line = lines[i].strip()
                if line.startswith('/**'):
                    return True
        
        return False
    
    def _is_child_of(self, child_node: Node, parent_node: Node) -> bool:
        """Check if child_node is a child of parent_node"""
        current = child_node.parent
        while current:
            if current == parent_node:
                return True
            current = current.parent
        return False

    def _check_missing_docstrings_simple(self, root_node: Node, code: str) -> List[Dict]:
        """Kiểm tra functions và classes thiếu docstring sử dụng simple approach"""
        issues = []
        
        # Get all functions
        function_captures = self.python_function_query.captures(root_node)
        
        # Process functions - match by finding name node within function node
        if 'function' in function_captures:
            for func_node in function_captures['function']:
                # Find the function name within this function node
                func_name = None
                for child in func_node.children:
                    if child.type == 'identifier':
                        func_name = self._get_node_text(child, code)
                        break
                
                if func_name:
                    # Skip dunder methods và private methods
                    if not (func_name.startswith('__') and func_name.endswith('__')) and not func_name.startswith('_'):
                        has_docstring = self._has_docstring(func_node, code)
                        if not has_docstring:
                            issues.append({
                                'type': 'missing_function_docstring',
                                'name': func_name,
                                'line': func_node.start_point[0] + 1,
                                'message': f"Function '{func_name}' thiếu docstring"
                            })
        
        # Get all classes
        class_captures = self.python_class_query.captures(root_node)
        
        # Process classes - match by finding name node within class node
        if 'class' in class_captures:
            for class_node in class_captures['class']:
                # Find the class name within this class node
                class_name = None
                for child in class_node.children:
                    if child.type == 'identifier':
                        class_name = self._get_node_text(child, code)
                        break
                
                if class_name:
                    has_docstring = self._has_docstring(class_node, code)
                    if not has_docstring:
                        issues.append({
                            'type': 'missing_class_docstring',
                            'name': class_name,
                            'line': class_node.start_point[0] + 1,
                            'message': f"Class '{class_name}' thiếu docstring"
                        })
        
        return issues
    
    def _check_unused_imports_simple(self, root_node: Node, code: str) -> List[Dict]:
        """Kiểm tra imports không được sử dụng"""
        issues = []
        
        # Collect all imports
        imports = set()
        captures = self.python_import_query.captures(root_node)
        
        # Process import statements
        for capture_name in ['import', 'from_import']:
            if capture_name in captures:
                for node in captures[capture_name]:
                    import_text = self._get_node_text(node, code)
                    # Extract import names from the statement
                    if 'import ' in import_text:
                        # Simple parsing of import statements
                        if import_text.startswith('from '):
                            # from module import name1, name2
                            parts = import_text.split(' import ')
                            if len(parts) == 2:
                                names_part = parts[1].strip()
                                for name in names_part.split(','):
                                    name = name.strip()
                                    if name and name != '*':
                                        imports.add(name)
                        else:
                            # import module1, module2
                            import_part = import_text.replace('import ', '').strip()
                            for module in import_part.split(','):
                                module = module.strip()
                                if module:
                                    if '.' in module:
                                        parts = module.split('.')
                                        imports.add(module)  # Full dotted name
                                        imports.add(parts[-1])   # Last part
                                    else:
                                        imports.add(module)
        
        # Check if imports are used in the code
        # Simple heuristic: search for import names in the code
        code_lines = code.split('\n')
        import_lines = []
        
        # Find import statements lines
        for i, line in enumerate(code_lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                import_lines.append(i + 1)
        
        # For each import, check if it's used
        for import_name in imports:
            if len(import_name) < 2:  # Skip very short names
                continue
                
            # Count occurrences (excluding import statements)
            usage_count = 0
            for i, line in enumerate(code_lines):
                line_num = i + 1
                if line_num in import_lines:
                    continue  # Skip import lines
                
                # Simple check: import name appears in line
                if import_name in line:
                    # More sophisticated check: not in comments or strings
                    stripped = line.strip()
                    if not stripped.startswith('#'):
                        usage_count += 1
            
            if usage_count == 0:
                # Find the line where this import is defined
                import_line = None
                for i, line in enumerate(code_lines):
                    if import_name in line and (line.strip().startswith('import ') or line.strip().startswith('from ')):
                        import_line = i + 1
                        break
                
                if import_line:
                    issues.append({
                        'type': 'unused_import',
                        'name': import_name,
                        'line': import_line,
                        'message': f"Import '{import_name}' không được sử dụng"
                    })
        
        return issues
    
    def _check_complex_functions_simple(self, root_node: Node, code: str) -> List[Dict]:
        """Kiểm tra functions phức tạp"""
        issues = []
        
        # Get all functions
        function_captures = self.python_function_query.captures(root_node)
        functions = {}
        
        # Process function nodes
        if 'function' in function_captures:
            for node in function_captures['function']:
                func_id = id(node)
                functions[func_id] = {
                    'node': node,
                    'name': None,
                    'param_count': 0,
                    'nested_functions': 0,
                    'if_statements': 0,
                    'line_count': 0
                }
                
                # Count lines
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                functions[func_id]['line_count'] = end_line - start_line + 1
                
                # Count parameters
                for child in node.children:
                    if child.type == 'parameters':
                        functions[func_id]['param_count'] = self._count_parameters(child, code)
                        break
                
                # Count nested functions and if statements
                functions[func_id]['nested_functions'] = self._count_nested_functions(node)
                functions[func_id]['if_statements'] = self._count_if_statements(node)
        
        # Process function names
        if 'func_name' in function_captures:
            for node in function_captures['func_name']:
                # Find parent function
                parent = node.parent
                while parent and parent.type != 'function_definition':
                    parent = parent.parent
                if parent:
                    func_id = id(parent)
                    if func_id in functions:
                        functions[func_id]['name'] = self._get_node_text(node, code)
        
        # Check for complexity issues
        for func_data in functions.values():
            if not func_data['name']:
                continue
            
            name = func_data['name']
            line = func_data['node'].start_point[0] + 1
            
            # Too many parameters
            if func_data['param_count'] > 5:
                issues.append({
                    'type': 'too_many_parameters',
                    'name': name,
                    'line': line,
                    'count': func_data['param_count'],
                    'message': f"Function '{name}' có quá nhiều parameters ({func_data['param_count']})"
                })
            
            # Too long function
            if func_data['line_count'] > 50:
                issues.append({
                    'type': 'long_function',
                    'name': name,
                    'line': line,
                    'count': func_data['line_count'],
                    'message': f"Function '{name}' quá dài ({func_data['line_count']} lines)"
                })
            
            # Too many nested functions
            if func_data['nested_functions'] > 2:
                issues.append({
                    'type': 'too_many_nested_functions',
                    'name': name,
                    'line': line,
                    'count': func_data['nested_functions'],
                    'message': f"Function '{name}' có quá nhiều nested functions ({func_data['nested_functions']})"
                })
            
            # High cyclomatic complexity (rough estimate)
            complexity = func_data['if_statements'] + 1  # Basic estimate
            if complexity > 10:
                issues.append({
                    'type': 'high_complexity',
                    'name': name,
                    'line': line,
                    'complexity': complexity,
                    'message': f"Function '{name}' có độ phức tạp cao (complexity: {complexity})"
                })
        
        return issues
    
    def _check_code_smells(self, root_node: Node, code: str, ast_analysis: Dict) -> List[Dict]:
        """Kiểm tra các code smells khác"""
        issues = []
        
        # Check for very long lines
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 120:  # PEP 8 recommends 79, but 120 is more practical
                issues.append({
                    'type': 'long_line',
                    'line': i + 1,
                    'length': len(line),
                    'message': f"Line quá dài ({len(line)} characters)"
                })
        
        # Check for too many global variables
        if ast_analysis.get('stats', {}).get('total_variables', 0) > 10:
            issues.append({
                'type': 'too_many_globals',
                'count': ast_analysis['stats']['total_variables'],
                'line': 1,
                'message': f"Quá nhiều global variables ({ast_analysis['stats']['total_variables']})"
            })
        
        # Check for classes with too many methods
        for class_info in ast_analysis.get('classes', []):
            if class_info.get('method_count', 0) > 20:
                issues.append({
                    'type': 'god_class',
                    'name': class_info['name'],
                    'line': class_info['start_line'],
                    'method_count': class_info['method_count'],
                    'message': f"Class '{class_info['name']}' có quá nhiều methods ({class_info['method_count']})"
                })
        
        return issues
    
    def _calculate_metrics(self, root_node: Node, code: str, ast_analysis: Dict) -> Dict:
        """Tính toán các metrics chất lượng code"""
        metrics = {
            'cyclomatic_complexity': 0,
            'maintainability_index': 0,
            'code_quality_score': 0,
            'lines_of_code': len(code.split('\n')),
            'comment_ratio': 0,
            'function_to_class_ratio': 0
        }
        
        try:
            # Calculate cyclomatic complexity (simplified)
            complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'with']
            total_complexity = 0
            
            for line in code.split('\n'):
                stripped = line.strip()
                for keyword in complexity_keywords:
                    if stripped.startswith(keyword + ' ') or stripped.startswith(keyword + '('):
                        total_complexity += 1
            
            metrics['cyclomatic_complexity'] = total_complexity
            
            # Calculate comment ratio
            comment_lines = sum(1 for line in code.split('\n') if line.strip().startswith('#'))
            total_lines = len(code.split('\n'))
            metrics['comment_ratio'] = comment_lines / total_lines if total_lines > 0 else 0
            
            # Function to class ratio
            total_functions = ast_analysis.get('stats', {}).get('total_functions', 0)
            total_classes = ast_analysis.get('stats', {}).get('total_classes', 0)
            if total_classes > 0:
                metrics['function_to_class_ratio'] = total_functions / total_classes
            else:
                metrics['function_to_class_ratio'] = total_functions
            
            # Simple maintainability index (0-100)
            # Based on lines of code, complexity, and comment ratio
            loc_factor = max(0, 100 - (total_lines / 10))  # Penalty for long files
            complexity_factor = max(0, 100 - (total_complexity * 2))  # Penalty for complexity
            comment_factor = min(100, metrics['comment_ratio'] * 200)  # Bonus for comments
            
            metrics['maintainability_index'] = (loc_factor + complexity_factor + comment_factor) / 3
            
            # Overall code quality score (0-100)
            # Combine various factors
            quality_score = metrics['maintainability_index']
            
            # Penalty for code smells would be calculated here
            # For now, use maintainability index as base
            metrics['code_quality_score'] = max(0, min(100, quality_score))
            
        except Exception as e:
            self.logger.warning(f"Error calculating metrics: {e}")
        
        return metrics
    
    def _generate_suggestions(self, analysis_result: Dict) -> List[str]:
        """Generate suggestions dựa trên kết quả phân tích cho Python"""
        suggestions = []
        
        issues = analysis_result['static_issues']
        metrics = analysis_result['metrics']
        language = analysis_result.get('language', 'python')
        
        # Suggestions for missing docstrings
        if issues['missing_docstrings']:
            doc_type = "docstrings" if language == 'python' else "Javadoc"
            suggestions.append(
                f"Thêm {doc_type} cho {len(issues['missing_docstrings'])} functions/classes "
                "để cải thiện documentation"
            )
        
        # Suggestions for unused imports
        if issues['unused_imports']:
            suggestions.append(
                f"Xóa {len(issues['unused_imports'])} unused imports để clean up code"
            )
        
        # Suggestions for complex functions
        if issues['complex_functions']:
            suggestions.append(
                f"Refactor {len(issues['complex_functions'])} complex functions "
                "để cải thiện maintainability"
            )
        
        # Suggestions for naming violations
        if issues['naming_violations']:
            suggestions.append(
                f"Sửa {len(issues['naming_violations'])} naming convention violations "
                f"để tuân thủ {language.title()} coding standards"
            )
        
        # Python-specific Google Style Guide suggestions
        if language == 'python' and issues['google_style_violations']:
            google_violations = issues['google_style_violations']
            lambda_assignments = [v for v in google_violations if v['type'] == 'lambda_assignment']
            bare_excepts = [v for v in google_violations if v['type'] == 'bare_except']
            old_formatting = [v for v in google_violations if v['type'] == 'old_string_formatting']
            
            if lambda_assignments:
                suggestions.append("Thay thế lambda assignments bằng def functions")
            if bare_excepts:
                suggestions.append("Specify exception types thay vì dùng bare except")
            if old_formatting:
                suggestions.append("Modernize string formatting: dùng f-strings hoặc .format()")
        
        # Suggestions based on metrics
        if metrics['comment_ratio'] < 0.1:
            suggestions.append("Thêm comments để giải thích logic phức tạp")
        
        if metrics['cyclomatic_complexity'] > 20:
            suggestions.append("Giảm cyclomatic complexity bằng cách chia nhỏ functions")
        
        if metrics['maintainability_index'] < 50:
            suggestions.append("Code cần refactoring để cải thiện maintainability")
        
        # Code smell suggestions
        code_smells = issues['code_smells']
        long_lines = [s for s in code_smells if s['type'] == 'long_line']
        if long_lines:
            style_guide = "PEP 8" if language == 'python' else "Java style guidelines"
            suggestions.append(f"Chia {len(long_lines)} long lines để tuân thủ {style_guide}")
        
        god_classes = [s for s in code_smells if s['type'] == 'god_class']
        if god_classes:
            suggestions.append("Chia nhỏ large classes theo Single Responsibility Principle")
        
        # Java-specific suggestions
        if language == 'java':
            empty_catches = [s for s in code_smells if s['type'] == 'empty_catch_block']
            if empty_catches:
                suggestions.append("Xử lý exceptions trong empty catch blocks")
        
        return suggestions
    
    # Helper methods
    def _get_node_text(self, node: Node, code: str) -> str:
        """Get text content của node"""
        return code[node.start_byte:node.end_byte]
    
    def _has_docstring(self, node: Node, code: str) -> bool:
        """Check if function/class has docstring"""
        # Look for first string literal in body
        for child in node.children:
            if child.type == 'block':
                for stmt in child.children:
                    if stmt.type == 'expression_statement':
                        for expr_child in stmt.children:
                            if expr_child.type == 'string':
                                return True
                        break  # Only check first statement
                break
        return False
    
    def _count_parameters(self, params_node: Node, code: str) -> int:
        """Count số lượng parameters trong function"""
        count = 0
        for child in params_node.children:
            if child.type in ['identifier', 'typed_parameter', 'default_parameter', 'typed_default_parameter']:
                count += 1
        return count
    
    def _count_nested_functions(self, func_node: Node) -> int:
        """Count nested functions trong function"""
        count = 0
        def traverse(node, depth=0):
            nonlocal count
            if depth > 0 and node.type == 'function_definition':
                count += 1
            for child in node.children:
                traverse(child, depth + 1)
        
        traverse(func_node, 0)
        return count
    
    def _count_if_statements(self, func_node: Node) -> int:
        """Count if statements trong function"""
        count = 0
        def traverse(node):
            nonlocal count
            if node.type == 'if_statement':
                count += 1
            for child in node.children:
                traverse(child)
        
        traverse(func_node)
        return count


def test_enhanced_static_analyzer():
    """Test function để test StaticAnalysisAgent với cả Python và Java"""
    
    # Python sample code với Google Style Guide violations
    python_sample = '''
import os
import sys
import unused_module
from typing import List, Dict
from collections import defaultdict

# Bad naming - should be CapWords
class badClassName:
    def __init__(self, initial_value=0):
        self.value = initial_value
    
    # Bad naming - should be snake_case
    def BadMethodName(self, x, y, z, a, b, c, d):  # Too many parameters
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

# Lambda assignment (discouraged)
my_lambda = lambda x: x * 2

# Bare except (bad practice)
try:
    risky_operation()
except:
    pass

# Old string formatting
message = "Hello %s, you have %d messages" % (name, count)

def function_without_docstring(x, y):
    return x + y

# Bad variable naming
BadVariableName = "should be snake_case"
'''
    
    # Java sample code
    java_sample = '''
import java.util.*;

// Missing Javadoc
public class badClassName {  // Should be PascalCase
    private int value;
    
    // Missing Javadoc
    public void BadMethodName(int param1, int param2, int param3, int param4, int param5, int param6) {  // Too many params, bad naming
        if (param1 > 0) {
            if (param2 > 0) {
                if (param3 > 0) {
                    if (param4 > 0) {
                        if (param5 > 0) {
                            if (param6 > 0) {
                                System.out.println("Very nested logic that should be refactored into smaller methods for better readability and maintainability");
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Empty catch block
    public void riskyMethod() {
        try {
            // Some risky operation
            int result = 10 / 0;
        } catch (Exception e) {
            // Empty catch - bad practice
        }
    }
    
    // Bad variable naming
    private String BadVariableName = "should be camelCase";
}
'''
    
    try:
        analyzer = StaticAnalysisAgent()
        
        print("🔍 === Enhanced Static Analysis Demo ===")
        print("Testing Python and Java code analysis with Google Style Guide rules\n")
        
        # Test Python code
        print("=" * 60)
        print("🐍 PYTHON CODE ANALYSIS")
        print("=" * 60)
        
        python_result = analyzer.analyze_code(python_sample, "demo.py")
        
        print(f"File: {python_result['filename']} ({python_result['language']})")
        print(f"Quality Score: {python_result['metrics']['code_quality_score']:.1f}/100")
        print(f"Maintainability Index: {python_result['metrics']['maintainability_index']:.1f}/100")
        print()
        
        print("📋 Issues Found:")
        for category, issues in python_result['static_issues'].items():
            if issues:
                print(f"\n  {category.replace('_', ' ').title()} ({len(issues)}):")
                for issue in issues[:3]:  # Show first 3 issues
                    print(f"    - Line {issue.get('line', '?')}: {issue['message']}")
                if len(issues) > 3:
                    print(f"    ... and {len(issues) - 3} more")
        
        print(f"\n💡 Python Suggestions ({len(python_result['suggestions'])}):")
        for suggestion in python_result['suggestions'][:5]:  # Show first 5
            print(f"  - {suggestion}")
        if len(python_result['suggestions']) > 5:
            print(f"  ... and {len(python_result['suggestions']) - 5} more")
        
        # Test Java code
        print("\n" + "=" * 60)
        print("☕ JAVA CODE ANALYSIS")
        print("=" * 60)
        
        java_result = analyzer.analyze_code(java_sample, "Demo.java")
        
        print(f"File: {java_result['filename']} ({java_result['language']})")
        print(f"Quality Score: {java_result['metrics']['code_quality_score']:.1f}/100")
        print(f"Maintainability Index: {java_result['metrics']['maintainability_index']:.1f}/100")
        print()
        
        print("📋 Issues Found:")
        for category, issues in java_result['static_issues'].items():
            if issues:
                print(f"\n  {category.replace('_', ' ').title()} ({len(issues)}):")
                for issue in issues[:3]:  # Show first 3 issues
                    print(f"    - Line {issue.get('line', '?')}: {issue['message']}")
                if len(issues) > 3:
                    print(f"    ... and {len(issues) - 3} more")
        
        print(f"\n💡 Java Suggestions ({len(java_result['suggestions'])}):")
        for suggestion in java_result['suggestions']:
            print(f"  - {suggestion}")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 COMPARISON SUMMARY")
        print("=" * 60)
        print(f"Python Quality Score: {python_result['metrics']['code_quality_score']:.1f}/100")
        print(f"Java Quality Score: {java_result['metrics']['code_quality_score']:.1f}/100")
        
        total_python_issues = sum(len(issues) for issues in python_result['static_issues'].values())
        total_java_issues = sum(len(issues) for issues in java_result['static_issues'].values())
        
        print(f"Python Total Issues: {total_python_issues}")
        print(f"Java Total Issues: {total_java_issues}")
        
        print("\n✨ New Features Demonstrated:")
        print("  ✓ Multi-language support (Python + Java)")
        print("  ✓ Google Python Style Guide compliance")
        print("  ✓ Java naming conventions")
        print("  ✓ Enhanced naming violation detection")
        print("  ✓ Language-specific code smell detection")
        print("  ✓ Javadoc vs Docstring detection")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_enhanced_static_analyzer()
    if success:
        print("\n✅ Enhanced StaticAnalysisAgent test completed successfully!")
    else:
        print("\n❌ Enhanced StaticAnalysisAgent test failed!")
        sys.exit(1) 