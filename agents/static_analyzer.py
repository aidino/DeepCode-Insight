"""StaticAnalysisAgent - Phân tích tĩnh code Python sử dụng Tree-sitter queries"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node, Query
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from parsers.ast_parser import ASTParsingAgent


class StaticAnalysisAgent:
    """Agent để thực hiện phân tích tĩnh code Python sử dụng Tree-sitter queries"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.python_language = Language(tspython.language())
            self.parser = Parser(self.python_language)
            self.ast_parser = ASTParsingAgent()
            
            # Initialize Tree-sitter queries
            self._init_queries()
            
            self.logger.info("StaticAnalysisAgent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize StaticAnalysisAgent: {e}")
            raise
    
    def _init_queries(self):
        """Initialize Tree-sitter queries cho các pattern phân tích"""
        
        # Query 1: Functions
        self.function_query = Query(
            self.python_language,
            """
            (function_definition
                name: (identifier) @func_name
            ) @function
            """
        )
        
        # Query 2: Classes
        self.class_query = Query(
            self.python_language,
            """
            (class_definition
                name: (identifier) @class_name
            ) @class
            """
        )
        
        # Query 3: Import statements
        self.import_query = Query(
            self.python_language,
            """
            (import_statement) @import
            (import_from_statement) @from_import
            """
        )
        
        # Query 4: String literals (for docstrings)
        self.string_query = Query(
            self.python_language,
            """
            (string) @string
            """
        )
        
        # Query 5: If statements (for complexity)
        self.if_query = Query(
            self.python_language,
            """
            (if_statement) @if_stmt
            """
        )
    
    def analyze_code(self, code: str, filename: str = "<string>") -> Dict[str, Any]:
        """
        Thực hiện phân tích tĩnh toàn diện cho Python code
        
        Args:
            code: Python source code
            filename: Tên file (để tracking)
            
        Returns:
            Dict chứa kết quả phân tích:
            {
                'filename': str,
                'ast_analysis': Dict,  # Từ ASTParsingAgent
                'static_issues': Dict,
                'metrics': Dict,
                'suggestions': List[str]
            }
        """
        result = {
            'filename': filename,
            'ast_analysis': {},
            'static_issues': {
                'missing_docstrings': [],
                'unused_imports': [],
                'complex_functions': [],
                'code_smells': []
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
            
            # Parse code với ASTParsingAgent
            result['ast_analysis'] = self.ast_parser.parse_code(code, filename)
            
            # Parse với Tree-sitter để chạy queries
            tree = self.parser.parse(bytes(code, 'utf8'))
            
            if tree.root_node.has_error:
                result['static_issues']['code_smells'].append({
                    'type': 'syntax_error',
                    'message': 'Code có syntax errors',
                    'line': 1
                })
                self.logger.warning(f"Syntax errors in {filename}")
                return result
            
            # Chạy các queries
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
            
            # Tính metrics
            result['metrics'] = self._calculate_metrics(
                tree.root_node, code, result['ast_analysis']
            )
            
            # Generate suggestions
            result['suggestions'] = self._generate_suggestions(result)
            
            self.logger.debug(f"Static analysis completed for {filename}")
            
        except Exception as e:
            error_msg = f"Static analysis error in {filename}: {e}"
            result['static_issues']['code_smells'].append({
                'type': 'analysis_error',
                'message': error_msg,
                'line': 1
            })
            self.logger.error(error_msg)
        
        return result
    
    def _check_missing_docstrings_simple(self, root_node: Node, code: str) -> List[Dict]:
        """Kiểm tra functions và classes thiếu docstring sử dụng simple approach"""
        issues = []
        
        # Get all functions
        function_captures = self.function_query.captures(root_node)
        
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
        class_captures = self.class_query.captures(root_node)
        
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
        captures = self.import_query.captures(root_node)
        
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
        function_captures = self.function_query.captures(root_node)
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
        """Generate suggestions dựa trên kết quả phân tích"""
        suggestions = []
        
        issues = analysis_result['static_issues']
        metrics = analysis_result['metrics']
        
        # Suggestions for missing docstrings
        if issues['missing_docstrings']:
            suggestions.append(
                f"Thêm docstrings cho {len(issues['missing_docstrings'])} functions/classes "
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
            suggestions.append(f"Chia {len(long_lines)} long lines để tuân thủ PEP 8")
        
        god_classes = [s for s in code_smells if s['type'] == 'god_class']
        if god_classes:
            suggestions.append("Chia nhỏ large classes theo Single Responsibility Principle")
        
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
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze một file Python
        
        Args:
            file_path: Path đến file Python
            
        Returns:
            Dict chứa kết quả phân tích
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            return self.analyze_code(code, file_path)
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return {
                'filename': file_path,
                'error': str(e),
                'static_issues': {'code_smells': [{'type': 'file_error', 'message': str(e), 'line': 1}]},
                'metrics': {},
                'suggestions': []
            }
    
    def analyze_repository(self, code_fetcher_agent, repo_url: str) -> Dict[str, Any]:
        """
        Analyze toàn bộ Python repository
        
        Args:
            code_fetcher_agent: Instance của CodeFetcherAgent
            repo_url: Repository URL
            
        Returns:
            Dict chứa kết quả phân tích repository
        """
        results = {
            'repository': repo_url,
            'files_analyzed': [],
            'summary': {
                'total_files': 0,
                'total_issues': 0,
                'total_suggestions': 0,
                'average_quality_score': 0,
                'issue_breakdown': {
                    'missing_docstrings': 0,
                    'unused_imports': 0,
                    'complex_functions': 0,
                    'code_smells': 0
                }
            },
            'repository_suggestions': [],
            'analysis_timestamp': None
        }
        
        try:
            from datetime import datetime
            results['analysis_timestamp'] = datetime.now().isoformat()
            
            # Get Python files từ repository
            self.logger.info(f"Starting static analysis of repository: {repo_url}")
            files = code_fetcher_agent.list_repository_files(repo_url)
            python_files = [f for f in files if f.endswith('.py')]
            
            self.logger.info(f"Found {len(python_files)} Python files to analyze")
            
            quality_scores = []
            
            for file_path in python_files:
                try:
                    content = code_fetcher_agent.get_file_content(repo_url, file_path)
                    if content:
                        analysis = self.analyze_code(content, file_path)
                        
                        results['files_analyzed'].append({
                            'file_path': file_path,
                            'analysis': analysis
                        })
                        
                        # Update summary
                        results['summary']['total_files'] += 1
                        
                        # Count issues
                        issues = analysis['static_issues']
                        file_issue_count = (
                            len(issues['missing_docstrings']) +
                            len(issues['unused_imports']) +
                            len(issues['complex_functions']) +
                            len(issues['code_smells'])
                        )
                        results['summary']['total_issues'] += file_issue_count
                        results['summary']['total_suggestions'] += len(analysis['suggestions'])
                        
                        # Update issue breakdown
                        results['summary']['issue_breakdown']['missing_docstrings'] += len(issues['missing_docstrings'])
                        results['summary']['issue_breakdown']['unused_imports'] += len(issues['unused_imports'])
                        results['summary']['issue_breakdown']['complex_functions'] += len(issues['complex_functions'])
                        results['summary']['issue_breakdown']['code_smells'] += len(issues['code_smells'])
                        
                        # Track quality scores
                        quality_score = analysis['metrics'].get('code_quality_score', 0)
                        quality_scores.append(quality_score)
                        
                except Exception as e:
                    self.logger.error(f"Error analyzing {file_path}: {e}")
            
            # Calculate average quality score
            if quality_scores:
                results['summary']['average_quality_score'] = sum(quality_scores) / len(quality_scores)
            
            # Generate repository-level suggestions
            results['repository_suggestions'] = self._generate_repository_suggestions(results)
            
            self.logger.info(f"Static analysis completed: {results['summary']}")
            
        except Exception as e:
            self.logger.error(f"Error analyzing repository: {e}")
            results['error'] = str(e)
        
        return results
    
    def _generate_repository_suggestions(self, repo_results: Dict) -> List[str]:
        """Generate suggestions cho toàn bộ repository"""
        suggestions = []
        summary = repo_results['summary']
        
        if summary['average_quality_score'] < 60:
            suggestions.append("Repository cần improvement tổng thể về code quality")
        
        if summary['issue_breakdown']['missing_docstrings'] > summary['total_files'] * 0.5:
            suggestions.append("Thiết lập documentation standards và thêm docstrings")
        
        if summary['issue_breakdown']['unused_imports'] > 10:
            suggestions.append("Sử dụng tools như autoflake để tự động xóa unused imports")
        
        if summary['issue_breakdown']['complex_functions'] > 5:
            suggestions.append("Implement code review process để catch complex functions sớm")
        
        if summary['total_issues'] > summary['total_files'] * 3:
            suggestions.append("Cân nhắc setup linting tools (flake8, pylint) trong CI/CD")
        
        return suggestions


def demo_static_analysis():
    """Demo function để test StaticAnalysisAgent"""
    sample_code = '''
import os
import sys
import unused_module
from typing import List, Dict
from collections import defaultdict

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
        # This is a very long line that exceeds the recommended line length limit and should be flagged by the static analyzer
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
    
    print("🔍 === Static Analysis Demo ===")
    print(f"File: {result['filename']}")
    print(f"Quality Score: {result['metrics']['code_quality_score']:.1f}/100")
    print(f"Maintainability Index: {result['metrics']['maintainability_index']:.1f}/100")
    print()
    
    print("📋 Issues Found:")
    for category, issues in result['static_issues'].items():
        if issues:
            print(f"\n  {category.replace('_', ' ').title()}:")
            for issue in issues:
                print(f"    - Line {issue.get('line', '?')}: {issue['message']}")
    
    print(f"\n💡 Suggestions ({len(result['suggestions'])}):")
    for suggestion in result['suggestions']:
        print(f"  - {suggestion}")
    
    print(f"\n📊 Metrics:")
    for metric, value in result['metrics'].items():
        if isinstance(value, float):
            print(f"  - {metric.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"  - {metric.replace('_', ' ').title()}: {value}")


if __name__ == "__main__":
    demo_static_analysis() 