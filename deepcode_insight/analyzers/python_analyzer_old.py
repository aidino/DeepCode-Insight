"""
Python-specific code analyzer

Analyzer chuyên biệt cho Python code
"""

import re
from typing import Dict, List, Any, Optional
from tree_sitter import Node

from ..core.interfaces import AnalysisResult, AnalysisLanguage
from .base_analyzer import BaseCodeAnalyzer


class PythonAnalyzer(BaseCodeAnalyzer):
    """Python-specific code analyzer"""
    
    def __init__(self):
        super().__init__(AnalysisLanguage.PYTHON)
        
        # Python naming conventions
        self.snake_case_pattern = re.compile(r'^[a-z_][a-z0-9_]*$')
        self.pascal_case_pattern = re.compile(r'^[A-Z][a-zA-Z0-9]*$')
        self.constant_pattern = re.compile(r'^[A-Z_][A-Z0-9_]*$')
    
    def _analyze_syntax(self, root_node: Node, code: str, result: AnalysisResult):
        """Analyze Python syntax issues"""
        # Check for missing docstrings
        self._check_missing_docstrings(root_node, code, result)
        
        # Check for bare except clauses
        self._check_bare_except(root_node, code, result)
        
        # Check for unused imports (simplified)
        self._check_unused_imports(root_node, code, result)
    
    def _analyze_style(self, root_node: Node, code: str, result: AnalysisResult):
        """Analyze Python style issues"""
        # Check naming conventions
        self._check_naming_conventions(root_node, code, result)
        
        # Check for lambda usage
        self._check_lambda_usage(root_node, code, result)
        
        # Check for comprehension complexity
        self._check_comprehension_complexity(root_node, code, result)
    
    def _analyze_complexity(self, root_node: Node, code: str, result: AnalysisResult):
        """Analyze Python complexity metrics"""
        # Find all functions
        functions_query = self.query_manager.get_query(self.language, 'functions')
        function_matches = functions_query.matches(root_node)
        
        total_complexity = 0
        function_count = 0
        max_complexity = 0
        
        for pattern_index, captures in function_matches:
            if 'function' in captures:
                for node in captures['function']:
                    complexity = self._calculate_cyclomatic_complexity(node)
                    total_complexity += complexity
                    function_count += 1
                    max_complexity = max(max_complexity, complexity)
                    
                    # Check for complex functions
                    if complexity > 10:
                        func_name = self._get_function_name(node, code)
                        result.add_issue(
                            'complex_function',
                            f"Function '{func_name}' has high cyclomatic complexity ({complexity})",
                            self._get_line_number(node),
                            'warning',
                            complexity=complexity,
                            function_name=func_name
                        )
        
        # Calculate metrics
        avg_complexity = total_complexity / function_count if function_count > 0 else 0
        
        result.add_metric('total_functions', function_count)
        result.add_metric('average_complexity', avg_complexity)
        result.add_metric('max_complexity', max_complexity)
        result.add_metric('total_complexity', total_complexity)
        
        # Calculate overall complexity score
        lines = code.split('\n')
        result.add_metric('lines_of_code', len([line for line in lines if line.strip()]))
        
        complexity_metrics = {
            'cyclomatic_complexity': avg_complexity,
            'lines_of_code': len(lines),
            'function_count': function_count
        }
        
        complexity_score = self._calculate_complexity_score(complexity_metrics)
        result.add_metric('complexity_score', complexity_score)
    
    def _check_missing_docstrings(self, root_node: Node, code: str, result: AnalysisResult):
        """Check for missing docstrings in functions and classes"""
        # Check functions
        functions_query = self.query_manager.get_query(self.language, 'functions')
        function_captures = functions_query.captures(root_node)
        
        for capture in function_captures:
            node, capture_name = capture
            if capture_name == 'function':
                if not self._has_docstring(node, code):
                    func_name = self._get_function_name(node, code)
                    result.add_issue(
                        'missing_docstring',
                        f"Function '{func_name}' is missing a docstring",
                        self._get_line_number(node),
                        'info',
                        element_type='function',
                        name=func_name
                    )
        
        # Check classes
        classes_query = self.query_manager.get_query(self.language, 'classes')
        class_captures = classes_query.captures(root_node)
        
        for capture in class_captures:
            node, capture_name = capture
            if capture_name == 'class':
                if not self._has_docstring(node, code):
                    class_name = self._get_class_name(node, code)
                    result.add_issue(
                        'missing_docstring',
                        f"Class '{class_name}' is missing a docstring",
                        self._get_line_number(node),
                        'info',
                        element_type='class',
                        name=class_name
                    )
    
    def _check_bare_except(self, root_node: Node, code: str, result: AnalysisResult):
        """Check for bare except clauses"""
        try:
            bare_except_query = self.query_manager.get_query(self.language, 'bare_except')
            captures = bare_except_query.captures(root_node)
            
            for capture in captures:
                node, capture_name = capture
                if capture_name == 'bare_except':
                    result.add_issue(
                        'bare_except',
                        "Bare except clause - consider catching specific exceptions",
                        self._get_line_number(node),
                        'warning'
                    )
        except ValueError:
            # Query not available, skip this check
            pass
    
    def _check_unused_imports(self, root_node: Node, code: str, result: AnalysisResult):
        """Check for potentially unused imports (simplified)"""
        imports_query = self.query_manager.get_query(self.language, 'imports')
        import_captures = imports_query.captures(root_node)
        
        imported_names = set()
        
        for capture in import_captures:
            node, capture_name = capture
            if capture_name in ['import', 'from_import']:
                import_text = self._get_node_text(node, code)
                # Extract imported names (simplified)
                if 'import' in import_text:
                    parts = import_text.split()
                    if 'as' in parts:
                        # Handle 'import x as y' or 'from x import y as z'
                        as_index = parts.index('as')
                        if as_index + 1 < len(parts):
                            imported_names.add(parts[as_index + 1])
                    else:
                        # Handle simple imports
                        for part in parts:
                            if part not in ['import', 'from', ',']:
                                imported_names.add(part.split('.')[0])
        
        # Check if imported names are used in the code
        # This is a simplified check - a more sophisticated implementation
        # would use proper scope analysis
        for name in imported_names:
            if name not in code or code.count(name) <= 1:
                result.add_issue(
                    'unused_import',
                    f"Import '{name}' appears to be unused",
                    1,  # Line number not easily determinable in this simplified version
                    'info',
                    import_name=name
                )
    
    def _check_naming_conventions(self, root_node: Node, code: str, result: AnalysisResult):
        """Check Python naming conventions"""
        # Check function names (should be snake_case)
        functions_query = self.query_manager.get_query(self.language, 'functions')
        function_captures = functions_query.captures(root_node)
        
        for capture in function_captures:
            node, capture_name = capture
            if capture_name == 'func_name':
                func_name = self._get_node_text(node, code)
                if not self.snake_case_pattern.match(func_name) and not func_name.startswith('_'):
                    result.add_issue(
                        'naming_violation',
                        f"Function '{func_name}' should use snake_case naming",
                        self._get_line_number(node),
                        'info',
                        element_type='function',
                        name=func_name,
                        convention='snake_case'
                    )
        
        # Check class names (should be PascalCase)
        classes_query = self.query_manager.get_query(self.language, 'classes')
        class_captures = classes_query.captures(root_node)
        
        for capture in class_captures:
            node, capture_name = capture
            if capture_name == 'class_name':
                class_name = self._get_node_text(node, code)
                if not self.pascal_case_pattern.match(class_name):
                    result.add_issue(
                        'naming_violation',
                        f"Class '{class_name}' should use PascalCase naming",
                        self._get_line_number(node),
                        'info',
                        element_type='class',
                        name=class_name,
                        convention='PascalCase'
                    )
    
    def _check_lambda_usage(self, root_node: Node, code: str, result: AnalysisResult):
        """Check for complex lambda expressions"""
        try:
            lambda_query = self.query_manager.get_query(self.language, 'lambda_functions')
            lambda_captures = lambda_query.captures(root_node)
            
            for capture in lambda_captures:
                node, capture_name = capture
                if capture_name == 'lambda_expr':
                    lambda_text = self._get_node_text(node, code)
                    # Check if lambda is too complex (simple heuristic)
                    if len(lambda_text) > 50 or lambda_text.count(':') > 1:
                        result.add_issue(
                            'complex_lambda',
                            "Complex lambda expression - consider using a regular function",
                            self._get_line_number(node),
                            'info'
                        )
        except ValueError:
            # Query not available, skip this check
            pass
    
    def _check_comprehension_complexity(self, root_node: Node, code: str, result: AnalysisResult):
        """Check for complex comprehensions"""
        try:
            comp_query = self.query_manager.get_query(self.language, 'comprehensions')
            comp_captures = comp_query.captures(root_node)
            
            for capture in comp_captures:
                node, capture_name = capture
                comp_text = self._get_node_text(node, code)
                # Check if comprehension is too complex (simple heuristic)
                if len(comp_text) > 80 or comp_text.count('for') > 1 or comp_text.count('if') > 1:
                    result.add_issue(
                        'complex_comprehension',
                        "Complex comprehension - consider breaking into multiple lines or using a loop",
                        self._get_line_number(node),
                        'info'
                    )
        except ValueError:
            # Query not available, skip this check
            pass
    
    def _has_docstring(self, node: Node, code: str) -> bool:
        """Check if a function or class has a docstring"""
        # Look for the first statement in the body
        for child in node.children:
            if child.type == 'block':
                for stmt in child.children:
                    if stmt.type == 'expression_statement':
                        for expr in stmt.children:
                            if expr.type == 'string':
                                return True
                    elif stmt.type not in ['pass_statement', 'comment']:
                        # Found a non-docstring statement
                        return False
        return False
    
    def _get_function_name(self, func_node: Node, code: str) -> str:
        """Extract function name from function node"""
        for child in func_node.children:
            if child.type == 'identifier':
                return self._get_node_text(child, code)
        return "unknown"
    
    def _get_class_name(self, class_node: Node, code: str) -> str:
        """Extract class name from class node"""
        for child in class_node.children:
            if child.type == 'identifier':
                return self._get_node_text(child, code)
        return "unknown"
    
    def _calculate_complexity_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate complexity score specific to Python"""
        from ..core.utils import calculate_complexity_score
        return calculate_complexity_score(metrics) 