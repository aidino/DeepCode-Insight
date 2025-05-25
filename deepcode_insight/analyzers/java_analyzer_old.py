"""
Java-specific code analyzer

Analyzer chuyên biệt cho Java code
"""

import re
from typing import Dict, List, Any, Optional
from tree_sitter import Node

from ..core.interfaces import AnalysisResult, AnalysisLanguage
from .base_analyzer import BaseCodeAnalyzer


class JavaAnalyzer(BaseCodeAnalyzer):
    """Java-specific code analyzer"""
    
    def __init__(self):
        super().__init__(AnalysisLanguage.JAVA)
        
        # Java naming conventions
        self.camel_case_pattern = re.compile(r'^[a-z][a-zA-Z0-9]*$')
        self.pascal_case_pattern = re.compile(r'^[A-Z][a-zA-Z0-9]*$')
        self.constant_pattern = re.compile(r'^[A-Z_][A-Z0-9_]*$')
        self.package_pattern = re.compile(r'^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$')
    
    def _analyze_syntax(self, root_node: Node, code: str, result: AnalysisResult):
        """Analyze Java syntax issues"""
        # Check for missing Javadoc
        self._check_missing_javadoc(root_node, code, result)
        
        # Check for empty catch blocks
        self._check_empty_catch_blocks(root_node, code, result)
        
        # Check for unused imports (simplified)
        self._check_unused_imports(root_node, code, result)
        
        # Check for magic numbers
        self._check_magic_numbers(root_node, code, result)
    
    def _analyze_style(self, root_node: Node, code: str, result: AnalysisResult):
        """Analyze Java style issues"""
        # Check naming conventions
        self._check_naming_conventions(root_node, code, result)
        
        # Check for long parameter lists
        self._check_long_parameter_lists(root_node, code, result)
        
        # Check for large classes
        self._check_large_classes(root_node, code, result)
    
    def _analyze_complexity(self, root_node: Node, code: str, result: AnalysisResult):
        """Analyze Java complexity metrics"""
        # Find all methods
        methods_query = self.query_manager.get_query(self.language, 'functions')
        method_matches = methods_query.matches(root_node)
        
        total_complexity = 0
        method_count = 0
        max_complexity = 0
        
        for pattern_index, captures in method_matches:
            if 'function' in captures:
                for node in captures['function']:
                    complexity = self._calculate_cyclomatic_complexity(node)
                    total_complexity += complexity
                    method_count += 1
                    max_complexity = max(max_complexity, complexity)
                    
                    # Check for complex methods
                    if complexity > 10:
                        method_name = self._get_method_name(node, code)
                        result.add_issue(
                            'complex_method',
                            f"Method '{method_name}' has high cyclomatic complexity ({complexity})",
                            self._get_line_number(node),
                            'warning',
                            complexity=complexity,
                            method_name=method_name
                        )
        
        # Calculate metrics
        avg_complexity = total_complexity / method_count if method_count > 0 else 0
        
        result.add_metric('total_methods', method_count)
        result.add_metric('average_complexity', avg_complexity)
        result.add_metric('max_complexity', max_complexity)
        result.add_metric('total_complexity', total_complexity)
        
        # Calculate overall complexity score
        lines = code.split('\n')
        result.add_metric('lines_of_code', len([line for line in lines if line.strip()]))
        
        complexity_metrics = {
            'cyclomatic_complexity': avg_complexity,
            'lines_of_code': len(lines),
            'method_count': method_count
        }
        
        complexity_score = self._calculate_complexity_score(complexity_metrics)
        result.add_metric('complexity_score', complexity_score)
    
    def _check_missing_javadoc(self, root_node: Node, code: str, result: AnalysisResult):
        """Check for missing Javadoc in public methods and classes"""
        # Check public methods
        methods_query = self.query_manager.get_query(self.language, 'functions')
        method_captures = methods_query.captures(root_node)
        
        for capture in method_captures:
            node, capture_name = capture
            if capture_name == 'function':
                if self._is_public_method(node, code) and not self._has_javadoc(node, code):
                    method_name = self._get_method_name(node, code)
                    result.add_issue(
                        'missing_javadoc',
                        f"Public method '{method_name}' is missing Javadoc",
                        self._get_line_number(node),
                        'info',
                        element_type='method',
                        name=method_name
                    )
        
        # Check public classes
        classes_query = self.query_manager.get_query(self.language, 'classes')
        class_captures = classes_query.captures(root_node)
        
        for capture in class_captures:
            node, capture_name = capture
            if capture_name == 'class':
                if self._is_public_class(node, code) and not self._has_javadoc(node, code):
                    class_name = self._get_class_name(node, code)
                    result.add_issue(
                        'missing_javadoc',
                        f"Public class '{class_name}' is missing Javadoc",
                        self._get_line_number(node),
                        'info',
                        element_type='class',
                        name=class_name
                    )
    
    def _check_empty_catch_blocks(self, root_node: Node, code: str, result: AnalysisResult):
        """Check for empty catch blocks"""
        try:
            catch_query = self.query_manager.get_query(self.language, 'catch_blocks')
            captures = catch_query.captures(root_node)
            
            for capture in captures:
                node, capture_name = capture
                if capture_name == 'catch_block':
                    block_text = self._get_node_text(node, code).strip()
                    # Check if catch block is empty or only contains comments
                    if self._is_empty_block(block_text):
                        result.add_issue(
                            'empty_catch_block',
                            "Empty catch block - consider logging or handling the exception",
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
        
        imported_classes = set()
        
        for capture in import_captures:
            node, capture_name = capture
            if capture_name == 'import':
                import_text = self._get_node_text(node, code)
                # Extract class name from import statement
                if 'import' in import_text:
                    parts = import_text.replace(';', '').split()
                    if len(parts) >= 2:
                        full_class_name = parts[-1]
                        class_name = full_class_name.split('.')[-1]
                        imported_classes.add(class_name)
        
        # Check if imported classes are used in the code
        for class_name in imported_classes:
            if class_name not in code or code.count(class_name) <= 1:
                result.add_issue(
                    'unused_import',
                    f"Import '{class_name}' appears to be unused",
                    1,  # Line number not easily determinable in this simplified version
                    'info',
                    import_name=class_name
                )
    
    def _check_magic_numbers(self, root_node: Node, code: str, result: AnalysisResult):
        """Check for magic numbers in the code"""
        # Simple regex to find numeric literals (excluding 0, 1, -1)
        magic_number_pattern = re.compile(r'\b(?!0\b|1\b|-1\b)\d{2,}\b')
        
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            matches = magic_number_pattern.findall(line)
            for match in matches:
                # Skip if it's in a comment or string
                if '//' in line and line.index('//') < line.index(match):
                    continue
                if '/*' in line or '*/' in line:
                    continue
                if '"' in line or "'" in line:
                    # Simple check - more sophisticated string detection needed
                    continue
                
                result.add_issue(
                    'magic_number',
                    f"Magic number '{match}' should be replaced with a named constant",
                    line_num,
                    'info',
                    number=match
                )
    
    def _check_naming_conventions(self, root_node: Node, code: str, result: AnalysisResult):
        """Check Java naming conventions"""
        # Check method names (should be camelCase)
        methods_query = self.query_manager.get_query(self.language, 'functions')
        method_captures = methods_query.captures(root_node)
        
        for capture in method_captures:
            node, capture_name = capture
            if capture_name == 'function':
                method_name = self._get_method_name(node, code)
                if method_name and not self.camel_case_pattern.match(method_name):
                    result.add_issue(
                        'naming_convention',
                        f"Method '{method_name}' should use camelCase naming",
                        self._get_line_number(node),
                        'info',
                        element_type='method',
                        name=method_name,
                        expected_pattern='camelCase'
                    )
        
        # Check class names (should be PascalCase)
        classes_query = self.query_manager.get_query(self.language, 'classes')
        class_captures = classes_query.captures(root_node)
        
        for capture in class_captures:
            node, capture_name = capture
            if capture_name == 'class':
                class_name = self._get_class_name(node, code)
                if class_name and not self.pascal_case_pattern.match(class_name):
                    result.add_issue(
                        'naming_convention',
                        f"Class '{class_name}' should use PascalCase naming",
                        self._get_line_number(node),
                        'info',
                        element_type='class',
                        name=class_name,
                        expected_pattern='PascalCase'
                    )
        
        # Check variable names (should be camelCase)
        variables_query = self.query_manager.get_query(self.language, 'variables')
        variable_captures = variables_query.captures(root_node)
        
        for capture in variable_captures:
            node, capture_name = capture
            if capture_name == 'variable':
                var_name = self._get_node_text(node, code)
                if var_name and not self.camel_case_pattern.match(var_name) and not self.constant_pattern.match(var_name):
                    result.add_issue(
                        'naming_convention',
                        f"Variable '{var_name}' should use camelCase naming",
                        self._get_line_number(node),
                        'info',
                        element_type='variable',
                        name=var_name,
                        expected_pattern='camelCase'
                    )
    
    def _check_long_parameter_lists(self, root_node: Node, code: str, result: AnalysisResult):
        """Check for methods with too many parameters"""
        methods_query = self.query_manager.get_query(self.language, 'functions')
        method_captures = methods_query.captures(root_node)
        
        for capture in method_captures:
            node, capture_name = capture
            if capture_name == 'function':
                param_count = self._count_parameters(node, code)
                if param_count > 5:  # Threshold for too many parameters
                    method_name = self._get_method_name(node, code)
                    result.add_issue(
                        'long_parameter_list',
                        f"Method '{method_name}' has too many parameters ({param_count})",
                        self._get_line_number(node),
                        'warning',
                        parameter_count=param_count,
                        method_name=method_name
                    )
    
    def _check_large_classes(self, root_node: Node, code: str, result: AnalysisResult):
        """Check for classes that are too large"""
        classes_query = self.query_manager.get_query(self.language, 'classes')
        class_captures = classes_query.captures(root_node)
        
        for capture in class_captures:
            node, capture_name = capture
            if capture_name == 'class':
                class_text = self._get_node_text(node, code)
                line_count = len(class_text.split('\n'))
                
                if line_count > 200:  # Threshold for large class
                    class_name = self._get_class_name(node, code)
                    result.add_issue(
                        'large_class',
                        f"Class '{class_name}' is too large ({line_count} lines)",
                        self._get_line_number(node),
                        'warning',
                        line_count=line_count,
                        class_name=class_name
                    )
    
    def _is_public_method(self, method_node: Node, code: str) -> bool:
        """Check if a method is public"""
        method_text = self._get_node_text(method_node, code)
        return 'public' in method_text
    
    def _is_public_class(self, class_node: Node, code: str) -> bool:
        """Check if a class is public"""
        class_text = self._get_node_text(class_node, code)
        return 'public' in class_text
    
    def _has_javadoc(self, node: Node, code: str) -> bool:
        """Check if a node has Javadoc comment"""
        # Look for /** comment before the node
        lines = code.split('\n')
        node_line = self._get_line_number(node)
        
        # Check a few lines before the node for Javadoc
        for i in range(max(0, node_line - 5), node_line):
            if i < len(lines):
                line = lines[i].strip()
                if line.startswith('/**'):
                    return True
        
        return False
    
    def _is_empty_block(self, block_text: str) -> bool:
        """Check if a code block is empty (ignoring comments)"""
        lines = block_text.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('//') and not stripped.startswith('/*') and not stripped.startswith('*'):
                if stripped not in ['{', '}']:
                    return False
        return True
    
    def _get_method_name(self, method_node: Node, code: str) -> str:
        """Extract method name from method node"""
        # This is a simplified implementation
        method_text = self._get_node_text(method_node, code)
        lines = method_text.split('\n')
        for line in lines:
            if '(' in line and not line.strip().startswith('//'):
                # Extract method name before the opening parenthesis
                parts = line.split('(')[0].split()
                if parts:
                    return parts[-1]
        return "unknown"
    
    def _get_class_name(self, class_node: Node, code: str) -> str:
        """Extract class name from class node"""
        # This is a simplified implementation
        class_text = self._get_node_text(class_node, code)
        lines = class_text.split('\n')
        for line in lines:
            if 'class' in line and not line.strip().startswith('//'):
                parts = line.split()
                class_index = -1
                for i, part in enumerate(parts):
                    if part == 'class':
                        class_index = i
                        break
                if class_index >= 0 and class_index + 1 < len(parts):
                    return parts[class_index + 1]
        return "unknown"
    
    def _count_parameters(self, method_node: Node, code: str) -> int:
        """Count the number of parameters in a method"""
        method_text = self._get_node_text(method_node, code)
        
        # Find the parameter list
        start_paren = method_text.find('(')
        end_paren = method_text.find(')', start_paren)
        
        if start_paren >= 0 and end_paren > start_paren:
            param_text = method_text[start_paren + 1:end_paren].strip()
            if not param_text:
                return 0
            
            # Simple parameter counting (split by comma)
            params = param_text.split(',')
            return len([p for p in params if p.strip()])
        
        return 0
    
    def _calculate_complexity_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall complexity score for Java code"""
        cyclomatic = metrics.get('cyclomatic_complexity', 0)
        lines = metrics.get('lines_of_code', 1)
        methods = metrics.get('method_count', 1)
        
        # Weighted complexity score
        score = (cyclomatic * 0.4) + (lines / 100 * 0.3) + (methods / 10 * 0.3)
        return min(score, 10.0)  # Cap at 10 