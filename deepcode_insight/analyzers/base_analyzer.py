"""
Base analyzer class for code analysis

Cung cấp foundation chung cho tất cả các analyzer
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from tree_sitter import Node, Parser
import logging

from ..core.interfaces import CodeAnalyzer, AnalysisResult, AnalysisLanguage
from ..core.utils import (
    detect_language_from_filename, normalize_line_endings,
    calculate_complexity_score, format_issue_message
)
from ..parsers.tree_sitter_queries import get_query_manager


class BaseCodeAnalyzer(CodeAnalyzer):
    """Base class for all code analyzers"""
    
    def __init__(self, language: AnalysisLanguage):
        self.language = language
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.query_manager = get_query_manager()
        
        if not self.query_manager.supports_language(language):
            raise ValueError(f"Language {language} not supported")
        
        self.parser = Parser(self.query_manager.get_language(language))
    
    def analyze(self, code: str, filename: str) -> AnalysisResult:
        """Analyze code and return standardized results"""
        # Normalize code
        code = normalize_line_endings(code)
        
        # Create result container
        result = AnalysisResult(filename, self.language)
        
        try:
            # Parse code to AST
            tree = self.parser.parse(bytes(code, 'utf8'))
            root_node = tree.root_node
            
            # Run language-specific analysis
            self._analyze_syntax(root_node, code, result)
            self._analyze_style(root_node, code, result)
            self._analyze_complexity(root_node, code, result)
            self._generate_suggestions(result)
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {filename}: {e}")
            result.success = False
            result.error = str(e)
        
        return result
    
    def supports_language(self, language: AnalysisLanguage) -> bool:
        """Check if analyzer supports the given language"""
        return language == self.language
    
    @abstractmethod
    def _analyze_syntax(self, root_node: Node, code: str, result: AnalysisResult):
        """Analyze syntax-related issues"""
        pass
    
    @abstractmethod
    def _analyze_style(self, root_node: Node, code: str, result: AnalysisResult):
        """Analyze style-related issues"""
        pass
    
    @abstractmethod
    def _analyze_complexity(self, root_node: Node, code: str, result: AnalysisResult):
        """Analyze complexity metrics"""
        pass
    
    def _generate_suggestions(self, result: AnalysisResult):
        """Generate suggestions based on analysis results"""
        suggestions = []
        
        # Count issues by type
        issue_counts = {}
        for issue in result.issues:
            issue_type = issue['type']
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        # Generate suggestions based on issue patterns
        if issue_counts.get('missing_docstring', 0) > 0:
            suggestions.append("Consider adding docstrings to improve code documentation")
        
        if issue_counts.get('complex_function', 0) > 0:
            suggestions.append("Consider breaking down complex functions into smaller, more manageable pieces")
        
        if issue_counts.get('naming_violation', 0) > 0:
            suggestions.append("Review naming conventions to improve code readability")
        
        # Add complexity-based suggestions
        complexity_score = result.metrics.get('complexity_score', 0)
        if complexity_score > 0.7:
            suggestions.append("High complexity detected - consider refactoring for better maintainability")
        elif complexity_score > 0.5:
            suggestions.append("Moderate complexity - monitor for potential refactoring opportunities")
        
        for suggestion in suggestions:
            result.add_suggestion(suggestion)
    
    def _get_node_text(self, node: Node, code: str) -> str:
        """Extract text content from a node"""
        return code[node.start_byte:node.end_byte]
    
    def _get_line_number(self, node: Node) -> int:
        """Get line number for a node (1-indexed)"""
        return node.start_point[0] + 1
    
    def _count_child_nodes(self, node: Node, node_type: str) -> int:
        """Count child nodes of a specific type"""
        count = 0
        
        def traverse(n):
            nonlocal count
            if n.type == node_type:
                count += 1
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return count
    
    def _find_nodes_by_type(self, root_node: Node, node_type: str) -> List[Node]:
        """Find all nodes of a specific type"""
        nodes = []
        
        def traverse(node):
            if node.type == node_type:
                nodes.append(node)
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return nodes
    
    def _calculate_cyclomatic_complexity(self, node: Node) -> int:
        """Calculate cyclomatic complexity for a function/method node"""
        # Base complexity is 1
        complexity = 1
        
        # Decision points that increase complexity
        decision_points = [
            'if_statement', 'elif_clause', 'else_clause',
            'for_statement', 'while_statement', 'do_statement',
            'try_statement', 'except_clause', 'catch_clause',
            'case_statement', 'switch_statement',
            'conditional_expression', 'ternary_expression'
        ]
        
        def traverse(n):
            nonlocal complexity
            if n.type in decision_points:
                complexity += 1
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return complexity
    
    def _get_function_parameters_count(self, func_node: Node, code: str) -> int:
        """Count parameters in a function"""
        # This is a simplified implementation
        # Subclasses should override for language-specific logic
        return 0
    
    def _calculate_nesting_depth(self, node: Node) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        
        def traverse(n, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            # Nodes that increase nesting depth
            nesting_nodes = [
                'if_statement', 'for_statement', 'while_statement',
                'try_statement', 'with_statement', 'function_definition',
                'class_definition', 'method_declaration'
            ]
            
            next_depth = depth + 1 if n.type in nesting_nodes else depth
            
            for child in n.children:
                traverse(child, next_depth)
        
        traverse(node)
        return max_depth 