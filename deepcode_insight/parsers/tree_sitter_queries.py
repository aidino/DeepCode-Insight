"""
Tree-sitter queries for different programming languages

Tập trung hóa các Tree-sitter queries để tái sử dụng và dễ bảo trì
"""

from typing import Dict, List
from tree_sitter import Language, Query
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
from ..core.interfaces import AnalysisLanguage


class TreeSitterQueryManager:
    """Manager for Tree-sitter queries across different languages"""
    
    def __init__(self):
        self.languages = {}
        self.queries = {}
        self._init_languages()
        self._init_queries()
    
    def _init_languages(self):
        """Initialize Tree-sitter languages"""
        try:
            self.languages[AnalysisLanguage.PYTHON] = Language(tspython.language())
            self.languages[AnalysisLanguage.JAVA] = Language(tsjava.language())
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Tree-sitter languages: {e}")
    
    def _init_queries(self):
        """Initialize all queries for all languages"""
        self._init_python_queries()
        self._init_java_queries()
    
    def _init_python_queries(self):
        """Initialize Python-specific queries"""
        lang = self.languages[AnalysisLanguage.PYTHON]
        
        self.queries[AnalysisLanguage.PYTHON] = {
            'functions': Query(lang, """
                (function_definition
                    name: (identifier) @func_name
                    parameters: (parameters) @params
                    body: (block) @body
                ) @function
            """),
            
            'classes': Query(lang, """
                (class_definition
                    name: (identifier) @class_name
                    superclasses: (argument_list)? @superclasses
                    body: (block) @body
                ) @class
            """),
            
            'imports': Query(lang, """
                (import_statement) @import
                (import_from_statement) @from_import
            """),
            
            'variables': Query(lang, """
                (assignment
                    left: (identifier) @var_name
                    right: (_) @value
                ) @assignment
                (assignment
                    left: (pattern_list (identifier) @var_name)
                ) @multi_assignment
            """),
            
            'docstrings': Query(lang, """
                (function_definition
                    body: (block
                        (expression_statement
                            (string) @docstring
                        )
                    )
                ) @function_with_docstring
                (class_definition
                    body: (block
                        (expression_statement
                            (string) @docstring
                        )
                    )
                ) @class_with_docstring
            """),
            
            'exceptions': Query(lang, """
                (try_statement) @try_stmt
                (except_clause) @except_clause
                (raise_statement) @raise_stmt
            """),
            
            'comprehensions': Query(lang, """
                (list_comprehension) @list_comp
                (dictionary_comprehension) @dict_comp
                (set_comprehension) @set_comp
                (generator_expression) @gen_expr
            """),
            
            'lambda_functions': Query(lang, """
                (lambda) @lambda_expr
            """),
            
            'string_literals': Query(lang, """
                (string) @string_literal
            """),
            
            'if_statements': Query(lang, """
                (if_statement) @if_stmt
            """),
            
            'for_loops': Query(lang, """
                (for_statement) @for_loop
            """),
            
            'while_loops': Query(lang, """
                (while_statement) @while_loop
            """),
            
            'method_calls': Query(lang, """
                (call
                    function: (attribute
                        object: (_) @object
                        attribute: (identifier) @method_name
                    )
                    arguments: (argument_list) @args
                ) @method_call
            """),
            
            'bare_except': Query(lang, """
                (except_clause
                    !type
                ) @bare_except
            """)
        }
    
    def _init_java_queries(self):
        """Initialize Java-specific queries"""
        lang = self.languages[AnalysisLanguage.JAVA]
        
        self.queries[AnalysisLanguage.JAVA] = {
            'classes': Query(lang, """
                (class_declaration
                    name: (identifier) @class_name
                ) @class
            """),
            
            'functions': Query(lang, """
                (method_declaration
                    name: (identifier) @method_name
                ) @function
            """),
            
            'methods': Query(lang, """
                (method_declaration
                    name: (identifier) @method_name
                ) @method
            """),
            
            'constructors': Query(lang, """
                (constructor_declaration
                    name: (identifier) @constructor_name
                ) @constructor
            """),
            
            'fields': Query(lang, """
                (field_declaration) @field
            """),
            
            'imports': Query(lang, """
                (import_declaration) @import
            """),
            
            'exceptions': Query(lang, """
                (try_statement) @try_stmt
                (catch_clause) @catch_clause
                (throw_statement) @throw_stmt
            """),
            
            'variables': Query(lang, """
                (local_variable_declaration) @local_var
            """),
            
            'method_calls': Query(lang, """
                (method_invocation) @method_call
            """),
            
            'if_statements': Query(lang, """
                (if_statement) @if_stmt
            """),
            
            'for_loops': Query(lang, """
                (for_statement) @for_loop
                (enhanced_for_statement) @enhanced_for_loop
            """),
            
            'while_loops': Query(lang, """
                (while_statement) @while_loop
                (do_statement) @do_while_loop
            """),
            
            'annotations': Query(lang, """
                (annotation
                    name: (identifier) @annotation_name
                ) @annotation
            """),
            
            'interfaces': Query(lang, """
                (interface_declaration
                    name: (identifier) @interface_name
                ) @interface
            """),
            
            'empty_catch': Query(lang, """
                (catch_clause) @catch_clause
            """)
        }
    
    def get_query(self, language: AnalysisLanguage, query_name: str) -> Query:
        """Get a specific query for a language"""
        if language not in self.queries:
            raise ValueError(f"Language {language} not supported")
        
        if query_name not in self.queries[language]:
            raise ValueError(f"Query '{query_name}' not found for language {language}")
        
        return self.queries[language][query_name]
    
    def get_all_queries(self, language: AnalysisLanguage) -> Dict[str, Query]:
        """Get all queries for a language"""
        if language not in self.queries:
            raise ValueError(f"Language {language} not supported")
        
        return self.queries[language].copy()
    
    def get_language(self, language: AnalysisLanguage) -> Language:
        """Get Tree-sitter language object"""
        if language not in self.languages:
            raise ValueError(f"Language {language} not supported")
        
        return self.languages[language]
    
    def supports_language(self, language: AnalysisLanguage) -> bool:
        """Check if language is supported"""
        return language in self.languages
    
    def get_supported_languages(self) -> List[AnalysisLanguage]:
        """Get list of supported languages"""
        return list(self.languages.keys())


# Global instance
query_manager = TreeSitterQueryManager()


def get_query_manager() -> TreeSitterQueryManager:
    """Get the global query manager instance"""
    return query_manager


# Convenience functions
def get_python_queries() -> Dict[str, Query]:
    """Get all Python queries"""
    return query_manager.get_all_queries(AnalysisLanguage.PYTHON)


def get_java_queries() -> Dict[str, Query]:
    """Get all Java queries"""
    return query_manager.get_all_queries(AnalysisLanguage.JAVA)


def get_query(language: AnalysisLanguage, query_name: str) -> Query:
    """Get a specific query"""
    return query_manager.get_query(language, query_name)


def get_language(language: AnalysisLanguage) -> Language:
    """Get Tree-sitter language"""
    return query_manager.get_language(language) 