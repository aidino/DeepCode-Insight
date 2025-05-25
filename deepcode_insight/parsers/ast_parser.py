"""ASTParsingAgent - Parse Python và Java code với tree-sitter"""

import logging
from typing import Dict, List, Optional, Any, Union
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Node


class ASTParsingAgent:
    """Agent để parse Python và Java code thành AST và extract thông tin chi tiết"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            # Initialize languages
            self.python_language = Language(tspython.language())
            self.java_language = Language(tsjava.language())
            
            # Initialize parsers
            self.python_parser = Parser(self.python_language)
            self.java_parser = Parser(self.java_language)
            
            self.logger.info("ASTParsingAgent initialized successfully with Python and Java support")
        except Exception as e:
            self.logger.error(f"Failed to initialize ASTParsingAgent: {e}")
            raise
    
    def parse_code(self, code: str, filename: str = "<string>", language: str = "python") -> Dict[str, Any]:
        """
        Parse code string và extract thông tin chi tiết
        
        Args:
            code: Source code (Python hoặc Java)
            filename: Tên file (để tracking)
            language: Programming language ("python" hoặc "java")
            
        Returns:
            Dict chứa parsed information:
            {
                'filename': str,
                'language': str,
                'functions': List[Dict],
                'classes': List[Dict], 
                'imports': List[Dict],
                'variables': List[Dict],
                'decorators': List[Dict],  # Python only
                'errors': List[str],
                'stats': Dict[str, int]
            }
        """
        result = {
            'filename': filename,
            'language': language.lower(),
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
            
            # Parse based on language
            if language.lower() == "python":
                return self._parse_python_code(code, filename, result)
            elif language.lower() == "java":
                return self._parse_java_code(code, filename, result)
            else:
                result['errors'].append(f"Unsupported language: {language}")
                self.logger.error(f"Unsupported language: {language}")
                return result
                
        except Exception as e:
            error_msg = f"Parse error in {filename}: {e}"
            result['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        return result
    
    def _parse_python_code(self, code: str, filename: str, result: Dict) -> Dict:
        """Parse Python code"""
        # Parse code
        tree = self.python_parser.parse(bytes(code, 'utf8'))
        
        if tree.root_node.has_error:
            result['errors'].append("Syntax errors detected in code")
            self.logger.warning(f"Syntax errors in {filename}")
        
        # Extract information
        result['functions'] = self._extract_python_functions(tree.root_node, code)
        result['classes'] = self._extract_python_classes(tree.root_node, code)
        result['imports'] = self._extract_python_imports(tree.root_node, code)
        result['variables'] = self._extract_python_variables(tree.root_node, code)
        result['decorators'] = self._extract_python_decorators(tree.root_node, code)
        
        # Update stats
        result['stats'].update({
            'total_functions': len(result['functions']),
            'total_classes': len(result['classes']),
            'total_imports': len(result['imports']),
            'total_variables': len(result['variables'])
        })
        
        self.logger.debug(f"Parsed Python {filename}: {result['stats']}")
        return result
    
    def _parse_java_code(self, code: str, filename: str, result: Dict) -> Dict:
        """Parse Java code"""
        # Parse code
        tree = self.java_parser.parse(bytes(code, 'utf8'))
        
        if tree.root_node.has_error:
            result['errors'].append("Syntax errors detected in code")
            self.logger.warning(f"Syntax errors in {filename}")
        
        # Extract information
        result['functions'] = self._extract_java_methods(tree.root_node, code)
        result['classes'] = self._extract_java_classes(tree.root_node, code)
        result['imports'] = self._extract_java_imports(tree.root_node, code)
        result['variables'] = self._extract_java_fields(tree.root_node, code)
        # Java doesn't have decorators like Python
        result['decorators'] = []
        
        # Update stats
        result['stats'].update({
            'total_functions': len(result['functions']),
            'total_classes': len(result['classes']),
            'total_imports': len(result['imports']),
            'total_variables': len(result['variables'])
        })
        
        self.logger.debug(f"Parsed Java {filename}: {result['stats']}")
        return result

    # Python parsing methods (renamed for clarity)
    def _extract_python_functions(self, root_node: Node, code: str) -> List[Dict]:
        """Extract Python functions với thông tin chi tiết"""
        functions = []
        
        def traverse(node: Node, class_name: Optional[str] = None):
            if node.type == 'function_definition':
                func_info = self._parse_python_function_definition(node, code, class_name)
                if func_info:
                    functions.append(func_info)
            
            # Recursively traverse children
            for child in node.children:
                current_class = class_name
                if child.type == 'class_definition':
                    current_class = self._get_python_class_name(child, code)
                traverse(child, current_class)
        
        traverse(root_node)
        return functions
    
    def _extract_python_classes(self, root_node: Node, code: str) -> List[Dict]:
        """Extract Python classes với thông tin chi tiết"""
        classes = []
        
        def traverse(node: Node):
            if node.type == 'class_definition':
                class_info = self._parse_python_class_definition(node, code)
                if class_info:
                    classes.append(class_info)
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return classes
    
    def _extract_python_imports(self, root_node: Node, code: str) -> List[Dict]:
        """Extract Python imports với thông tin chi tiết"""
        imports = []
        
        def traverse(node: Node):
            if node.type in ['import_statement', 'import_from_statement']:
                import_info = self._parse_python_import_statement(node, code)
                if import_info:
                    imports.append(import_info)
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return imports
    
    def _extract_python_variables(self, root_node: Node, code: str) -> List[Dict]:
        """Extract Python global variables"""
        variables = []
        
        def traverse(node: Node, depth: int = 0):
            # Only get top-level assignments (global variables)
            if depth == 0 and node.type == 'assignment':
                var_info = self._parse_python_assignment(node, code)
                if var_info:
                    variables.append(var_info)
            
            for child in node.children:
                new_depth = depth + 1 if node.type in ['function_definition', 'class_definition'] else depth
                traverse(child, new_depth)
        
        traverse(root_node)
        return variables
    
    def _extract_python_decorators(self, root_node: Node, code: str) -> List[Dict]:
        """Extract Python decorators"""
        decorators = []
        
        def traverse(node: Node):
            if node.type == 'decorator':
                decorator_info = self._parse_python_decorator(node, code)
                if decorator_info:
                    decorators.append(decorator_info)
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return decorators

    # Java parsing methods
    def _extract_java_classes(self, root_node: Node, code: str) -> List[Dict]:
        """Extract Java classes và interfaces"""
        classes = []
        
        def traverse(node: Node):
            if node.type in ['class_declaration', 'interface_declaration', 'enum_declaration']:
                class_info = self._parse_java_class_declaration(node, code)
                if class_info:
                    classes.append(class_info)
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return classes
    
    def _extract_java_methods(self, root_node: Node, code: str) -> List[Dict]:
        """Extract Java methods"""
        methods = []
        
        def traverse(node: Node, class_name: Optional[str] = None):
            if node.type == 'method_declaration':
                method_info = self._parse_java_method_declaration(node, code, class_name)
                if method_info:
                    methods.append(method_info)
            elif node.type == 'constructor_declaration':
                constructor_info = self._parse_java_constructor_declaration(node, code, class_name)
                if constructor_info:
                    methods.append(constructor_info)
            
            # Recursively traverse children
            for child in node.children:
                current_class = class_name
                if child.type in ['class_declaration', 'interface_declaration']:
                    current_class = self._get_java_class_name(child, code)
                traverse(child, current_class)
        
        traverse(root_node)
        return methods
    
    def _extract_java_imports(self, root_node: Node, code: str) -> List[Dict]:
        """Extract Java imports"""
        imports = []
        
        def traverse(node: Node):
            if node.type == 'import_declaration':
                import_info = self._parse_java_import_declaration(node, code)
                if import_info:
                    imports.append(import_info)
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return imports
    
    def _extract_java_fields(self, root_node: Node, code: str) -> List[Dict]:
        """Extract Java fields (class variables)"""
        fields = []
        
        def traverse(node: Node, class_name: Optional[str] = None):
            if node.type == 'field_declaration':
                field_info = self._parse_java_field_declaration(node, code, class_name)
                if field_info:
                    fields.append(field_info)
            
            # Recursively traverse children
            for child in node.children:
                current_class = class_name
                if child.type in ['class_declaration', 'interface_declaration']:
                    current_class = self._get_java_class_name(child, code)
                traverse(child, current_class)
        
        traverse(root_node)
        return fields

    # Legacy methods for backward compatibility (renamed to Python-specific)
    def _extract_functions(self, root_node: Node, code: str) -> List[Dict]:
        """Legacy method - delegates to Python functions"""
        return self._extract_python_functions(root_node, code)
    
    def _extract_classes(self, root_node: Node, code: str) -> List[Dict]:
        """Legacy method - delegates to Python classes"""
        return self._extract_python_classes(root_node, code)
    
    def _extract_imports(self, root_node: Node, code: str) -> List[Dict]:
        """Legacy method - delegates to Python imports"""
        return self._extract_python_imports(root_node, code)
    
    def _extract_variables(self, root_node: Node, code: str) -> List[Dict]:
        """Legacy method - delegates to Python variables"""
        return self._extract_python_variables(root_node, code)
    
    def _extract_decorators(self, root_node: Node, code: str) -> List[Dict]:
        """Legacy method - delegates to Python decorators"""
        return self._extract_python_decorators(root_node, code)

    # Python parsing implementation methods
    def _parse_python_function_definition(self, node: Node, code: str, class_name: Optional[str] = None) -> Optional[Dict]:
        """Parse Python function definition với chi tiết"""
        try:
            name = self._get_python_function_name(node, code)
            if not name:
                return None
            
            # Get parameters
            parameters = self._get_python_function_parameters(node, code)
            
            # Get docstring
            docstring = self._get_python_function_docstring(node, code)
            
            # Get decorators
            decorators = self._get_python_function_decorators(node, code)
            
            # Get return type annotation if exists
            return_type = self._get_python_return_type_annotation(node, code)
            
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
            self.logger.warning(f"Error parsing Python function: {e}")
            return None
    
    def _parse_python_class_definition(self, node: Node, code: str) -> Optional[Dict]:
        """Parse Python class definition với chi tiết"""
        try:
            name = self._get_python_class_name(node, code)
            if not name:
                return None
            
            # Get base classes
            base_classes = self._get_python_base_classes(node, code)
            
            # Get docstring
            docstring = self._get_python_class_docstring(node, code)
            
            # Get decorators
            decorators = self._get_python_class_decorators(node, code)
            
            # Get methods
            methods = []
            for child in node.children:
                if child.type == 'block':
                    for grandchild in child.children:
                        if grandchild.type == 'function_definition':
                            method_info = self._parse_python_function_definition(grandchild, code, name)
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
            self.logger.warning(f"Error parsing Python class: {e}")
            return None
    
    def _parse_python_import_statement(self, node: Node, code: str) -> Optional[Dict]:
        """Parse Python import statement với chi tiết"""
        try:
            import_text = self._get_node_text(node, code).strip()
            
            if node.type == 'import_statement':
                # import module1, module2
                modules = []
                for child in node.children:
                    if child.type == 'dotted_as_names' or child.type == 'dotted_name':
                        modules.extend(self._extract_python_module_names(child, code))
                
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
                        names.extend(self._extract_python_import_names(child, code))
                
                return {
                    'type': 'from_import',
                    'module': module,
                    'names': names,
                    'text': import_text,
                    'line': node.start_point[0] + 1
                }
        except Exception as e:
            self.logger.warning(f"Error parsing Python import: {e}")
            return None
    
    def _parse_python_assignment(self, node: Node, code: str) -> Optional[Dict]:
        """Parse Python assignment statement"""
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
            self.logger.warning(f"Error parsing Python assignment: {e}")
            return None
    
    def _parse_python_decorator(self, node: Node, code: str) -> Optional[Dict]:
        """Parse Python decorator"""
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
            self.logger.warning(f"Error parsing Python decorator: {e}")
            return None

    # Java parsing implementation methods
    def _parse_java_class_declaration(self, node: Node, code: str) -> Optional[Dict]:
        """Parse Java class/interface declaration"""
        try:
            name = self._get_java_class_name(node, code)
            if not name:
                return None
            
            # Get modifiers
            modifiers = self._get_java_modifiers(node, code)
            
            # Determine type
            class_type = node.type  # class_declaration, interface_declaration, enum_declaration
            
            # Get superclass and interfaces
            superclass = None
            interfaces = []
            
            for child in node.children:
                if child.type == 'superclass':
                    superclass = self._get_node_text(child, code).replace('extends', '').strip()
                elif child.type == 'super_interfaces':
                    interfaces = self._get_java_interface_names(child, code)
            
            # Get methods and fields
            methods = []
            fields = []
            
            for child in node.children:
                if child.type == 'class_body':
                    for body_child in child.children:
                        if body_child.type == 'method_declaration':
                            method_info = self._parse_java_method_declaration(body_child, code, name)
                            if method_info:
                                methods.append(method_info)
                        elif body_child.type == 'constructor_declaration':
                            constructor_info = self._parse_java_constructor_declaration(body_child, code, name)
                            if constructor_info:
                                methods.append(constructor_info)
                        elif body_child.type == 'field_declaration':
                            field_info = self._parse_java_field_declaration(body_child, code, name)
                            if field_info:
                                fields.append(field_info)
            
            return {
                'name': name,
                'type': class_type,
                'modifiers': modifiers,
                'superclass': superclass,
                'interfaces': interfaces,
                'methods': methods,
                'fields': fields,
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1,
                'is_interface': class_type == 'interface_declaration',
                'is_enum': class_type == 'enum_declaration',
                'is_abstract': 'abstract' in modifiers,
                'is_public': 'public' in modifiers,
                'method_count': len(methods),
                'field_count': len(fields)
            }
        except Exception as e:
            self.logger.warning(f"Error parsing Java class: {e}")
            return None
    
    def _parse_java_method_declaration(self, node: Node, code: str, class_name: Optional[str] = None) -> Optional[Dict]:
        """Parse Java method declaration"""
        try:
            name = self._get_java_method_name(node, code)
            if not name:
                return None
            
            # Get modifiers
            modifiers = self._get_java_modifiers(node, code)
            
            # Get return type
            return_type = self._get_java_method_return_type(node, code)
            
            # Get parameters
            parameters = self._get_java_method_parameters(node, code)
            
            return {
                'name': name,
                'class_name': class_name,
                'modifiers': modifiers,
                'return_type': return_type,
                'parameters': parameters,
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1,
                'is_method': True,
                'is_constructor': False,
                'is_static': 'static' in modifiers,
                'is_abstract': 'abstract' in modifiers,
                'is_public': 'public' in modifiers,
                'is_private': 'private' in modifiers,
                'is_protected': 'protected' in modifiers
            }
        except Exception as e:
            self.logger.warning(f"Error parsing Java method: {e}")
            return None
    
    def _parse_java_constructor_declaration(self, node: Node, code: str, class_name: Optional[str] = None) -> Optional[Dict]:
        """Parse Java constructor declaration"""
        try:
            name = self._get_java_constructor_name(node, code)
            if not name:
                name = class_name or "Constructor"
            
            # Get modifiers
            modifiers = self._get_java_modifiers(node, code)
            
            # Get parameters
            parameters = self._get_java_method_parameters(node, code)
            
            return {
                'name': name,
                'class_name': class_name,
                'modifiers': modifiers,
                'return_type': 'void',  # Constructors don't have return types
                'parameters': parameters,
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1,
                'is_method': True,
                'is_constructor': True,
                'is_static': False,  # Constructors can't be static
                'is_abstract': False,  # Constructors can't be abstract
                'is_public': 'public' in modifiers,
                'is_private': 'private' in modifiers,
                'is_protected': 'protected' in modifiers
            }
        except Exception as e:
            self.logger.warning(f"Error parsing Java constructor: {e}")
            return None
    
    def _parse_java_field_declaration(self, node: Node, code: str, class_name: Optional[str] = None) -> Optional[Dict]:
        """Parse Java field declaration"""
        try:
            # Get modifiers
            modifiers = self._get_java_modifiers(node, code)
            
            # Get type and variable names
            field_type = None
            variable_names = []
            
            for child in node.children:
                if child.type in ['integral_type', 'floating_point_type', 'boolean_type', 'type_identifier', 'generic_type']:
                    field_type = self._get_node_text(child, code)
                elif child.type == 'variable_declarator':
                    var_name = self._get_java_variable_name(child, code)
                    if var_name:
                        variable_names.append(var_name)
            
            # Return info for each variable (Java allows multiple variables in one declaration)
            if variable_names:
                return {
                    'names': variable_names,
                    'type': field_type,
                    'class_name': class_name,
                    'modifiers': modifiers,
                    'start_line': node.start_point[0] + 1,
                    'is_static': 'static' in modifiers,
                    'is_final': 'final' in modifiers,
                    'is_public': 'public' in modifiers,
                    'is_private': 'private' in modifiers,
                    'is_protected': 'protected' in modifiers
                }
        except Exception as e:
            self.logger.warning(f"Error parsing Java field: {e}")
            return None
    
    def _parse_java_import_declaration(self, node: Node, code: str) -> Optional[Dict]:
        """Parse Java import declaration"""
        try:
            import_text = self._get_node_text(node, code).strip()
            
            # Extract the imported package/class
            imported = None
            is_static = False
            
            for child in node.children:
                if child.type == 'static':
                    is_static = True
                elif child.type in ['scoped_identifier', 'identifier']:
                    imported = self._get_node_text(child, code)
            
            return {
                'type': 'static_import' if is_static else 'import',
                'imported': imported,
                'text': import_text,
                'line': node.start_point[0] + 1,
                'is_static': is_static
            }
        except Exception as e:
            self.logger.warning(f"Error parsing Java import: {e}")
            return None

    # Helper methods - Common
    def _get_node_text(self, node: Node, code: str) -> str:
        """Get text content của node"""
        return code[node.start_byte:node.end_byte]

    # Python helper methods
    def _get_python_function_name(self, node: Node, code: str) -> Optional[str]:
        """Get Python function name"""
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
    
    def _get_python_class_name(self, node: Node, code: str) -> Optional[str]:
        """Get Python class name"""
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
    
    def _get_python_function_parameters(self, node: Node, code: str) -> List[Dict]:
        """Get Python function parameters"""
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
    
    def _get_python_function_docstring(self, node: Node, code: str) -> Optional[str]:
        """Get Python function docstring"""
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
    
    def _get_python_class_docstring(self, node: Node, code: str) -> Optional[str]:
        """Get Python class docstring"""
        return self._get_python_function_docstring(node, code)  # Same logic
    
    def _get_python_function_decorators(self, node: Node, code: str) -> List[str]:
        """Get Python function decorators"""
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
    
    def _get_python_class_decorators(self, node: Node, code: str) -> List[str]:
        """Get Python class decorators"""
        return self._get_python_function_decorators(node, code)  # Same logic
    
    def _get_python_return_type_annotation(self, node: Node, code: str) -> Optional[str]:
        """Get Python return type annotation"""
        for child in node.children:
            if child.type == 'type':
                return self._get_node_text(child, code)
        return None
    
    def _get_python_base_classes(self, node: Node, code: str) -> List[str]:
        """Get Python base classes"""
        base_classes = []
        
        for child in node.children:
            if child.type == 'argument_list':
                for arg_child in child.children:
                    if arg_child.type == 'identifier':
                        base_classes.append(self._get_node_text(arg_child, code))
        
        return base_classes
    
    def _extract_python_module_names(self, node: Node, code: str) -> List[str]:
        """Extract Python module names từ import statement"""
        modules = []
        
        if node.type == 'dotted_name':
            modules.append(self._get_node_text(node, code))
        elif node.type == 'dotted_as_names':
            for child in node.children:
                if child.type == 'dotted_name':
                    modules.append(self._get_node_text(child, code))
        
        return modules
    
    def _extract_python_import_names(self, node: Node, code: str) -> List[str]:
        """Extract Python import names từ from import statement"""
        names = []
        
        if node.type == 'wildcard_import':
            names.append('*')
        elif node.type == 'import_list':
            for child in node.children:
                if child.type == 'identifier':
                    names.append(self._get_node_text(child, code))
        
        return names

    # Java helper methods
    def _get_java_class_name(self, node: Node, code: str) -> Optional[str]:
        """Get Java class name"""
        for child in node.children:
            if child.type == 'identifier':
                return self._get_node_text(child, code)
        return None
    
    def _get_java_method_name(self, node: Node, code: str) -> Optional[str]:
        """Get Java method name"""
        for child in node.children:
            if child.type == 'identifier':
                return self._get_node_text(child, code)
        return None
    
    def _get_java_constructor_name(self, node: Node, code: str) -> Optional[str]:
        """Get Java constructor name"""
        for child in node.children:
            if child.type == 'identifier':
                return self._get_node_text(child, code)
        return None
    
    def _get_java_modifiers(self, node: Node, code: str) -> List[str]:
        """Get Java modifiers (public, private, static, etc.)"""
        modifiers = []
        
        for child in node.children:
            if child.type == 'modifiers':
                for modifier_child in child.children:
                    if modifier_child.type in ['public', 'private', 'protected', 'static', 'final', 'abstract', 'synchronized', 'native', 'strictfp']:
                        modifiers.append(modifier_child.type)
        
        return modifiers
    
    def _get_java_method_return_type(self, node: Node, code: str) -> Optional[str]:
        """Get Java method return type"""
        for child in node.children:
            if child.type in ['void_type', 'integral_type', 'floating_point_type', 'boolean_type', 'type_identifier', 'generic_type', 'array_type']:
                return self._get_node_text(child, code)
        return None
    
    def _get_java_method_parameters(self, node: Node, code: str) -> List[Dict]:
        """Get Java method parameters"""
        parameters = []
        
        for child in node.children:
            if child.type == 'formal_parameters':
                for param_child in child.children:
                    if param_child.type == 'formal_parameter':
                        param_info = self._parse_java_formal_parameter(param_child, code)
                        if param_info:
                            parameters.append(param_info)
        
        return parameters
    
    def _parse_java_formal_parameter(self, node: Node, code: str) -> Optional[Dict]:
        """Parse Java formal parameter"""
        param_type = None
        param_name = None
        
        for child in node.children:
            if child.type in ['integral_type', 'floating_point_type', 'boolean_type', 'type_identifier', 'generic_type', 'array_type']:
                param_type = self._get_node_text(child, code)
            elif child.type == 'variable_declarator':
                param_name = self._get_java_variable_name(child, code)
            elif child.type == 'identifier':
                param_name = self._get_node_text(child, code)
        
        if param_name:
            return {
                'name': param_name,
                'type': param_type,
                'default': None  # Java doesn't have default parameters like Python
            }
        
        return None
    
    def _get_java_variable_name(self, node: Node, code: str) -> Optional[str]:
        """Get Java variable name from variable_declarator"""
        for child in node.children:
            if child.type == 'identifier':
                return self._get_node_text(child, code)
        return None
    
    def _get_java_interface_names(self, node: Node, code: str) -> List[str]:
        """Get Java interface names from super_interfaces"""
        interfaces = []
        
        for child in node.children:
            if child.type == 'type_list':
                for type_child in child.children:
                    if type_child.type == 'type_identifier':
                        interfaces.append(self._get_node_text(type_child, code))
        
        return interfaces

    # Legacy helper methods for backward compatibility
    def _get_function_name(self, node: Node, code: str) -> Optional[str]:
        """Legacy method - delegates to Python function name"""
        return self._get_python_function_name(node, code)
    
    def _get_class_name(self, node: Node, code: str) -> Optional[str]:
        """Legacy method - delegates to Python class name"""
        return self._get_python_class_name(node, code)
    
    def _get_function_parameters(self, node: Node, code: str) -> List[Dict]:
        """Legacy method - delegates to Python function parameters"""
        return self._get_python_function_parameters(node, code)
    
    def _get_function_docstring(self, node: Node, code: str) -> Optional[str]:
        """Legacy method - delegates to Python function docstring"""
        return self._get_python_function_docstring(node, code)
    
    def _get_class_docstring(self, node: Node, code: str) -> Optional[str]:
        """Legacy method - delegates to Python class docstring"""
        return self._get_python_class_docstring(node, code)
    
    def _get_function_decorators(self, node: Node, code: str) -> List[str]:
        """Legacy method - delegates to Python function decorators"""
        return self._get_python_function_decorators(node, code)
    
    def _get_class_decorators(self, node: Node, code: str) -> List[str]:
        """Legacy method - delegates to Python class decorators"""
        return self._get_python_class_decorators(node, code)
    
    def _get_return_type_annotation(self, node: Node, code: str) -> Optional[str]:
        """Legacy method - delegates to Python return type annotation"""
        return self._get_python_return_type_annotation(node, code)
    
    def _get_base_classes(self, node: Node, code: str) -> List[str]:
        """Legacy method - delegates to Python base classes"""
        return self._get_python_base_classes(node, code)
    
    def _extract_module_names(self, node: Node, code: str) -> List[str]:
        """Legacy method - delegates to Python module names"""
        return self._extract_python_module_names(node, code)
    
    def _extract_import_names(self, node: Node, code: str) -> List[str]:
        """Legacy method - delegates to Python import names"""
        return self._extract_python_import_names(node, code)

    # Legacy parsing methods for backward compatibility
    def _parse_function_definition(self, node: Node, code: str, class_name: Optional[str] = None) -> Optional[Dict]:
        """Legacy method - delegates to Python function definition"""
        return self._parse_python_function_definition(node, code, class_name)
    
    def _parse_class_definition(self, node: Node, code: str) -> Optional[Dict]:
        """Legacy method - delegates to Python class definition"""
        return self._parse_python_class_definition(node, code)
    
    def _parse_import_statement(self, node: Node, code: str) -> Optional[Dict]:
        """Legacy method - delegates to Python import statement"""
        return self._parse_python_import_statement(node, code)
    
    def _parse_assignment(self, node: Node, code: str) -> Optional[Dict]:
        """Legacy method - delegates to Python assignment"""
        return self._parse_python_assignment(node, code)
    
    def _parse_decorator(self, node: Node, code: str) -> Optional[Dict]:
        """Legacy method - delegates to Python decorator"""
        return self._parse_python_decorator(node, code)


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