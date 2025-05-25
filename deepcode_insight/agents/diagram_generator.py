"""
Diagram Generation Agent - Generates PlantUML diagrams từ AST analysis
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DiagramType(Enum):
    """Types of diagrams that can be generated"""
    CLASS_DIAGRAM = "class"
    SEQUENCE_DIAGRAM = "sequence"


@dataclass
class ClassInfo:
    """Information about a class extracted từ AST"""
    name: str
    file_path: str
    line_number: int
    is_abstract: bool = False
    is_interface: bool = False
    superclasses: List[str] = None
    interfaces: List[str] = None
    fields: List['FieldInfo'] = None
    methods: List['MethodInfo'] = None
    inner_classes: List[str] = None
    visibility: str = "public"  # public, private, protected
    
    def __post_init__(self):
        if self.superclasses is None:
            self.superclasses = []
        if self.interfaces is None:
            self.interfaces = []
        if self.fields is None:
            self.fields = []
        if self.methods is None:
            self.methods = []
        if self.inner_classes is None:
            self.inner_classes = []


@dataclass
class FieldInfo:
    """Information about a class field"""
    name: str
    type_hint: str
    visibility: str = "public"  # public, private, protected
    is_static: bool = False
    is_final: bool = False
    default_value: Optional[str] = None


@dataclass
class MethodInfo:
    """Information about a class method"""
    name: str
    return_type: str
    parameters: List['ParameterInfo'] = None
    visibility: str = "public"
    is_static: bool = False
    is_abstract: bool = False
    is_constructor: bool = False
    is_destructor: bool = False
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []


@dataclass
class ParameterInfo:
    """Information about method parameters"""
    name: str
    type_hint: str
    default_value: Optional[str] = None
    is_varargs: bool = False


@dataclass
class RelationshipInfo:
    """Information about relationships between classes"""
    source_class: str
    target_class: str
    relationship_type: str  # inheritance, composition, aggregation, association
    label: Optional[str] = None
    multiplicity: Optional[str] = None


class DiagramGenerationAgent:
    """
    Agent để generate PlantUML diagrams từ AST analysis
    """
    
    def __init__(self, 
                 include_private_members: bool = False,
                 include_method_parameters: bool = True,
                 include_field_types: bool = True,
                 max_classes_per_diagram: int = 20):
        """
        Initialize DiagramGenerationAgent
        
        Args:
            include_private_members: Whether to include private fields/methods
            include_method_parameters: Whether to show method parameters
            include_field_types: Whether to show field types
            max_classes_per_diagram: Maximum classes per diagram
        """
        self.include_private_members = include_private_members
        self.include_method_parameters = include_method_parameters
        self.include_field_types = include_field_types
        self.max_classes_per_diagram = max_classes_per_diagram
        
        logger.info("DiagramGenerationAgent initialized")
    
    def extract_class_info_from_ast(self, 
                                   ast_data: Dict[str, Any], 
                                   file_path: str,
                                   language: str = "python") -> List[ClassInfo]:
        """
        Extract class information từ AST data
        
        Args:
            ast_data: AST data từ ASTParsingAgent
            file_path: Path to source file
            language: Programming language (python, java)
            
        Returns:
            List of ClassInfo objects
        """
        try:
            if language.lower() == "python":
                return self._extract_python_classes(ast_data, file_path)
            elif language.lower() == "java":
                return self._extract_java_classes(ast_data, file_path)
            else:
                logger.warning(f"Unsupported language: {language}")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting class info: {e}")
            return []
    
    def _extract_python_classes(self, ast_data: Dict[str, Any], file_path: str) -> List[ClassInfo]:
        """Extract class information từ Python AST"""
        classes = []
        
        # Handle different AST formats
        if 'classes' in ast_data:
            # Direct classes list
            class_nodes = ast_data['classes']
        elif 'ast' in ast_data and 'body' in ast_data['ast']:
            # Full AST structure
            class_nodes = self._find_class_nodes_python(ast_data['ast']['body'])
        else:
            logger.warning("No class information found in AST data")
            return []
        
        for class_node in class_nodes:
            try:
                class_info = self._parse_python_class(class_node, file_path)
                if class_info:
                    classes.append(class_info)
            except Exception as e:
                logger.error(f"Error parsing Python class: {e}")
                continue
        
        return classes
    
    def _extract_java_classes(self, ast_data: Dict[str, Any], file_path: str) -> List[ClassInfo]:
        """Extract class information từ Java AST"""
        classes = []
        
        # Handle different AST formats
        if 'classes' in ast_data:
            class_nodes = ast_data['classes']
        elif 'ast' in ast_data:
            class_nodes = self._find_class_nodes_java(ast_data['ast'])
        else:
            logger.warning("No class information found in Java AST data")
            return []
        
        for class_node in class_nodes:
            try:
                class_info = self._parse_java_class(class_node, file_path)
                if class_info:
                    classes.append(class_info)
            except Exception as e:
                logger.error(f"Error parsing Java class: {e}")
                continue
        
        return classes
    
    def _find_class_nodes_python(self, body: List[Dict]) -> List[Dict]:
        """Find class nodes trong Python AST body"""
        class_nodes = []
        
        for node in body:
            if isinstance(node, dict) and node.get('type') == 'ClassDef':
                class_nodes.append(node)
            elif isinstance(node, dict) and 'body' in node:
                # Recursive search trong nested structures
                class_nodes.extend(self._find_class_nodes_python(node['body']))
        
        return class_nodes
    
    def _find_class_nodes_java(self, ast_node: Dict) -> List[Dict]:
        """Find class nodes trong Java AST"""
        class_nodes = []
        
        if isinstance(ast_node, dict):
            if ast_node.get('type') in ['class_declaration', 'interface_declaration']:
                class_nodes.append(ast_node)
            
            # Recursive search
            for key, value in ast_node.items():
                if isinstance(value, list):
                    for item in value:
                        class_nodes.extend(self._find_class_nodes_java(item))
                elif isinstance(value, dict):
                    class_nodes.extend(self._find_class_nodes_java(value))
        
        return class_nodes
    
    def _parse_python_class(self, class_node: Dict, file_path: str) -> Optional[ClassInfo]:
        """Parse Python class node thành ClassInfo"""
        try:
            name = class_node.get('name', 'UnknownClass')
            line_number = class_node.get('lineno', 0)
            
            # Extract superclasses
            superclasses = []
            if 'bases' in class_node:
                for base in class_node['bases']:
                    if isinstance(base, dict) and 'id' in base:
                        superclasses.append(base['id'])
                    elif isinstance(base, str):
                        superclasses.append(base)
            
            # Extract fields và methods
            fields = []
            methods = []
            
            if 'body' in class_node:
                for item in class_node['body']:
                    if isinstance(item, dict):
                        if item.get('type') == 'FunctionDef':
                            method_info = self._parse_python_method(item)
                            if method_info:
                                methods.append(method_info)
                        elif item.get('type') == 'AnnAssign':
                            # Type annotated assignment
                            field_info = self._parse_python_field(item)
                            if field_info:
                                fields.append(field_info)
                        elif item.get('type') == 'Assign':
                            # Regular assignment
                            field_info = self._parse_python_assignment(item)
                            if field_info:
                                fields.append(field_info)
            
            return ClassInfo(
                name=name,
                file_path=file_path,
                line_number=line_number,
                superclasses=superclasses,
                fields=fields,
                methods=methods
            )
            
        except Exception as e:
            logger.error(f"Error parsing Python class node: {e}")
            return None
    
    def _parse_java_class(self, class_node: Dict, file_path: str) -> Optional[ClassInfo]:
        """Parse Java class node thành ClassInfo"""
        try:
            name = class_node.get('name', 'UnknownClass')
            line_number = class_node.get('start_point', {}).get('row', 0)
            
            is_interface = class_node.get('type') == 'interface_declaration'
            is_abstract = 'abstract' in class_node.get('modifiers', [])
            
            # Extract superclass và interfaces
            superclasses = []
            interfaces = []
            
            if 'superclass' in class_node:
                superclasses.append(class_node['superclass'])
            
            if 'interfaces' in class_node:
                interfaces.extend(class_node['interfaces'])
            
            # Extract fields và methods
            fields = []
            methods = []
            
            if 'body' in class_node:
                for item in class_node['body']:
                    if isinstance(item, dict):
                        # Check for method - could be 'method_declaration' or have 'name' and 'modifiers'
                        if (item.get('type') == 'method_declaration' or 
                            ('name' in item and 'modifiers' in item and 'parameters' in item)):
                            method_info = self._parse_java_method(item)
                            if method_info:
                                methods.append(method_info)
                        elif (item.get('type') == 'field_declaration' or 
                              ('name' in item and 'modifiers' in item and 'type' in item and 'parameters' not in item)):
                            field_info = self._parse_java_field(item)
                            if field_info:
                                fields.append(field_info)
            
            return ClassInfo(
                name=name,
                file_path=file_path,
                line_number=line_number,
                is_abstract=is_abstract,
                is_interface=is_interface,
                superclasses=superclasses,
                interfaces=interfaces,
                fields=fields,
                methods=methods
            )
            
        except Exception as e:
            logger.error(f"Error parsing Java class node: {e}")
            return None
    
    def _parse_python_method(self, method_node: Dict) -> Optional[MethodInfo]:
        """Parse Python method node"""
        try:
            name = method_node.get('name', 'unknown_method')
            
            # Determine visibility
            visibility = "private" if name.startswith('_') else "public"
            
            # Check if constructor
            is_constructor = name == '__init__'
            is_destructor = name == '__del__'
            
            # Extract parameters
            parameters = []
            if 'args' in method_node:
                args = method_node['args']
                if 'args' in args:
                    for arg in args['args']:
                        if isinstance(arg, dict) and 'arg' in arg:
                            param_name = arg['arg']
                            if param_name != 'self':  # Skip self parameter
                                param_type = arg.get('annotation', {}).get('id', 'Any')
                                parameters.append(ParameterInfo(
                                    name=param_name,
                                    type_hint=param_type
                                ))
            
            # Extract return type
            return_type = "None"
            if 'returns' in method_node and method_node['returns']:
                if isinstance(method_node['returns'], dict):
                    return_type = method_node['returns'].get('id', 'Any')
            
            return MethodInfo(
                name=name,
                return_type=return_type,
                parameters=parameters,
                visibility=visibility,
                is_constructor=is_constructor,
                is_destructor=is_destructor
            )
            
        except Exception as e:
            logger.error(f"Error parsing Python method: {e}")
            return None
    
    def _parse_java_method(self, method_node: Dict) -> Optional[MethodInfo]:
        """Parse Java method node"""
        try:
            name = method_node.get('name', 'unknown_method')
            
            # Extract modifiers
            modifiers = method_node.get('modifiers', [])
            visibility = "public"
            if 'private' in modifiers:
                visibility = "private"
            elif 'protected' in modifiers:
                visibility = "protected"
            
            is_static = 'static' in modifiers
            is_abstract = 'abstract' in modifiers
            
            # Extract return type
            return_type = method_node.get('type', 'void')
            
            # Extract parameters
            parameters = []
            if 'parameters' in method_node:
                for param in method_node['parameters']:
                    if isinstance(param, dict):
                        param_name = param.get('name', 'param')
                        param_type = param.get('type', 'Object')
                        parameters.append(ParameterInfo(
                            name=param_name,
                            type_hint=param_type
                        ))
            
            return MethodInfo(
                name=name,
                return_type=return_type,
                parameters=parameters,
                visibility=visibility,
                is_static=is_static,
                is_abstract=is_abstract
            )
            
        except Exception as e:
            logger.error(f"Error parsing Java method: {e}")
            return None
    
    def _parse_python_field(self, field_node: Dict) -> Optional[FieldInfo]:
        """Parse Python annotated field"""
        try:
            if 'target' in field_node and 'id' in field_node['target']:
                name = field_node['target']['id']
                
                # Extract type annotation
                type_hint = "Any"
                if 'annotation' in field_node:
                    annotation = field_node['annotation']
                    if isinstance(annotation, dict) and 'id' in annotation:
                        type_hint = annotation['id']
                
                # Determine visibility
                visibility = "private" if name.startswith('_') else "public"
                
                return FieldInfo(
                    name=name,
                    type_hint=type_hint,
                    visibility=visibility
                )
        except Exception as e:
            logger.error(f"Error parsing Python field: {e}")
        
        return None
    
    def _parse_python_assignment(self, assign_node: Dict) -> Optional[FieldInfo]:
        """Parse Python assignment as field"""
        try:
            if 'targets' in assign_node and assign_node['targets']:
                target = assign_node['targets'][0]
                if isinstance(target, dict) and 'id' in target:
                    name = target['id']
                    
                    # Determine visibility
                    visibility = "private" if name.startswith('_') else "public"
                    
                    return FieldInfo(
                        name=name,
                        type_hint="Any",
                        visibility=visibility
                    )
        except Exception as e:
            logger.error(f"Error parsing Python assignment: {e}")
        
        return None
    
    def _parse_java_field(self, field_node: Dict) -> Optional[FieldInfo]:
        """Parse Java field node"""
        try:
            name = field_node.get('name', 'unknown_field')
            type_hint = field_node.get('type', 'Object')
            
            # Extract modifiers
            modifiers = field_node.get('modifiers', [])
            visibility = "public"
            if 'private' in modifiers:
                visibility = "private"
            elif 'protected' in modifiers:
                visibility = "protected"
            
            is_static = 'static' in modifiers
            is_final = 'final' in modifiers
            
            return FieldInfo(
                name=name,
                type_hint=type_hint,
                visibility=visibility,
                is_static=is_static,
                is_final=is_final
            )
            
        except Exception as e:
            logger.error(f"Error parsing Java field: {e}")
            return None
    
    def generate_class_diagram(self, 
                              classes: List[ClassInfo],
                              title: str = "Class Diagram",
                              include_relationships: bool = True) -> str:
        """
        Generate PlantUML class diagram từ class information
        
        Args:
            classes: List of ClassInfo objects
            title: Diagram title
            include_relationships: Whether to include inheritance relationships
            
        Returns:
            PlantUML text for class diagram
        """
        try:
            plantuml_lines = []
            
            # Header
            plantuml_lines.append("@startuml")
            plantuml_lines.append(f"title {title}")
            plantuml_lines.append("")
            
            # Styling
            plantuml_lines.extend([
                "skinparam classAttributeIconSize 0",
                "skinparam classFontStyle bold",
                "skinparam classBackgroundColor lightblue",
                "skinparam classBorderColor darkblue",
                ""
            ])
            
            # Generate classes
            for class_info in classes[:self.max_classes_per_diagram]:
                class_uml = self._generate_class_uml(class_info)
                plantuml_lines.extend(class_uml)
                plantuml_lines.append("")
            
            # Generate relationships
            if include_relationships:
                relationships = self._extract_relationships(classes)
                for relationship in relationships:
                    rel_uml = self._generate_relationship_uml(relationship)
                    if rel_uml:
                        plantuml_lines.append(rel_uml)
                
                if relationships:
                    plantuml_lines.append("")
            
            # Footer
            plantuml_lines.append("@enduml")
            
            return "\n".join(plantuml_lines)
            
        except Exception as e:
            logger.error(f"Error generating class diagram: {e}")
            return f"@startuml\nnote: Error generating diagram: {e}\n@enduml"
    
    def _generate_class_uml(self, class_info: ClassInfo) -> List[str]:
        """Generate PlantUML for single class"""
        lines = []
        
        # Class declaration
        class_type = "interface" if class_info.is_interface else "class"
        abstract_modifier = "abstract " if class_info.is_abstract else ""
        
        lines.append(f"{abstract_modifier}{class_type} {class_info.name} {{")
        
        # Fields
        if class_info.fields:
            for field in class_info.fields:
                if self.include_private_members or field.visibility != "private":
                    field_uml = self._generate_field_uml(field)
                    lines.append(f"  {field_uml}")
        
        # Separator between fields và methods
        if class_info.fields and class_info.methods:
            lines.append("  --")
        
        # Methods
        if class_info.methods:
            for method in class_info.methods:
                if self.include_private_members or method.visibility != "private":
                    method_uml = self._generate_method_uml(method)
                    lines.append(f"  {method_uml}")
        
        lines.append("}")
        
        return lines
    
    def _generate_field_uml(self, field: FieldInfo) -> str:
        """Generate PlantUML for field"""
        # Visibility symbol
        visibility_symbol = {
            "public": "+",
            "private": "-",
            "protected": "#"
        }.get(field.visibility, "+")
        
        # Static modifier
        static_modifier = "{static} " if field.is_static else ""
        
        # Type information
        type_info = f": {field.type_hint}" if self.include_field_types and field.type_hint != "Any" else ""
        
        return f"{visibility_symbol} {static_modifier}{field.name}{type_info}"
    
    def _generate_method_uml(self, method: MethodInfo) -> str:
        """Generate PlantUML for method"""
        # Visibility symbol
        visibility_symbol = {
            "public": "+",
            "private": "-",
            "protected": "#"
        }.get(method.visibility, "+")
        
        # Static modifier
        static_modifier = "{static} " if method.is_static else ""
        
        # Abstract modifier
        abstract_modifier = "{abstract} " if method.is_abstract else ""
        
        # Parameters
        params_str = ""
        if self.include_method_parameters and method.parameters:
            param_strs = []
            for param in method.parameters:
                param_str = param.name
                if param.type_hint and param.type_hint != "Any":
                    param_str += f": {param.type_hint}"
                param_strs.append(param_str)
            params_str = ", ".join(param_strs)
        
        # Return type
        return_type = f": {method.return_type}" if method.return_type and method.return_type != "None" else ""
        
        return f"{visibility_symbol} {static_modifier}{abstract_modifier}{method.name}({params_str}){return_type}"
    
    def _extract_relationships(self, classes: List[ClassInfo]) -> List[RelationshipInfo]:
        """Extract relationships between classes"""
        relationships = []
        class_names = {cls.name for cls in classes}
        
        for class_info in classes:
            # Inheritance relationships
            for superclass in class_info.superclasses:
                if superclass in class_names:
                    relationships.append(RelationshipInfo(
                        source_class=class_info.name,
                        target_class=superclass,
                        relationship_type="inheritance"
                    ))
            
            # Interface implementation
            for interface in class_info.interfaces:
                if interface in class_names:
                    relationships.append(RelationshipInfo(
                        source_class=class_info.name,
                        target_class=interface,
                        relationship_type="implementation"
                    ))
        
        return relationships
    
    def _generate_relationship_uml(self, relationship: RelationshipInfo) -> Optional[str]:
        """Generate PlantUML for relationship"""
        if relationship.relationship_type == "inheritance":
            return f"{relationship.source_class} --|> {relationship.target_class}"
        elif relationship.relationship_type == "implementation":
            return f"{relationship.source_class} ..|> {relationship.target_class}"
        elif relationship.relationship_type == "composition":
            return f"{relationship.source_class} *-- {relationship.target_class}"
        elif relationship.relationship_type == "aggregation":
            return f"{relationship.source_class} o-- {relationship.target_class}"
        elif relationship.relationship_type == "association":
            return f"{relationship.source_class} --> {relationship.target_class}"
        
        return None
    
    def process_files(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph node function để process files và generate diagrams
        
        Args:
            state: LangGraph state containing AST data
            
        Returns:
            Updated state với diagram data
        """
        try:
            logger.info("DiagramGenerationAgent processing files...")
            
            # Extract AST data từ state
            ast_results = state.get('ast_results', {})
            if not ast_results:
                logger.warning("No AST results found in state")
                return {
                    **state,
                    'diagrams': {},
                    'current_agent': 'diagram_generator',
                    'processing_status': 'no_ast_data'
                }
            
            all_classes = []
            diagrams = {}
            
            # Process each file
            for file_path, ast_data in ast_results.items():
                try:
                    # Determine language từ file extension
                    language = "python" if file_path.endswith('.py') else "java"
                    
                    # Extract class information
                    classes = self.extract_class_info_from_ast(ast_data, file_path, language)
                    all_classes.extend(classes)
                    
                    # Generate diagram cho individual file
                    if classes:
                        diagram_title = f"Classes in {file_path}"
                        diagram_uml = self.generate_class_diagram(classes, diagram_title)
                        diagrams[file_path] = {
                            'type': 'class_diagram',
                            'title': diagram_title,
                            'uml': diagram_uml,
                            'classes': len(classes)
                        }
                
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue
            
            # Generate overall project diagram
            if all_classes:
                overall_diagram = self.generate_class_diagram(
                    all_classes, 
                    "Project Class Diagram"
                )
                diagrams['project_overview'] = {
                    'type': 'class_diagram',
                    'title': 'Project Class Diagram',
                    'uml': overall_diagram,
                    'classes': len(all_classes)
                }
            
            logger.info(f"Generated {len(diagrams)} diagrams for {len(all_classes)} classes")
            
            return {
                **state,
                'diagrams': diagrams,
                'extracted_classes': [
                    {
                        'name': cls.name,
                        'file_path': cls.file_path,
                        'methods': len(cls.methods),
                        'fields': len(cls.fields),
                        'superclasses': cls.superclasses,
                        'is_abstract': cls.is_abstract,
                        'is_interface': cls.is_interface
                    }
                    for cls in all_classes
                ],
                'current_agent': 'diagram_generator',
                'processing_status': 'diagram_generation_completed',
                'diagram_metadata': {
                    'total_diagrams': len(diagrams),
                    'total_classes': len(all_classes),
                    'files_processed': len(ast_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in DiagramGenerationAgent.process_files: {e}")
            return {
                **state,
                'diagrams': {},
                'current_agent': 'diagram_generator',
                'processing_status': 'diagram_generation_failed',
                'error': str(e)
            }


def create_diagram_generator_agent(**kwargs) -> DiagramGenerationAgent:
    """
    Factory function để create DiagramGenerationAgent
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        DiagramGenerationAgent instance
    """
    return DiagramGenerationAgent(**kwargs)


def diagram_generator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function cho diagram generation
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state với diagram data
    """
    agent = create_diagram_generator_agent()
    return agent.process_files(state) 