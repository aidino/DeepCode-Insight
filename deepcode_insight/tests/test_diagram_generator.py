"""
Tests for DiagramGenerationAgent
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from deepcode_insight.agents.diagram_generator import (
    DiagramGenerationAgent,
    ClassInfo,
    FieldInfo,
    MethodInfo,
    ParameterInfo,
    RelationshipInfo,
    DiagramType,
    create_diagram_generator_agent,
    diagram_generator_node
)


class TestDiagramGenerationAgent:
    """Test cases for DiagramGenerationAgent"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.agent = DiagramGenerationAgent(
            include_private_members=True,
            include_method_parameters=True,
            include_field_types=True,
            max_classes_per_diagram=10
        )
    
    def test_initialization_default_settings(self):
        """Test agent initialization với default settings"""
        agent = DiagramGenerationAgent()
        
        assert agent.include_private_members is False
        assert agent.include_method_parameters is True
        assert agent.include_field_types is True
        assert agent.max_classes_per_diagram == 20
    
    def test_initialization_custom_settings(self):
        """Test agent initialization với custom settings"""
        agent = DiagramGenerationAgent(
            include_private_members=True,
            include_method_parameters=False,
            include_field_types=False,
            max_classes_per_diagram=5
        )
        
        assert agent.include_private_members is True
        assert agent.include_method_parameters is False
        assert agent.include_field_types is False
        assert agent.max_classes_per_diagram == 5
    
    def test_extract_python_classes_direct_format(self):
        """Test extracting Python classes từ direct format"""
        ast_data = {
            'classes': [
                {
                    'name': 'TestClass',
                    'lineno': 1,
                    'bases': [{'id': 'BaseClass'}],
                    'body': [
                        {
                            'type': 'FunctionDef',
                            'name': '__init__',
                            'args': {
                                'args': [
                                    {'arg': 'self'},
                                    {'arg': 'value', 'annotation': {'id': 'str'}}
                                ]
                            },
                            'returns': None
                        },
                        {
                            'type': 'AnnAssign',
                            'target': {'id': 'name'},
                            'annotation': {'id': 'str'}
                        }
                    ]
                }
            ]
        }
        
        classes = self.agent.extract_class_info_from_ast(ast_data, 'test.py', 'python')
        
        assert len(classes) == 1
        assert classes[0].name == 'TestClass'
        assert classes[0].file_path == 'test.py'
        assert classes[0].line_number == 1
        assert classes[0].superclasses == ['BaseClass']
        assert len(classes[0].methods) == 1
        assert len(classes[0].fields) == 1
        assert classes[0].methods[0].name == '__init__'
        assert classes[0].methods[0].is_constructor is True
        assert classes[0].fields[0].name == 'name'
        assert classes[0].fields[0].type_hint == 'str'
    
    def test_extract_python_classes_full_ast_format(self):
        """Test extracting Python classes từ full AST format"""
        ast_data = {
            'ast': {
                'body': [
                    {
                        'type': 'ClassDef',
                        'name': 'Calculator',
                        'lineno': 5,
                        'bases': [],
                        'body': [
                            {
                                'type': 'FunctionDef',
                                'name': 'add',
                                'args': {
                                    'args': [
                                        {'arg': 'self'},
                                        {'arg': 'a', 'annotation': {'id': 'int'}},
                                        {'arg': 'b', 'annotation': {'id': 'int'}}
                                    ]
                                },
                                'returns': {'id': 'int'}
                            },
                            {
                                'type': 'Assign',
                                'targets': [{'id': 'result'}]
                            }
                        ]
                    }
                ]
            }
        }
        
        classes = self.agent.extract_class_info_from_ast(ast_data, 'calculator.py', 'python')
        
        assert len(classes) == 1
        assert classes[0].name == 'Calculator'
        assert classes[0].superclasses == []
        assert len(classes[0].methods) == 1
        assert len(classes[0].fields) == 1
        assert classes[0].methods[0].name == 'add'
        assert classes[0].methods[0].return_type == 'int'
        assert len(classes[0].methods[0].parameters) == 2
        assert classes[0].fields[0].name == 'result'
    
    def test_extract_java_classes_basic(self):
        """Test extracting Java classes"""
        ast_data = {
            'classes': [
                {
                    'type': 'class_declaration',
                    'name': 'Person',
                    'start_point': {'row': 10},
                    'modifiers': ['public'],
                    'superclass': 'Object',
                    'interfaces': ['Serializable'],
                    'body': [
                        {
                            'type': 'field_declaration',
                            'name': 'name',
                            'type': 'String',
                            'modifiers': ['private']
                        },
                        {
                            'type': 'method_declaration',
                            'name': 'getName',
                            'type': 'String',
                            'modifiers': ['public'],
                            'parameters': []
                        }
                    ]
                }
            ]
        }
        
        classes = self.agent.extract_class_info_from_ast(ast_data, 'Person.java', 'java')
        
        assert len(classes) == 1
        assert classes[0].name == 'Person'
        assert classes[0].file_path == 'Person.java'
        assert classes[0].line_number == 10
        assert classes[0].is_interface is False
        assert classes[0].superclasses == ['Object']
        assert classes[0].interfaces == ['Serializable']
        assert len(classes[0].fields) == 1
        assert len(classes[0].methods) == 1
        assert classes[0].fields[0].name == 'name'
        assert classes[0].fields[0].type_hint == 'String'
        assert classes[0].fields[0].visibility == 'private'
        assert classes[0].methods[0].name == 'getName'
        assert classes[0].methods[0].return_type == 'String'
        assert classes[0].methods[0].visibility == 'public'
    
    def test_extract_java_interface(self):
        """Test extracting Java interface"""
        ast_data = {
            'classes': [
                {
                    'type': 'interface_declaration',
                    'name': 'Drawable',
                    'start_point': {'row': 1},
                    'modifiers': ['public'],
                    'body': [
                        {
                            'type': 'method_declaration',
                            'name': 'draw',
                            'type': 'void',
                            'modifiers': ['public', 'abstract'],
                            'parameters': [
                                {'name': 'canvas', 'type': 'Canvas'}
                            ]
                        }
                    ]
                }
            ]
        }
        
        classes = self.agent.extract_class_info_from_ast(ast_data, 'Drawable.java', 'java')
        
        assert len(classes) == 1
        assert classes[0].name == 'Drawable'
        assert classes[0].is_interface is True
        assert classes[0].is_abstract is False  # Interface không set abstract flag
        assert len(classes[0].methods) == 1
        assert classes[0].methods[0].name == 'draw'
        assert classes[0].methods[0].is_abstract is True
        assert len(classes[0].methods[0].parameters) == 1
        assert classes[0].methods[0].parameters[0].name == 'canvas'
        assert classes[0].methods[0].parameters[0].type_hint == 'Canvas'
    
    def test_extract_unsupported_language(self):
        """Test extracting từ unsupported language"""
        ast_data = {'classes': []}
        
        classes = self.agent.extract_class_info_from_ast(ast_data, 'test.js', 'javascript')
        
        assert classes == []
    
    def test_extract_empty_ast_data(self):
        """Test extracting từ empty AST data"""
        ast_data = {}
        
        classes = self.agent.extract_class_info_from_ast(ast_data, 'test.py', 'python')
        
        assert classes == []
    
    def test_extract_malformed_ast_data(self):
        """Test extracting từ malformed AST data"""
        ast_data = {
            'classes': [
                {
                    # Missing required fields
                    'body': []
                }
            ]
        }
        
        classes = self.agent.extract_class_info_from_ast(ast_data, 'test.py', 'python')
        
        # Should handle gracefully và return partial results
        assert len(classes) == 1
        assert classes[0].name == 'UnknownClass'
    
    def test_generate_simple_class_diagram(self):
        """Test generating simple class diagram"""
        classes = [
            ClassInfo(
                name='SimpleClass',
                file_path='test.py',
                line_number=1,
                fields=[
                    FieldInfo(name='value', type_hint='int', visibility='public')
                ],
                methods=[
                    MethodInfo(
                        name='get_value',
                        return_type='int',
                        visibility='public',
                        parameters=[]
                    )
                ]
            )
        ]
        
        diagram = self.agent.generate_class_diagram(classes, "Test Diagram")
        
        expected_lines = [
            "@startuml",
            "title Test Diagram",
            "",
            "skinparam classAttributeIconSize 0",
            "skinparam classFontStyle bold",
            "skinparam classBackgroundColor lightblue",
            "skinparam classBorderColor darkblue",
            "",
            "class SimpleClass {",
            "  + value: int",
            "  --",
            "  + get_value(): int",
            "}",
            "",
            "@enduml"
        ]
        
        assert diagram == "\n".join(expected_lines)
    
    def test_generate_class_diagram_with_inheritance(self):
        """Test generating class diagram với inheritance"""
        classes = [
            ClassInfo(
                name='BaseClass',
                file_path='base.py',
                line_number=1,
                methods=[
                    MethodInfo(name='base_method', return_type='void', visibility='public')
                ]
            ),
            ClassInfo(
                name='DerivedClass',
                file_path='derived.py',
                line_number=1,
                superclasses=['BaseClass'],
                methods=[
                    MethodInfo(name='derived_method', return_type='void', visibility='public')
                ]
            )
        ]
        
        diagram = self.agent.generate_class_diagram(classes, "Inheritance Diagram")
        
        # Check for inheritance relationship
        assert "DerivedClass --|> BaseClass" in diagram
        assert "class BaseClass" in diagram
        assert "class DerivedClass" in diagram
    
    def test_generate_class_diagram_with_interface(self):
        """Test generating class diagram với interface implementation"""
        classes = [
            ClassInfo(
                name='MyInterface',
                file_path='interface.java',
                line_number=1,
                is_interface=True,
                methods=[
                    MethodInfo(name='interface_method', return_type='void', is_abstract=True)
                ]
            ),
            ClassInfo(
                name='Implementation',
                file_path='impl.java',
                line_number=1,
                interfaces=['MyInterface'],
                methods=[
                    MethodInfo(name='interface_method', return_type='void', visibility='public')
                ]
            )
        ]
        
        diagram = self.agent.generate_class_diagram(classes, "Interface Diagram")
        
        # Check for interface implementation relationship
        assert "Implementation ..|> MyInterface" in diagram
        assert "interface MyInterface" in diagram
        assert "class Implementation" in diagram
    
    def test_generate_class_diagram_exclude_private_members(self):
        """Test generating diagram excluding private members"""
        agent = DiagramGenerationAgent(include_private_members=False)
        
        classes = [
            ClassInfo(
                name='TestClass',
                file_path='test.py',
                line_number=1,
                fields=[
                    FieldInfo(name='public_field', type_hint='str', visibility='public'),
                    FieldInfo(name='_private_field', type_hint='int', visibility='private')
                ],
                methods=[
                    MethodInfo(name='public_method', return_type='void', visibility='public'),
                    MethodInfo(name='_private_method', return_type='void', visibility='private')
                ]
            )
        ]
        
        diagram = agent.generate_class_diagram(classes)
        
        assert "public_field" in diagram
        assert "_private_field" not in diagram
        assert "public_method" in diagram
        assert "_private_method" not in diagram
    
    def test_generate_class_diagram_without_parameters(self):
        """Test generating diagram without method parameters"""
        agent = DiagramGenerationAgent(include_method_parameters=False)
        
        classes = [
            ClassInfo(
                name='TestClass',
                file_path='test.py',
                line_number=1,
                methods=[
                    MethodInfo(
                        name='method_with_params',
                        return_type='str',
                        parameters=[
                            ParameterInfo(name='param1', type_hint='int'),
                            ParameterInfo(name='param2', type_hint='str')
                        ]
                    )
                ]
            )
        ]
        
        diagram = agent.generate_class_diagram(classes)
        
        assert "+ method_with_params(): str" in diagram
        assert "param1" not in diagram
        assert "param2" not in diagram
    
    def test_generate_class_diagram_without_field_types(self):
        """Test generating diagram without field types"""
        agent = DiagramGenerationAgent(include_field_types=False)
        
        classes = [
            ClassInfo(
                name='TestClass',
                file_path='test.py',
                line_number=1,
                fields=[
                    FieldInfo(name='typed_field', type_hint='str', visibility='public')
                ]
            )
        ]
        
        diagram = agent.generate_class_diagram(classes)
        
        assert "+ typed_field" in diagram
        assert ": str" not in diagram
    
    def test_generate_class_diagram_max_classes_limit(self):
        """Test max classes per diagram limit"""
        agent = DiagramGenerationAgent(max_classes_per_diagram=2)
        
        classes = [
            ClassInfo(name=f'Class{i}', file_path=f'class{i}.py', line_number=1)
            for i in range(5)
        ]
        
        diagram = agent.generate_class_diagram(classes)
        
        # Should only include first 2 classes
        assert "class Class0" in diagram
        assert "class Class1" in diagram
        assert "class Class2" not in diagram
        assert "class Class3" not in diagram
        assert "class Class4" not in diagram
    
    def test_generate_field_uml_variations(self):
        """Test generating UML for different field variations"""
        # Public field với type
        field1 = FieldInfo(name='public_field', type_hint='str', visibility='public')
        uml1 = self.agent._generate_field_uml(field1)
        assert uml1 == "+ public_field: str"
        
        # Private static field
        field2 = FieldInfo(name='_private_static', type_hint='int', visibility='private', is_static=True)
        uml2 = self.agent._generate_field_uml(field2)
        assert uml2 == "- {static} _private_static: int"
        
        # Protected field without type
        field3 = FieldInfo(name='protected_field', type_hint='Any', visibility='protected')
        uml3 = self.agent._generate_field_uml(field3)
        assert uml3 == "# protected_field"
    
    def test_generate_method_uml_variations(self):
        """Test generating UML for different method variations"""
        # Simple public method
        method1 = MethodInfo(name='simple_method', return_type='void', visibility='public')
        uml1 = self.agent._generate_method_uml(method1)
        assert uml1 == "+ simple_method()"
        
        # Static method với parameters
        method2 = MethodInfo(
            name='static_method',
            return_type='str',
            visibility='public',
            is_static=True,
            parameters=[
                ParameterInfo(name='param1', type_hint='int'),
                ParameterInfo(name='param2', type_hint='str')
            ]
        )
        uml2 = self.agent._generate_method_uml(method2)
        assert uml2 == "+ {static} static_method(param1: int, param2: str): str"
        
        # Abstract method
        method3 = MethodInfo(
            name='abstract_method',
            return_type='void',
            visibility='protected',
            is_abstract=True
        )
        uml3 = self.agent._generate_method_uml(method3)
        assert uml3 == "# {abstract} abstract_method()"
    
    def test_extract_relationships(self):
        """Test extracting relationships between classes"""
        classes = [
            ClassInfo(name='BaseClass', file_path='base.py', line_number=1),
            ClassInfo(name='Interface1', file_path='interface.py', line_number=1),
            ClassInfo(
                name='DerivedClass',
                file_path='derived.py',
                line_number=1,
                superclasses=['BaseClass'],
                interfaces=['Interface1']
            ),
            ClassInfo(
                name='UnrelatedClass',
                file_path='unrelated.py',
                line_number=1,
                superclasses=['ExternalClass']  # Not trong classes list
            )
        ]
        
        relationships = self.agent._extract_relationships(classes)
        
        assert len(relationships) == 2
        
        # Check inheritance relationship
        inheritance_rel = next(r for r in relationships if r.relationship_type == 'inheritance')
        assert inheritance_rel.source_class == 'DerivedClass'
        assert inheritance_rel.target_class == 'BaseClass'
        
        # Check implementation relationship
        impl_rel = next(r for r in relationships if r.relationship_type == 'implementation')
        assert impl_rel.source_class == 'DerivedClass'
        assert impl_rel.target_class == 'Interface1'
    
    def test_process_files_success(self):
        """Test successful file processing"""
        state = {
            'ast_results': {
                'test.py': {
                    'classes': [
                        {
                            'name': 'TestClass',
                            'lineno': 1,
                            'bases': [],
                            'body': [
                                {
                                    'type': 'FunctionDef',
                                    'name': 'test_method',
                                    'args': {'args': [{'arg': 'self'}]},
                                    'returns': None
                                }
                            ]
                        }
                    ]
                },
                'Calculator.java': {
                    'classes': [
                        {
                            'type': 'class_declaration',
                            'name': 'Calculator',
                            'start_point': {'row': 1},
                            'modifiers': ['public'],
                            'body': []
                        }
                    ]
                }
            }
        }
        
        result = self.agent.process_files(state)
        
        assert result['current_agent'] == 'diagram_generator'
        assert result['processing_status'] == 'diagram_generation_completed'
        assert 'diagrams' in result
        assert 'extracted_classes' in result
        assert 'diagram_metadata' in result
        
        # Check diagrams
        diagrams = result['diagrams']
        assert 'test.py' in diagrams
        assert 'Calculator.java' in diagrams
        assert 'project_overview' in diagrams
        
        # Check metadata
        metadata = result['diagram_metadata']
        assert metadata['total_diagrams'] == 3
        assert metadata['total_classes'] == 2
        assert metadata['files_processed'] == 2
        
        # Check extracted classes
        extracted_classes = result['extracted_classes']
        assert len(extracted_classes) == 2
        assert any(cls['name'] == 'TestClass' for cls in extracted_classes)
        assert any(cls['name'] == 'Calculator' for cls in extracted_classes)
    
    def test_process_files_no_ast_data(self):
        """Test processing files với no AST data"""
        state = {}
        
        result = self.agent.process_files(state)
        
        assert result['current_agent'] == 'diagram_generator'
        assert result['processing_status'] == 'no_ast_data'
        assert result['diagrams'] == {}
    
    def test_process_files_empty_ast_results(self):
        """Test processing files với empty AST results"""
        state = {'ast_results': {}}
        
        result = self.agent.process_files(state)
        
        assert result['current_agent'] == 'diagram_generator'
        assert result['processing_status'] == 'no_ast_data'
        assert result['diagrams'] == {}
    
    def test_process_files_with_errors(self):
        """Test processing files với errors trong individual files"""
        state = {
            'ast_results': {
                'good_file.py': {
                    'classes': [
                        {
                            'name': 'GoodClass',
                            'lineno': 1,
                            'bases': [],
                            'body': []
                        }
                    ]
                },
                'bad_file.py': {
                    'classes': [
                        {
                            # Malformed class data
                            'invalid': 'data'
                        }
                    ]
                }
            }
        }
        
        result = self.agent.process_files(state)
        
        # Should still process successfully với partial results
        assert result['processing_status'] == 'diagram_generation_completed'
        assert 'good_file.py' in result['diagrams']
        # bad_file.py should be skipped
        assert 'bad_file.py' not in result['diagrams']
    
    @patch('deepcode_insight.agents.diagram_generator.logger')
    def test_process_files_exception_handling(self, mock_logger):
        """Test exception handling trong process_files"""
        # Create agent với mocked method that raises exception
        agent = DiagramGenerationAgent()
        
        with patch.object(agent, 'extract_class_info_from_ast', side_effect=Exception("Test error")):
            state = {
                'ast_results': {
                    'test.py': {'classes': []}
                }
            }
            
            result = agent.process_files(state)
            
            assert result['processing_status'] == 'diagram_generation_failed'
            assert 'error' in result
            assert result['diagrams'] == {}
    
    def test_factory_function(self):
        """Test factory function"""
        agent = create_diagram_generator_agent(
            include_private_members=True,
            max_classes_per_diagram=5
        )
        
        assert isinstance(agent, DiagramGenerationAgent)
        assert agent.include_private_members is True
        assert agent.max_classes_per_diagram == 5
    
    def test_langraph_node_function(self):
        """Test LangGraph node function"""
        state = {
            'ast_results': {
                'test.py': {
                    'classes': [
                        {
                            'name': 'NodeTestClass',
                            'lineno': 1,
                            'bases': [],
                            'body': []
                        }
                    ]
                }
            }
        }
        
        result = diagram_generator_node(state)
        
        assert result['current_agent'] == 'diagram_generator'
        assert 'diagrams' in result
        assert 'test.py' in result['diagrams']


class TestDataClasses:
    """Test cases for data classes"""
    
    def test_class_info_initialization(self):
        """Test ClassInfo initialization"""
        class_info = ClassInfo(
            name='TestClass',
            file_path='test.py',
            line_number=10
        )
        
        assert class_info.name == 'TestClass'
        assert class_info.file_path == 'test.py'
        assert class_info.line_number == 10
        assert class_info.is_abstract is False
        assert class_info.is_interface is False
        assert class_info.superclasses == []
        assert class_info.interfaces == []
        assert class_info.fields == []
        assert class_info.methods == []
        assert class_info.inner_classes == []
        assert class_info.visibility == 'public'
    
    def test_field_info_initialization(self):
        """Test FieldInfo initialization"""
        field_info = FieldInfo(
            name='test_field',
            type_hint='str'
        )
        
        assert field_info.name == 'test_field'
        assert field_info.type_hint == 'str'
        assert field_info.visibility == 'public'
        assert field_info.is_static is False
        assert field_info.is_final is False
        assert field_info.default_value is None
    
    def test_method_info_initialization(self):
        """Test MethodInfo initialization"""
        method_info = MethodInfo(
            name='test_method',
            return_type='void'
        )
        
        assert method_info.name == 'test_method'
        assert method_info.return_type == 'void'
        assert method_info.parameters == []
        assert method_info.visibility == 'public'
        assert method_info.is_static is False
        assert method_info.is_abstract is False
        assert method_info.is_constructor is False
        assert method_info.is_destructor is False
    
    def test_parameter_info_initialization(self):
        """Test ParameterInfo initialization"""
        param_info = ParameterInfo(
            name='param',
            type_hint='int'
        )
        
        assert param_info.name == 'param'
        assert param_info.type_hint == 'int'
        assert param_info.default_value is None
        assert param_info.is_varargs is False
    
    def test_relationship_info_initialization(self):
        """Test RelationshipInfo initialization"""
        rel_info = RelationshipInfo(
            source_class='Child',
            target_class='Parent',
            relationship_type='inheritance'
        )
        
        assert rel_info.source_class == 'Child'
        assert rel_info.target_class == 'Parent'
        assert rel_info.relationship_type == 'inheritance'
        assert rel_info.label is None
        assert rel_info.multiplicity is None


class TestComplexScenarios:
    """Test cases for complex real-world scenarios"""
    
    def test_complex_python_class_extraction(self):
        """Test extracting complex Python class với multiple features"""
        ast_data = {
            'classes': [
                {
                    'name': 'ComplexClass',
                    'lineno': 1,
                    'bases': [{'id': 'BaseClass'}, {'id': 'Mixin'}],
                    'body': [
                        # Constructor
                        {
                            'type': 'FunctionDef',
                            'name': '__init__',
                            'args': {
                                'args': [
                                    {'arg': 'self'},
                                    {'arg': 'name', 'annotation': {'id': 'str'}},
                                    {'arg': 'age', 'annotation': {'id': 'int'}}
                                ]
                            },
                            'returns': None
                        },
                        # Public method
                        {
                            'type': 'FunctionDef',
                            'name': 'get_info',
                            'args': {'args': [{'arg': 'self'}]},
                            'returns': {'id': 'str'}
                        },
                        # Private method
                        {
                            'type': 'FunctionDef',
                            'name': '_validate',
                            'args': {
                                'args': [
                                    {'arg': 'self'},
                                    {'arg': 'value', 'annotation': {'id': 'Any'}}
                                ]
                            },
                            'returns': {'id': 'bool'}
                        },
                        # Annotated field
                        {
                            'type': 'AnnAssign',
                            'target': {'id': 'name'},
                            'annotation': {'id': 'str'}
                        },
                        # Regular assignment
                        {
                            'type': 'Assign',
                            'targets': [{'id': '_private_data'}]
                        }
                    ]
                }
            ]
        }
        
        agent = DiagramGenerationAgent(include_private_members=True)
        classes = agent.extract_class_info_from_ast(ast_data, 'complex.py', 'python')
        
        assert len(classes) == 1
        complex_class = classes[0]
        
        # Check basic info
        assert complex_class.name == 'ComplexClass'
        assert complex_class.superclasses == ['BaseClass', 'Mixin']
        
        # Check methods
        assert len(complex_class.methods) == 3
        method_names = [m.name for m in complex_class.methods]
        assert '__init__' in method_names
        assert 'get_info' in method_names
        assert '_validate' in method_names
        
        # Check constructor
        constructor = next(m for m in complex_class.methods if m.name == '__init__')
        assert constructor.is_constructor is True
        assert len(constructor.parameters) == 2
        assert constructor.parameters[0].name == 'name'
        assert constructor.parameters[0].type_hint == 'str'
        
        # Check fields
        assert len(complex_class.fields) == 2
        field_names = [f.name for f in complex_class.fields]
        assert 'name' in field_names
        assert '_private_data' in field_names
        
        # Check visibility
        name_field = next(f for f in complex_class.fields if f.name == 'name')
        assert name_field.visibility == 'public'
        private_field = next(f for f in complex_class.fields if f.name == '_private_data')
        assert private_field.visibility == 'private'
    
    def test_complex_java_class_extraction(self):
        """Test extracting complex Java class"""
        ast_data = {
            'classes': [
                {
                    'type': 'class_declaration',
                    'name': 'ComplexJavaClass',
                    'start_point': {'row': 5},
                    'modifiers': ['public', 'abstract'],
                    'superclass': 'AbstractBase',
                    'interfaces': ['Serializable', 'Comparable'],
                    'body': [
                        # Static field
                        {
                            'type': 'field_declaration',
                            'name': 'CONSTANT',
                            'type': 'String',
                            'modifiers': ['public', 'static', 'final']
                        },
                        # Private field
                        {
                            'type': 'field_declaration',
                            'name': 'data',
                            'type': 'List<String>',
                            'modifiers': ['private']
                        },
                        # Public method
                        {
                            'type': 'method_declaration',
                            'name': 'processData',
                            'type': 'void',
                            'modifiers': ['public'],
                            'parameters': [
                                {'name': 'input', 'type': 'String'},
                                {'name': 'options', 'type': 'Map<String, Object>'}
                            ]
                        },
                        # Abstract method
                        {
                            'type': 'method_declaration',
                            'name': 'abstractMethod',
                            'type': 'int',
                            'modifiers': ['protected', 'abstract'],
                            'parameters': []
                        }
                    ]
                }
            ]
        }
        
        agent = DiagramGenerationAgent(include_private_members=True)
        classes = agent.extract_class_info_from_ast(ast_data, 'Complex.java', 'java')
        
        assert len(classes) == 1
        complex_class = classes[0]
        
        # Check basic info
        assert complex_class.name == 'ComplexJavaClass'
        assert complex_class.is_abstract is True
        assert complex_class.superclasses == ['AbstractBase']
        assert complex_class.interfaces == ['Serializable', 'Comparable']
        
        # Check fields
        assert len(complex_class.fields) == 2
        constant_field = next(f for f in complex_class.fields if f.name == 'CONSTANT')
        assert constant_field.is_static is True
        assert constant_field.is_final is True
        assert constant_field.visibility == 'public'
        
        data_field = next(f for f in complex_class.fields if f.name == 'data')
        assert data_field.visibility == 'private'
        assert data_field.type_hint == 'List<String>'
        
        # Check methods
        assert len(complex_class.methods) == 2
        process_method = next(m for m in complex_class.methods if m.name == 'processData')
        assert process_method.visibility == 'public'
        assert len(process_method.parameters) == 2
        
        abstract_method = next(m for m in complex_class.methods if m.name == 'abstractMethod')
        assert abstract_method.is_abstract is True
        assert abstract_method.visibility == 'protected'
    
    def test_generate_complex_diagram_with_all_features(self):
        """Test generating complex diagram với all features"""
        classes = [
            # Abstract base class
            ClassInfo(
                name='AbstractShape',
                file_path='shape.py',
                line_number=1,
                is_abstract=True,
                fields=[
                    FieldInfo(name='color', type_hint='str', visibility='protected'),
                    FieldInfo(name='_id', type_hint='int', visibility='private', is_static=True)
                ],
                methods=[
                    MethodInfo(
                        name='__init__',
                        return_type='None',
                        visibility='public',
                        is_constructor=True,
                        parameters=[
                            ParameterInfo(name='color', type_hint='str')
                        ]
                    ),
                    MethodInfo(
                        name='area',
                        return_type='float',
                        visibility='public',
                        is_abstract=True
                    ),
                    MethodInfo(
                        name='_validate_color',
                        return_type='bool',
                        visibility='private',
                        parameters=[
                            ParameterInfo(name='color', type_hint='str')
                        ]
                    )
                ]
            ),
            # Interface
            ClassInfo(
                name='Drawable',
                file_path='drawable.py',
                line_number=1,
                is_interface=True,
                methods=[
                    MethodInfo(
                        name='draw',
                        return_type='void',
                        visibility='public',
                        is_abstract=True,
                        parameters=[
                            ParameterInfo(name='canvas', type_hint='Canvas')
                        ]
                    )
                ]
            ),
            # Concrete implementation
            ClassInfo(
                name='Circle',
                file_path='circle.py',
                line_number=1,
                superclasses=['AbstractShape'],
                interfaces=['Drawable'],
                fields=[
                    FieldInfo(name='radius', type_hint='float', visibility='public'),
                    FieldInfo(name='PI', type_hint='float', visibility='public', is_static=True, is_final=True)
                ],
                methods=[
                    MethodInfo(
                        name='__init__',
                        return_type='None',
                        visibility='public',
                        is_constructor=True,
                        parameters=[
                            ParameterInfo(name='radius', type_hint='float'),
                            ParameterInfo(name='color', type_hint='str', default_value='red')
                        ]
                    ),
                    MethodInfo(
                        name='area',
                        return_type='float',
                        visibility='public'
                    ),
                    MethodInfo(
                        name='draw',
                        return_type='void',
                        visibility='public',
                        parameters=[
                            ParameterInfo(name='canvas', type_hint='Canvas')
                        ]
                    ),
                    MethodInfo(
                        name='calculate_circumference',
                        return_type='float',
                        visibility='public',
                        is_static=True
                    )
                ]
            )
        ]
        
        agent = DiagramGenerationAgent(include_private_members=True)
        diagram = agent.generate_class_diagram(classes, "Complex Shape Hierarchy")
        
        # Check that all classes are included
        assert "abstract class AbstractShape" in diagram
        assert "interface Drawable" in diagram
        assert "class Circle" in diagram
        
        # Check relationships
        assert "Circle --|> AbstractShape" in diagram
        assert "Circle ..|> Drawable" in diagram
        
        # Check fields với different modifiers
        assert "# color: str" in diagram  # protected
        assert "- {static} _id: int" in diagram  # private static
        assert "+ radius: float" in diagram  # public
        assert "+ {static} PI: float" in diagram  # public static final
        
        # Check methods với different modifiers
        assert "+ __init__(radius: float, color: str)" in diagram  # constructor với params
        assert "+ {abstract} area(): float" in diagram  # abstract method
        assert "- _validate_color(color: str): bool" in diagram  # private method
        assert "+ {static} calculate_circumference(): float" in diagram  # static method
        
        # Check interface method
        assert "+ {abstract} draw(canvas: Canvas)" in diagram


if __name__ == '__main__':
    pytest.main([__file__]) 