"""
Edge cases và performance tests for DiagramGenerationAgent
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from deepcode_insight.agents.diagram_generator import (
    DiagramGenerationAgent,
    ClassInfo,
    FieldInfo,
    MethodInfo,
    ParameterInfo,
    RelationshipInfo
)


class TestEdgeCases:
    """Test edge cases và unusual scenarios"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.agent = DiagramGenerationAgent(
            include_private_members=True,
            include_method_parameters=True,
            include_field_types=True
        )
    
    def test_malformed_python_ast_graceful_handling(self):
        """Test graceful handling của malformed Python AST"""
        # Malformed AST với missing required fields
        malformed_ast = {
            'classes': [
                {
                    # Missing 'name' field
                    'lineno': 1,
                    'body': [
                        {
                            'type': 'FunctionDef',
                            # Missing 'name' field
                            'args': {'args': [{'arg': 'self'}]},
                            'returns': None
                        }
                    ]
                },
                {
                    'name': 'ValidClass',
                    # Missing 'lineno'
                    'bases': [],
                    'body': []
                },
                {
                    'name': 'ClassWithBadMethod',
                    'lineno': 10,
                    'bases': [],
                    'body': [
                        {
                            'type': 'FunctionDef',
                            'name': 'good_method',
                            'args': {'args': [{'arg': 'self'}]},
                            'returns': None
                        },
                        {
                            # Malformed method
                            'type': 'FunctionDef',
                            'name': 'bad_method',
                            # Missing 'args'
                            'returns': None
                        }
                    ]
                }
            ]
        }
        
        # Should handle gracefully without crashing
        classes = self.agent.extract_class_info_from_ast(malformed_ast, 'malformed.py', 'python')
        
        # Should extract what it can
        assert len(classes) >= 1  # At least ValidClass should be extracted
        
        # Find ValidClass
        valid_class = next((cls for cls in classes if cls.name == 'ValidClass'), None)
        assert valid_class is not None
        assert valid_class.line_number == 0  # Default value
        
        # Find ClassWithBadMethod
        bad_method_class = next((cls for cls in classes if cls.name == 'ClassWithBadMethod'), None)
        if bad_method_class:
            # Should have extracted good_method but not bad_method
            method_names = [m.name for m in bad_method_class.methods]
            assert 'good_method' in method_names
            # bad_method might be skipped or have default values
    
    def test_malformed_java_ast_graceful_handling(self):
        """Test graceful handling của malformed Java AST"""
        malformed_ast = {
            'classes': [
                {
                    'type': 'class_declaration',
                    # Missing 'name'
                    'start_point': {'row': 1},
                    'body': []
                },
                {
                    'type': 'interface_declaration',
                    'name': 'ValidInterface',
                    # Missing 'start_point'
                    'body': [
                        {
                            'type': 'method_declaration',
                            'name': 'validMethod',
                            'type': 'void',
                            'modifiers': ['public'],
                            'parameters': []
                        },
                        {
                            # Malformed method
                            'type': 'method_declaration',
                            # Missing 'name'
                            'type': 'int',
                            'modifiers': ['public'],
                            'parameters': []
                        }
                    ]
                }
            ]
        }
        
        # Should handle gracefully
        classes = self.agent.extract_class_info_from_ast(malformed_ast, 'malformed.java', 'java')
        
        # Should extract what it can
        valid_interface = next((cls for cls in classes if cls.name == 'ValidInterface'), None)
        if valid_interface:
            assert valid_interface.is_interface is True
            assert valid_interface.line_number == 0  # Default value
            # Should have extracted validMethod
            method_names = [m.name for m in valid_interface.methods]
            assert 'validMethod' in method_names
    
    def test_deeply_nested_ast_structure(self):
        """Test deeply nested AST structure"""
        # AST với nested classes (inner classes)
        nested_ast = {
            'ast': {
                'body': [
                    {
                        'type': 'ClassDef',
                        'name': 'OuterClass',
                        'lineno': 1,
                        'bases': [],
                        'body': [
                            {
                                'type': 'FunctionDef',
                                'name': 'outer_method',
                                'args': {'args': [{'arg': 'self'}]},
                                'returns': None
                            },
                            {
                                'type': 'ClassDef',
                                'name': 'InnerClass',
                                'lineno': 10,
                                'bases': [],
                                'body': [
                                    {
                                        'type': 'FunctionDef',
                                        'name': 'inner_method',
                                        'args': {'args': [{'arg': 'self'}]},
                                        'returns': None
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }
        
        # Extract classes
        classes = self.agent.extract_class_info_from_ast(nested_ast, 'nested.py', 'python')
        
        # Should find both outer và inner classes
        class_names = [cls.name for cls in classes]
        assert 'OuterClass' in class_names
        assert 'InnerClass' in class_names
        
        # Verify methods
        outer_class = next(cls for cls in classes if cls.name == 'OuterClass')
        inner_class = next(cls for cls in classes if cls.name == 'InnerClass')
        
        assert len(outer_class.methods) == 1
        assert outer_class.methods[0].name == 'outer_method'
        
        assert len(inner_class.methods) == 1
        assert inner_class.methods[0].name == 'inner_method'
    
    def test_unicode_and_special_characters_in_names(self):
        """Test Unicode và special characters trong class/method names"""
        unicode_ast = {
            'classes': [
                {
                    'name': 'ClassWithUnicode_测试',
                    'lineno': 1,
                    'bases': [],
                    'body': [
                        {
                            'type': 'FunctionDef',
                            'name': 'method_with_unicode_测试',
                            'args': {'args': [{'arg': 'self'}]},
                            'returns': None
                        },
                        {
                            'type': 'FunctionDef',
                            'name': '__special__method__',
                            'args': {'args': [{'arg': 'self'}]},
                            'returns': None
                        },
                        {
                            'type': 'AnnAssign',
                            'target': {'id': 'field_with_unicode_测试'},
                            'annotation': {'id': 'str'}
                        }
                    ]
                }
            ]
        }
        
        # Extract classes
        classes = self.agent.extract_class_info_from_ast(unicode_ast, 'unicode.py', 'python')
        
        # Should handle Unicode characters
        assert len(classes) == 1
        cls = classes[0]
        assert cls.name == 'ClassWithUnicode_测试'
        
        # Check methods
        method_names = [m.name for m in cls.methods]
        assert 'method_with_unicode_测试' in method_names
        assert '__special__method__' in method_names
        
        # Check fields
        field_names = [f.name for f in cls.fields]
        assert 'field_with_unicode_测试' in field_names
        
        # Generate PlantUML - should not crash
        diagram = self.agent.generate_class_diagram(classes, "Unicode Test")
        assert 'ClassWithUnicode_测试' in diagram
    
    def test_extremely_long_parameter_lists(self):
        """Test methods với extremely long parameter lists"""
        long_params_ast = {
            'classes': [
                {
                    'name': 'LongParamsClass',
                    'lineno': 1,
                    'bases': [],
                    'body': [
                        {
                            'type': 'FunctionDef',
                            'name': 'method_with_many_params',
                            'args': {
                                'args': [
                                    {'arg': 'self'},
                                    *[{'arg': f'param_{i}', 'annotation': {'id': 'str'}} for i in range(20)]
                                ]
                            },
                            'returns': {'id': 'bool'}
                        }
                    ]
                }
            ]
        }
        
        # Extract classes
        classes = self.agent.extract_class_info_from_ast(long_params_ast, 'long_params.py', 'python')
        
        # Should handle long parameter lists
        assert len(classes) == 1
        cls = classes[0]
        assert len(cls.methods) == 1
        method = cls.methods[0]
        assert len(method.parameters) == 20
        
        # Generate PlantUML - should not crash
        diagram = self.agent.generate_class_diagram(classes, "Long Parameters")
        assert 'method_with_many_params' in diagram
        
        # Test với include_method_parameters=False
        agent_no_params = DiagramGenerationAgent(include_method_parameters=False)
        diagram_no_params = agent_no_params.generate_class_diagram(classes, "No Parameters")
        assert 'method_with_many_params(): bool' in diagram_no_params
        assert 'param_0' not in diagram_no_params
    
    def test_circular_inheritance_detection(self):
        """Test detection của circular inheritance"""
        # Note: This tests the relationship extraction, not actual circular inheritance
        # which would be invalid in most languages
        circular_ast = {
            'classes': [
                {
                    'name': 'ClassA',
                    'lineno': 1,
                    'bases': [{'id': 'ClassB'}],
                    'body': []
                },
                {
                    'name': 'ClassB',
                    'lineno': 5,
                    'bases': [{'id': 'ClassC'}],
                    'body': []
                },
                {
                    'name': 'ClassC',
                    'lineno': 10,
                    'bases': [{'id': 'ClassA'}],  # Creates circular reference
                    'body': []
                }
            ]
        }
        
        # Extract classes
        classes = self.agent.extract_class_info_from_ast(circular_ast, 'circular.py', 'python')
        
        # Should extract all classes
        assert len(classes) == 3
        class_names = [cls.name for cls in classes]
        assert 'ClassA' in class_names
        assert 'ClassB' in class_names
        assert 'ClassC' in class_names
        
        # Generate PlantUML - should handle circular references
        diagram = self.agent.generate_class_diagram(classes, "Circular Inheritance")
        
        # Should show all relationships
        assert 'ClassA --|> ClassB' in diagram
        assert 'ClassB --|> ClassC' in diagram
        assert 'ClassC --|> ClassA' in diagram
    
    def test_empty_and_none_values_handling(self):
        """Test handling của empty và None values"""
        empty_values_ast = {
            'classes': [
                {
                    'name': 'TestClass',
                    'lineno': 1,
                    'bases': None,  # None instead of list
                    'body': [
                        {
                            'type': 'FunctionDef',
                            'name': 'method_with_none_return',
                            'args': {
                                'args': [
                                    {'arg': 'self'},
                                    {'arg': 'param_no_annotation'}  # No annotation
                                ]
                            },
                            'returns': None
                        },
                        {
                            'type': 'AnnAssign',
                            'target': {'id': 'field_no_annotation'},
                            'annotation': None  # None annotation
                        }
                    ]
                }
            ]
        }
        
        # Should handle None values gracefully
        classes = self.agent.extract_class_info_from_ast(empty_values_ast, 'empty_values.py', 'python')
        
        assert len(classes) == 1
        cls = classes[0]
        assert cls.superclasses == []  # Should default to empty list
        
        # Check method
        assert len(cls.methods) == 1
        method = cls.methods[0]
        assert method.return_type == 'None'
        assert len(method.parameters) == 1
        assert method.parameters[0].type_hint == 'Any'  # Default for missing annotation
        
        # Check field
        assert len(cls.fields) == 1
        field = cls.fields[0]
        assert field.type_hint == 'Any'  # Default for None annotation
    
    def test_very_large_class_count_performance(self):
        """Test performance với very large number of classes"""
        # Generate AST với many classes
        large_ast = {
            'classes': [
                {
                    'name': f'Class_{i:04d}',
                    'lineno': i,
                    'bases': [{'id': f'Class_{(i-1):04d}'}] if i > 0 else [],
                    'body': [
                        {
                            'type': 'FunctionDef',
                            'name': f'method_{j}',
                            'args': {'args': [{'arg': 'self'}]},
                            'returns': None
                        }
                        for j in range(3)  # 3 methods per class
                    ]
                }
                for i in range(100)  # 100 classes
            ]
        }
        
        # Extract classes - should complete in reasonable time
        import time
        start_time = time.time()
        classes = self.agent.extract_class_info_from_ast(large_ast, 'large.py', 'python')
        extraction_time = time.time() - start_time
        
        assert len(classes) == 100
        assert extraction_time < 5.0  # Should complete within 5 seconds
        
        # Generate diagram với max_classes_per_diagram limit
        agent_limited = DiagramGenerationAgent(max_classes_per_diagram=10)
        start_time = time.time()
        diagram = agent_limited.generate_class_diagram(classes, "Large Project")
        generation_time = time.time() - start_time
        
        assert generation_time < 2.0  # Should complete within 2 seconds
        assert diagram.count('class Class_') == 10  # Should limit to 10 classes
    
    def test_memory_usage_with_large_diagrams(self):
        """Test memory usage với large diagrams"""
        # Create classes với many fields và methods
        memory_test_ast = {
            'classes': [
                {
                    'name': f'MemoryTestClass_{i}',
                    'lineno': i,
                    'bases': [],
                    'body': [
                        # Many fields
                        *[
                            {
                                'type': 'AnnAssign',
                                'target': {'id': f'field_{j}'},
                                'annotation': {'id': 'str'}
                            }
                            for j in range(50)
                        ],
                        # Many methods
                        *[
                            {
                                'type': 'FunctionDef',
                                'name': f'method_{k}',
                                'args': {
                                    'args': [
                                        {'arg': 'self'},
                                        *[{'arg': f'param_{l}', 'annotation': {'id': 'int'}} for l in range(10)]
                                    ]
                                },
                                'returns': {'id': 'bool'}
                            }
                            for k in range(20)
                        ]
                    ]
                }
                for i in range(10)
            ]
        }
        
        # Extract classes
        classes = self.agent.extract_class_info_from_ast(memory_test_ast, 'memory_test.py', 'python')
        
        # Verify extraction
        assert len(classes) == 10
        for cls in classes:
            assert len(cls.fields) == 50
            assert len(cls.methods) == 20
            for method in cls.methods:
                assert len(method.parameters) == 10
        
        # Generate diagram - should not cause memory issues
        diagram = self.agent.generate_class_diagram(classes, "Memory Test")
        
        # Verify diagram contains expected content
        assert len(diagram.split('\n')) > 100  # Should be a large diagram
        assert diagram.count('class MemoryTestClass_') == 10


class TestErrorRecovery:
    """Test error recovery scenarios"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.agent = DiagramGenerationAgent()
    
    def test_partial_ast_processing_with_errors(self):
        """Test partial processing khi một số classes có errors"""
        mixed_ast = {
            'classes': [
                # Valid class
                {
                    'name': 'ValidClass1',
                    'lineno': 1,
                    'bases': [],
                    'body': [
                        {
                            'type': 'FunctionDef',
                            'name': 'valid_method',
                            'args': {'args': [{'arg': 'self'}]},
                            'returns': None
                        }
                    ]
                },
                # Invalid class (will cause error)
                {
                    'name': 'InvalidClass',
                    'lineno': 'not_a_number',  # Invalid line number
                    'bases': 'not_a_list',     # Invalid bases
                    'body': [
                        {
                            'type': 'FunctionDef',
                            'name': 'method',
                            'args': 'not_a_dict',  # Invalid args
                            'returns': None
                        }
                    ]
                },
                # Another valid class
                {
                    'name': 'ValidClass2',
                    'lineno': 10,
                    'bases': [],
                    'body': []
                }
            ]
        }
        
        # Should extract valid classes despite errors
        classes = self.agent.extract_class_info_from_ast(mixed_ast, 'mixed.py', 'python')
        
        # Should have extracted at least the valid classes
        class_names = [cls.name for cls in classes]
        assert 'ValidClass1' in class_names
        assert 'ValidClass2' in class_names
        
        # Generate diagram với partial data
        diagram = self.agent.generate_class_diagram(classes, "Partial Recovery")
        assert 'ValidClass1' in diagram
        assert 'ValidClass2' in diagram
    
    @patch('deepcode_insight.agents.diagram_generator.logger')
    def test_logging_during_error_recovery(self, mock_logger):
        """Test logging during error recovery"""
        # AST that will cause errors
        error_ast = {
            'classes': [
                {
                    'name': 'ErrorClass',
                    'lineno': 1,
                    'bases': [],
                    'body': [
                        {
                            'type': 'FunctionDef',
                            'name': 'error_method',
                            'args': None,  # Will cause error
                            'returns': None
                        }
                    ]
                }
            ]
        }
        
        # Extract classes - should log errors
        classes = self.agent.extract_class_info_from_ast(error_ast, 'error.py', 'python')
        
        # Verify logging was called
        assert mock_logger.error.called
    
    def test_state_consistency_after_errors(self):
        """Test state consistency sau khi có errors"""
        # Process valid state first
        valid_state = {
            'ast_results': {
                'valid.py': {
                    'classes': [
                        {
                            'name': 'ValidClass',
                            'lineno': 1,
                            'bases': [],
                            'body': []
                        }
                    ]
                }
            }
        }
        
        result1 = self.agent.process_files(valid_state)
        assert result1['processing_status'] == 'diagram_generation_completed'
        assert len(result1['diagrams']) > 0
        
        # Process invalid state
        invalid_state = {
            'ast_results': {
                'invalid.py': {
                    'classes': [
                        {
                            'name': None,  # Invalid name
                            'lineno': 'invalid',
                            'body': 'not_a_list'
                        }
                    ]
                }
            }
        }
        
        result2 = self.agent.process_files(invalid_state)
        # Should handle gracefully
        assert 'processing_status' in result2
        assert 'diagrams' in result2
        
        # Process valid state again - should still work
        result3 = self.agent.process_files(valid_state)
        assert result3['processing_status'] == 'diagram_generation_completed'


if __name__ == '__main__':
    pytest.main([__file__]) 