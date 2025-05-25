"""
Debug script để test DiagramGenerationAgent output
"""

from deepcode_insight.agents.diagram_generator import DiagramGenerationAgent

def test_simple_case():
    """Test simple case để debug"""
    agent = DiagramGenerationAgent(
        include_private_members=True, 
        include_method_parameters=True, 
        include_field_types=True
    )
    
    python_ast = {
        'classes': [
            {
                'name': 'Calculator',
                'lineno': 1,
                'bases': [],
                'body': [
                    {
                        'type': 'FunctionDef',
                        'name': '__init__',
                        'args': {
                            'args': [
                                {'arg': 'self'}
                            ]
                        },
                        'returns': None
                    },
                    {
                        'type': 'AnnAssign',
                        'target': {'id': 'result'},
                        'annotation': {'id': 'int'}
                    }
                ]
            }
        ]
    }
    
    classes = agent.extract_class_info_from_ast(python_ast, 'calculator.py', 'python')
    print('Extracted classes:', len(classes))
    if classes:
        cls = classes[0]
        print('Class name:', cls.name)
        print('Fields:', [(f.name, f.visibility, f.type_hint) for f in cls.fields])
        print('Methods:', [(m.name, m.visibility, m.is_constructor) for m in cls.methods])
    
    diagram = agent.generate_class_diagram(classes, 'Calculator Class')
    print('\nGenerated diagram:')
    print(diagram)
    print('\n' + '='*50)

def test_java_case():
    """Test Java case"""
    agent = DiagramGenerationAgent(
        include_private_members=True, 
        include_method_parameters=True, 
        include_field_types=True
    )
    
    java_ast = {
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
                        'parameters': []
                    }
                ]
            }
        ]
    }
    
    classes = agent.extract_class_info_from_ast(java_ast, 'shapes.java', 'java')
    print('Java classes:', len(classes))
    if classes:
        cls = classes[0]
        print('Class name:', cls.name)
        print('Is interface:', cls.is_interface)
        print('Methods:', [(m.name, m.visibility, m.is_abstract) for m in cls.methods])
    
    diagram = agent.generate_class_diagram(classes, 'Java Interface')
    print('\nJava diagram:')
    print(diagram)

if __name__ == '__main__':
    test_simple_case()
    test_java_case() 