"""
Debug Java fields parsing
"""

from deepcode_insight.agents.diagram_generator import DiagramGenerationAgent

def test_java_fields():
    """Test Java fields parsing"""
    agent = DiagramGenerationAgent(
        include_private_members=True, 
        include_method_parameters=True, 
        include_field_types=True
    )
    
    java_ast = {
        'classes': [
            {
                'type': 'class_declaration',
                'name': 'Circle',
                'start_point': {'row': 10},
                'modifiers': ['public'],
                'interfaces': ['Drawable'],
                'body': [
                    {
                        'type': 'field_declaration',
                        'name': 'radius',
                        'type': 'double',
                        'modifiers': ['private']
                    },
                    {
                        'name': 'draw',
                        'type': 'void',
                        'modifiers': ['public'],
                        'parameters': []
                    }
                ]
            }
        ]
    }
    
    print("Input AST:")
    print(java_ast)
    
    classes = agent.extract_class_info_from_ast(java_ast, 'shapes.java', 'java')
    print('\nExtracted classes:', len(classes))
    
    if classes:
        cls = classes[0]
        print('Class name:', cls.name)
        print('Body from AST:', java_ast['classes'][0]['body'])
        print('Extracted fields:', len(cls.fields))
        for field in cls.fields:
            print(f'  Field: {field.name}, type: {field.type_hint}, visibility: {field.visibility}')
        print('Extracted methods:', len(cls.methods))
        for method in cls.methods:
            print(f'  Method: {method.name}, type: {method.return_type}')

if __name__ == '__main__':
    test_java_fields() 