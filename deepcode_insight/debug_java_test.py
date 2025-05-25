"""
Debug Java parsing
"""

from deepcode_insight.agents.diagram_generator import DiagramGenerationAgent

def test_java_parsing():
    """Test Java parsing"""
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
    
    print("Input AST:")
    print(java_ast)
    
    classes = agent.extract_class_info_from_ast(java_ast, 'shapes.java', 'java')
    print('\nExtracted classes:', len(classes))
    
    if classes:
        cls = classes[0]
        print('Class name:', cls.name)
        print('Is interface:', cls.is_interface)
        print('Body from AST:', java_ast['classes'][0]['body'])
        print('Extracted methods:', len(cls.methods))
        for method in cls.methods:
            print(f'  Method: {method.name}, visibility: {method.visibility}, abstract: {method.is_abstract}')
        
        # Debug the parsing
        print('\nDebugging _parse_java_method:')
        method_node = java_ast['classes'][0]['body'][0]
        print('Method node:', method_node)
        
        parsed_method = agent._parse_java_method(method_node)
        print('Parsed method:', parsed_method)

if __name__ == '__main__':
    test_java_parsing() 