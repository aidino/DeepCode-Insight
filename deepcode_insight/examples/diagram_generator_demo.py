"""
Demo script for DiagramGenerationAgent
Showcases class diagram generation từ sample AST data
"""

import json
from typing import Dict, Any
from deepcode_insight.agents.diagram_generator import (
    DiagramGenerationAgent,
    create_diagram_generator_agent,
    diagram_generator_node
)


def create_sample_python_ast() -> Dict[str, Any]:
    """Create sample Python AST data"""
    return {
        'classes': [
            {
                'name': 'Animal',
                'lineno': 1,
                'bases': [],
                'body': [
                    {
                        'type': 'FunctionDef',
                        'name': '__init__',
                        'args': {
                            'args': [
                                {'arg': 'self'},
                                {'arg': 'name', 'annotation': {'id': 'str'}},
                                {'arg': 'species', 'annotation': {'id': 'str'}}
                            ]
                        },
                        'returns': None
                    },
                    {
                        'type': 'FunctionDef',
                        'name': 'make_sound',
                        'args': {'args': [{'arg': 'self'}]},
                        'returns': {'id': 'str'}
                    },
                    {
                        'type': 'FunctionDef',
                        'name': '_validate_name',
                        'args': {
                            'args': [
                                {'arg': 'self'},
                                {'arg': 'name', 'annotation': {'id': 'str'}}
                            ]
                        },
                        'returns': {'id': 'bool'}
                    },
                    {
                        'type': 'AnnAssign',
                        'target': {'id': 'name'},
                        'annotation': {'id': 'str'}
                    },
                    {
                        'type': 'AnnAssign',
                        'target': {'id': 'species'},
                        'annotation': {'id': 'str'}
                    },
                    {
                        'type': 'Assign',
                        'targets': [{'id': '_id'}]
                    }
                ]
            },
            {
                'name': 'Dog',
                'lineno': 25,
                'bases': [{'id': 'Animal'}],
                'body': [
                    {
                        'type': 'FunctionDef',
                        'name': '__init__',
                        'args': {
                            'args': [
                                {'arg': 'self'},
                                {'arg': 'name', 'annotation': {'id': 'str'}},
                                {'arg': 'breed', 'annotation': {'id': 'str'}}
                            ]
                        },
                        'returns': None
                    },
                    {
                        'type': 'FunctionDef',
                        'name': 'make_sound',
                        'args': {'args': [{'arg': 'self'}]},
                        'returns': {'id': 'str'}
                    },
                    {
                        'type': 'FunctionDef',
                        'name': 'fetch',
                        'args': {
                            'args': [
                                {'arg': 'self'},
                                {'arg': 'item', 'annotation': {'id': 'str'}}
                            ]
                        },
                        'returns': {'id': 'bool'}
                    },
                    {
                        'type': 'AnnAssign',
                        'target': {'id': 'breed'},
                        'annotation': {'id': 'str'}
                    }
                ]
            }
        ]
    }


def create_sample_java_ast() -> Dict[str, Any]:
    """Create sample Java AST data"""
    return {
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
                            {'name': 'graphics', 'type': 'Graphics2D'}
                        ]
                    },
                    {
                        'type': 'method_declaration',
                        'name': 'getBounds',
                        'type': 'Rectangle',
                        'modifiers': ['public', 'abstract'],
                        'parameters': []
                    }
                ]
            },
            {
                'type': 'class_declaration',
                'name': 'Shape',
                'start_point': {'row': 10},
                'modifiers': ['public', 'abstract'],
                'interfaces': ['Drawable'],
                'body': [
                    {
                        'type': 'field_declaration',
                        'name': 'color',
                        'type': 'Color',
                        'modifiers': ['protected']
                    },
                    {
                        'type': 'field_declaration',
                        'name': 'SHAPE_COUNT',
                        'type': 'int',
                        'modifiers': ['private', 'static']
                    },
                    {
                        'type': 'method_declaration',
                        'name': 'getColor',
                        'type': 'Color',
                        'modifiers': ['public'],
                        'parameters': []
                    },
                    {
                        'type': 'method_declaration',
                        'name': 'setColor',
                        'type': 'void',
                        'modifiers': ['public'],
                        'parameters': [
                            {'name': 'color', 'type': 'Color'}
                        ]
                    },
                    {
                        'type': 'method_declaration',
                        'name': 'calculateArea',
                        'type': 'double',
                        'modifiers': ['public', 'abstract'],
                        'parameters': []
                    }
                ]
            },
            {
                'type': 'class_declaration',
                'name': 'Circle',
                'start_point': {'row': 40},
                'modifiers': ['public'],
                'superclass': 'Shape',
                'body': [
                    {
                        'type': 'field_declaration',
                        'name': 'radius',
                        'type': 'double',
                        'modifiers': ['private']
                    },
                    {
                        'type': 'field_declaration',
                        'name': 'PI',
                        'type': 'double',
                        'modifiers': ['public', 'static', 'final']
                    },
                    {
                        'type': 'method_declaration',
                        'name': 'Circle',
                        'type': 'void',
                        'modifiers': ['public'],
                        'parameters': [
                            {'name': 'radius', 'type': 'double'}
                        ]
                    },
                    {
                        'type': 'method_declaration',
                        'name': 'calculateArea',
                        'type': 'double',
                        'modifiers': ['public'],
                        'parameters': []
                    },
                    {
                        'type': 'method_declaration',
                        'name': 'draw',
                        'type': 'void',
                        'modifiers': ['public'],
                        'parameters': [
                            {'name': 'graphics', 'type': 'Graphics2D'}
                        ]
                    },
                    {
                        'type': 'method_declaration',
                        'name': 'getBounds',
                        'type': 'Rectangle',
                        'modifiers': ['public'],
                        'parameters': []
                    }
                ]
            }
        ]
    }


def demo_basic_functionality():
    """Demo basic DiagramGenerationAgent functionality"""
    print("=" * 80)
    print("DEMO: DiagramGenerationAgent Basic Functionality")
    print("=" * 80)
    
    # Create agent với different configurations
    print("\n1. Creating DiagramGenerationAgent với different configurations...")
    
    # Default configuration
    agent_default = DiagramGenerationAgent()
    print(f"   Default agent: private_members={agent_default.include_private_members}, "
          f"parameters={agent_default.include_method_parameters}, "
          f"field_types={agent_default.include_field_types}")
    
    # Custom configuration
    agent_custom = DiagramGenerationAgent(
        include_private_members=True,
        include_method_parameters=True,
        include_field_types=True,
        max_classes_per_diagram=5
    )
    print(f"   Custom agent: private_members={agent_custom.include_private_members}, "
          f"parameters={agent_custom.include_method_parameters}, "
          f"field_types={agent_custom.include_field_types}")


def demo_python_class_extraction():
    """Demo Python class extraction"""
    print("\n" + "=" * 80)
    print("DEMO: Python Class Extraction")
    print("=" * 80)
    
    agent = DiagramGenerationAgent(include_private_members=True)
    python_ast = create_sample_python_ast()
    
    print("\n2. Sample Python AST structure:")
    print(json.dumps(python_ast, indent=2)[:500] + "...")
    
    print("\n3. Extracting class information...")
    classes = agent.extract_class_info_from_ast(python_ast, 'animals.py', 'python')
    
    print(f"\n   Extracted {len(classes)} classes:")
    for cls in classes:
        print(f"   - {cls.name} (line {cls.line_number})")
        print(f"     Superclasses: {cls.superclasses}")
        print(f"     Fields: {[f.name + ':' + f.type_hint for f in cls.fields]}")
        print(f"     Methods: {[m.name + '()' for m in cls.methods]}")
        print()


def demo_java_class_extraction():
    """Demo Java class extraction"""
    print("\n" + "=" * 80)
    print("DEMO: Java Class Extraction")
    print("=" * 80)
    
    agent = DiagramGenerationAgent(include_private_members=True)
    java_ast = create_sample_java_ast()
    
    print("\n4. Sample Java AST structure:")
    print(json.dumps(java_ast, indent=2)[:500] + "...")
    
    print("\n5. Extracting class information...")
    classes = agent.extract_class_info_from_ast(java_ast, 'shapes/Shape.java', 'java')
    
    print(f"\n   Extracted {len(classes)} classes:")
    for cls in classes:
        print(f"   - {cls.name} (line {cls.line_number})")
        print(f"     Type: {'Interface' if cls.is_interface else 'Abstract Class' if cls.is_abstract else 'Class'}")
        print(f"     Superclasses: {cls.superclasses}")
        print(f"     Interfaces: {cls.interfaces}")
        print(f"     Fields: {[f.name + ':' + f.type_hint for f in cls.fields]}")
        print(f"     Methods: {[m.name + '()' for m in cls.methods]}")
        print()


def demo_plantuml_generation():
    """Demo PlantUML diagram generation"""
    print("\n" + "=" * 80)
    print("DEMO: PlantUML Diagram Generation")
    print("=" * 80)
    
    agent = DiagramGenerationAgent(include_private_members=True)
    
    # Extract classes từ both Python và Java
    python_ast = create_sample_python_ast()
    java_ast = create_sample_java_ast()
    
    python_classes = agent.extract_class_info_from_ast(python_ast, 'animals.py', 'python')
    java_classes = agent.extract_class_info_from_ast(java_ast, 'shapes/Shape.java', 'java')
    
    print("\n6. Generating PlantUML diagrams...")
    
    # Python diagram
    print("\n   Python Animal Hierarchy Diagram:")
    python_diagram = agent.generate_class_diagram(python_classes, "Python Animal Hierarchy")
    print("   " + "─" * 60)
    for line in python_diagram.split('\n'):
        print(f"   {line}")
    print("   " + "─" * 60)
    
    # Java diagram
    print("\n   Java Shape Hierarchy Diagram:")
    java_diagram = agent.generate_class_diagram(java_classes, "Java Shape Hierarchy")
    print("   " + "─" * 60)
    for line in java_diagram.split('\n'):
        print(f"   {line}")
    print("   " + "─" * 60)
    
    # Combined diagram
    print("\n   Combined Multi-Language Diagram:")
    all_classes = python_classes + java_classes
    combined_diagram = agent.generate_class_diagram(all_classes, "Multi-Language Class Diagram")
    print("   " + "─" * 60)
    for line in combined_diagram.split('\n'):
        print(f"   {line}")
    print("   " + "─" * 60)


def demo_configuration_variations():
    """Demo different configuration variations"""
    print("\n" + "=" * 80)
    print("DEMO: Configuration Variations")
    print("=" * 80)
    
    python_ast = create_sample_python_ast()
    classes = DiagramGenerationAgent().extract_class_info_from_ast(python_ast, 'test.py', 'python')
    
    print("\n7. Different configuration outputs:")
    
    # Without private members
    agent1 = DiagramGenerationAgent(include_private_members=False)
    diagram1 = agent1.generate_class_diagram(classes, "Without Private Members")
    print("\n   Without Private Members:")
    print("   " + "─" * 40)
    for line in diagram1.split('\n')[7:15]:  # Show only class content
        if line.strip():
            print(f"   {line}")
    print("   " + "─" * 40)
    
    # Without method parameters
    agent2 = DiagramGenerationAgent(include_method_parameters=False)
    diagram2 = agent2.generate_class_diagram(classes, "Without Method Parameters")
    print("\n   Without Method Parameters:")
    print("   " + "─" * 40)
    for line in diagram2.split('\n')[7:15]:  # Show only class content
        if line.strip():
            print(f"   {line}")
    print("   " + "─" * 40)
    
    # Without field types
    agent3 = DiagramGenerationAgent(include_field_types=False)
    diagram3 = agent3.generate_class_diagram(classes, "Without Field Types")
    print("\n   Without Field Types:")
    print("   " + "─" * 40)
    for line in diagram3.split('\n')[7:15]:  # Show only class content
        if line.strip():
            print(f"   {line}")
    print("   " + "─" * 40)


def demo_langraph_integration():
    """Demo LangGraph integration"""
    print("\n" + "=" * 80)
    print("DEMO: LangGraph Integration")
    print("=" * 80)
    
    print("\n8. LangGraph node function demo...")
    
    # Create sample state
    state = {
        'ast_results': {
            'animals.py': create_sample_python_ast(),
            'shapes/Shape.java': create_sample_java_ast()
        },
        'current_agent': 'ast_parser',
        'processing_status': 'ast_parsing_completed'
    }
    
    print(f"   Input state keys: {list(state.keys())}")
    print(f"   AST results files: {list(state['ast_results'].keys())}")
    
    # Process với LangGraph node
    result = diagram_generator_node(state)
    
    print(f"\n   Output state keys: {list(result.keys())}")
    print(f"   Current agent: {result['current_agent']}")
    print(f"   Processing status: {result['processing_status']}")
    print(f"   Generated diagrams: {list(result['diagrams'].keys())}")
    print(f"   Extracted classes: {len(result['extracted_classes'])}")
    
    # Show diagram metadata
    metadata = result['diagram_metadata']
    print(f"\n   Diagram metadata:")
    print(f"   - Total diagrams: {metadata['total_diagrams']}")
    print(f"   - Total classes: {metadata['total_classes']}")
    print(f"   - Files processed: {metadata['files_processed']}")
    
    # Show sample diagram
    if 'project_overview' in result['diagrams']:
        overview_diagram = result['diagrams']['project_overview']
        print(f"\n   Project overview diagram (first 10 lines):")
        lines = overview_diagram['uml'].split('\n')
        for i, line in enumerate(lines[:10]):
            print(f"   {i+1:2d}: {line}")
        if len(lines) > 10:
            print(f"   ... ({len(lines)-10} more lines)")


def demo_error_handling():
    """Demo error handling scenarios"""
    print("\n" + "=" * 80)
    print("DEMO: Error Handling")
    print("=" * 80)
    
    agent = DiagramGenerationAgent()
    
    print("\n9. Error handling scenarios:")
    
    # Empty AST data
    print("\n   Empty AST data:")
    classes = agent.extract_class_info_from_ast({}, 'empty.py', 'python')
    print(f"   Result: {len(classes)} classes extracted")
    
    # Unsupported language
    print("\n   Unsupported language:")
    classes = agent.extract_class_info_from_ast({'classes': []}, 'test.js', 'javascript')
    print(f"   Result: {len(classes)} classes extracted")
    
    # Malformed AST data
    print("\n   Malformed AST data:")
    malformed_ast = {
        'classes': [
            {'invalid': 'data', 'body': []}
        ]
    }
    classes = agent.extract_class_info_from_ast(malformed_ast, 'malformed.py', 'python')
    print(f"   Result: {len(classes)} classes extracted")
    if classes:
        print(f"   Class name: {classes[0].name}")
    
    # Process files với no AST data
    print("\n   LangGraph processing với no AST data:")
    result = agent.process_files({})
    print(f"   Status: {result['processing_status']}")
    print(f"   Diagrams: {len(result['diagrams'])}")


def main():
    """Main demo function"""
    print("DiagramGenerationAgent Demo")
    print("Showcasing class diagram generation từ AST analysis")
    print("Generated by DeepCode-Insight")
    
    try:
        demo_basic_functionality()
        demo_python_class_extraction()
        demo_java_class_extraction()
        demo_plantuml_generation()
        demo_configuration_variations()
        demo_langraph_integration()
        demo_error_handling()
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Features Demonstrated:")
        print("✓ Python và Java class extraction từ AST")
        print("✓ PlantUML diagram generation với inheritance relationships")
        print("✓ Configurable output (private members, parameters, field types)")
        print("✓ LangGraph integration với state management")
        print("✓ Error handling và graceful degradation")
        print("✓ Multi-language support trong single diagram")
        print("\nThe DiagramGenerationAgent is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Demo failed với error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 