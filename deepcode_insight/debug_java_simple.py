#!/usr/bin/env python3
"""
Simple debug script để test Java integration từng agent một
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ast_parser():
    print("🧪 Testing ASTParsingAgent...")
    try:
        from parsers.ast_parser import ASTParsingAgent
        
        java_code = '''
public class Test {
    private int value;
    
    public Test(int value) {
        this.value = value;
    }
    
    public int getValue() {
        return value;
    }
}
'''
        
        agent = ASTParsingAgent()
        result = agent.parse_code(java_code, "Test.java", language="java")
        
        print(f"  ✅ Language: {result['language']}")
        print(f"  ✅ Classes: {result['stats']['total_classes']}")
        print(f"  ✅ Methods: {result['stats']['total_functions']}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_static_analyzer():
    print("\n🧪 Testing StaticAnalysisAgent...")
    try:
        from agents.static_analyzer import StaticAnalysisAgent
        
        java_code = '''
public class Test {
    private int value;
    
    public Test(int value) {
        this.value = value;
    }
    
    public int getValue() {
        return value;
    }
}
'''
        
        agent = StaticAnalysisAgent()
        result = agent.analyze_code(java_code, "Test.java")
        
        print(f"  ✅ Language: {result['language']}")
        print(f"  ✅ AST analysis: {'ast_analysis' in result}")
        print(f"  ✅ Issues: {len(result['static_issues'])}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_diagram_generator():
    print("\n🧪 Testing DiagramGenerationAgent...")
    try:
        from agents.diagram_generator import DiagramGenerationAgent
        
        # Sample AST data
        ast_data = {
            'language': 'java',
            'classes': [
                {
                    'type': 'class_declaration',
                    'name': 'Test',
                    'start_point': {'row': 1},
                    'modifiers': ['public'],
                    'body': [
                        {
                            'name': 'value',
                            'type': 'int',
                            'modifiers': ['private']
                        },
                        {
                            'name': 'Test',
                            'type': 'void',
                            'modifiers': ['public'],
                            'parameters': [{'name': 'value', 'type': 'int'}]
                        },
                        {
                            'name': 'getValue',
                            'type': 'int',
                            'modifiers': ['public'],
                            'parameters': []
                        }
                    ]
                }
            ]
        }
        
        agent = DiagramGenerationAgent()
        classes = agent.extract_class_info_from_ast(ast_data, 'Test.java', 'java')
        
        print(f"  ✅ Classes extracted: {len(classes)}")
        
        if classes:
            diagram = agent.generate_class_diagram(classes, "Test Diagram")
            print(f"  ✅ Diagram generated: {len(diagram)} characters")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Simple Java Integration Debug")
    print("=" * 40)
    
    tests = [
        test_ast_parser,
        test_static_analyzer,
        test_diagram_generator
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} passed")
    return passed == len(tests)

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1) 