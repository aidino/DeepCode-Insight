#!/usr/bin/env python3
"""
Comprehensive test script để verify Java integration cho ASTParsingAgent, StaticAnalysisAgent, và DiagramGenerationAgent
"""

import logging
import sys
import os
from typing import Dict, Any

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_ast_parsing_agent_java():
    """Test ASTParsingAgent với Java code"""
    print("🧪 Testing ASTParsingAgent với Java support...")
    
    try:
        from parsers.ast_parser import ASTParsingAgent
        
        # Sample Java code
        java_code = '''
package com.example;

import java.util.List;
import java.util.ArrayList;

/**
 * A simple calculator class
 */
public class Calculator {
    private static final int MAX_VALUE = 1000;
    private List<Integer> history;
    
    public Calculator() {
        this.history = new ArrayList<>();
    }
    
    /**
     * Add two numbers
     * @param a first number
     * @param b second number
     * @return sum of a and b
     */
    public int add(int a, int b) {
        int result = a + b;
        history.add(result);
        return result;
    }
    
    public static void main(String[] args) {
        Calculator calc = new Calculator();
        System.out.println(calc.add(5, 3));
    }
}
'''
        
        # Initialize agent
        agent = ASTParsingAgent()
        
        # Parse Java code
        result = agent.parse_code(java_code, "Calculator.java", language="java")
        
        # Verify results
        print(f"  ✅ Language detected: {result['language']}")
        print(f"  ✅ Classes found: {result['stats']['total_classes']}")
        print(f"  ✅ Methods found: {result['stats']['total_functions']}")
        print(f"  ✅ Imports found: {result['stats']['total_imports']}")
        print(f"  ✅ Fields found: {result['stats']['total_variables']}")
        
        # Check specific details
        if result['classes']:
            cls = result['classes'][0]
            print(f"  ✅ Class name: {cls['name']}")
            print(f"  ✅ Class type: {cls['type']}")
            print(f"  ✅ Is interface: {cls['is_interface']}")
            print(f"  ✅ Methods in class: {cls['method_count']}")
            print(f"  ✅ Fields in class: {cls['field_count']}")
        
        if result['functions']:
            print(f"  ✅ Method names: {[m['name'] for m in result['functions']]}")
        
        if result['imports']:
            print(f"  ✅ Imported packages: {[imp['imported'] for imp in result['imports']]}")
        
        assert result['stats']['total_classes'] >= 1, "Should find at least 1 class"
        assert result['stats']['total_functions'] >= 3, "Should find at least 3 methods"
        assert result['stats']['total_imports'] >= 2, "Should find at least 2 imports"
        
        print("  🎉 ASTParsingAgent Java test PASSED!")
        return True
        
    except Exception as e:
        print(f"  ❌ ASTParsingAgent Java test FAILED: {e}")
        return False

def test_static_analysis_agent_java():
    """Test StaticAnalysisAgent với Java code"""
    print("\n🧪 Testing StaticAnalysisAgent với Java support...")
    
    try:
        from agents.static_analyzer import StaticAnalysisAgent
        
        # Sample Java code với some issues
        java_code = '''
public class badClassName {  // Naming violation
    private int x;  // Missing Javadoc
    
    public void methodWithoutJavadoc() {  // Missing Javadoc
        try {
            int result = 10 / 0;
        } catch (Exception e) {
            // Empty catch block
        }
    }
    
    public void BADLY_NAMED_METHOD() {  // Naming violation
        // Method body
    }
}
'''
        
        # Initialize agent
        agent = StaticAnalysisAgent()
        
        # Analyze Java code
        result = agent.analyze_code(java_code, "BadExample.java")
        
        # Verify results
        print(f"  ✅ Language detected: {result['language']}")
        print(f"  ✅ AST analysis completed: {'ast_analysis' in result}")
        print(f"  ✅ Static issues found: {len(result['static_issues'])}")
        
        # Check specific issues
        naming_violations = result['static_issues'].get('naming_violations', [])
        missing_docs = result['static_issues'].get('missing_docstrings', [])
        code_smells = result['static_issues'].get('code_smells', [])
        
        print(f"  ✅ Naming violations: {len(naming_violations)}")
        print(f"  ✅ Missing Javadocs: {len(missing_docs)}")
        print(f"  ✅ Code smells: {len(code_smells)}")
        
        # Check metrics
        metrics = result.get('metrics', {})
        print(f"  ✅ Metrics calculated: {len(metrics)} metrics")
        
        # Check suggestions
        suggestions = result.get('suggestions', [])
        print(f"  ✅ Suggestions generated: {len(suggestions)}")
        
        assert result['language'] == 'java', "Should detect Java language"
        assert 'ast_analysis' in result, "Should include AST analysis"
        assert len(naming_violations) > 0, "Should find naming violations"
        assert len(missing_docs) > 0, "Should find missing Javadocs"
        
        print("  🎉 StaticAnalysisAgent Java test PASSED!")
        return True
        
    except Exception as e:
        print(f"  ❌ StaticAnalysisAgent Java test FAILED: {e}")
        return False

def test_diagram_generation_agent_java():
    """Test DiagramGenerationAgent với Java code"""
    print("\n🧪 Testing DiagramGenerationAgent với Java support...")
    
    try:
        from agents.diagram_generator import DiagramGenerationAgent
        
        # Sample Java AST (simulating output from ASTParsingAgent)
        java_ast = {
            'language': 'java',
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
                },
                {
                    'type': 'class_declaration',
                    'name': 'Circle',
                    'start_point': {'row': 10},
                    'modifiers': ['public'],
                    'interfaces': ['Drawable'],
                    'body': [
                        {
                            'name': 'radius',
                            'type': 'double',
                            'modifiers': ['private']
                        },
                        {
                            'name': 'draw',
                            'type': 'void',
                            'modifiers': ['public'],
                            'parameters': []
                        },
                        {
                            'name': 'getRadius',
                            'type': 'double',
                            'modifiers': ['public'],
                            'parameters': []
                        }
                    ]
                }
            ]
        }
        
        # Initialize agent
        agent = DiagramGenerationAgent(
            include_private_members=True,
            include_method_parameters=True,
            include_field_types=True
        )
        
        # Extract class information
        classes = agent.extract_class_info_from_ast(java_ast, 'shapes.java', 'java')
        
        print(f"  ✅ Classes extracted: {len(classes)}")
        
        if classes:
            for cls in classes:
                print(f"    - {cls.name} ({'interface' if cls.is_interface else 'class'})")
                print(f"      Methods: {len(cls.methods)}")
                print(f"      Fields: {len(cls.fields)}")
        
        # Generate PlantUML diagram
        diagram = agent.generate_class_diagram(classes, "Java Shape Example")
        
        print(f"  ✅ PlantUML diagram generated ({len(diagram)} characters)")
        
        # Verify diagram content
        expected_components = [
            "@startuml",
            "title Java Shape Example",
            "interface Drawable",
            "class Circle",
            "draw()",
            "getRadius()",
            "Circle ..|> Drawable",
            "@enduml"
        ]
        
        for component in expected_components:
            assert component in diagram, f"Missing component: {component}"
            print(f"    ✅ Found: {component}")
        
        # Test LangGraph integration
        state = {
            'ast_results': {
                'shapes.java': java_ast
            }
        }
        
        result = agent.process_files(state)
        
        print(f"  ✅ LangGraph integration: {result['processing_status']}")
        print(f"  ✅ Diagrams generated: {len(result.get('diagrams', {}))}")
        
        assert result['processing_status'] == 'diagram_generation_completed'
        assert len(result.get('diagrams', {})) > 0
        
        print("  🎉 DiagramGenerationAgent Java test PASSED!")
        return True
        
    except Exception as e:
        print(f"  ❌ DiagramGenerationAgent Java test FAILED: {e}")
        return False

def test_end_to_end_java_integration():
    """Test end-to-end Java integration với tất cả 3 agents"""
    print("\n🧪 Testing End-to-End Java Integration...")
    
    try:
        from parsers.ast_parser import ASTParsingAgent
        from agents.static_analyzer import StaticAnalysisAgent
        from agents.diagram_generator import DiagramGenerationAgent
        
        # Sample Java code
        java_code = '''
package com.example.shapes;

import java.util.List;

/**
 * Abstract shape class
 */
public abstract class Shape {
    protected String color;
    
    public Shape(String color) {
        this.color = color;
    }
    
    public abstract double getArea();
    
    public String getColor() {
        return color;
    }
}

/**
 * Circle implementation
 */
public class Circle extends Shape {
    private double radius;
    
    public Circle(String color, double radius) {
        super(color);
        this.radius = radius;
    }
    
    @Override
    public double getArea() {
        return Math.PI * radius * radius;
    }
    
    public double getRadius() {
        return radius;
    }
}
'''
        
        # Step 1: Parse với ASTParsingAgent
        print("  📝 Step 1: Parsing Java code...")
        ast_agent = ASTParsingAgent()
        ast_result = ast_agent.parse_code(java_code, "shapes.java", language="java")
        
        print(f"    ✅ Found {ast_result['stats']['total_classes']} classes")
        print(f"    ✅ Found {ast_result['stats']['total_functions']} methods")
        
        # Step 2: Analyze với StaticAnalysisAgent
        print("  🔍 Step 2: Static analysis...")
        static_agent = StaticAnalysisAgent()
        static_result = static_agent.analyze_code(java_code, "shapes.java")
        
        print(f"    ✅ Language: {static_result['language']}")
        print(f"    ✅ Issues found: {sum(len(issues) for issues in static_result['static_issues'].values())}")
        print(f"    ✅ Suggestions: {len(static_result.get('suggestions', []))}")
        
        # Step 3: Generate diagrams với DiagramGenerationAgent
        print("  📊 Step 3: Generating diagrams...")
        diagram_agent = DiagramGenerationAgent()
        
        # Simulate LangGraph state
        state = {
            'ast_results': {
                'shapes.java': ast_result
            }
        }
        
        diagram_result = diagram_agent.process_files(state)
        
        print(f"    ✅ Status: {diagram_result['processing_status']}")
        print(f"    ✅ Diagrams: {len(diagram_result.get('diagrams', {}))}")
        print(f"    ✅ Classes extracted: {len(diagram_result.get('extracted_classes', []))}")
        
        # Verify integration
        assert ast_result['language'] == 'java'
        assert static_result['language'] == 'java'
        assert diagram_result['processing_status'] == 'diagram_generation_completed'
        
        print("  🎉 End-to-End Java Integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"  ❌ End-to-End Java Integration test FAILED: {e}")
        return False

def main():
    """Run all Java integration tests"""
    print("🚀 Java Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_ast_parsing_agent_java,
        test_static_analysis_agent_java,
        test_diagram_generation_agent_java,
        test_end_to_end_java_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Test {test.__name__} CRASHED: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%" if (passed+failed) > 0 else "N/A")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Java integration is working correctly.")
        print("✅ ASTParsingAgent, StaticAnalysisAgent, và DiagramGenerationAgent")
        print("   đều handle Java code correctly với tree-sitter-java integration.")
    else:
        print(f"\n⚠️  {failed} tests failed. Please review and fix issues.")
    
    return failed == 0

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1) 