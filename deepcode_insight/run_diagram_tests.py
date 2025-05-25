#!/usr/bin/env python3
"""
Comprehensive test runner for DiagramGenerationAgent
"""

import subprocess
import sys
import os

def run_tests():
    """Run all DiagramGenerationAgent tests"""
    
    # Set PYTHONPATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root
    
    print("🧪 Running DiagramGenerationAgent Tests")
    print("=" * 50)
    
    test_files = [
        "deepcode_insight/tests/test_diagram_generator.py",
        "deepcode_insight/tests/test_diagram_generator_plantuml.py", 
        "deepcode_insight/tests/test_diagram_generator_edge_cases.py"
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_file in test_files:
        print(f"\n📋 Running {test_file}")
        print("-" * 40)
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
            ], cwd=project_root, env=env, capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            # Parse results
            lines = result.stdout.split('\n')
            for line in lines:
                if 'passed' in line and 'failed' in line:
                    # Extract numbers
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'passed' in part and i > 0:
                            try:
                                passed = int(parts[i-1])
                                total_passed += passed
                            except:
                                pass
                        if 'failed' in part and i > 0:
                            try:
                                failed = int(parts[i-1])
                                total_failed += failed
                            except:
                                pass
                elif 'passed' in line and 'failed' not in line:
                    # Only passed tests
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'passed' in part and i > 0:
                            try:
                                passed = int(parts[i-1])
                                total_passed += passed
                            except:
                                pass
            
        except Exception as e:
            print(f"❌ Error running {test_file}: {e}")
            total_failed += 1
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Total Passed: {total_passed}")
    print(f"❌ Total Failed: {total_failed}")
    print(f"📈 Success Rate: {total_passed/(total_passed+total_failed)*100:.1f}%" if (total_passed+total_failed) > 0 else "N/A")
    
    if total_failed == 0:
        print("\n🎉 All tests passed! DiagramGenerationAgent is ready for production.")
    else:
        print(f"\n⚠️  {total_failed} tests failed. Please review and fix.")
    
    return total_failed == 0

def demo_functionality():
    """Demo DiagramGenerationAgent functionality"""
    print("\n" + "=" * 50)
    print("🚀 DEMO: DiagramGenerationAgent Functionality")
    print("=" * 50)
    
    try:
        from deepcode_insight.agents.diagram_generator import DiagramGenerationAgent
        
        # Create agent
        agent = DiagramGenerationAgent(
            include_private_members=True,
            include_method_parameters=True,
            include_field_types=True
        )
        print("✅ Agent created successfully")
        
        # Sample Python AST
        python_ast = {
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
                                    {'arg': 'name', 'annotation': {'id': 'str'}}
                                ]
                            },
                            'returns': None
                        },
                        {
                            'type': 'AnnAssign',
                            'target': {'id': 'value'},
                            'annotation': {'id': 'int'}
                        }
                    ]
                }
            ]
        }
        
        # Extract classes
        classes = agent.extract_class_info_from_ast(python_ast, 'test.py', 'python')
        print(f"✅ Extracted {len(classes)} classes")
        
        # Generate diagram
        diagram = agent.generate_class_diagram(classes, "Test Diagram")
        print(f"✅ Generated PlantUML diagram ({len(diagram)} characters)")
        
        # Test LangGraph integration
        state = {
            'ast_results': {
                'test.py': python_ast
            }
        }
        result = agent.process_files(state)
        print(f"✅ LangGraph integration working ({len(result.get('diagrams', {}))} diagrams generated)")
        
        print("\n📋 Sample PlantUML Output:")
        print("-" * 30)
        print(diagram[:200] + "..." if len(diagram) > 200 else diagram)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False
    
    return True

if __name__ == '__main__':
    print("🔧 DiagramGenerationAgent Test Suite")
    print("=" * 50)
    
    # Run tests
    tests_passed = run_tests()
    
    # Run demo
    demo_passed = demo_functionality()
    
    print("\n" + "=" * 50)
    print("🏁 FINAL RESULTS")
    print("=" * 50)
    
    if tests_passed and demo_passed:
        print("🎉 SUCCESS: All tests passed and demo working!")
        print("✅ DiagramGenerationAgent is ready for production use.")
        sys.exit(0)
    else:
        print("❌ FAILURE: Some tests failed or demo not working.")
        print("⚠️  Please review and fix issues before production use.")
        sys.exit(1) 