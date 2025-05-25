#!/usr/bin/env python3
"""
Test runner cho StaticAnalysisAgent
Chạy tất cả test suites và tạo báo cáo
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def run_command(command, description):
    """Run command và return kết quả"""
    print(f"\n🔍 {description}")
    print(f"Command: {command}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏱️  Duration: {duration:.2f}s")
        
        if result.returncode == 0:
            print("✅ PASSED")
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
        else:
            print("❌ FAILED")
            if result.stderr:
                print("\nError:")
                print(result.stderr)
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
        
        return result.returncode == 0, result.stdout, result.stderr
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False, "", str(e)

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔧 Checking dependencies...")
    
    dependencies = [
        ("pytest", "pytest --version"),
        ("tree-sitter", "python -c 'import tree_sitter; print(\"tree-sitter available\")'"),
        ("tree-sitter-python", "python -c 'import tree_sitter_python; print(\"tree-sitter-python available\")'"),
    ]
    
    all_good = True
    
    for dep_name, check_cmd in dependencies:
        success, stdout, stderr = run_command(check_cmd, f"Checking {dep_name}")
        if not success:
            print(f"❌ {dep_name} not available")
            all_good = False
        else:
            print(f"✅ {dep_name} available")
    
    return all_good

def run_basic_test():
    """Run basic functionality test"""
    print("\n" + "="*80)
    print("🧪 BASIC FUNCTIONALITY TEST")
    print("="*80)
    
    success, stdout, stderr = run_command(
        "python test_static_analyzer.py",
        "Running basic functionality test"
    )
    
    return success

def run_comprehensive_tests():
    """Run comprehensive test suite với pytest"""
    print("\n" + "="*80)
    print("🧪 COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest not available. Installing...")
        success, _, _ = run_command(
            "pip install pytest",
            "Installing pytest"
        )
        if not success:
            print("❌ Failed to install pytest")
            return False
    
    # Run comprehensive tests
    success, stdout, stderr = run_command(
        "python -m pytest test_static_analysis_comprehensive.py -v --tb=short",
        "Running comprehensive test suite"
    )
    
    return success

def run_tree_sitter_tests():
    """Run Tree-sitter specific tests"""
    print("\n" + "="*80)
    print("🌳 TREE-SITTER QUERY TESTS")
    print("="*80)
    
    success, stdout, stderr = run_command(
        "python -m pytest test_tree_sitter_queries.py -v --tb=short",
        "Running Tree-sitter query tests"
    )
    
    return success

def run_demo():
    """Run demo script"""
    print("\n" + "="*80)
    print("🎬 DEMO SCRIPT")
    print("="*80)
    
    success, stdout, stderr = run_command(
        "python demo_static_analyzer.py",
        "Running demo script"
    )
    
    return success

def generate_test_report(results):
    """Generate test report"""
    print("\n" + "="*80)
    print("📊 TEST REPORT")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for success, _ in results if success)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {failed_tests} ❌")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\nDetailed Results:")
    for i, (success, test_name) in enumerate(results, 1):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {i}. {test_name}: {status}")
    
    if failed_tests == 0:
        print("\n🎉 All tests passed! StaticAnalysisAgent is working correctly.")
    else:
        print(f"\n⚠️  {failed_tests} test(s) failed. Please check the output above.")
    
    return failed_tests == 0

def main():
    """Main test runner function"""
    print("🚀 StaticAnalysisAgent Test Runner")
    print("="*80)
    
    # Check if we're in the right directory
    if not Path("agents/static_analyzer.py").exists():
        print("❌ Error: agents/static_analyzer.py not found!")
        print("Please run this script from the project root directory.")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Some dependencies are missing. Please install them first.")
        print("Try: pip install pytest tree-sitter tree-sitter-python")
        return 1
    
    # Run tests
    results = []
    
    # Basic test
    success = run_basic_test()
    results.append((success, "Basic Functionality Test"))
    
    # Comprehensive tests
    success = run_comprehensive_tests()
    results.append((success, "Comprehensive Test Suite"))
    
    # Tree-sitter tests
    success = run_tree_sitter_tests()
    results.append((success, "Tree-sitter Query Tests"))
    
    # Demo
    success = run_demo()
    results.append((success, "Demo Script"))
    
    # Generate report
    all_passed = generate_test_report(results)
    
    if all_passed:
        print("\n🎯 All tests completed successfully!")
        print("\n📚 StaticAnalysisAgent Features Verified:")
        print("  ✅ Tree-sitter query execution")
        print("  ✅ Missing docstring detection")
        print("  ✅ Unused import detection")
        print("  ✅ Complex function analysis")
        print("  ✅ Code quality metrics calculation")
        print("  ✅ Suggestion generation")
        print("  ✅ AST integration")
        print("  ✅ File analysis")
        print("  ✅ Error handling")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    exit(main()) 