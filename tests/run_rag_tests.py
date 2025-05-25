#!/usr/bin/env python3
"""
Test runner cho tất cả RAGContextAgent tests
Chạy unit tests, mocked tests, performance tests, và integration tests
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add paths
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(project_root, 'deepcode_insight'))
sys.path.append(project_root)

def run_command(command, description):
    """Run a command và return results"""
    print(f"\n🧪 {description}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Command: {command}")
        print(f"Duration: {duration:.2f}s")
        print(f"Exit code: {result.returncode}")
        
        if result.stdout:
            print("\nOutput:")
            print(result.stdout)
        
        if result.stderr:
            print("\nErrors:")
            print(result.stderr)
        
        return result.returncode == 0, duration
        
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False, 0

def check_prerequisites():
    """Check if prerequisites are available"""
    print("🔍 Checking prerequisites...")
    
    # Check if pytest is available
    try:
        import pytest
        print("✅ pytest available")
    except ImportError:
        print("❌ pytest not available. Install with: pip install pytest")
        return False
    
    # Check if psutil is available (for performance tests)
    try:
        import psutil
        print("✅ psutil available")
    except ImportError:
        print("⚠️ psutil not available. Performance tests may fail. Install with: pip install psutil")
    
    # Check if test files exist
    test_files = [
        "tests/test_rag_simple.py",
        "tests/test_rag_context.py", 
        "tests/test_rag_context_mocked.py",
        "tests/test_rag_performance.py"
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"✅ {test_file} found")
        else:
            print(f"❌ {test_file} not found")
            return False
    
    return True

def run_unit_tests():
    """Run basic unit tests"""
    print("\n🧪 === Running Unit Tests ===")
    
    tests = [
        ("python tests/test_rag_simple.py", "Component Tests (No API required)"),
        ("python tests/test_rag_context.py", "Basic RAG Tests (Config integration)")
    ]
    
    results = []
    total_duration = 0
    
    for command, description in tests:
        success, duration = run_command(command, description)
        results.append((description, success))
        total_duration += duration
    
    print(f"\n📊 Unit Test Summary:")
    print("=" * 40)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {description}")
    
    print(f"\n🎯 Unit Tests: {passed}/{total} passed")
    print(f"⏱️ Total duration: {total_duration:.2f}s")
    
    return passed == total

def run_mocked_tests():
    """Run mocked tests với pytest"""
    print("\n🧪 === Running Mocked Tests ===")
    
    commands = [
        ("python -m pytest tests/test_rag_context_mocked.py -v", "Mocked Unit Tests"),
        ("python -m pytest tests/test_rag_context_mocked.py::TestRAGContextAgentMocked::test_query_success -v", "Query Test"),
        ("python -m pytest tests/test_rag_context_mocked.py::TestRAGContextAgentMocked::test_index_code_file_success -v", "Indexing Test")
    ]
    
    results = []
    total_duration = 0
    
    for command, description in commands:
        success, duration = run_command(command, description)
        results.append((description, success))
        total_duration += duration
    
    print(f"\n📊 Mocked Test Summary:")
    print("=" * 40)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {description}")
    
    print(f"\n🎯 Mocked Tests: {passed}/{total} passed")
    print(f"⏱️ Total duration: {total_duration:.2f}s")
    
    return passed == total

def run_performance_tests():
    """Run performance tests"""
    print("\n🧪 === Running Performance Tests ===")
    
    commands = [
        ("python tests/test_rag_performance.py", "Performance & Integration Tests"),
        ("python -m pytest tests/test_rag_performance.py::TestRAGPerformance::test_indexing_performance -v", "Indexing Performance"),
        ("python -m pytest tests/test_rag_performance.py::TestRAGPerformance::test_query_performance -v", "Query Performance")
    ]
    
    results = []
    total_duration = 0
    
    for command, description in commands:
        success, duration = run_command(command, description)
        results.append((description, success))
        total_duration += duration
    
    print(f"\n📊 Performance Test Summary:")
    print("=" * 40)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {description}")
    
    print(f"\n🎯 Performance Tests: {passed}/{total} passed")
    print(f"⏱️ Total duration: {total_duration:.2f}s")
    
    return passed == total

def run_real_data_tests():
    """Run real data tests (requires API keys)"""
    print("\n🧪 === Running Real Data Tests ===")
    
    # Check if API keys are configured
    from deepcode_insight.config import config
    
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your_openai_api_key_here":
        print("⚠️ OpenAI API key not configured. Skipping real data tests.")
        print("   Set OPENAI_API_KEY in .env file to run real data tests.")
        return True
    
    commands = [
        ("python tests/test_rag_real_data.py", "Real Data Tests (với OpenAI API)")
    ]
    
    results = []
    total_duration = 0
    
    for command, description in commands:
        success, duration = run_command(command, description)
        results.append((description, success))
        total_duration += duration
    
    print(f"\n📊 Real Data Test Summary:")
    print("=" * 40)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {description}")
    
    print(f"\n🎯 Real Data Tests: {passed}/{total} passed")
    print(f"⏱️ Total duration: {total_duration:.2f}s")
    
    return passed == total

def run_all_pytest_tests():
    """Run all tests với pytest"""
    print("\n🧪 === Running All Tests với Pytest ===")
    
    command = "python -m pytest tests/test_rag_*.py -v --tb=short"
    success, duration = run_command(command, "All RAG Tests với Pytest")
    
    return success

def generate_test_report():
    """Generate comprehensive test report"""
    print("\n📊 === Generating Test Report ===")
    
    # Run pytest với coverage nếu available
    try:
        import coverage
        command = "python -m pytest tests/test_rag_*.py --cov=deepcode_insight.agents.rag_context --cov-report=html --cov-report=term"
        success, duration = run_command(command, "Test Coverage Report")
        
        if success:
            print("✅ Coverage report generated in htmlcov/")
        
    except ImportError:
        print("⚠️ coverage not available. Install with: pip install pytest-cov")
        
        # Run basic pytest với detailed output
        command = "python -m pytest tests/test_rag_*.py -v --tb=long --durations=10"
        success, duration = run_command(command, "Detailed Test Report")

def main():
    """Main test runner"""
    print("🚀 RAGContextAgent Comprehensive Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Exiting.")
        sys.exit(1)
    
    # Test results
    test_results = {}
    
    # Run different test suites
    print("\n🎯 Running Test Suites...")
    
    # 1. Unit tests
    test_results["Unit Tests"] = run_unit_tests()
    
    # 2. Mocked tests
    test_results["Mocked Tests"] = run_mocked_tests()
    
    # 3. Performance tests
    test_results["Performance Tests"] = run_performance_tests()
    
    # 4. Real data tests (optional)
    test_results["Real Data Tests"] = run_real_data_tests()
    
    # 5. All pytest tests
    test_results["Pytest Suite"] = run_all_pytest_tests()
    
    # Generate report
    generate_test_report()
    
    # Final summary
    end_time = time.time()
    total_duration = end_time - start_time
    
    print("\n🎉 === Final Test Summary ===")
    print("=" * 60)
    
    passed_suites = sum(1 for success in test_results.values() if success)
    total_suites = len(test_results)
    
    for suite_name, success in test_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {suite_name}")
    
    print(f"\n🎯 Overall Result: {passed_suites}/{total_suites} test suites passed")
    print(f"⏱️ Total execution time: {total_duration:.2f}s")
    
    if passed_suites == total_suites:
        print("\n🎉 All test suites passed! RAGContextAgent is working correctly.")
        return True
    else:
        print(f"\n❌ {total_suites - passed_suites} test suite(s) failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    
    print(f"\n📚 Test Coverage Summary:")
    print(f"  ✓ Unit Tests: Component functionality")
    print(f"  ✓ Mocked Tests: Logic verification without external dependencies")
    print(f"  ✓ Performance Tests: Speed, memory, concurrency")
    print(f"  ✓ Integration Tests: End-to-end workflows")
    print(f"  ✓ Real Data Tests: Actual API integration")
    
    print(f"\n🔧 Individual Test Commands:")
    print(f"  python tests/test_rag_simple.py")
    print(f"  python tests/test_rag_context.py")
    print(f"  python -m pytest tests/test_rag_context_mocked.py -v")
    print(f"  python tests/test_rag_performance.py")
    print(f"  python tests/test_rag_real_data.py")
    
    print(f"\n🎯 Pytest Commands:")
    print(f"  python -m pytest tests/test_rag_*.py -v")
    print(f"  python -m pytest tests/test_rag_context_mocked.py::TestRAGContextAgentMocked::test_query_success -v")
    print(f"  python -m pytest tests/ -k 'rag' --tb=short")
    
    if not success:
        sys.exit(1) 