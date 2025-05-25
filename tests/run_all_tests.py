#!/usr/bin/env python3
"""Master test runner cho tất cả StaticAnalysisAgent test suites"""

import sys
import os
import time
import subprocess
from typing import List, Tuple

def run_test_file(test_file: str) -> Tuple[bool, str, float]:
    """
    Chạy một test file và trả về kết quả
    
    Args:
        test_file: Path đến test file
        
    Returns:
        Tuple of (success, output, duration)
    """
    print(f"\n🚀 Running {test_file}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        duration = time.time() - start_time
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        if success:
            print(f"✅ {test_file} PASSED ({duration:.2f}s)")
        else:
            print(f"❌ {test_file} FAILED ({duration:.2f}s)")
            print(f"Error output:\n{result.stderr}")
        
        return success, output, duration
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ {test_file} TIMEOUT ({duration:.2f}s)")
        return False, "Test timed out", duration
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 {test_file} ERROR ({duration:.2f}s): {e}")
        return False, str(e), duration


def main():
    """Main test runner"""
    print("🧪 === StaticAnalysisAgent Complete Test Suite ===")
    print("Running all test files to validate enhanced functionality\n")
    
    # List of test files to run
    test_files = [
        "test_enhanced_static_analyzer.py",
        "test_java_rules.py", 
        "test_comprehensive_static_analyzer.py",
        "test_edge_cases_static_analyzer.py"
    ]
    
    # Check if test files exist
    missing_files = []
    for test_file in test_files:
        if not os.path.exists(test_file):
            missing_files.append(test_file)
    
    if missing_files:
        print(f"❌ Missing test files: {missing_files}")
        print("Please ensure all test files are in the current directory.")
        return 1
    
    # Run all tests
    results = []
    total_start_time = time.time()
    
    for test_file in test_files:
        success, output, duration = run_test_file(test_file)
        results.append((test_file, success, output, duration))
    
    total_duration = time.time() - total_start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 COMPLETE TEST SUITE SUMMARY")
    print("=" * 80)
    
    passed_tests = [r for r in results if r[1]]
    failed_tests = [r for r in results if not r[1]]
    
    print(f"Total Test Files: {len(test_files)}")
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {(len(passed_tests)/len(test_files))*100:.1f}%")
    print(f"Total Duration: {total_duration:.2f}s")
    
    # Detailed results
    print(f"\n📋 Detailed Results:")
    for test_file, success, output, duration in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_file} ({duration:.2f}s)")
    
    if failed_tests:
        print(f"\n❌ Failed Tests Details:")
        for test_file, success, output, duration in failed_tests:
            print(f"\n--- {test_file} ---")
            # Show last few lines of output for debugging
            lines = output.split('\n')
            relevant_lines = [line for line in lines[-20:] if line.strip()]
            for line in relevant_lines:
                print(f"  {line}")
    
    # Feature coverage summary
    print(f"\n✨ Enhanced Features Tested:")
    print(f"  🐍 Python Google Style Guide Rules:")
    print(f"    ✓ Naming conventions (PascalCase, snake_case)")
    print(f"    ✓ Lambda assignment detection")
    print(f"    ✓ Exception handling patterns (bare except)")
    print(f"    ✓ String formatting modernization")
    print(f"    ✓ Line length validation (79 chars)")
    print(f"    ✓ Complex comprehension detection")
    
    print(f"\n  ☕ Java Code Analysis Rules:")
    print(f"    ✓ Missing Javadoc detection")
    print(f"    ✓ Empty catch block detection")
    print(f"    ✓ Naming conventions (PascalCase, camelCase)")
    print(f"    ✓ Long line detection (120 chars)")
    print(f"    ✓ Basic metrics calculation")
    
    print(f"\n  🔧 Multi-language Support:")
    print(f"    ✓ Language detection from file extensions")
    print(f"    ✓ Language-specific rule application")
    print(f"    ✓ Unsupported language handling")
    
    print(f"\n  🔬 Edge Cases & Robustness:")
    print(f"    ✓ Dunder methods handling")
    print(f"    ✓ Private method docstring requirements")
    print(f"    ✓ Constants vs variables distinction")
    print(f"    ✓ Unicode and encoding support")
    print(f"    ✓ Error handling and recovery")
    print(f"    ✓ Large file performance")
    print(f"    ✓ Deeply nested code structures")
    
    print(f"\n  📊 Quality Metrics:")
    print(f"    ✓ Cyclomatic complexity calculation")
    print(f"    ✓ Maintainability index")
    print(f"    ✓ Code quality scoring")
    print(f"    ✓ Comment ratio analysis")
    
    # Final verdict
    if len(failed_tests) == 0:
        print(f"\n🎉 ALL TESTS PASSED! 🎉")
        print(f"Enhanced StaticAnalysisAgent is working correctly with:")
        print(f"  • Google Python Style Guide compliance")
        print(f"  • Java code analysis capabilities")
        print(f"  • Multi-language support")
        print(f"  • Robust error handling")
        print(f"  • Comprehensive rule coverage")
        print(f"\n✅ Ready for Giai đoạn 3 of the roadmap!")
        return 0
    else:
        print(f"\n💥 {len(failed_tests)} TEST(S) FAILED!")
        print(f"Please review the failed tests and fix any issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 