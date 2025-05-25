#!/usr/bin/env python3
"""
End-to-End Test Runner cho DeepCode-Insight LangGraph Workflow
Chạy comprehensive test với sample PR URL và kiểm tra Markdown report generation
"""

import sys
import os
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(__file__))

def main():
    """Main function để run end-to-end test"""
    
    parser = argparse.ArgumentParser(
        description="🚀 End-to-End Test Runner cho DeepCode-Insight LangGraph Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_end_to_end_test.py                    # Run basic end-to-end test
  python run_end_to_end_test.py --verbose          # Run với verbose output
  python run_end_to_end_test.py --all-tests        # Run tất cả test scenarios
  python run_end_to_end_test.py --performance      # Run performance test only
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--all-tests', '-a',
        action='store_true',
        help='Run tất cả test scenarios (main + error handling + performance)'
    )
    
    parser.add_argument(
        '--performance', '-p',
        action='store_true',
        help='Run performance test only'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='test_reports',
        help='Output directory cho test reports (default: test_reports)'
    )
    
    args = parser.parse_args()
    
    print("🚀 DeepCode-Insight End-to-End Test Runner")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Output directory: {args.output_dir}")
    
    if args.verbose:
        print("🔧 Verbose mode enabled")
    
    try:
        # Import test functions
        from ..tests.test_end_to_end_workflow import (
            TestEndToEndWorkflow,
            run_end_to_end_test
        )
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create test instance
        test_instance = TestEndToEndWorkflow()
        
        if args.performance:
            print("\n⚡ Running Performance Test Only...")
            run_performance_test(test_instance, args.output_dir, args.verbose)
            
        elif args.all_tests:
            print("\n🧪 Running All Test Scenarios...")
            run_all_tests(test_instance, args.output_dir, args.verbose)
            
        else:
            print("\n🎯 Running Main End-to-End Test...")
            run_main_test(test_instance, args.output_dir, args.verbose)
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"📁 Check '{args.output_dir}' directory for generated reports")
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_main_test(test_instance, output_dir, verbose):
    """Run main end-to-end test"""
    
    print("📋 Test Scenario: Complete Workflow với PR URL")
    print("-" * 50)
    
    # Get sample data
    sample_data = test_instance.sample_pr_data()
    
    if verbose:
        print(f"📂 Repository: {sample_data['repo_url']}")
        print(f"🔀 PR ID: {sample_data['pr_id']}")
        print(f"📄 Target file: {sample_data['target_file']}")
        print(f"📊 Code lines: {len(sample_data['code_content'].splitlines())}")
    
    # Run test
    result = test_instance.test_complete_end_to_end_workflow_with_pr_url(
        output_dir
    )
    
    # Summary
    print(f"\n📊 Test Results Summary:")
    print(f"  ✅ Workflow Status: {result['processing_status']}")
    print(f"  ✅ Report Generated: {'Yes' if result.get('report') else 'No'}")
    
    if result.get('report'):
        report_path = result['report']['output_path']
        if os.path.exists(report_path):
            file_size = os.path.getsize(report_path)
            print(f"  ✅ Report Size: {file_size} bytes")
            print(f"  ✅ Report Location: {report_path}")


def run_all_tests(test_instance, output_dir, verbose):
    """Run all test scenarios"""
    
    test_scenarios = [
        ("Main Workflow Test", lambda: run_main_test(test_instance, output_dir, verbose)),
        ("Error Handling Test", lambda: test_instance.test_end_to_end_workflow_with_error_handling(output_dir)),
        ("Performance Test", lambda: run_performance_test(test_instance, output_dir, verbose))
    ]
    
    results = {}
    
    for scenario_name, test_func in test_scenarios:
        print(f"\n🧪 Running: {scenario_name}")
        print("-" * 50)
        
        try:
            start_time = datetime.now()
            test_func()
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            results[scenario_name] = {
                'status': 'PASSED',
                'execution_time': execution_time
            }
            
            print(f"✅ {scenario_name}: PASSED ({execution_time:.2f}s)")
            
        except Exception as e:
            results[scenario_name] = {
                'status': 'FAILED',
                'error': str(e)
            }
            
            print(f"❌ {scenario_name}: FAILED - {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    # Final summary
    print(f"\n📊 All Tests Summary:")
    print("=" * 40)
    
    passed_count = sum(1 for r in results.values() if r['status'] == 'PASSED')
    total_count = len(results)
    
    for scenario, result in results.items():
        status_icon = "✅" if result['status'] == 'PASSED' else "❌"
        if result['status'] == 'PASSED':
            print(f"{status_icon} {scenario}: {result['status']} ({result['execution_time']:.2f}s)")
        else:
            print(f"{status_icon} {scenario}: {result['status']} - {result['error']}")
    
    print(f"\n🎯 Overall Result: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("🎉 All tests PASSED!")
    else:
        print("⚠️ Some tests FAILED!")
        sys.exit(1)


def run_performance_test(test_instance, output_dir, verbose):
    """Run performance test"""
    
    print("📋 Test Scenario: Performance Metrics")
    print("-" * 40)
    
    # Get sample data
    sample_data = test_instance.sample_pr_data()
    
    # Run performance test
    test_instance.test_end_to_end_performance_metrics(output_dir)
    
    print("✅ Performance test completed")


def check_dependencies():
    """Check required dependencies"""
    
    required_modules = [
        'langgraph',
        'pytest',
        'tree_sitter'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing required modules: {', '.join(missing_modules)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_modules)}")
        sys.exit(1)
    
    print("✅ All required dependencies are available")


if __name__ == "__main__":
    # Check dependencies first
    check_dependencies()
    
    # Run main function
    main() 