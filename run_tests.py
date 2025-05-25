#!/usr/bin/env python3
"""
Simple test runner for DeepCode-Insight
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n🧪 {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PASSED")
            if result.stdout:
                print(result.stdout)
        else:
            print("❌ FAILED")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            if result.stdout:
                print("Standard output:")
                print(result.stdout)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def main():
    """Main test runner"""
    print("🚀 DeepCode-Insight Test Runner")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Test commands
    tests = [
        ("python -c 'from deepcode_insight.config import config; print(\"Config import:\", \"✅ OK\")'", 
         "Testing config import"),
        ("python -c 'from deepcode_insight.parsers.ast_parser import ASTParsingAgent; print(\"AST Parser import:\", \"✅ OK\")'", 
         "Testing AST parser import"),
        ("python -c 'from deepcode_insight.agents.rag_context import RAGContextAgent; print(\"RAG Context import:\", \"✅ OK\")'", 
         "Testing RAG context import"),
        ("python -m pytest tests/test_rag_context.py -v", 
         "Running RAG context tests"),
        ("python -m pytest tests/test_enhanced_static_analyzer.py -v", 
         "Running static analyzer tests"),
    ]
    
    results = []
    
    for cmd, description in tests:
        success = run_command(cmd, description)
        results.append((description, success))
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {description}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 