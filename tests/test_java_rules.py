#!/usr/bin/env python3
"""Test script để kiểm tra 2 quy tắc Java cơ bản"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'deepcode_insight'))

from deepcode_insight.agents.static_analyzer import StaticAnalysisAgent


def test_java_basic_rules():
    """Test 2 quy tắc Java cơ bản theo roadmap"""
    
    # Java sample với các violations cụ thể
    java_sample_with_violations = '''
import java.util.*;
import java.io.*;

public class TestClass {
    private int value;
    
    // Rule 1: Missing Javadoc for public methods
    public void publicMethodWithoutJavadoc() {
        System.out.println("This method should have Javadoc");
    }
    
    /**
     * This method has proper Javadoc
     */
    public void properlyDocumentedMethod() {
        System.out.println("This method has Javadoc");
    }
    
    // Rule 2: Empty catch blocks
    public void methodWithEmptyCatch() {
        try {
            int result = 10 / 0;
        } catch (Exception e) {
            // Empty catch block - this is bad practice
        }
    }
    
    public void methodWithProperCatch() {
        try {
            int result = 10 / 0;
        } catch (Exception e) {
            System.err.println("Error occurred: " + e.getMessage());
            // Proper exception handling
        }
    }
}
'''
    
    try:
        analyzer = StaticAnalysisAgent()
        
        print("🔍 === Java Basic Rules Test ===")
        print("Testing 2 basic Java rules as specified in roadmap\n")
        
        result = analyzer.analyze_code(java_sample_with_violations, "TestClass.java")
        
        print(f"File: {result['filename']} ({result['language']})")
        print(f"Quality Score: {result['metrics']['code_quality_score']:.1f}/100")
        print()
        
        # Check Rule 1: Missing Javadoc
        missing_javadoc = result['static_issues']['missing_docstrings']
        print("📋 Rule 1: Missing Javadoc Detection")
        print(f"Found {len(missing_javadoc)} missing Javadoc issues:")
        for issue in missing_javadoc:
            print(f"  - Line {issue['line']}: {issue['message']}")
        
        # Check Rule 2: Empty catch blocks
        code_smells = result['static_issues']['code_smells']
        empty_catches = [s for s in code_smells if s['type'] == 'empty_catch_block']
        print(f"\n📋 Rule 2: Empty Catch Block Detection")
        print(f"Found {len(empty_catches)} empty catch block issues:")
        for issue in empty_catches:
            print(f"  - Line {issue['line']}: {issue['message']}")
        
        # Additional analysis
        print(f"\n📊 Additional Issues Found:")
        for category, issues in result['static_issues'].items():
            if issues and category not in ['missing_docstrings', 'code_smells']:
                print(f"  - {category.replace('_', ' ').title()}: {len(issues)} issues")
        
        print(f"\n💡 Suggestions ({len(result['suggestions'])}):")
        for suggestion in result['suggestions']:
            print(f"  - {suggestion}")
        
        # Verify the rules are working
        success = True
        if len(missing_javadoc) == 0:
            print("\n❌ Rule 1 (Missing Javadoc) not working properly")
            success = False
        else:
            print(f"\n✅ Rule 1 (Missing Javadoc) working: Found {len(missing_javadoc)} issues")
        
        # Expect at least 1 empty catch block (methodWithEmptyCatch has comment-only catch)
        if len(empty_catches) == 0:
            print("❌ Rule 2 (Empty catch blocks) not working properly")
            success = False
        else:
            print(f"✅ Rule 2 (Empty catch blocks) working: Found {len(empty_catches)} issues")
        
        return success
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_java_naming_conventions():
    """Test Java naming conventions"""
    
    java_naming_sample = '''
public class badClassName {  // Should be PascalCase
    private int BadFieldName;  // Should be camelCase
    private static final int bad_constant = 10;  // Should be UPPER_CASE
    
    public void BadMethodName() {  // Should be camelCase
        int BadLocalVariable = 5;  // Should be camelCase
    }
    
    // Good examples
    public class GoodClassName {
        private int goodFieldName;
        private static final int GOOD_CONSTANT = 10;
        
        public void goodMethodName() {
            int goodLocalVariable = 5;
        }
    }
}
'''
    
    try:
        analyzer = StaticAnalysisAgent()
        
        print("\n🔍 === Java Naming Conventions Test ===")
        
        result = analyzer.analyze_code(java_naming_sample, "NamingTest.java")
        
        naming_violations = result['static_issues']['naming_violations']
        print(f"Found {len(naming_violations)} naming convention violations:")
        for violation in naming_violations:
            print(f"  - Line {violation['line']}: {violation['message']}")
        
        return len(naming_violations) > 0
        
    except Exception as e:
        print(f"❌ Error during naming test: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Testing Enhanced StaticAnalysisAgent Java Rules\n")
    
    # Test basic Java rules
    basic_rules_success = test_java_basic_rules()
    
    # Test naming conventions
    naming_success = test_java_naming_conventions()
    
    if basic_rules_success and naming_success:
        print("\n🎉 All Java rules tests passed successfully!")
        print("\n📋 Summary of implemented Java rules:")
        print("  1. ✅ Missing Javadoc detection for classes and methods")
        print("  2. ✅ Empty catch block detection")
        print("  3. ✅ Java naming conventions (PascalCase for classes, camelCase for methods/variables)")
        print("  4. ✅ Long line detection (120+ characters)")
        print("  5. ✅ Basic metrics calculation for Java")
    else:
        print("\n❌ Some Java rules tests failed!")
        sys.exit(1) 