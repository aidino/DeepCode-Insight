"""
Simplified End-to-End Test cho Complete LangGraph Workflow
Test đơn giản để kiểm tra workflow từ PR URL đến Markdown report
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ..core.graph import create_analysis_workflow
from ..core.state import AgentState, DEFAULT_AGENT_STATE


def test_end_to_end_workflow_with_pr_url():
    """
    Test complete end-to-end workflow với PR URL
    """
    print("\n🚀 Starting End-to-End Workflow Test")
    print("=" * 60)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Sample PR data
        sample_pr_data = {
            'repo_url': 'https://github.com/test-user/sample-repo',
            'pr_id': '123',
            'target_file': 'src/calculator.py',
            'repository_info': {
                'full_name': 'test-user/sample-repo',
                'platform': 'github',
                'owner': 'test-user',
                'repo_name': 'sample-repo'
            },
            'pr_diff': {
                'files_changed': ['src/calculator.py'],
                'stats': {'additions': 45, 'deletions': 12, 'files': 1}
            },
            'code_content': '''
class Calculator:
    """A simple calculator class."""
    
    def add(self, a, b):
        """Add two numbers."""
        return a + b
    
    def subtract(self, a, b):
        """Subtract second number from first."""
        return a - b

def very_long_function_name_that_exceeds_the_recommended_line_length_limit():
    """This function name is too long."""
    pass

def calculate_average(numbers):
    return sum(numbers) / len(numbers) if numbers else 0
'''
        }
        
        # Create workflow graph
        graph = create_analysis_workflow()
        
        # Prepare initial state
        initial_state: AgentState = {
            **DEFAULT_AGENT_STATE,
            'repo_url': sample_pr_data['repo_url'],
            'pr_id': sample_pr_data['pr_id'],
            'target_file': sample_pr_data['target_file'],
            'config': {
                'output_dir': temp_dir,
                'test_mode': True
            }
        }
        
        print(f"📂 Repository: {sample_pr_data['repo_url']}")
        print(f"🔀 PR ID: {sample_pr_data['pr_id']}")
        print(f"📄 Target file: {sample_pr_data['target_file']}")
        
        # Mock CodeFetcherAgent
        with patch('deepcode_insight.core.graph.CodeFetcherAgent') as mock_fetcher_class:
            mock_fetcher = Mock()
            mock_fetcher.get_repository_info.return_value = sample_pr_data['repository_info']
            mock_fetcher.get_pr_diff.return_value = sample_pr_data['pr_diff']
            mock_fetcher.list_repository_files.return_value = ['src/calculator.py', 'README.md']
            mock_fetcher.get_file_content.return_value = sample_pr_data['code_content']
            mock_fetcher.cleanup.return_value = None
            mock_fetcher_class.return_value = mock_fetcher
            
            # Mock LLM (skip for simplicity)
            with patch('deepcode_insight.core.graph.create_llm_orchestrator_agent') as mock_create_llm:
                mock_llm_agent = Mock()
                mock_llm_agent.check_llm_health.return_value = False  # Skip LLM
                mock_create_llm.return_value = mock_llm_agent
                
                print("\n🔄 Running complete workflow...")
                
                # Execute workflow
                result = graph.invoke(initial_state)
                
                print(f"⏱️ Workflow completed")
                
                # ===== VERIFICATION =====
                print("\n📋 Verifying Results:")
                print("-" * 30)
                
                # 1. Verify workflow completion
                assert result['finished'] == True, f"Workflow should be finished, got {result.get('finished')}"
                print("✅ Workflow marked as finished")
                
                # 2. Verify processing status
                expected_status = 'report_generated'
                actual_status = result.get('processing_status')
                assert actual_status == expected_status, f"Expected '{expected_status}', got '{actual_status}'"
                print(f"✅ Processing status: {actual_status}")
                
                # 3. Verify code content
                assert result.get('code_content') == sample_pr_data['code_content'], "Code content should match"
                print("✅ Code content preserved")
                
                # 4. Verify static analysis
                assert 'static_analysis_results' in result, "Static analysis results should be present"
                static_results = result['static_analysis_results']
                assert 'static_issues' in static_results, "Static issues should be present"
                print("✅ Static analysis completed")
                
                # 5. Verify report generation
                assert 'report' in result, "Report should be present"
                report = result['report']
                assert 'content' in report, "Report content should be present"
                assert 'output_path' in report, "Report output path should be present"
                
                # 6. Verify report file exists
                report_path = report['output_path']
                assert os.path.exists(report_path), f"Report file should exist at {report_path}"
                print(f"✅ Report file created: {os.path.basename(report_path)}")
                
                # 7. Verify report content
                report_content = report['content']
                assert '# 📊 Code Analysis Report' in report_content, "Report should have proper header"
                assert 'calculate_average' in report_content, "Report should mention function from code"
                assert 'Missing Docstrings' in report_content, "Report should show static analysis issues"
                assert len(report_content) > 500, f"Report should be substantial, got {len(report_content)} chars"
                print(f"✅ Report content valid ({len(report_content)} characters)")
                
                # 8. Verify agent interactions
                mock_fetcher.get_repository_info.assert_called_once()
                mock_fetcher.get_file_content.assert_called_once()
                mock_fetcher.cleanup.assert_called_once()
                print("✅ All agent interactions verified")
                
                print(f"\n🎉 End-to-End Test PASSED!")
                print(f"📁 Report saved to: {report_path}")
                
                return result
                
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_end_to_end_workflow_with_local_code():
    """
    Test workflow với local code content (không cần repository)
    """
    print("\n🚀 Testing Local Code Analysis")
    print("=" * 40)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Sample code
        sample_code = '''
def hello_world():
    print("Hello, World!")

class SimpleClass:
    def method_without_docstring(self):
        pass

def very_long_function_name_that_definitely_exceeds_line_length_limits():
    return "too long"
'''
        
        # Create workflow
        graph = create_analysis_workflow()
        
        # Initial state với code content
        initial_state: AgentState = {
            **DEFAULT_AGENT_STATE,
            'code_content': sample_code,
            'filename': 'test_local.py',
            'config': {'output_dir': temp_dir}
        }
        
        # Mock LLM (skip)
        with patch('deepcode_insight.core.graph.create_llm_orchestrator_agent') as mock_create_llm:
            mock_llm_agent = Mock()
            mock_llm_agent.check_llm_health.return_value = False
            mock_create_llm.return_value = mock_llm_agent
            
            print("🔄 Running local code analysis...")
            
            # Execute workflow
            result = graph.invoke(initial_state)
            
            # Verify results
            assert result['finished'] == True
            assert result['processing_status'] == 'report_generated'
            assert 'static_analysis_results' in result
            assert 'report' in result
            
            report_path = result['report']['output_path']
            assert os.path.exists(report_path)
            
            print(f"✅ Local code analysis completed")
            print(f"📁 Report: {os.path.basename(report_path)}")
            
            return result
            
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("🚀 Running End-to-End Tests")
    print("=" * 50)
    
    try:
        # Test 1: PR URL workflow
        print("\n1️⃣ Testing PR URL Workflow...")
        test_end_to_end_workflow_with_pr_url()
        
        # Test 2: Local code workflow
        print("\n2️⃣ Testing Local Code Workflow...")
        test_end_to_end_workflow_with_local_code()
        
        print(f"\n🎉 All End-to-End Tests PASSED!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 