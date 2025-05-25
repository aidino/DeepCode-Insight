"""
End-to-End Test Script cho Complete LangGraph Workflow
Kiểm tra toàn bộ pipeline từ PR URL đến Markdown report generation
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ..core.graph import create_analysis_workflow
from ..core.state import AgentState, DEFAULT_AGENT_STATE


class TestEndToEndWorkflow:
    """End-to-end tests cho complete LangGraph workflow"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory cho reports"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def sample_pr_data(self):
        """Mock PR data để simulate real GitHub PR"""
        return {
            'repo_url': 'https://github.com/test-user/sample-repo',
            'pr_id': '123',
            'target_file': 'src/calculator.py',
            'repository_info': {
                'full_name': 'test-user/sample-repo',
                'platform': 'github',
                'owner': 'test-user',
                'repo_name': 'sample-repo',
                'latest_commit': {
                    'message': 'Add calculator functionality',
                    'author': 'test-user',
                    'date': '2024-01-15T10:30:00Z',
                    'sha': 'abc123def456'
                },
                'branches': ['main', 'develop', 'feature/calculator'],
                'tags': ['v1.0.0', 'v1.1.0']
            },
            'pr_diff': {
                'files_changed': ['src/calculator.py', 'tests/test_calculator.py'],
                'stats': {
                    'additions': 45,
                    'deletions': 12,
                    'files': 2
                },
                'commits': [
                    {
                        'sha': 'abc123def456',
                        'message': 'Add calculator functionality',
                        'author': 'test-user'
                    }
                ],
                'error': None
            },
            'code_content': '''
"""
Calculator module với basic arithmetic operations
"""

class Calculator:
    """A simple calculator class for basic operations."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a, b):
        """Subtract second number from first."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a, b):
        """Divide first number by second."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def get_history(self):
        """Get calculation history."""
        return self.history
    
    def clear_history(self):
        """Clear calculation history."""
        self.history = []

def very_long_function_name_that_exceeds_the_recommended_line_length_limit_and_should_be_flagged_by_static_analysis():
    """This function name is intentionally too long to trigger static analysis warnings."""
    pass

# Missing docstring function
def calculate_average(numbers):
    return sum(numbers) / len(numbers) if numbers else 0
'''
        }
    
    def test_complete_end_to_end_workflow_with_pr_url(self, temp_output_dir):
        """
        Test complete end-to-end workflow với PR URL
        Kiểm tra từ input PR URL đến final Markdown report
        """
        print("\n🚀 Starting End-to-End Workflow Test")
        print("=" * 60)
        
        # Get sample data
        sample_pr_data = self.sample_pr_data()
        
        # Create workflow graph
        graph = create_analysis_workflow()
        
        # Prepare initial state với PR URL
        initial_state: AgentState = {
            **DEFAULT_AGENT_STATE,
            'repo_url': sample_pr_data['repo_url'],
            'pr_id': sample_pr_data['pr_id'],
            'target_file': sample_pr_data['target_file'],
            'config': {
                'output_dir': temp_output_dir,
                'test_mode': True,
                'end_to_end_test': True
            }
        }
        
        print(f"📂 Repository: {sample_pr_data['repo_url']}")
        print(f"🔀 PR ID: {sample_pr_data['pr_id']}")
        print(f"📄 Target file: {sample_pr_data['target_file']}")
        print(f"📁 Output directory: {temp_output_dir}")
        
        # Mock CodeFetcherAgent để simulate repository operations
        with patch('deepcode_insight.core.graph.CodeFetcherAgent') as mock_fetcher_class:
            mock_fetcher = Mock()
            
            # Mock repository operations
            mock_fetcher.get_repository_info.return_value = sample_pr_data['repository_info']
            mock_fetcher.get_pr_diff.return_value = sample_pr_data['pr_diff']
            mock_fetcher.list_repository_files.return_value = [
                'src/calculator.py',
                'tests/test_calculator.py',
                'README.md',
                'requirements.txt'
            ]
            mock_fetcher.get_file_content.return_value = sample_pr_data['code_content']
            mock_fetcher.cleanup.return_value = None
            
            mock_fetcher_class.return_value = mock_fetcher
            
            # Mock LLMOrchestratorAgent để simulate LLM analysis
            with patch('deepcode_insight.core.graph.create_llm_orchestrator_agent') as mock_create_llm:
                mock_llm_agent = Mock()
                mock_llm_agent.check_llm_health.return_value = True
                
                # Mock comprehensive LLM analysis result
                mock_llm_result = {
                    **initial_state,
                    'current_agent': 'llm_orchestrator',
                    'processing_status': 'llm_analysis_completed',
                    'llm_analysis': {
                        'filename': sample_pr_data['target_file'],
                        'summary': 'The calculator module demonstrates good object-oriented design with proper encapsulation. However, there are some documentation and style issues that should be addressed.',
                        'detailed_analysis': '''
The code shows a well-structured Calculator class with clear method definitions and proper error handling for division by zero. The class maintains calculation history which is a nice feature. However, there are several areas for improvement:

1. Missing docstrings for some functions
2. Line length violations in function names
3. Could benefit from type hints
4. Error handling could be more comprehensive
                        ''',
                        'priority_issues': [
                            {
                                'priority': 'High',
                                'description': 'Missing docstring for calculate_average function reduces code maintainability',
                                'action': 'Add comprehensive docstring following Google or NumPy style guide'
                            },
                            {
                                'priority': 'Medium',
                                'description': 'Function name exceeds recommended line length limit',
                                'action': 'Refactor function name to be more concise while maintaining clarity'
                            },
                            {
                                'priority': 'Low',
                                'description': 'Missing type hints reduce code clarity',
                                'action': 'Add type hints to function parameters and return values'
                            }
                        ],
                        'recommendations': [
                            {
                                'description': 'Add comprehensive docstrings to all functions and methods',
                                'effort': 'Medium'
                            },
                            {
                                'description': 'Implement type hints throughout the codebase',
                                'effort': 'High'
                            },
                            {
                                'description': 'Set up automated code formatting with black and isort',
                                'effort': 'Low'
                            },
                            {
                                'description': 'Add unit tests for edge cases and error conditions',
                                'effort': 'High'
                            },
                            {
                                'description': 'Consider using dataclasses for configuration objects',
                                'effort': 'Medium'
                            }
                        ],
                        'code_quality_assessment': 'The code quality is above average with good structure and error handling. Main areas for improvement are documentation completeness and adherence to style guidelines.',
                        'improvement_suggestions': [
                            {
                                'title': 'Documentation Enhancement',
                                'description': 'Add docstrings following Google style guide for all functions',
                                'effort': 'Medium'
                            },
                            {
                                'title': 'Type Safety Implementation',
                                'description': 'Add type hints and use mypy for static type checking',
                                'effort': 'High'
                            },
                            {
                                'title': 'Code Style Consistency',
                                'description': 'Use automated formatters and linters for consistent style',
                                'effort': 'Low'
                            },
                            {
                                'title': 'Error Handling Enhancement',
                                'description': 'Implement more comprehensive error handling and logging',
                                'effort': 'Medium'
                            }
                        ],
                        'llm_metadata': {
                            'model_used': 'codellama',
                            'analysis_type': 'comprehensive_pr_review',
                            'processing_time': '2.3s',
                            'confidence_score': 0.87
                        }
                    }
                }
                
                mock_llm_agent.process_findings.return_value = mock_llm_result
                mock_create_llm.return_value = mock_llm_agent
                
                print("\n🔄 Running complete workflow...")
                
                # Execute workflow
                start_time = datetime.now()
                result = graph.invoke(initial_state)
                end_time = datetime.now()
                
                execution_time = (end_time - start_time).total_seconds()
                print(f"⏱️ Workflow execution time: {execution_time:.2f} seconds")
                
                # ===== COMPREHENSIVE VERIFICATION =====
                
                print("\n📋 Verifying Workflow Results:")
                print("-" * 40)
                
                # 1. Verify workflow completion
                assert result['finished'] == True, "Workflow should be marked as finished"
                assert result['processing_status'] == 'report_generated', f"Expected 'report_generated', got '{result['processing_status']}'"
                print("✅ Workflow completed successfully")
                
                # 2. Verify state progression
                assert result['current_agent'] == 'reporter', f"Final agent should be 'reporter', got '{result['current_agent']}'"
                print("✅ State progression correct")
                
                # 3. Verify input data preservation
                assert result['repo_url'] == sample_pr_data['repo_url'], "Repository URL should be preserved"
                assert result['pr_id'] == sample_pr_data['pr_id'], "PR ID should be preserved"
                assert result['target_file'] == sample_pr_data['target_file'], "Target file should be preserved"
                print("✅ Input data preserved")
                
                # 4. Verify code fetching results
                assert result['code_content'] == sample_pr_data['code_content'], "Code content should match"
                assert result['filename'] == sample_pr_data['target_file'], "Filename should match target file"
                assert result['repository_info'] == sample_pr_data['repository_info'], "Repository info should match"
                assert result['pr_diff'] == sample_pr_data['pr_diff'], "PR diff should match"
                print("✅ Code fetching successful")
                
                # 5. Verify static analysis results
                assert 'static_analysis_results' in result, "Static analysis results should be present"
                static_results = result['static_analysis_results']
                assert 'static_issues' in static_results, "Static issues should be present"
                assert 'metrics' in static_results, "Metrics should be present"
                assert 'suggestions' in static_results, "Suggestions should be present"
                
                # Check for expected issues
                static_issues = static_results['static_issues']
                assert len(static_issues) > 0, "Should find some static analysis issues"
                print(f"✅ Static analysis found {len(static_issues)} issue types")
                
                # 6. Verify LLM analysis results
                assert 'llm_analysis' in result, "LLM analysis should be present"
                llm_analysis = result['llm_analysis']
                assert llm_analysis is not None, "LLM analysis should not be None"
                assert 'summary' in llm_analysis, "LLM summary should be present"
                assert 'priority_issues' in llm_analysis, "Priority issues should be present"
                assert 'recommendations' in llm_analysis, "Recommendations should be present"
                assert len(llm_analysis['priority_issues']) > 0, "Should have priority issues"
                assert len(llm_analysis['recommendations']) > 0, "Should have recommendations"
                print(f"✅ LLM analysis completed with {len(llm_analysis['priority_issues'])} priority issues")
                
                # 7. Verify report generation
                assert 'report' in result, "Report should be present"
                report = result['report']
                assert 'filename' in report, "Report filename should be present"
                assert 'content' in report, "Report content should be present"
                assert 'output_path' in report, "Report output path should be present"
                assert 'generated_at' in report, "Report generation timestamp should be present"
                
                # 8. Verify report file exists
                report_path = report['output_path']
                assert os.path.exists(report_path), f"Report file should exist at {report_path}"
                print(f"✅ Report file created: {os.path.basename(report_path)}")
                
                # 9. Verify report content structure
                report_content = report['content']
                
                # Check for required sections
                required_sections = [
                    '# 📊 Code Analysis Report',
                    '## 🎯 Executive Summary',
                    '## 🔍 Static Analysis Results',
                    '## 🤖 AI-Powered Analysis',
                    '## 📋 Action Items & Recommendations',
                    '## 📝 Report Information'
                ]
                
                for section in required_sections:
                    assert section in report_content, f"Report should contain section: {section}"
                
                print("✅ Report contains all required sections")
                
                # 10. Verify specific content elements
                content_checks = [
                    (sample_pr_data['target_file'], "Target filename"),
                    (sample_pr_data['repository_info']['full_name'], "Repository name"),
                    ('Calculator', "Class name from code"),
                    ('calculate_average', "Function name from code"),
                    ('Missing docstring', "Static analysis issue"),
                    ('The calculator module demonstrates', "LLM summary"),
                    ('High Priority', "Priority classification"),
                    ('Medium Priority', "Priority classification"),
                    ('Low Priority', "Priority classification"),
                    ('codellama', "LLM model used")
                ]
                
                for content, description in content_checks:
                    assert content in report_content, f"Report should contain {description}: {content}"
                
                print("✅ Report contains expected content elements")
                
                # 11. Verify report file size and format
                report_size = len(report_content)
                assert report_size > 1000, f"Report should be substantial (>1000 chars), got {report_size}"
                assert report_content.startswith('# 📊 Code Analysis Report'), "Report should start with proper header"
                assert report_content.endswith('*This report was automatically generated. Please review findings and recommendations carefully.*'), "Report should end with disclaimer"
                print(f"✅ Report format valid ({report_size} characters)")
                
                # 12. Verify configuration preservation
                assert result['config']['test_mode'] == True, "Test mode config should be preserved"
                assert result['config']['end_to_end_test'] == True, "End-to-end test flag should be preserved"
                print("✅ Configuration preserved")
                
                # 13. Verify agent interactions
                mock_fetcher.get_repository_info.assert_called_once_with(sample_pr_data['repo_url'])
                mock_fetcher.get_pr_diff.assert_called_once_with(sample_pr_data['repo_url'], int(sample_pr_data['pr_id']))
                mock_fetcher.list_repository_files.assert_called_once_with(sample_pr_data['repo_url'])
                mock_fetcher.get_file_content.assert_called_once_with(sample_pr_data['repo_url'], sample_pr_data['target_file'])
                mock_fetcher.cleanup.assert_called_once()
                
                mock_llm_agent.check_llm_health.assert_called_once()
                mock_llm_agent.process_findings.assert_called_once()
                
                print("✅ All agent interactions verified")
                
                # ===== DETAILED REPORT ANALYSIS =====
                
                print(f"\n📊 Detailed Report Analysis:")
                print("-" * 40)
                
                # Count sections and content
                sections_found = sum(1 for section in required_sections if section in report_content)
                print(f"📋 Report sections: {sections_found}/{len(required_sections)}")
                
                # Count issues and recommendations
                static_issues_count = sum(len(issues) for issues in static_results['static_issues'].values())
                llm_issues_count = len(llm_analysis['priority_issues'])
                recommendations_count = len(llm_analysis['recommendations'])
                
                print(f"🔍 Static analysis issues: {static_issues_count}")
                print(f"🤖 LLM priority issues: {llm_issues_count}")
                print(f"💡 Recommendations: {recommendations_count}")
                
                # Quality metrics
                if 'metrics' in static_results:
                    quality_score = static_results['metrics'].get('code_quality_score', 'N/A')
                    print(f"📊 Code quality score: {quality_score}")
                
                # Report metadata
                print(f"📄 Report filename: {report['filename']}")
                print(f"📁 Report path: {report_path}")
                print(f"⏰ Generated at: {report['generated_at']}")
                
                print(f"\n🎉 End-to-End Test PASSED!")
                print("=" * 60)
                
                return result
    
    def test_end_to_end_workflow_with_error_handling(self, temp_output_dir):
        """
        Test end-to-end workflow với error scenarios
        """
        print("\n🧪 Testing Error Handling Scenarios")
        print("=" * 50)
        
        graph = create_analysis_workflow()
        
        # Test 1: Invalid repository URL
        print("\n1️⃣ Testing invalid repository URL...")
        invalid_state = {
            **DEFAULT_AGENT_STATE,
            'repo_url': 'https://github.com/nonexistent/repo',
            'pr_id': '999',
            'config': {'output_dir': temp_output_dir}
        }
        
        with patch('deepcode_insight.agents.code_fetcher.CodeFetcherAgent') as mock_fetcher_class:
            mock_fetcher = Mock()
            mock_fetcher.get_repository_info.side_effect = Exception("Repository not found")
            mock_fetcher_class.return_value = mock_fetcher
            
            result = graph.invoke(invalid_state)
            
            assert result['finished'] == True
            assert result['processing_status'] == 'error'
            assert 'Repository not found' in result['error']
            print("✅ Invalid repository URL handled correctly")
        
        # Test 2: Empty repository (no Python files)
        print("\n2️⃣ Testing empty repository...")
        empty_repo_state = {
            **DEFAULT_AGENT_STATE,
            'repo_url': 'https://github.com/test/empty-repo',
            'config': {'output_dir': temp_output_dir}
        }
        
        with patch('deepcode_insight.agents.code_fetcher.CodeFetcherAgent') as mock_fetcher_class:
            mock_fetcher = Mock()
            mock_fetcher.get_repository_info.return_value = {'name': 'empty-repo'}
            mock_fetcher.list_repository_files.return_value = ['README.md', 'LICENSE']  # No Python files
            mock_fetcher_class.return_value = mock_fetcher
            
            result = graph.invoke(empty_repo_state)
            
            assert result['finished'] == True
            assert result['processing_status'] == 'error'
            assert 'No Python files found' in result['error']
            print("✅ Empty repository handled correctly")
        
        # Test 3: LLM service unavailable
        print("\n3️⃣ Testing LLM service unavailable...")
        llm_unavailable_state = {
            **DEFAULT_AGENT_STATE,
            'code_content': 'def test(): pass',
            'filename': 'test.py',
            'config': {'output_dir': temp_output_dir}
        }
        
        with patch('deepcode_insight.agents.llm_orchestrator.create_llm_orchestrator_agent') as mock_create_llm:
            mock_llm_agent = Mock()
            mock_llm_agent.check_llm_health.return_value = False  # LLM unavailable
            mock_create_llm.return_value = mock_llm_agent
            
            result = graph.invoke(llm_unavailable_state)
            
            assert result['finished'] == True
            assert result['processing_status'] == 'report_generated'  # Should continue without LLM
            assert result['llm_analysis'] is None
            assert 'report' in result
            print("✅ LLM unavailable handled correctly (graceful degradation)")
        
        print("\n✅ All error handling scenarios passed!")
    
    def test_end_to_end_performance_metrics(self, temp_output_dir):
        """
        Test performance metrics của end-to-end workflow
        """
        print("\n⚡ Performance Testing")
        print("=" * 30)
        
        # Get sample data
        sample_pr_data = self.sample_pr_data()
        
        graph = create_analysis_workflow()
        
        initial_state = {
            **DEFAULT_AGENT_STATE,
            'repo_url': sample_pr_data['repo_url'],
            'pr_id': sample_pr_data['pr_id'],
            'code_content': sample_pr_data['code_content'],  # Provide code directly for faster test
            'filename': sample_pr_data['target_file'],
            'config': {'output_dir': temp_output_dir}
        }
        
        # Mock LLM for consistent timing
        with patch('deepcode_insight.agents.llm_orchestrator.create_llm_orchestrator_agent') as mock_create_llm:
            mock_llm_agent = Mock()
            mock_llm_agent.check_llm_health.return_value = False  # Skip LLM for performance test
            mock_create_llm.return_value = mock_llm_agent
            
            # Measure execution time
            start_time = datetime.now()
            result = graph.invoke(initial_state)
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            # Performance assertions
            assert execution_time < 10.0, f"Workflow should complete within 10 seconds, took {execution_time:.2f}s"
            assert result['finished'] == True
            assert 'report' in result
            
            # Report size check
            report_size = len(result['report']['content'])
            assert report_size > 500, f"Report should be substantial, got {report_size} characters"
            
            print(f"⏱️ Execution time: {execution_time:.2f} seconds")
            print(f"📄 Report size: {report_size} characters")
            print(f"🚀 Performance: {'PASS' if execution_time < 5.0 else 'ACCEPTABLE'}")


def run_end_to_end_test():
    """
    Standalone function để run end-to-end test
    """
    print("🚀 Running End-to-End LangGraph Workflow Test")
    print("=" * 60)
    
    # Create test instance
    test_instance = TestEndToEndWorkflow()
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Get sample data
        sample_data = test_instance.sample_pr_data()
        
        print("📋 Test Configuration:")
        print(f"  Repository: {sample_data['repo_url']}")
        print(f"  PR ID: {sample_data['pr_id']}")
        print(f"  Target file: {sample_data['target_file']}")
        print(f"  Output directory: {temp_dir}")
        
        # Run main test
        result = test_instance.test_complete_end_to_end_workflow_with_pr_url(temp_dir, sample_data)
        
        print(f"\n📊 Final Results:")
        print(f"  Workflow status: {result['processing_status']}")
        print(f"  Report generated: {'Yes' if result.get('report') else 'No'}")
        
        if result.get('report'):
            report_path = result['report']['output_path']
            print(f"  Report location: {report_path}")
            
            if os.path.exists(report_path):
                print(f"  Report file size: {os.path.getsize(report_path)} bytes")
        
        print(f"\n✅ End-to-End Test COMPLETED SUCCESSFULLY!")
        
        return result
        
    except Exception as e:
        print(f"\n❌ End-to-End Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary directory")


if __name__ == "__main__":
    # Run standalone test
    run_end_to_end_test() 