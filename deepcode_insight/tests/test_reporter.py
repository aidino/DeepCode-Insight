"""
Tests cho ReportingAgent
Kiểm tra việc tạo báo cáo Markdown từ findings và LLM summaries
"""

import pytest
import os
import tempfile
import shutil
from datetime import datetime
from unittest.mock import Mock, patch
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ..agents.reporter import ReportingAgent, create_reporting_agent, reporting_node


class TestReportingAgentInitialization:
    """Test ReportingAgent initialization"""
    
    def test_init_default_output_dir(self):
        """Test initialization với default output directory"""
        agent = ReportingAgent()
        
        assert agent.output_dir == "reports"
        assert os.path.exists(agent.output_dir)
        
        # Cleanup
        if os.path.exists("reports"):
            shutil.rmtree("reports")
    
    def test_init_custom_output_dir(self):
        """Test initialization với custom output directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_dir = os.path.join(temp_dir, "custom_reports")
            agent = ReportingAgent(output_dir=custom_dir)
            
            assert agent.output_dir == custom_dir
            assert os.path.exists(custom_dir)
    
    def test_init_creates_output_directory(self):
        """Test rằng output directory được tạo nếu chưa tồn tại"""
        with tempfile.TemporaryDirectory() as temp_dir:
            non_existent_dir = os.path.join(temp_dir, "new_reports")
            
            # Verify directory doesn't exist
            assert not os.path.exists(non_existent_dir)
            
            # Create agent
            agent = ReportingAgent(output_dir=non_existent_dir)
            
            # Verify directory was created
            assert os.path.exists(non_existent_dir)


class TestGenerateReport:
    """Test generate_report method"""
    
    @pytest.fixture
    def agent(self):
        """Create agent với temporary directory"""
        temp_dir = tempfile.mkdtemp()
        agent = ReportingAgent(output_dir=temp_dir)
        yield agent
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_state_full(self):
        """Sample state với đầy đủ static và LLM analysis"""
        return {
            'static_analysis_results': {
                'filename': 'test.py',
                'static_issues': {
                    'missing_docstrings': [
                        {
                            'type': 'missing_function_docstring',
                            'name': 'func1',
                            'line': 5,
                            'message': 'Function lacks docstring'
                        }
                    ]
                },
                'metrics': {
                    'code_quality_score': 75.5,
                    'maintainability_index': 68.2
                },
                'suggestions': ['Add docstrings']
            },
            'llm_analysis': {
                'filename': 'test.py',
                'summary': 'Code needs improvement',
                'detailed_analysis': 'Detailed analysis here',
                'priority_issues': [
                    {
                        'priority': 'High',
                        'description': 'Critical issue',
                        'action': 'Fix immediately'
                    }
                ],
                'recommendations': [
                    {
                        'description': 'Add documentation',
                        'effort': 'Medium'
                    }
                ],
                'code_quality_assessment': 'Needs improvement',
                'improvement_suggestions': [
                    {
                        'title': 'Documentation',
                        'description': 'Add docstrings',
                        'effort': 'Low'
                    }
                ],
                'llm_metadata': {
                    'model_used': 'codellama',
                    'analysis_type': 'comprehensive'
                }
            },
            'filename': 'test.py',
            'current_agent': 'llm_orchestrator',
            'processing_status': 'llm_analysis_completed'
        }
    
    @pytest.fixture
    def sample_state_static_only(self):
        """Sample state với chỉ static analysis"""
        return {
            'static_analysis_results': {
                'filename': 'test.py',
                'static_issues': {
                    'code_style': [
                        {
                            'type': 'line_too_long',
                            'line': 10,
                            'message': 'Line too long'
                        }
                    ]
                },
                'metrics': {
                    'code_quality_score': 80.0
                },
                'suggestions': ['Fix line length']
            },
            'filename': 'test.py'
        }
    
    def test_generate_report_full_data(self, agent, sample_state_full):
        """Test generate report với đầy đủ data"""
        result = agent.generate_report(sample_state_full)
        
        # Verify state update
        assert result['processing_status'] == 'report_generated'
        assert result['current_agent'] == 'reporter'
        assert 'report' in result
        
        # Verify report info
        report_info = result['report']
        assert 'filename' in report_info
        assert 'content' in report_info
        assert 'generated_at' in report_info
        assert 'output_path' in report_info
        
        # Verify file was created
        assert os.path.exists(report_info['output_path'])
        
        # Verify content structure
        content = report_info['content']
        assert '# 📊 Code Analysis Report' in content
        assert '## 🎯 Executive Summary' in content
        assert '## 🔍 Static Analysis Results' in content
        assert '## 🤖 AI-Powered Analysis' in content
        assert '## 📋 Action Items & Recommendations' in content
        assert 'test.py' in content
    
    def test_generate_report_static_only(self, agent, sample_state_static_only):
        """Test generate report với chỉ static analysis"""
        result = agent.generate_report(sample_state_static_only)
        
        assert result['processing_status'] == 'report_generated'
        
        content = result['report']['content']
        assert '# 📊 Code Analysis Report' in content
        assert '## 🔍 Static Analysis Results' in content
        # Should not have LLM sections
        assert '## 🎯 Executive Summary' not in content
        assert '## 🤖 AI-Powered Analysis' not in content
    
    def test_generate_report_empty_state(self, agent):
        """Test generate report với empty state"""
        empty_state = {
            'filename': 'test.py'
        }
        
        result = agent.generate_report(empty_state)
        
        assert result['processing_status'] == 'report_error'
        assert 'error' in result
        assert 'No analysis results available' in result['error']
    
    def test_generate_report_no_filename(self, agent, sample_state_static_only):
        """Test generate report without filename"""
        del sample_state_static_only['filename']
        
        result = agent.generate_report(sample_state_static_only)
        
        assert result['processing_status'] == 'report_generated'
        # Should use default filename
        assert 'unknown_file' in result['report']['content']


class TestMarkdownReportCreation:
    """Test _create_markdown_report method"""
    
    @pytest.fixture
    def agent(self):
        return ReportingAgent(output_dir=tempfile.mkdtemp())
    
    def test_create_markdown_report_structure(self, agent):
        """Test basic Markdown report structure"""
        static_results = {
            'filename': 'test.py',
            'static_issues': {},
            'metrics': {'score': 85},
            'suggestions': []
        }
        llm_analysis = {
            'summary': 'Good code quality'
        }
        
        content = agent._create_markdown_report(static_results, llm_analysis, 'test.py')
        
        # Check header structure
        assert content.startswith('# 📊 Code Analysis Report')
        assert '**File:** `test.py`' in content
        assert '**Generated:**' in content
        assert '**Analysis Tool:** DeepCode-Insight' in content
        
        # Check sections
        assert '## 🎯 Executive Summary' in content
        assert '## 🔍 Static Analysis Results' in content
        assert '## 📝 Report Information' in content
    
    def test_create_markdown_report_with_metrics(self, agent):
        """Test report với metrics table"""
        static_results = {
            'metrics': {
                'code_quality_score': 75.5,
                'maintainability_index': 68.2,
                'complexity_score': 3
            }
        }
        
        content = agent._create_markdown_report(static_results, {}, 'test.py')
        
        assert '### 📈 Code Metrics' in content
        assert '| Metric | Value |' in content
        assert '| Code Quality Score | 75.50 |' in content
        assert '| Maintainability Index | 68.20 |' in content
        assert '| Complexity Score | 3 |' in content
    
    def test_create_markdown_report_with_issues(self, agent):
        """Test report với static issues"""
        static_results = {
            'static_issues': {
                'missing_docstrings': [
                    {
                        'type': 'missing_function_docstring',
                        'name': 'func1',
                        'line': 5,
                        'message': 'Function lacks docstring'
                    }
                ],
                'code_style': [
                    {
                        'type': 'line_too_long',
                        'line': 10
                    }
                ]
            }
        }
        
        content = agent._create_markdown_report(static_results, {}, 'test.py')
        
        assert '### ⚠️ Issues Found' in content
        assert '#### Missing Docstrings' in content
        assert '#### Code Style' in content
        assert 'missing_function_docstring' in content
        assert 'func1' in content
        assert 'line 5' in content
    
    def test_create_markdown_report_no_issues(self, agent):
        """Test report khi không có issues"""
        static_results = {
            'static_issues': {}
        }
        
        content = agent._create_markdown_report(static_results, {}, 'test.py')
        
        assert '### ✅ No Issues Found' in content
        assert 'Static analysis did not identify any issues' in content
    
    def test_create_markdown_report_with_llm_priority_issues(self, agent):
        """Test report với LLM priority issues"""
        llm_analysis = {
            'priority_issues': [
                {
                    'priority': 'High',
                    'description': 'Critical security issue',
                    'action': 'Fix immediately'
                },
                {
                    'priority': 'Medium',
                    'description': 'Performance issue',
                    'action': 'Optimize code'
                }
            ]
        }
        
        content = agent._create_markdown_report({}, llm_analysis, 'test.py')
        
        assert '### 🚨 Priority Issues' in content
        assert '#### 🔴 High Priority' in content
        assert '#### 🟡 Medium Priority' in content
        assert 'Critical security issue' in content
        assert 'Fix immediately' in content
    
    def test_create_markdown_report_with_recommendations(self, agent):
        """Test report với recommendations grouped by priority"""
        llm_analysis = {
            'recommendations': [
                {
                    'description': 'Critical fix needed',
                    'effort': 'High'
                },
                {
                    'description': 'Easy improvement',
                    'effort': 'Low'
                },
                {
                    'description': 'Standard improvement',
                    'effort': 'Medium'
                }
            ]
        }
        
        content = agent._create_markdown_report({}, llm_analysis, 'test.py')
        
        assert '## 📋 Action Items & Recommendations' in content
        assert '### 🔴 High Priority Actions' in content
        assert '### 🟡 Medium Priority Actions' in content
        assert '### 🟢 Low Priority Actions' in content
        assert '- [ ] Critical fix needed' in content
        assert '- [ ] Easy improvement' in content


class TestUtilityMethods:
    """Test utility methods"""
    
    @pytest.fixture
    def agent(self):
        return ReportingAgent(output_dir=tempfile.mkdtemp())
    
    def test_format_issue_description_complete(self, agent):
        """Test format issue description với complete data"""
        issue = {
            'type': 'missing_docstring',
            'name': 'function_name',
            'line': 42,
            'message': 'Function needs documentation'
        }
        
        result = agent._format_issue_description(issue)
        
        assert '**missing_docstring**' in result
        assert '`function_name`' in result
        assert '(line 42)' in result
        assert 'Function needs documentation' in result
    
    def test_format_issue_description_partial(self, agent):
        """Test format issue description với partial data"""
        issue = {
            'type': 'style_issue',
            'line': 10
        }
        
        result = agent._format_issue_description(issue)
        
        assert '**style_issue**' in result
        assert '(line 10)' in result
    
    def test_format_issue_description_empty(self, agent):
        """Test format issue description với empty dict"""
        issue = {}
        
        result = agent._format_issue_description(issue)
        
        assert result == "{}"
    
    def test_save_report_filename_generation(self, agent):
        """Test report filename generation"""
        content = "Test report content"
        
        # Mock datetime để có consistent timestamp
        with patch('deepcode_insight.agents.reporter.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
            
            filename = agent._save_report(content, "test_file.py")
            
            assert filename == "report_test_file_20240101_120000.md"
    
    def test_save_report_file_creation(self, agent):
        """Test rằng report file được tạo với correct content"""
        content = "# Test Report\n\nThis is a test report."
        
        filename = agent._save_report(content, "test.py")
        
        # Verify file exists
        file_path = os.path.join(agent.output_dir, filename)
        assert os.path.exists(file_path)
        
        # Verify content
        with open(file_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()
        
        assert saved_content == content
    
    def test_update_state_with_error(self, agent):
        """Test error state update"""
        original_state = {
            'filename': 'test.py',
            'some_data': 'value'
        }
        
        result = agent._update_state_with_error(original_state, "Test error message")
        
        # Original state should be preserved
        assert result['filename'] == 'test.py'
        assert result['some_data'] == 'value'
        
        # Error info should be added
        assert result['error'] == "Test error message"
        assert result['current_agent'] == 'reporter'
        assert result['processing_status'] == 'report_error'


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_create_reporting_agent_default(self):
        """Test create_reporting_agent với default parameters"""
        agent = create_reporting_agent()
        
        assert isinstance(agent, ReportingAgent)
        assert agent.output_dir == "reports"
        
        # Cleanup
        if os.path.exists("reports"):
            shutil.rmtree("reports")
    
    def test_create_reporting_agent_custom_dir(self):
        """Test create_reporting_agent với custom directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_dir = os.path.join(temp_dir, "custom")
            agent = create_reporting_agent(output_dir=custom_dir)
            
            assert isinstance(agent, ReportingAgent)
            assert agent.output_dir == custom_dir
    
    def test_reporting_node_function(self):
        """Test reporting_node wrapper function"""
        sample_state = {
            'static_analysis_results': {
                'filename': 'test.py',
                'static_issues': {},
                'metrics': {'score': 85}
            },
            'filename': 'test.py'
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('deepcode_insight.agents.reporter.create_reporting_agent') as mock_create:
                mock_agent = Mock()
                mock_agent.generate_report.return_value = {
                    'processing_status': 'report_generated',
                    'report': {'filename': 'test_report.md'}
                }
                mock_create.return_value = mock_agent
                
                result = reporting_node(sample_state)
                
                # Verify agent was created and called
                mock_create.assert_called_once()
                mock_agent.generate_report.assert_called_once_with(sample_state)
                
                assert result['processing_status'] == 'report_generated'


class TestEdgeCases:
    """Test edge cases và error scenarios"""
    
    @pytest.fixture
    def agent(self):
        temp_dir = tempfile.mkdtemp()
        agent = ReportingAgent(output_dir=temp_dir)
        yield agent
        shutil.rmtree(temp_dir)
    
    def test_generate_report_with_unicode_content(self, agent):
        """Test report generation với Unicode characters"""
        state = {
            'static_analysis_results': {
                'filename': 'tệp_tiếng_việt.py',
                'static_issues': {
                    'unicode_test': [
                        {
                            'type': 'unicode_issue',
                            'name': 'hàm_tiếng_việt',
                            'message': 'Thông báo có dấu tiếng Việt 🚀'
                        }
                    ]
                }
            },
            'llm_analysis': {
                'summary': 'Phân tích code với emoji 📊 và tiếng Việt'
            },
            'filename': 'tệp_tiếng_việt.py'
        }
        
        result = agent.generate_report(state)
        
        assert result['processing_status'] == 'report_generated'
        
        # Verify Unicode content is preserved
        content = result['report']['content']
        assert 'tệp_tiếng_việt.py' in content
        assert 'hàm_tiếng_việt' in content
        assert '🚀' in content
        assert '📊' in content
    
    def test_generate_report_with_malformed_data(self, agent):
        """Test report generation với malformed data structures"""
        state = {
            'static_analysis_results': {
                'static_issues': {
                    'malformed': [
                        "string instead of dict",
                        {'incomplete': 'data'},
                        None
                    ]
                },
                'metrics': {
                    'invalid_metric': 'not_a_number'
                }
            },
            'llm_analysis': {
                'priority_issues': [
                    "string instead of dict",
                    {'incomplete': 'priority_data'}
                ],
                'recommendations': [
                    {'description': 'valid recommendation'},
                    "invalid recommendation format"
                ]
            },
            'filename': 'malformed_test.py'
        }
        
        # Should handle malformed data gracefully
        result = agent.generate_report(state)
        
        assert result['processing_status'] == 'report_generated'
        
        content = result['report']['content']
        assert 'malformed_test.py' in content
        # Should include the malformed data as strings
        assert 'string instead of dict' in content
    
    def test_generate_report_file_write_error(self, agent):
        """Test error handling khi không thể write file"""
        # Make output directory read-only
        os.chmod(agent.output_dir, 0o444)
        
        state = {
            'static_analysis_results': {'filename': 'test.py'},
            'filename': 'test.py'
        }
        
        try:
            result = agent.generate_report(state)
            
            # Should handle write error gracefully
            assert result['processing_status'] == 'report_error'
            assert 'error' in result
            
        finally:
            # Restore permissions for cleanup
            os.chmod(agent.output_dir, 0o755)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 