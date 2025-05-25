"""
Tests cho CLI module
Kiểm tra click commands và validation functions
"""

import pytest
import sys
import os
from click.testing import CliRunner
from unittest.mock import patch

# Thêm root directory vào Python path để import cli
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cli import cli, validate_repo_url, validate_pr_id
import click


class TestValidationFunctions:
    """Test validation functions"""
    
    def test_validate_repo_url_valid_github(self):
        """Test valid GitHub URLs"""
        valid_urls = [
            "https://github.com/user/repo",
            "https://github.com/user-name/repo-name",
            "https://github.com/user.name/repo.name",
            "https://github.com/user/repo/",  # with trailing slash
            "https://github.com/user/repo.git"  # with .git
        ]
        
        for url in valid_urls:
            result = validate_repo_url(None, None, url)
            assert result.startswith("https://github.com")
            assert not result.endswith('.git')
            assert not result.endswith('/')
    
    def test_validate_repo_url_valid_gitlab(self):
        """Test valid GitLab URLs"""
        valid_urls = [
            "https://gitlab.com/user/repo",
            "https://gitlab.com/user-name/repo-name"
        ]
        
        for url in valid_urls:
            result = validate_repo_url(None, None, url)
            assert result.startswith("https://gitlab.com")
    
    def test_validate_repo_url_valid_bitbucket(self):
        """Test valid Bitbucket URLs"""
        valid_urls = [
            "https://bitbucket.org/user/repo",
            "https://bitbucket.org/user-name/repo-name"
        ]
        
        for url in valid_urls:
            result = validate_repo_url(None, None, url)
            assert result.startswith("https://bitbucket.org")
    
    def test_validate_repo_url_invalid(self):
        """Test invalid repository URLs"""
        invalid_urls = [
            "",
            "not-a-url",
            "https://example.com/repo",
            "github.com/user/repo",  # missing https
            "https://github.com/user",  # missing repo name
            "https://github.com/",  # missing user and repo
            "https://invalid-host.com/user/repo"
        ]
        
        for url in invalid_urls:
            with pytest.raises(click.BadParameter):
                validate_repo_url(None, None, url)
    
    def test_validate_pr_id_valid(self):
        """Test valid PR IDs"""
        valid_ids = ["1", "123", "9999", "1000"]
        
        for pr_id in valid_ids:
            result = validate_pr_id(None, None, pr_id)
            assert isinstance(result, int)
            assert result > 0
    
    def test_validate_pr_id_invalid(self):
        """Test invalid PR IDs"""
        invalid_ids = [
            "",
            "0",
            "-1",
            "abc", 
            "12.5",
            "1a"
        ]
        
        for pr_id in invalid_ids:
            with pytest.raises(click.BadParameter):
                validate_pr_id(None, None, pr_id)


class TestCLICommands:
    """Test CLI commands using Click testing"""
    
    def setup_method(self):
        """Setup test runner"""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test main CLI help"""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'LangGraph Demo CLI' in result.output
        assert 'analyze' in result.output
        assert 'demo' in result.output
        assert 'validate' in result.output
    
    def test_analyze_help(self):
        """Test analyze command help"""
        result = self.runner.invoke(cli, ['analyze', '--help'])
        assert result.exit_code == 0
        assert 'repo-url' in result.output
        assert 'pr-id' in result.output
        assert 'output-format' in result.output
    
    def test_analyze_command_valid_inputs(self):
        """Test analyze command với valid inputs"""
        result = self.runner.invoke(cli, [
            'analyze',
            '--repo-url', 'https://github.com/user/repo',
            '--pr-id', '123',
            '--output-format', 'text'
        ])
        
        assert result.exit_code == 0
        assert 'Analysis completed' in result.output
        assert 'https://github.com/user/repo' in result.output
        assert '123' in result.output
    
    def test_analyze_command_json_output(self):
        """Test analyze command với JSON output"""
        result = self.runner.invoke(cli, [
            'analyze',
            '--repo-url', 'https://github.com/user/repo',
            '--pr-id', '456',
            '--output-format', 'json'
        ])
        
        assert result.exit_code == 0
        assert 'JSON' in result.output
        assert '"repository"' in result.output or 'repository' in result.output
    
    def test_analyze_command_markdown_output(self):
        """Test analyze command với Markdown output"""
        result = self.runner.invoke(cli, [
            'analyze',
            '--repo-url', 'https://github.com/user/repo', 
            '--pr-id', '789',
            '--output-format', 'markdown'
        ])
        
        assert result.exit_code == 0
        assert '# PR Analysis Report' in result.output
        assert '**Repository:**' in result.output
    
    def test_analyze_command_verbose(self):
        """Test analyze command với verbose flag"""
        result = self.runner.invoke(cli, [
            'analyze',
            '--repo-url', 'https://github.com/user/repo',
            '--pr-id', '999',
            '--verbose'
        ])
        
        assert result.exit_code == 0
        assert 'Starting analysis' in result.output
        assert 'Repository:' in result.output
    
    def test_analyze_command_invalid_repo_url(self):
        """Test analyze command với invalid repo URL"""
        result = self.runner.invoke(cli, [
            'analyze',
            '--repo-url', 'invalid-url',
            '--pr-id', '123'
        ])
        
        assert result.exit_code != 0
        assert 'Invalid repository URL' in result.output
    
    def test_analyze_command_invalid_pr_id(self):
        """Test analyze command với invalid PR ID"""
        result = self.runner.invoke(cli, [
            'analyze',
            '--repo-url', 'https://github.com/user/repo',
            '--pr-id', 'invalid'
        ])
        
        assert result.exit_code != 0
        assert 'must be a positive integer' in result.output
    
    @patch('src.graph.run_demo')
    def test_demo_command_valid_inputs(self, mock_run_demo):
        """Test demo command với valid inputs"""
        result = self.runner.invoke(cli, [
            'demo',
            '--repo-url', 'https://github.com/user/repo',
            '--pr-id', '123'
        ])
        
        assert result.exit_code == 0
        assert 'Running LangGraph demo' in result.output
        assert 'Demo completed successfully' in result.output
        mock_run_demo.assert_called_once()
    
    def test_demo_command_invalid_inputs(self):
        """Test demo command với invalid inputs"""
        result = self.runner.invoke(cli, [
            'demo',
            '--repo-url', 'invalid',
            '--pr-id', '0'
        ])
        
        assert result.exit_code != 0
    
    def test_validate_command(self):
        """Test validate command"""
        result = self.runner.invoke(cli, ['validate'])
        
        assert result.exit_code == 0
        assert 'Testing validation functions' in result.output
        assert 'Testing valid URLs' in result.output
        assert 'Testing invalid URLs' in result.output
        assert 'Testing valid PR IDs' in result.output
        assert 'Testing invalid PR IDs' in result.output
    
    def test_analyze_prompt_mode(self):
        """Test analyze command trong prompt mode"""
        result = self.runner.invoke(cli, ['analyze'], input='https://github.com/user/repo\n123\n')
        
        assert result.exit_code == 0
        assert 'Repository URL:' in result.output
        assert 'PR/MR ID:' in result.output


class TestCLIIntegration:
    """Integration tests cho CLI"""
    
    def setup_method(self):
        """Setup test runner"""
        self.runner = CliRunner()
    
    def test_full_workflow_text_format(self):
        """Test complete workflow với text format"""
        result = self.runner.invoke(cli, [
            'analyze',
            '--repo-url', 'https://github.com/microsoft/vscode',
            '--pr-id', '42',
            '--output-format', 'text',
            '--verbose'
        ])
        
        assert result.exit_code == 0
        assert 'microsoft/vscode' in result.output
        assert '42' in result.output
        assert 'Agent 1 Review' in result.output
        assert 'Agent 2 Feedback' in result.output
    
    def test_edge_case_large_pr_id(self):
        """Test với PR ID lớn"""
        result = self.runner.invoke(cli, [
            'analyze',
            '--repo-url', 'https://gitlab.com/group/project',
            '--pr-id', '999999'
        ])
        
        assert result.exit_code == 0
        assert '999999' in result.output
    
    def test_different_repo_hosts(self):
        """Test với different repository hosts"""
        hosts = [
            'https://github.com/user/repo',
            'https://gitlab.com/user/repo', 
            'https://bitbucket.org/user/repo'
        ]
        
        for repo_url in hosts:
            result = self.runner.invoke(cli, [
                'demo',
                '--repo-url', repo_url,
                '--pr-id', '1'
            ])
            
            assert result.exit_code == 0
            assert repo_url in result.output


if __name__ == "__main__":
    # Chạy tests khi file được execute trực tiếp
    pytest.main([__file__, "-v"]) 