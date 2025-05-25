"""
Comprehensive Edge Case Tests cho CLI - Fixed Version
Focus vào error handling, missing arguments, và edge cases
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


class TestMissingArguments:
    """Test cases cho missing arguments"""
    
    def setup_method(self):
        """Setup test runner"""
        self.runner = CliRunner()
    
    def test_analyze_missing_repo_url(self):
        """Test analyze command thiếu repo-url"""
        result = self.runner.invoke(cli, [
            'analyze',
            '--pr-id', '123'
        ])
        
        assert result.exit_code != 0
        assert 'Missing option' in result.output or 'Repository URL:' in result.output
    
    def test_analyze_missing_pr_id(self):
        """Test analyze command thiếu pr-id"""
        result = self.runner.invoke(cli, [
            'analyze',
            '--repo-url', 'https://github.com/user/repo'
        ])
        
        assert result.exit_code != 0
        assert 'Missing option' in result.output or 'PR/MR ID:' in result.output


class TestValidationEdgeCasesFixed:
    """Test validation edge cases with proper formatting"""
    
    def test_repo_url_edge_cases(self):
        """Test repository URL edge cases"""
        valid_cases = [
            "https://github.com/user-123/repo_name",
            "https://github.com/user.name/repo.name", 
            "https://github.com/123user/repo123",
        ]
        
        invalid_cases = [
            "https://github.com/user/repo/issues",
            "https://github.com/user/repo?tab=readme",
            "https://github.com/user/repo#readme",
            "https://GITHUB.com/user/repo",
            "https://github.com/user/",
        ]
        
        for url in valid_cases:
            try:
                result = validate_repo_url(None, None, url)
                assert result is not None
            except click.BadParameter:
                pytest.fail(f"URL {url} should be valid but validation failed")
        
        for url in invalid_cases:
            with pytest.raises(click.BadParameter):
                validate_repo_url(None, None, url)
    
    def test_pr_id_edge_cases(self):
        """Test PR ID edge cases"""
        valid_cases = [
            "999999999",  # Large number
            "1000000000", # Very large number
            "001",        # Leading zeros
            "0123",       # More leading zeros
        ]
        
        invalid_cases = [
            "0",          # Zero
            "-1",         # Negative
            "abc",        # Non-numeric
            "12.3",       # Float
            "12,3",       # Comma
            "123abc",     # Mixed
            "1.0",        # Float format
        ]
        
        for pr_id in valid_cases:
            try:
                result = validate_pr_id(None, None, pr_id)
                assert result > 0
            except click.BadParameter:
                pytest.fail(f"PR ID {pr_id} should be valid but validation failed")
        
        for pr_id in invalid_cases:
            with pytest.raises(click.BadParameter):
                validate_pr_id(None, None, pr_id)


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def setup_method(self):
        """Setup test runner"""
        self.runner = CliRunner()
    
    def test_analyze_keyboard_interrupt(self):
        """Test analyze command với simulated KeyboardInterrupt"""
        with patch('time.sleep', side_effect=KeyboardInterrupt):
            result = self.runner.invoke(cli, [
                'analyze',
                '--repo-url', 'https://github.com/user/repo',
                '--pr-id', '123'
            ])
            
            assert result.exit_code == 1
            assert 'cancelled by user' in result.output
    
    def test_invalid_output_format(self):
        """Test analyze với invalid output format"""
        result = self.runner.invoke(cli, [
            'analyze',
            '--repo-url', 'https://github.com/user/repo',
            '--pr-id', '123',
            '--output-format', 'invalid-format'
        ])
        
        assert result.exit_code != 0
        assert 'Invalid value' in result.output or 'is not one of' in result.output


class TestInteractiveMode:
    """Test interactive mode scenarios"""
    
    def setup_method(self):
        """Setup test runner"""
        self.runner = CliRunner()
    
    def test_analyze_interactive_valid_inputs(self):
        """Test analyze command với valid interactive inputs"""
        result = self.runner.invoke(cli, ['analyze'], 
                                  input='https://github.com/user/repo\n123\n')
        
        assert result.exit_code == 0
        assert 'Repository URL:' in result.output
        assert 'PR/MR ID:' in result.output
        assert 'Analysis completed' in result.output
    
    def test_analyze_interactive_invalid_repo(self):
        """Test analyze command với invalid repo trong interactive mode"""
        result = self.runner.invoke(cli, ['analyze'], 
                                  input='invalid-url\n')
        
        assert result.exit_code != 0
        assert 'Invalid repository URL' in result.output


if __name__ == "__main__":
    # Chạy tests khi file được execute trực tiếp
    pytest.main([__file__, "-v"]) 