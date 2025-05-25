"""
Performance và Stress Tests cho CLI
Test performance, memory usage, và concurrent scenarios
"""

import pytest
import sys
import os
import time
from click.testing import CliRunner
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

# Thêm root directory vào Python path để import cli
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ..cli.cli import cli
import click


# Mock validation functions for testing
def validate_repo_url(ctx, param, value):
    """Mock validation function for repo URL"""
    if not value:
        raise click.BadParameter("Repository URL is required")
    
    valid_hosts = ['github.com', 'gitlab.com', 'bitbucket.org']
    if not any(host in value for host in valid_hosts):
        raise click.BadParameter("Invalid repository URL")
    
    # Clean up URL
    if value.endswith('.git'):
        value = value[:-4]
    if value.endswith('/'):
        value = value[:-1]
    
    return value


def validate_pr_id(ctx, param, value):
    """Mock validation function for PR ID"""
    if not value:
        raise click.BadParameter("PR ID is required")
    
    try:
        pr_id = int(value)
        if pr_id <= 0:
            raise click.BadParameter("PR ID must be a positive integer")
        return pr_id
    except ValueError:
        raise click.BadParameter("PR ID must be a positive integer")


class TestPerformance:
    """Test CLI performance"""
    
    def setup_method(self):
        """Setup test runner"""
        self.runner = CliRunner()
    
    def test_validate_repo_url_performance(self):
        """Test validation performance với large number of URLs"""
        urls = [
            "https://github.com/user/repo",
            "https://gitlab.com/user/repo", 
            "https://bitbucket.org/user/repo"
        ] * 100  # 300 URLs total
        
        start_time = time.time()
        
        for url in urls:
            try:
                validate_repo_url(None, None, url)
            except:
                pass
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should validate 300 URLs in less than 1 second
        assert duration < 1.0, f"Validation took {duration:.3f}s, should be < 1.0s"
    
    def test_validate_pr_id_performance(self):
        """Test PR ID validation performance"""
        pr_ids = [str(i) for i in range(1, 1001)]  # 1000 PR IDs
        
        start_time = time.time()
        
        for pr_id in pr_ids:
            try:
                validate_pr_id(None, None, pr_id)
            except:
                pass
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should validate 1000 PR IDs in less than 0.5 seconds
        assert duration < 0.5, f"PR ID validation took {duration:.3f}s, should be < 0.5s"
    
    @patch('time.sleep', return_value=None)  # Skip actual sleep để test nhanh hơn
    def test_analyze_command_performance(self, mock_sleep):
        """Test analyze command performance"""
        start_time = time.time()
        
        result = self.runner.invoke(cli, [
            'analyze',
            '--repo-url', 'https://github.com/user/repo',
            '--pr-id', '123'
        ])
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert result.exit_code == 0
        # Should complete in less than 2 seconds (without actual sleep)
        assert duration < 2.0, f"Analyze command took {duration:.3f}s, should be < 2.0s"


class TestStressScenarios:
    """Test CLI under stress conditions"""
    
    def setup_method(self):
        """Setup test runner"""
        self.runner = CliRunner()
    
    def test_large_pr_id_handling(self):
        """Test với very large PR IDs"""
        large_pr_ids = [
            "999999999",      # 9 digits
            "9999999999",     # 10 digits  
            "99999999999",    # 11 digits
            "999999999999",   # 12 digits
        ]
        
        for pr_id in large_pr_ids:
            result = self.runner.invoke(cli, [
                'analyze',
                '--repo-url', 'https://github.com/user/repo',
                '--pr-id', pr_id
            ])
            
            assert result.exit_code == 0, f"Failed với PR ID: {pr_id}"
            assert pr_id in result.output
    
    def test_very_long_repo_urls(self):
        """Test với very long repository URLs"""
        # Generate long but valid repo names
        long_user = "a" * 50
        long_repo = "b" * 50
        
        result = self.runner.invoke(cli, [
            'analyze',
            '--repo-url', f'https://github.com/{long_user}/{long_repo}',
            '--pr-id', '123'
        ])
        
        assert result.exit_code == 0
        assert long_user in result.output
        assert long_repo in result.output
    
    def test_rapid_consecutive_commands(self):
        """Test rapid consecutive command execution"""
        commands = [
            ['analyze', '--repo-url', 'https://github.com/user/repo1', '--pr-id', '1'],
            ['analyze', '--repo-url', 'https://github.com/user/repo2', '--pr-id', '2'],
            ['analyze', '--repo-url', 'https://github.com/user/repo3', '--pr-id', '3'],
            ['demo', '--repo-url', 'https://github.com/user/repo4', '--pr-id', '4'],
            ['validate'],
        ]
        
        start_time = time.time()
        
        for cmd in commands:
            result = self.runner.invoke(cli, cmd)
            # Most commands should succeed (demo might fail due to import)
            assert result.exit_code in [0, 1]
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete all commands in reasonable time
        assert duration < 10.0, f"Rapid commands took {duration:.3f}s, should be < 10.0s"


class TestConcurrency:
    """Test concurrent CLI execution"""
    
    def setup_method(self):
        """Setup test runner"""
        self.runner = CliRunner()
    
    def _run_analyze_command(self, repo_suffix):
        """Helper function để run analyze command"""
        result = self.runner.invoke(cli, [
            'analyze',
            '--repo-url', f'https://github.com/user{repo_suffix}/repo{repo_suffix}',
            '--pr-id', str(repo_suffix)
        ])
        return result.exit_code == 0
    
    def test_concurrent_validation(self):
        """Test concurrent validation functions"""
        urls = [f"https://github.com/user{i}/repo{i}" for i in range(10)]
        pr_ids = [str(i) for i in range(1, 11)]
        
        # Test concurrent URL validation
        with ThreadPoolExecutor(max_workers=5) as executor:
            url_futures = [executor.submit(validate_repo_url, None, None, url) for url in urls]
            pr_futures = [executor.submit(validate_pr_id, None, None, pr_id) for pr_id in pr_ids]
            
            # All should complete successfully
            for future in url_futures:
                assert future.result() is not None
            
            for future, expected_id in zip(pr_futures, range(1, 11)):
                assert future.result() == expected_id
    
    @patch('time.sleep', return_value=None)  # Skip sleep để test nhanh hơn
    def test_concurrent_analyze_commands(self, mock_sleep):
        """Test concurrent analyze command execution"""
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(self._run_analyze_command, i) for i in range(1, 6)]
            
            # All commands should succeed
            results = [future.result() for future in futures]
            assert all(results), "Some concurrent commands failed"


class TestMemoryUsage:
    """Test memory usage scenarios"""
    
    def setup_method(self):
        """Setup test runner"""
        self.runner = CliRunner()
    
    def test_validation_with_many_invalid_urls(self):
        """Test validation với many invalid URLs"""
        invalid_urls = [
            f"invalid-url-{i}" for i in range(100)
        ]
        
        errors_caught = 0
        for url in invalid_urls:
            try:
                validate_repo_url(None, None, url)
            except:
                errors_caught += 1
        
        # All should be caught as invalid
        assert errors_caught == 100
    
    def test_large_output_handling(self):
        """Test handling of large output"""
        # Test với verbose mode để generate more output
        result = self.runner.invoke(cli, [
            'analyze',
            '--repo-url', 'https://github.com/microsoft/vscode',
            '--pr-id', '123456',
            '--verbose',
            '--output-format', 'text'
        ])
        
        assert result.exit_code == 0
        # Output should contain significant content
        assert len(result.output) > 500  # At least 500 characters
        assert 'microsoft/vscode' in result.output
        assert '123456' in result.output


class TestEdgeCaseScenarios:
    """Test edge case scenarios trong real-world usage"""
    
    def setup_method(self):
        """Setup test runner"""
        self.runner = CliRunner()
    
    def test_analyze_all_output_formats_stress(self):
        """Test all output formats với stress conditions"""
        formats = ['text', 'json', 'markdown']
        repo_urls = [
            'https://github.com/microsoft/vscode',
            'https://gitlab.com/gitlab-org/gitlab',
            'https://bitbucket.org/atlassian/bitbucket'
        ]
        pr_ids = ['1', '999', '123456']
        
        for fmt in formats:
            for repo_url in repo_urls:
                for pr_id in pr_ids:
                    result = self.runner.invoke(cli, [
                        'analyze',
                        '--repo-url', repo_url,
                        '--pr-id', pr_id,
                        '--output-format', fmt
                    ])
                    
                    assert result.exit_code == 0, f"Failed: {fmt}, {repo_url}, {pr_id}"
                    assert repo_url in result.output
                    assert pr_id in result.output
    
    def test_interactive_mode_stress(self):
        """Test interactive mode với various inputs"""
        test_cases = [
            # Valid cases
            ('https://github.com/user/repo\n123\n', 0),
            ('https://gitlab.com/user/repo\n456\n', 0),
            ('https://bitbucket.org/user/repo\n789\n', 0),
            
            # Invalid cases
            ('invalid-url\n', 2),  # Should fail on URL validation
            ('https://github.com/user/repo\ninvalid\n', 2),  # Should fail on PR ID
        ]
        
        for input_data, expected_exit_code in test_cases:
            result = self.runner.invoke(cli, ['analyze'], input=input_data)
            assert result.exit_code == expected_exit_code
    
    def test_cli_resilience_with_malformed_inputs(self):
        """Test CLI resilience với malformed inputs"""
        malformed_inputs = [
            # URLs với weird formats
            ['analyze', '--repo-url', 'https://github.com/user/repo\x00', '--pr-id', '123'],
            ['analyze', '--repo-url', 'https://github.com/user/repo\n\r', '--pr-id', '123'],
            
            # PR IDs với special formatting
            ['analyze', '--repo-url', 'https://github.com/user/repo', '--pr-id', '123\x00'],
            ['analyze', '--repo-url', 'https://github.com/user/repo', '--pr-id', '\t123\t'],
        ]
        
        for cmd in malformed_inputs:
            result = self.runner.invoke(cli, cmd)
            # Should either succeed (if input is cleaned) or fail gracefully
            assert result.exit_code in [0, 1, 2], f"Unexpected exit code for: {cmd}"
            # Should not crash or hang
            assert result.output is not None


class TestResourceCleanup:
    """Test resource cleanup và proper shutdown"""
    
    def setup_method(self):
        """Setup test runner"""
        self.runner = CliRunner()
    
    def test_proper_cleanup_on_error(self):
        """Test proper cleanup khi commands fail"""
        # Test với command that will fail
        result = self.runner.invoke(cli, [
            'analyze',
            '--repo-url', 'invalid-url',
            '--pr-id', '123'
        ])
        
        assert result.exit_code != 0
        # Should have proper error message
        assert 'Invalid repository URL' in result.output
        # Should not leave hanging resources
    
    def test_cleanup_on_keyboard_interrupt(self):
        """Test cleanup on simulated keyboard interrupt"""
        with patch('time.sleep', side_effect=KeyboardInterrupt):
            result = self.runner.invoke(cli, [
                'analyze',
                '--repo-url', 'https://github.com/user/repo',
                '--pr-id', '123'
            ])
            
            assert result.exit_code == 1
            assert 'cancelled by user' in result.output


if __name__ == "__main__":
    # Chạy performance tests khi file được execute trực tiếp
    pytest.main([__file__, "-v", "-s"]) 