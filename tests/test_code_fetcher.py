"""
Tests cho CodeFetcherAgent
Test Git operations, repository cloning, và PR diff fetching
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import patch, MagicMock
import sys

# Thêm root directory vào Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.code_fetcher import CodeFetcherAgent, validate_git_url, extract_pr_number_from_url
from git.exc import GitError


class TestCodeFetcherAgent:
    """Test CodeFetcherAgent functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="test_codefetcher_")
        self.agent = CodeFetcherAgent(workspace_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.agent.cleanup()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test CodeFetcherAgent initialization"""
        assert self.agent.workspace_dir == self.temp_dir
        assert os.path.exists(self.temp_dir)
        assert isinstance(self.agent.cloned_repos, dict)
        assert len(self.agent.cloned_repos) == 0
    
    def test_init_with_default_workspace(self):
        """Test initialization với default workspace"""
        agent = CodeFetcherAgent()
        assert agent.workspace_dir.startswith(tempfile.gettempdir())
        assert "codefetcher_" in agent.workspace_dir
        agent.cleanup()
    
    def test_parse_repo_url_github(self):
        """Test parsing GitHub URLs"""
        test_cases = [
            ("https://github.com/user/repo", {
                'platform': 'github',
                'owner': 'user',
                'repo_name': 'repo',
                'full_name': 'user/repo',
                'clone_url': 'https://github.com/user/repo.git'
            }),
            ("https://github.com/user/repo.git", {
                'platform': 'github',
                'owner': 'user',
                'repo_name': 'repo',
                'full_name': 'user/repo',
                'clone_url': 'https://github.com/user/repo.git'
            }),
            ("https://github.com/user/repo/", {
                'platform': 'github',
                'owner': 'user',
                'repo_name': 'repo',
                'full_name': 'user/repo',
                'clone_url': 'https://github.com/user/repo.git'
            })
        ]
        
        for url, expected in test_cases:
            result = self.agent._parse_repo_url(url)
            assert result == expected
    
    def test_parse_repo_url_gitlab(self):
        """Test parsing GitLab URLs"""
        url = "https://gitlab.com/user/project"
        result = self.agent._parse_repo_url(url)
        
        assert result['platform'] == 'gitlab'
        assert result['owner'] == 'user'
        assert result['repo_name'] == 'project'
        assert result['clone_url'] == 'https://gitlab.com/user/project.git'
    
    def test_parse_repo_url_bitbucket(self):
        """Test parsing Bitbucket URLs"""
        url = "https://bitbucket.org/user/repo"
        result = self.agent._parse_repo_url(url)
        
        assert result['platform'] == 'bitbucket'
        assert result['owner'] == 'user'
        assert result['repo_name'] == 'repo'
        assert result['clone_url'] == 'https://bitbucket.org/user/repo.git'
    
    def test_parse_repo_url_invalid(self):
        """Test parsing invalid URLs"""
        invalid_urls = [
            "",
            "not-a-url",
            "https://example.com/repo",
            "https://github.com/user",
            "https://github.com/",
            "ftp://github.com/user/repo"
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValueError):
                self.agent._parse_repo_url(url)
    
    def test_get_repo_local_path(self):
        """Test getting local repository path"""
        repo_info = {
            'platform': 'github',
            'full_name': 'user/repo'
        }
        
        expected_path = os.path.join(self.temp_dir, "github_user_repo")
        result = self.agent._get_repo_local_path(repo_info)
        
        assert result == expected_path
    
    @patch('git.Repo.clone_from')
    def test_clone_repository_success(self, mock_clone):
        """Test successful repository cloning"""
        mock_repo = MagicMock()
        mock_clone.return_value = mock_repo
        
        repo_url = "https://github.com/user/repo"
        result = self.agent.clone_repository(repo_url)
        
        assert result == mock_repo
        assert "user/repo" in self.agent.cloned_repos
        mock_clone.assert_called_once()
    
    @patch('git.Repo.clone_from')
    def test_clone_repository_already_exists(self, mock_clone):
        """Test cloning repository that already exists"""
        mock_repo = MagicMock()
        mock_repo.remotes.origin.fetch = MagicMock()
        
        repo_url = "https://github.com/user/repo"
        self.agent.cloned_repos["user/repo"] = mock_repo
        
        result = self.agent.clone_repository(repo_url)
        
        assert result == mock_repo
        mock_repo.remotes.origin.fetch.assert_called_once()
        mock_clone.assert_not_called()
    
    @patch('git.Repo.clone_from')
    def test_clone_repository_force_refresh(self, mock_clone):
        """Test force refresh cloning"""
        mock_repo = MagicMock()
        mock_clone.return_value = mock_repo
        
        repo_url = "https://github.com/user/repo"
        
        # Create fake directory
        repo_info = self.agent._parse_repo_url(repo_url)
        local_path = self.agent._get_repo_local_path(repo_info)
        os.makedirs(local_path, exist_ok=True)
        
        result = self.agent.clone_repository(repo_url, force_refresh=True)
        
        assert result == mock_repo
        mock_clone.assert_called_once()
    
    def test_get_pr_diff_structure(self):
        """Test PR diff result structure"""
        repo_url = "https://github.com/user/repo"
        pr_id = 123
        
        # Mock để avoid actual Git operations
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            mock_repo = MagicMock()
            mock_clone.return_value = mock_repo
            mock_repo.remotes.origin.fetch.side_effect = Exception("Mocked error")
            
            result = self.agent.get_pr_diff(repo_url, pr_id)
            
            # Check result structure
            assert 'pr_id' in result
            assert 'repo_url' in result
            assert 'diff' in result
            assert 'files_changed' in result
            assert 'stats' in result
            assert 'commits' in result
            assert 'error' in result
            
            assert result['pr_id'] == pr_id
            assert result['repo_url'] == repo_url
    
    def test_get_file_content_not_found(self):
        """Test getting content of non-existent file"""
        repo_url = "https://github.com/user/repo"
        
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            mock_repo = MagicMock()
            mock_clone.return_value = mock_repo
            mock_repo.commit.return_value.tree.__getitem__.side_effect = KeyError("File not found")
            
            result = self.agent.get_file_content(repo_url, "nonexistent.txt")
            
            assert result is None
    
    def test_list_repository_files_empty(self):
        """Test listing files trong empty repository"""
        repo_url = "https://github.com/user/repo"
        
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            mock_repo = MagicMock()
            mock_clone.return_value = mock_repo
            mock_repo.commit.return_value.tree.traverse.return_value = []
            
            result = self.agent.list_repository_files(repo_url)
            
            assert result == []
    
    def test_get_repository_info_error(self):
        """Test getting repository info với error"""
        repo_url = "https://github.com/user/repo"
        
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            mock_clone.side_effect = GitError("Mocked Git error")
            
            result = self.agent.get_repository_info(repo_url)
            
            assert 'error' in result
            assert 'Mocked Git error' in result['error']
    
    def test_cleanup(self):
        """Test cleanup functionality"""
        # Add mock repo
        mock_repo = MagicMock()
        self.agent.cloned_repos['test/repo'] = mock_repo
        
        self.agent.cleanup()
        
        # Verify repo close was called
        mock_repo.close.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_validate_git_url_valid(self):
        """Test validating valid Git URLs"""
        valid_urls = [
            "https://github.com/user/repo",
            "https://gitlab.com/user/project",
            "https://bitbucket.org/user/repo"
        ]
        
        for url in valid_urls:
            assert validate_git_url(url) == True
    
    def test_validate_git_url_invalid(self):
        """Test validating invalid Git URLs"""
        invalid_urls = [
            "",
            "not-a-url",
            "https://example.com/repo",
            "ftp://github.com/user/repo"
        ]
        
        for url in invalid_urls:
            assert validate_git_url(url) == False
    
    def test_extract_pr_number_from_url(self):
        """Test extracting PR numbers từ URLs"""
        test_cases = [
            ("https://github.com/user/repo/pull/123", 123),
            ("https://gitlab.com/user/project/merge_requests/456", 456),
            ("https://bitbucket.org/user/repo/pullrequests/789", 789),
            ("https://github.com/user/repo/pull/123/files", 123),
            ("https://github.com/user/repo", None),
            ("invalid-url", None),
            ("", None)
        ]
        
        for url, expected in test_cases:
            result = extract_pr_number_from_url(url)
            assert result == expected


class TestCodeFetcherAgentIntegration:
    """Integration tests với real Git operations (nếu có network)"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="test_integration_")
        self.agent = CodeFetcherAgent(workspace_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.agent.cleanup()
    
    @pytest.mark.slow
    def test_clone_public_repository(self):
        """Test cloning a real public repository"""
        # Use a small, stable public repository
        repo_url = "https://github.com/octocat/Hello-World"
        
        try:
            repo = self.agent.clone_repository(repo_url)
            assert repo is not None
            assert "octocat/Hello-World" in self.agent.cloned_repos
            
            # Test getting repository info
            info = self.agent.get_repository_info(repo_url)
            assert info['platform'] == 'github'
            assert info['owner'] == 'octocat'
            assert info['repo_name'] == 'Hello-World'
            
        except Exception as e:
            pytest.skip(f"Network test failed: {e}")
    
    @pytest.mark.slow
    def test_list_files_public_repository(self):
        """Test listing files trong public repository"""
        repo_url = "https://github.com/octocat/Hello-World"
        
        try:
            files = self.agent.list_repository_files(repo_url)
            assert isinstance(files, list)
            # Hello-World repo should have README
            assert any('README' in f for f in files)
            
        except Exception as e:
            pytest.skip(f"Network test failed: {e}")


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="test_error_")
        self.agent = CodeFetcherAgent(workspace_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.agent.cleanup()
    
    def test_clone_invalid_repository(self):
        """Test cloning invalid repository"""
        invalid_repo = "https://github.com/nonexistent/repository"
        
        with pytest.raises(GitError):
            self.agent.clone_repository(invalid_repo)
    
    def test_get_pr_diff_invalid_repo(self):
        """Test getting PR diff từ invalid repository"""
        invalid_repo = "https://github.com/nonexistent/repository"
        
        result = self.agent.get_pr_diff(invalid_repo, 1)
        
        assert result['error'] is not None
        assert result['diff'] == ''
        assert result['files_changed'] == []
    
    def test_workspace_permission_error(self):
        """Test handling workspace permission errors"""
        # Try to create agent với invalid workspace
        with pytest.raises(Exception):
            CodeFetcherAgent(workspace_dir="/root/invalid_path")


if __name__ == "__main__":
    # Chạy tests khi file được execute trực tiếp
    pytest.main([__file__, "-v"]) 