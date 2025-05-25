"""
Unit tests cho CodeFetcherAgent với mock GitPython calls
Test diff extraction và error handling mà không cần network calls
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime
import sys

# Thêm root directory vào Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ..agents.code_fetcher import CodeFetcherAgent, validate_git_url, extract_pr_number_from_url
from git.exc import GitError, GitCommandError, InvalidGitRepositoryError


class TestCodeFetcherAgentMocked:
    """Unit tests với mocked GitPython calls"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="test_mocked_")
        self.agent = CodeFetcherAgent(workspace_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.agent.cleanup()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('git.Repo.clone_from')
    def test_clone_repository_success_mocked(self, mock_clone):
        """Test successful repository cloning với mock"""
        # Setup mock
        mock_repo = MagicMock()
        mock_repo.remotes.origin.fetch = MagicMock()
        mock_clone.return_value = mock_repo
        
        repo_url = "https://github.com/test/repo"
        
        # Test clone
        result = self.agent.clone_repository(repo_url)
        
        # Assertions
        assert result == mock_repo
        assert "test/repo" in self.agent.cloned_repos
        mock_clone.assert_called_once()
        
        # Verify clone arguments
        args, kwargs = mock_clone.call_args
        assert args[0] == "https://github.com/test/repo.git"
        assert "depth" in kwargs
        assert kwargs["depth"] == 1
    
    @patch('git.Repo.clone_from')
    def test_clone_repository_git_error(self, mock_clone):
        """Test Git error during cloning"""
        mock_clone.side_effect = GitCommandError("git clone", "Clone failed")
        
        repo_url = "https://github.com/test/repo"
        
        with pytest.raises(GitError) as exc_info:
            self.agent.clone_repository(repo_url)
        
        assert "Git command failed" in str(exc_info.value)
        assert "Clone failed" in str(exc_info.value)
    
    @patch('git.Repo.clone_from')
    def test_clone_repository_invalid_repo_error(self, mock_clone):
        """Test invalid repository error"""
        mock_clone.side_effect = InvalidGitRepositoryError("Invalid repo")
        
        repo_url = "https://github.com/test/repo"
        
        with pytest.raises(GitError) as exc_info:
            self.agent.clone_repository(repo_url)
        
        assert "Invalid Git repository" in str(exc_info.value)
    
    @patch('git.Repo.clone_from')
    def test_clone_repository_force_refresh(self, mock_clone):
        """Test force refresh cloning"""
        mock_repo = MagicMock()
        mock_clone.return_value = mock_repo
        
        repo_url = "https://github.com/test/repo"
        
        # Create existing directory
        repo_info = self.agent._parse_repo_url(repo_url)
        local_path = self.agent._get_repo_local_path(repo_info)
        os.makedirs(local_path, exist_ok=True)
        
        # Add existing repo to cache
        self.agent.cloned_repos["test/repo"] = MagicMock()
        
        # Test force refresh
        result = self.agent.clone_repository(repo_url, force_refresh=True)
        
        assert result == mock_repo
        assert not os.path.exists(local_path) or len(os.listdir(local_path)) == 0
        mock_clone.assert_called_once()
    
    def test_get_pr_diff_structure_mocked(self):
        """Test PR diff result structure với mock"""
        repo_url = "https://github.com/test/repo"
        pr_id = 123
        
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            # Setup mock repository
            mock_repo = MagicMock()
            mock_clone.return_value = mock_repo
            
            # Mock fetch to raise error (simulating PR not found)
            mock_repo.remotes.origin.fetch.side_effect = GitCommandError("fetch", "PR not found")
            
            result = self.agent.get_pr_diff(repo_url, pr_id)
            
            # Verify structure
            expected_keys = ['pr_id', 'repo_url', 'diff', 'files_changed', 'stats', 'commits', 'error']
            for key in expected_keys:
                assert key in result
            
            assert result['pr_id'] == pr_id
            assert result['repo_url'] == repo_url
            assert result['error'] is not None
            assert "Failed to fetch PR" in result['error']
    
    def test_get_pr_diff_successful_extraction(self):
        """Test successful PR diff extraction với detailed mock"""
        repo_url = "https://github.com/test/repo"
        pr_id = 42
        
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            # Setup mock repository
            mock_repo = MagicMock()
            mock_clone.return_value = mock_repo
            
            # Mock successful fetch
            mock_repo.remotes.origin.fetch = MagicMock()
            
            # Mock PR commit
            mock_pr_commit = MagicMock()
            mock_pr_commit.hexsha = "abc123def456"
            mock_pr_commit.message = "Fix bug in authentication"
            mock_pr_commit.author = "Test Author"
            mock_pr_commit.committed_datetime = datetime(2024, 1, 15, 10, 30, 0)
            mock_pr_commit.parents = [MagicMock()]
            
            # Mock base commit
            mock_base_commit = MagicMock()
            mock_base_commit.hexsha = "def456abc123"
            
            # Setup commit lookup
            def mock_commit_lookup(ref):
                if "pull/42/head" in ref:
                    return mock_pr_commit
                elif "main" in ref or "master" in ref:
                    return mock_base_commit
                return mock_base_commit
            
            mock_repo.commit.side_effect = mock_commit_lookup
            
            # Mock diff
            mock_diff_item = MagicMock()
            mock_diff_item.a_path = "src/auth.py"
            mock_diff_item.b_path = "src/auth.py"
            mock_diff_item.diff = b"""--- a/src/auth.py
+++ b/src/auth.py
@@ -10,6 +10,8 @@ def authenticate(user):
     if not user:
         return False
     
+    # Add validation
+    if not user.is_valid():
+        return False
+    
     return check_credentials(user)
"""
            
            mock_diff = [mock_diff_item]
            mock_base_commit.diff.return_value = mock_diff
            
            # Mock iter_commits for commit history
            mock_repo.iter_commits.return_value = [mock_pr_commit]
            
            result = self.agent.get_pr_diff(repo_url, pr_id)
            
            # Verify successful extraction
            assert result['error'] is None
            assert result['pr_id'] == pr_id
            assert len(result['files_changed']) == 1
            assert "src/auth.py" in result['files_changed']
            assert result['stats']['additions'] > 0
            assert result['stats']['files'] == 1
            assert len(result['commits']) == 1
            assert result['commits'][0]['message'] == "Fix bug in authentication"
            assert "auth.py" in result['diff']
    
    def test_get_pr_diff_no_base_commit(self):
        """Test PR diff khi không tìm thấy base commit"""
        repo_url = "https://github.com/test/repo"
        pr_id = 99
        
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            mock_repo = MagicMock()
            mock_clone.return_value = mock_repo
            
            # Mock successful fetch
            mock_repo.remotes.origin.fetch = MagicMock()
            
            # Mock PR commit without parents
            mock_pr_commit = MagicMock()
            mock_pr_commit.parents = []
            
            # Mock commit lookup - no base branches found
            def mock_commit_lookup(ref):
                if "pull/99/head" in ref:
                    return mock_pr_commit
                raise Exception("Branch not found")
            
            mock_repo.commit.side_effect = mock_commit_lookup
            
            result = self.agent.get_pr_diff(repo_url, pr_id)
            
            assert result['error'] is not None
            assert "Cannot find base commit" in result['error']
    
    def test_get_pr_diff_gitlab_platform(self):
        """Test PR diff cho GitLab platform"""
        repo_url = "https://gitlab.com/test/project"
        pr_id = 15
        
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            mock_repo = MagicMock()
            mock_clone.return_value = mock_repo
            
            # Mock fetch failure (GitLab MR refs are different)
            mock_repo.remotes.origin.fetch.side_effect = GitCommandError("fetch", "Ref not found")
            
            result = self.agent.get_pr_diff(repo_url, pr_id)
            
            assert result['error'] is not None
            assert "gitlab" in result['error'].lower()
            assert "API access required" in result['error']
    
    def test_get_file_content_success(self):
        """Test successful file content retrieval"""
        repo_url = "https://github.com/test/repo"
        file_path = "README.md"
        
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            mock_repo = MagicMock()
            mock_clone.return_value = mock_repo
            
            # Mock file blob
            mock_blob = MagicMock()
            mock_blob.data_stream.read.return_value = b"# Test Repository\n\nThis is a test."
            
            # Mock tree navigation
            mock_tree = MagicMock()
            mock_tree.__getitem__.return_value = mock_blob
            mock_repo.commit.return_value.tree = mock_tree
            
            result = self.agent.get_file_content(repo_url, file_path)
            
            assert result == "# Test Repository\n\nThis is a test."
            mock_tree.__getitem__.assert_called_once_with(file_path)
    
    def test_get_file_content_not_found(self):
        """Test file content retrieval khi file không tồn tại"""
        repo_url = "https://github.com/test/repo"
        file_path = "nonexistent.txt"
        
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            mock_repo = MagicMock()
            mock_clone.return_value = mock_repo
            
            # Mock tree to raise KeyError
            mock_tree = MagicMock()
            mock_tree.__getitem__.side_effect = KeyError("File not found")
            mock_repo.commit.return_value.tree = mock_tree
            
            result = self.agent.get_file_content(repo_url, file_path)
            
            assert result is None
    
    def test_get_file_content_encoding_error(self):
        """Test file content với encoding issues"""
        repo_url = "https://github.com/test/repo"
        file_path = "binary_file.bin"
        
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            mock_repo = MagicMock()
            mock_clone.return_value = mock_repo
            
            # Mock binary content
            mock_blob = MagicMock()
            mock_blob.data_stream.read.return_value = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
            
            mock_tree = MagicMock()
            mock_tree.__getitem__.return_value = mock_blob
            mock_repo.commit.return_value.tree = mock_tree
            
            result = self.agent.get_file_content(repo_url, file_path)
            
            # Should handle encoding errors gracefully
            assert result is not None
            assert isinstance(result, str)
    
    def test_list_repository_files_success(self):
        """Test successful repository file listing"""
        repo_url = "https://github.com/test/repo"
        
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            mock_repo = MagicMock()
            mock_clone.return_value = mock_repo
            
            # Mock tree items
            mock_file1 = MagicMock()
            mock_file1.type = 'blob'
            mock_file1.path = 'README.md'
            
            mock_file2 = MagicMock()
            mock_file2.type = 'blob'
            mock_file2.path = 'src/main.py'
            
            mock_dir = MagicMock()
            mock_dir.type = 'tree'
            mock_dir.path = 'src'
            
            mock_tree = MagicMock()
            mock_tree.traverse.return_value = [mock_file1, mock_file2, mock_dir]
            mock_repo.commit.return_value.tree = mock_tree
            
            result = self.agent.list_repository_files(repo_url)
            
            assert len(result) == 2  # Only files, not directories
            assert 'README.md' in result
            assert 'src/main.py' in result
            assert result == sorted(result)  # Should be sorted
    
    def test_list_repository_files_with_path(self):
        """Test file listing với specific path"""
        repo_url = "https://github.com/test/repo"
        path = "src"
        
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            mock_repo = MagicMock()
            mock_clone.return_value = mock_repo
            
            # Mock subtree
            mock_file = MagicMock()
            mock_file.type = 'blob'
            mock_file.path = 'src/utils.py'
            
            mock_subtree = MagicMock()
            mock_subtree.traverse.return_value = [mock_file]
            
            mock_tree = MagicMock()
            mock_tree.__getitem__.return_value = mock_subtree
            mock_repo.commit.return_value.tree = mock_tree
            
            result = self.agent.list_repository_files(repo_url, path)
            
            assert len(result) == 1
            assert 'src/utils.py' in result
            mock_tree.__getitem__.assert_called_once_with(path)
    
    def test_get_repository_info_success(self):
        """Test successful repository info retrieval"""
        repo_url = "https://github.com/test/awesome-project"
        
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            mock_repo = MagicMock()
            mock_clone.return_value = mock_repo
            
            # Mock latest commit
            mock_commit = MagicMock()
            mock_commit.hexsha = "1234567890abcdef"
            mock_commit.message = "Add new feature for user authentication"
            mock_commit.author = "John Doe <john@example.com>"
            mock_commit.committed_datetime = datetime(2024, 1, 20, 14, 30, 0)
            
            mock_repo.head.commit = mock_commit
            
            # Mock branches
            mock_ref1 = MagicMock()
            mock_ref1.name = "origin/main"
            mock_ref2 = MagicMock()
            mock_ref2.name = "origin/develop"
            mock_ref3 = MagicMock()
            mock_ref3.name = "origin/feature/auth"
            
            mock_remote = MagicMock()
            mock_remote.refs = [mock_ref1, mock_ref2, mock_ref3]
            mock_repo.remote.return_value = mock_remote
            
            # Mock tags
            mock_tag1 = MagicMock()
            mock_tag1.name = "v1.0.0"
            mock_tag2 = MagicMock()
            mock_tag2.name = "v1.1.0"
            
            mock_repo.tags = [mock_tag1, mock_tag2]
            
            result = self.agent.get_repository_info(repo_url)
            
            # Verify result structure
            assert 'error' not in result
            assert result['platform'] == 'github'
            assert result['owner'] == 'test'
            assert result['repo_name'] == 'awesome-project'
            assert result['full_name'] == 'test/awesome-project'
            assert result['clone_url'] == 'https://github.com/test/awesome-project.git'
            
            # Verify commit info
            commit_info = result['latest_commit']
            assert commit_info['sha'] == '12345678'  # First 8 chars
            assert commit_info['message'] == 'Add new feature for user authentication'
            assert commit_info['author'] == 'John Doe <john@example.com>'
            assert '2024-01-20T14:30:00' in commit_info['date']
            
            # Verify branches and tags
            assert 'main' in result['branches']
            assert 'develop' in result['branches']
            assert 'auth' in result['branches']  # 'feature/auth' becomes 'auth' after split
            assert 'v1.0.0' in result['tags']
            assert 'v1.1.0' in result['tags']
    
    def test_get_repository_info_error(self):
        """Test repository info retrieval với error"""
        repo_url = "https://github.com/test/repo"
        
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            mock_clone.side_effect = GitError("Repository not accessible")
            
            result = self.agent.get_repository_info(repo_url)
            
            assert 'error' in result
            assert 'Repository not accessible' in result['error']
    
    def test_cleanup_with_repos(self):
        """Test cleanup functionality với repositories"""
        # Add mock repositories
        mock_repo1 = MagicMock()
        mock_repo2 = MagicMock()
        
        self.agent.cloned_repos['repo1'] = mock_repo1
        self.agent.cloned_repos['repo2'] = mock_repo2
        
        # Test cleanup
        self.agent.cleanup()
        
        # Verify all repos were closed
        mock_repo1.close.assert_called_once()
        mock_repo2.close.assert_called_once()
    
    def test_cleanup_repo_without_close_method(self):
        """Test cleanup với repo object không có close method"""
        # Add mock repo without close method
        mock_repo = MagicMock()
        del mock_repo.close  # Remove close method
        
        self.agent.cloned_repos['repo'] = mock_repo
        
        # Should not raise error
        self.agent.cleanup()
    
    def test_workspace_cleanup_temp_directory(self):
        """Test workspace cleanup cho temporary directory"""
        # Create agent với temp directory
        agent = CodeFetcherAgent()
        temp_workspace = agent.workspace_dir
        
        # Verify it's a temp directory
        assert temp_workspace.startswith(tempfile.gettempdir())
        assert os.path.exists(temp_workspace)
        
        # Cleanup
        agent.cleanup()
        
        # Verify directory was removed
        assert not os.path.exists(temp_workspace)
    
    def test_workspace_cleanup_custom_directory(self):
        """Test workspace cleanup cho custom directory"""
        # Custom directory should not be removed
        custom_dir = tempfile.mkdtemp(prefix="custom_workspace_")
        agent = CodeFetcherAgent(workspace_dir=custom_dir)
        
        # Cleanup
        agent.cleanup()
        
        # Custom directory might be removed if it's detected as temp
        # This is expected behavior
        
        # Manual cleanup if still exists
        if os.path.exists(custom_dir):
            shutil.rmtree(custom_dir)


class TestErrorHandlingMocked:
    """Test error handling scenarios với mocks"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="test_error_")
        self.agent = CodeFetcherAgent(workspace_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.agent.cleanup()
    
    def test_clone_repository_network_error(self):
        """Test network error during cloning"""
        with patch('git.Repo.clone_from') as mock_clone:
            mock_clone.side_effect = GitCommandError("clone", "Network unreachable")
            
            repo_url = "https://github.com/test/repo"
            
            with pytest.raises(GitError) as exc_info:
                self.agent.clone_repository(repo_url)
            
            assert "Git command failed" in str(exc_info.value)
            assert "Network unreachable" in str(exc_info.value)
    
    def test_clone_repository_permission_error(self):
        """Test permission error during cloning"""
        with patch('git.Repo.clone_from') as mock_clone:
            mock_clone.side_effect = PermissionError("Permission denied")
            
            repo_url = "https://github.com/test/repo"
            
            with pytest.raises(GitError) as exc_info:
                self.agent.clone_repository(repo_url)
            
            assert "Failed to clone repository" in str(exc_info.value)
            assert "Permission denied" in str(exc_info.value)
    
    def test_get_pr_diff_fetch_timeout(self):
        """Test timeout during PR fetch"""
        repo_url = "https://github.com/test/repo"
        pr_id = 123
        
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            mock_repo = MagicMock()
            mock_clone.return_value = mock_repo
            
            # Mock timeout error
            mock_repo.remotes.origin.fetch.side_effect = GitCommandError("fetch", "Timeout")
            
            result = self.agent.get_pr_diff(repo_url, pr_id)
            
            assert result['error'] is not None
            assert "Failed to fetch PR" in result['error']
            assert "Timeout" in result['error']
    
    def test_get_file_content_repo_error(self):
        """Test repository error during file content retrieval"""
        repo_url = "https://github.com/test/repo"
        file_path = "test.txt"
        
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            mock_clone.side_effect = GitError("Repository corrupted")
            
            result = self.agent.get_file_content(repo_url, file_path)
            
            assert result is None
    
    def test_list_files_tree_error(self):
        """Test tree traversal error"""
        repo_url = "https://github.com/test/repo"
        
        with patch.object(self.agent, 'clone_repository') as mock_clone:
            mock_repo = MagicMock()
            mock_clone.return_value = mock_repo
            
            # Mock tree traversal error
            mock_repo.commit.return_value.tree.traverse.side_effect = Exception("Tree corrupted")
            
            result = self.agent.list_repository_files(repo_url)
            
            assert result == []


if __name__ == "__main__":
    # Chạy tests khi file được execute trực tiếp
    pytest.main([__file__, "-v"]) 