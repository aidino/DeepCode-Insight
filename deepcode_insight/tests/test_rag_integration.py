"""Integration tests cho RAGContextAgent"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from ..agents.rag_context import RAGContextAgent
from ..parsers.ast_parser import ASTParsingAgent
from config import config

@pytest.fixture
def test_repo():
    """Create temporary test repository"""
    temp_dir = tempfile.mkdtemp()
    repo_dir = Path(temp_dir) / "test_repo"
    repo_dir.mkdir()
    
    # Create test files
    test_files = {
        "main.py": """
def main():
    print("Hello World")
    helper_function()
    
def helper_function():
    return True
""",
        "utils/helper.py": """
class Helper:
    def __init__(self):
        self.value = 0
        
    def increment(self):
        self.value += 1
        return self.value
""",
        "tests/test_main.py": """
import pytest
from main import main, helper_function

def test_main():
    assert helper_function() is True
"""
    }
    
    # Write test files
    for filepath, content in test_files.items():
        file_path = repo_dir / filepath
        file_path.parent.mkdir(exist_ok=True, parents=True)
        file_path.write_text(content)
    
    yield repo_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def integration_agent():
    """RAGContextAgent với real dependencies"""
    agent = RAGContextAgent(
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="integration_test"
    )
    yield agent
    # Cleanup
    agent.clear_collection()

def test_end_to_end_workflow(integration_agent, test_repo):
    """Test end-to-end workflow từ indexing đến querying"""
    # Index entire repository
    class MockCodeFetcher:
        def get_repository_files(self, repo_path, patterns):
            files = []
            for pattern in patterns:
                files.extend(Path(repo_path).rglob(pattern))
            return [(f.read_text(), f.name) for f in files]
    
    mock_fetcher = MockCodeFetcher()
    index_results = integration_agent.index_repository(
        mock_fetcher,
        str(test_repo),
        ["*.py"]
    )
    
    assert index_results["success"] is True
    assert index_results["total_files"] > 0
    
    # Test basic querying
    results = integration_agent.query(
        "function that prints Hello World",
        top_k=1
    )
    assert len(results["matches"]) > 0
    assert "main.py" in results["matches"][0]["metadata"]["filename"]
    
    # Test context-aware querying
    results = integration_agent.query_with_context(
        "How does the Helper class work?",
        top_k=2,
        generate_response=True
    )
    assert len(results["matches"]) > 0
    assert "Helper" in results["generated_response"]
    
    # Test filtering
    results = integration_agent.query(
        "test functions",
        filters={"file_type": "test"},
        top_k=1
    )
    assert len(results["matches"]) > 0
    assert "test" in results["matches"][0]["metadata"]["filename"]

def test_error_recovery(integration_agent, test_repo):
    """Test error recovery scenarios"""
    # Test với invalid file
    invalid_code = "def invalid_syntax("
    with pytest.raises(Exception):
        integration_agent.index_code_file(invalid_code, "invalid.py")
    
    # Verify system still works after error
    valid_code = "def valid_function(): pass"
    success = integration_agent.index_code_file(valid_code, "valid.py")
    assert success is True
    
    # Test với network failure simulation
    with patch('qdrant_client.QdrantClient.upload_collection') as mock_upload:
        mock_upload.side_effect = Exception("Network error")
        # Should handle error gracefully
        with pytest.raises(Exception):
            integration_agent.index_code_file("def test(): pass", "test.py")
    
    # Verify system recovers
    results = integration_agent.query("valid function")
    assert len(results["matches"]) > 0

def test_multi_language_support(integration_agent):
    """Test support cho multiple programming languages"""
    test_files = {
        "python": ("""
def python_function():
    return "Python"
""", "test.py"),
        "javascript": ("""
function javascriptFunction() {
    return "JavaScript";
}
""", "test.js"),
        "java": ("""
public class Test {
    public String javaMethod() {
        return "Java";
    }
}
""", "Test.java")
    }
    
    # Index files in different languages
    for lang, (code, filename) in test_files.items():
        success = integration_agent.index_code_file(
            code=code,
            filename=filename,
            language=lang
        )
        assert success is True
    
    # Test language-specific querying
    for lang in test_files.keys():
        results = integration_agent.query(
            f"function in {lang}",
            filters={"language": lang}
        )
        assert len(results["matches"]) > 0
        assert any(lang in m["metadata"]["language"] for m in results["matches"])

def test_collection_persistence(integration_agent, test_repo):
    """Test collection persistence và recovery"""
    # Index some test data
    test_code = "def test_persistence(): pass"
    integration_agent.index_code_file(test_code, "persistence.py")
    
    # Create new agent instance với same collection
    new_agent = RAGContextAgent(
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name=integration_agent.collection_name
    )
    
    # Verify data persists
    results = new_agent.query("test persistence")
    assert len(results["matches"]) > 0
    assert "persistence.py" in results["matches"][0]["metadata"]["filename"]
    
    # Test collection stats
    stats = new_agent.get_collection_stats()
    assert stats["total_vectors"] > 0
    
    # Cleanup
    new_agent.clear_collection()
    
    # Verify cleanup
    stats = new_agent.get_collection_stats()
    assert stats["total_vectors"] == 0

def test_api_failure_handling(integration_agent):
    """Test handling của API failures"""
    # Test OpenAI API failure
    with patch('llama_index.embeddings.openai.OpenAIEmbedding.get_text_embedding') as mock_embed:
        mock_embed.side_effect = Exception("API Error")
        
        with pytest.raises(Exception):
            integration_agent.index_code_file("def test(): pass", "test.py")
    
    # Test Qdrant API failure
    with patch('qdrant_client.QdrantClient.upsert') as mock_upsert:
        mock_upsert.side_effect = Exception("Qdrant Error")
        
        with pytest.raises(Exception):
            integration_agent.index_code_file("def test(): pass", "test.py")
    
    # Verify system state after failures
    stats = integration_agent.get_collection_stats()
    assert isinstance(stats, dict)  # Should still return valid stats 