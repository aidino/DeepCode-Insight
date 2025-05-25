"""Unit tests cho RAGContextAgent với mocked dependencies"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from datetime import datetime

from ..agents.rag_context import RAGContextAgent
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.schema import TextNode
from qdrant_client.models import Distance, VectorParams
from config import config

@pytest.fixture
def mock_qdrant_client():
    """Mock cho Qdrant client"""
    mock_client = Mock()
    mock_client.get_collections.return_value = Mock(collections=[])
    return mock_client

@pytest.fixture
def mock_openai():
    """Mock cho OpenAI embeddings và completions"""
    with patch('llama_index.embeddings.openai.OpenAIEmbedding') as mock_embed:
        with patch('llama_index.llms.openai.OpenAI') as mock_llm:
            mock_embed.return_value = Mock()
            mock_llm.return_value = Mock()
            yield (mock_embed, mock_llm)

@pytest.fixture
def mock_ast_parser():
    """Mock cho AST parser"""
    mock_parser = Mock()
    mock_parser.parse_code.return_value = {
        'stats': {'num_functions': 5, 'num_classes': 2},
        'classes': [{'name': 'TestClass1'}, {'name': 'TestClass2'}],
        'functions': [{'name': 'test_func', 'start_line': 1, 'end_line': 10}]
    }
    return mock_parser

@pytest.fixture
def rag_agent(mock_qdrant_client, mock_openai, mock_ast_parser):
    """Fixture cho RAGContextAgent với mocked dependencies"""
    with patch('deepcode_insight.agents.rag_context.QdrantClient', return_value=mock_qdrant_client):
        with patch('deepcode_insight.agents.rag_context.ASTParsingAgent', return_value=mock_ast_parser):
            agent = RAGContextAgent(
                qdrant_host="mock_host",
                qdrant_port=6333,
                collection_name="test_collection"
            )
            return agent

def test_initialization(rag_agent, mock_qdrant_client):
    """Test khởi tạo RAGContextAgent"""
    assert rag_agent.collection_name == "test_collection"
    assert rag_agent.qdrant_client == mock_qdrant_client
    mock_qdrant_client.create_collection.assert_called_once()

def test_chunk_code_file(rag_agent, mock_ast_parser):
    """Test chunking code file"""
    test_code = '''
def test_function():
    print("Hello World")
    
class TestClass:
    def method(self):
        pass
'''
    documents = rag_agent.chunk_code_file(
        code=test_code,
        filename="test.py",
        language="python"
    )
    
    assert len(documents) > 0
    assert isinstance(documents[0], Document)
    assert "filename" in documents[0].metadata
    assert documents[0].metadata["language"] == "python"

def test_index_code_file(rag_agent):
    """Test indexing code file"""
    test_code = "def test(): pass"
    
    success = rag_agent.index_code_file(
        code=test_code,
        filename="test.py",
        language="python"
    )
    
    assert success is True
    # Verify vector store was called
    assert rag_agent.vector_store.add.called

def test_query(rag_agent):
    """Test querying indexed code"""
    # Mock vector store response
    mock_node = TextNode(text="test code", metadata={"filename": "test.py"})
    rag_agent.index.as_retriever = Mock(return_value=Mock(retrieve=Mock(return_value=[mock_node])))
    
    results = rag_agent.query(
        query_text="test query",
        top_k=1
    )
    
    assert len(results["matches"]) == 1
    assert "filename" in results["matches"][0]["metadata"]

def test_error_handling(rag_agent):
    """Test error handling trong các operations chính"""
    # Test indexing với invalid code
    rag_agent.ast_parser.parse_code.side_effect = Exception("Parse error")
    
    with pytest.raises(Exception):
        rag_agent.index_code_file("invalid code", "test.py")
    
    # Test querying với failed vector store
    rag_agent.vector_store.query.side_effect = Exception("Query error")
    
    with pytest.raises(Exception):
        rag_agent.query("test query")

def test_collection_management(rag_agent):
    """Test collection management operations"""
    # Test clear collection
    rag_agent.clear_collection()
    rag_agent.qdrant_client.delete_collection.assert_called_once_with(
        collection_name=rag_agent.collection_name
    )
    
    # Test get collection stats
    mock_stats = {"vectors_count": 100}
    rag_agent.qdrant_client.get_collection.return_value = Mock(vectors_count=100)
    
    stats = rag_agent.get_collection_stats()
    assert stats["total_vectors"] == 100 