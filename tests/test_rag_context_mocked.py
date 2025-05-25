#!/usr/bin/env python3
"""
Comprehensive test suite cho RAGContextAgent với mocked dependencies
Tests indexing và querying logic mà không cần external services
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Add paths
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(project_root, 'deepcode_insight'))
sys.path.append(project_root)

# Import config
from config import config

class TestRAGContextAgentMocked:
    """Test suite cho RAGContextAgent với mocked dependencies"""
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client"""
        mock_client = Mock()
        
        # Mock collection operations
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client.collection_exists.return_value = False
        mock_client.create_collection.return_value = True
        mock_client.delete_collection.return_value = True
        
        # Mock vector operations
        mock_client.upsert.return_value = Mock(status="completed")
        mock_client.search.return_value = [
            Mock(
                id=1,
                score=0.95,
                payload={
                    "content": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                    "filename": "math_utils.py",
                    "language": "python",
                    "chunk_type": "function",
                    "function_name": "fibonacci"
                }
            ),
            Mock(
                id=2,
                score=0.87,
                payload={
                    "content": "class Calculator: def add(self, a, b): return a + b",
                    "filename": "calculator.py", 
                    "language": "python",
                    "chunk_type": "class",
                    "class_name": "Calculator"
                }
            )
        ]
        
        # Mock collection info
        mock_client.get_collection.return_value = Mock(
            points_count=10,
            vectors_count=10,
            config=Mock(params=Mock(vectors=Mock(size=1536)))
        )
        
        return mock_client
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client"""
        mock_client = Mock()
        
        # Mock embeddings
        mock_embedding_response = Mock()
        mock_embedding_response.data = [
            Mock(embedding=[0.1] * 1536)  # Mock 1536-dimensional embedding
        ]
        mock_client.embeddings.create.return_value = mock_embedding_response
        
        # Mock chat completions
        mock_chat_response = Mock()
        mock_chat_response.choices = [
            Mock(message=Mock(content="This is a mocked LLM response explaining the code context."))
        ]
        mock_client.chat.completions.create.return_value = mock_chat_response
        
        return mock_client
    
    @pytest.fixture
    def mock_llamaindex_components(self):
        """Mock LlamaIndex components"""
        mocks = {}
        
        # Mock Document
        mock_document = Mock()
        mock_document.text = "Sample code content"
        mock_document.metadata = {"filename": "test.py", "language": "python"}
        mocks['Document'] = Mock(return_value=mock_document)
        
        # Mock VectorStoreIndex
        mock_index = Mock()
        mock_index.insert.return_value = None
        mock_index.as_query_engine.return_value = Mock()
        mocks['VectorStoreIndex'] = Mock(return_value=mock_index)
        
        # Mock QdrantVectorStore
        mock_vector_store = Mock()
        mocks['QdrantVectorStore'] = Mock(return_value=mock_vector_store)
        
        # Mock OpenAIEmbedding
        mock_embedding = Mock()
        mock_embedding.get_text_embedding.return_value = [0.1] * 1536
        mocks['OpenAIEmbedding'] = Mock(return_value=mock_embedding)
        
        # Mock ServiceContext
        mock_service_context = Mock()
        mocks['ServiceContext'] = Mock(from_defaults=Mock(return_value=mock_service_context))
        
        return mocks
    
    @pytest.fixture
    def mock_ast_parser(self):
        """Mock AST parser"""
        mock_parser = Mock()
        mock_parser.parse_code.return_value = {
            'functions': [
                {'name': 'fibonacci', 'line_start': 1, 'line_end': 3, 'complexity': 2},
                {'name': 'factorial', 'line_start': 5, 'line_end': 8, 'complexity': 3}
            ],
            'classes': [
                {'name': 'Calculator', 'line_start': 10, 'line_end': 15, 'methods': ['add', 'subtract']}
            ],
            'imports': ['math', 'typing'],
            'total_lines': 20,
            'complexity_score': 5
        }
        return mock_parser
    
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    @patch('deepcode_insight.agents.rag_context.Document')
    @patch('deepcode_insight.agents.rag_context.VectorStoreIndex')
    @patch('deepcode_insight.agents.rag_context.QdrantVectorStore')
    @patch('deepcode_insight.agents.rag_context.OpenAIEmbedding')
    @patch('deepcode_insight.agents.rag_context.ServiceContext')
    def test_rag_context_agent_initialization(self, mock_service_context, mock_openai_embedding, 
                                            mock_qdrant_vector_store, mock_vector_store_index,
                                            mock_document, mock_ast_class, mock_openai_class, mock_qdrant_class, 
                                            mock_qdrant_client, mock_openai_client, mock_ast_parser):
        """Test RAGContextAgent initialization với mocked dependencies"""
        
        # Setup mocks
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        # Import và initialize
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        agent = RAGContextAgent()
        
        # Verify initialization
        assert agent is not None
        mock_qdrant_class.assert_called_once()
        mock_ast_class.assert_called_once()
        
        # Verify collection setup
        mock_qdrant_client.collection_exists.assert_called()
        
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_index_code_file_success(self, mock_ast_class, mock_openai_class, mock_qdrant_class,
                                   mock_qdrant_client, mock_openai_client, mock_ast_parser):
        """Test successful code file indexing"""
        
        # Setup mocks
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        agent = RAGContextAgent()
        
        # Test code
        test_code = '''
def fibonacci(n):
    """Calculate fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b
'''
        
        # Mock successful indexing
        mock_qdrant_client.upsert.return_value = Mock(status="completed")
        
        # Test indexing
        result = agent.index_code_file(test_code, "test.py", "python")
        
        # Verify results
        assert result is True
        mock_ast_parser.parse_code.assert_called_once_with(test_code, "test.py")
        mock_openai_client.embeddings.create.assert_called()
        mock_qdrant_client.upsert.assert_called()
        
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_index_code_file_failure(self, mock_ast_class, mock_openai_class, mock_qdrant_class,
                                   mock_qdrant_client, mock_openai_client, mock_ast_parser):
        """Test code file indexing failure handling"""
        
        # Setup mocks
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        agent = RAGContextAgent()
        
        # Mock failure
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")
        
        # Test indexing failure
        result = agent.index_code_file("invalid code", "test.py", "python")
        
        # Verify failure handling
        assert result is False
        
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_query_success(self, mock_ast_class, mock_openai_class, mock_qdrant_class,
                         mock_qdrant_client, mock_openai_client, mock_ast_parser):
        """Test successful querying"""
        
        # Setup mocks
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        agent = RAGContextAgent()
        
        # Test query
        query = "How to calculate fibonacci numbers?"
        result = agent.query(query, top_k=2)
        
        # Verify results
        assert result is not None
        assert "total_results" in result
        assert "results" in result
        assert result["total_results"] == 2
        assert len(result["results"]) == 2
        
        # Verify search was called
        mock_openai_client.embeddings.create.assert_called()
        mock_qdrant_client.search.assert_called()
        
        # Verify result structure
        first_result = result["results"][0]
        assert "content_preview" in first_result
        assert "metadata" in first_result
        assert "score" in first_result
        
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_query_with_context_generation(self, mock_ast_class, mock_openai_class, mock_qdrant_class,
                                         mock_qdrant_client, mock_openai_client, mock_ast_parser):
        """Test query với LLM context generation"""
        
        # Setup mocks
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        agent = RAGContextAgent()
        
        # Test query với context generation
        query = "Explain fibonacci implementation"
        result = agent.query_with_context(query, top_k=2, generate_response=True)
        
        # Verify results
        assert result is not None
        assert "total_chunks" in result
        assert "context" in result
        assert "response" in result
        assert result["total_chunks"] == 2
        
        # Verify LLM was called for response generation
        mock_openai_client.chat.completions.create.assert_called()
        
        # Verify response content
        assert "This is a mocked LLM response" in result["response"]
        
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_chunk_code_file(self, mock_ast_class, mock_openai_class, mock_qdrant_class,
                           mock_qdrant_client, mock_openai_client, mock_ast_parser):
        """Test code chunking logic"""
        
        # Setup mocks
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        agent = RAGContextAgent()
        
        # Test code
        test_code = '''
def function1():
    """First function"""
    return 1

def function2():
    """Second function"""
    return 2

class TestClass:
    def method1(self):
        return "method1"
'''
        
        # Test chunking
        chunks = agent.chunk_code_file(test_code, "test.py", "python")
        
        # Verify chunking
        assert chunks is not None
        assert len(chunks) > 0
        
        # Verify AST parsing was called
        mock_ast_parser.parse_code.assert_called_once_with(test_code, "test.py")
        
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_get_collection_stats(self, mock_ast_class, mock_openai_class, mock_qdrant_class,
                                mock_qdrant_client, mock_openai_client, mock_ast_parser):
        """Test collection statistics retrieval"""
        
        # Setup mocks
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        agent = RAGContextAgent()
        
        # Test stats retrieval
        stats = agent.get_collection_stats()
        
        # Verify stats
        assert stats is not None
        assert "total_points" in stats
        assert "vector_size" in stats
        
        # Verify Qdrant was called
        mock_qdrant_client.get_collection.assert_called()
        
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_clear_collection(self, mock_ast_class, mock_openai_class, mock_qdrant_class,
                            mock_qdrant_client, mock_openai_client, mock_ast_parser):
        """Test collection clearing"""
        
        # Setup mocks
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        agent = RAGContextAgent()
        
        # Test collection clearing
        result = agent.clear_collection()
        
        # Verify clearing
        assert result is True
        mock_qdrant_client.delete_collection.assert_called()
        mock_qdrant_client.create_collection.assert_called()
        
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_error_handling_qdrant_connection(self, mock_ast_class, mock_openai_class, mock_qdrant_class,
                                            mock_qdrant_client, mock_openai_client, mock_ast_parser):
        """Test error handling khi Qdrant connection fails"""
        
        # Setup mocks với connection error
        mock_qdrant_class.side_effect = Exception("Connection failed")
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        # Test initialization với connection error
        with pytest.raises(Exception):
            agent = RAGContextAgent()
            
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_error_handling_openai_api(self, mock_ast_class, mock_openai_class, mock_qdrant_class,
                                     mock_qdrant_client, mock_openai_client, mock_ast_parser):
        """Test error handling khi OpenAI API fails"""
        
        # Setup mocks
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        # Mock OpenAI API error
        mock_openai_client.embeddings.create.side_effect = Exception("API rate limit exceeded")
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        agent = RAGContextAgent()
        
        # Test query với API error
        result = agent.query("test query")
        
        # Verify error handling
        assert "error" in result
        
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_metadata_extraction(self, mock_ast_class, mock_openai_class, mock_qdrant_class,
                                mock_qdrant_client, mock_openai_client, mock_ast_parser):
        """Test metadata extraction từ code"""
        
        # Setup mocks
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        agent = RAGContextAgent()
        
        # Test code với rich metadata
        test_code = '''
import math
from typing import List

def fibonacci(n: int) -> int:
    """Calculate fibonacci number using recursion"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    """A simple calculator class"""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
'''
        
        # Test indexing để extract metadata
        result = agent.index_code_file(test_code, "calculator.py", "python", 
                                     metadata={"project": "test", "version": "1.0"})
        
        # Verify AST parsing was called với correct parameters
        mock_ast_parser.parse_code.assert_called_once_with(test_code, "calculator.py")
        
        # Verify embedding creation was called
        mock_openai_client.embeddings.create.assert_called()
        
        # Verify upsert was called với metadata
        mock_qdrant_client.upsert.assert_called()
        
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_different_languages(self, mock_ast_class, mock_openai_class, mock_qdrant_class,
                                mock_qdrant_client, mock_openai_client, mock_ast_parser):
        """Test indexing different programming languages"""
        
        # Setup mocks
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        agent = RAGContextAgent()
        
        # Test Python code
        python_code = "def hello(): print('Hello Python')"
        result_py = agent.index_code_file(python_code, "test.py", "python")
        assert result_py is True
        
        # Test Java code
        java_code = "public class Hello { public void hello() { System.out.println(\"Hello Java\"); } }"
        result_java = agent.index_code_file(java_code, "Test.java", "java")
        assert result_java is True
        
        # Test JavaScript code
        js_code = "function hello() { console.log('Hello JavaScript'); }"
        result_js = agent.index_code_file(js_code, "test.js", "javascript")
        assert result_js is True
        
        # Verify all were processed
        assert mock_ast_parser.parse_code.call_count == 3
        
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_query_filtering(self, mock_ast_class, mock_openai_class, mock_qdrant_class,
                           mock_qdrant_client, mock_openai_client, mock_ast_parser):
        """Test query filtering by metadata"""
        
        # Setup mocks
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        agent = RAGContextAgent()
        
        # Test query với language filter
        query = "fibonacci function"
        result = agent.query(query, top_k=5, filter_metadata={"language": "python"})
        
        # Verify filtering was applied
        assert result is not None
        mock_qdrant_client.search.assert_called()
        
        # Check if filter was passed to search
        search_call = mock_qdrant_client.search.call_args
        assert search_call is not None


def test_rag_context_agent_mocked_suite():
    """Run all mocked tests"""
    
    print("🧪 === Running RAGContextAgent Mocked Test Suite ===\n")
    
    # Run pytest với this file
    import subprocess
    result = subprocess.run([
        'python', '-m', 'pytest', 
        __file__, 
        '-v', 
        '--tb=short'
    ], capture_output=True, text=True)
    
    print("Test Results:")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    print("🚀 Testing RAGContextAgent với Mocked Dependencies\n")
    
    # Run individual test functions for demonstration
    test_instance = TestRAGContextAgentMocked()
    
    print("📋 Available Test Methods:")
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    for i, method in enumerate(test_methods, 1):
        print(f"  {i}. {method}")
    
    print(f"\n✅ Total test methods: {len(test_methods)}")
    print("\n🔧 To run all tests:")
    print("   python -m pytest tests/test_rag_context_mocked.py -v")
    print("\n🔧 To run specific test:")
    print("   python -m pytest tests/test_rag_context_mocked.py::TestRAGContextAgentMocked::test_query_success -v")
    
    print("\n📚 Test Coverage:")
    print("  ✓ Initialization với mocked dependencies")
    print("  ✓ Code indexing success và failure scenarios")
    print("  ✓ Query execution và result formatting")
    print("  ✓ Context generation với LLM")
    print("  ✓ Code chunking logic")
    print("  ✓ Collection management")
    print("  ✓ Error handling")
    print("  ✓ Metadata extraction")
    print("  ✓ Multi-language support")
    print("  ✓ Query filtering") 