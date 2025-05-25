#!/usr/bin/env python3
"""
Performance và Integration tests cho RAGContextAgent
Tests performance metrics, memory usage, và integration scenarios
"""

import sys
import os
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Add paths
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(project_root, 'deepcode_insight'))
sys.path.append(project_root)

# Import config
from deepcode_insight.config import config

class TestRAGPerformance:
    """Performance tests cho RAGContextAgent"""
    
    def setup_method(self):
        """Setup cho mỗi test method"""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
    def teardown_method(self):
        """Cleanup sau mỗi test method"""
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - self.start_time
        memory_diff = end_memory - self.start_memory
        
        print(f"Test duration: {duration:.2f}s, Memory change: {memory_diff:.2f}MB")
    
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_indexing_performance(self, mock_ast_class, mock_openai_class, mock_qdrant_class):
        """Test indexing performance với large code files"""
        
        # Setup mocks
        mock_qdrant_client = Mock()
        mock_openai_client = Mock()
        mock_ast_parser = Mock()
        
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        # Mock responses
        mock_qdrant_client.collection_exists.return_value = True
        mock_qdrant_client.upsert.return_value = Mock(status="completed")
        mock_openai_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1] * 1536)]
        )
        mock_ast_parser.parse_code.return_value = {
            'functions': [{'name': f'func_{i}', 'line_start': i*10, 'line_end': i*10+5} for i in range(50)],
            'classes': [{'name': f'Class_{i}', 'line_start': i*20, 'line_end': i*20+10} for i in range(10)],
            'imports': ['os', 'sys', 'typing'],
            'total_lines': 1000,
            'complexity_score': 25
        }
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        agent = RAGContextAgent()
        
        # Generate large code file
        large_code = self._generate_large_code_file(1000)  # 1000 lines
        
        # Measure indexing performance
        start_time = time.time()
        result = agent.index_code_file(large_code, "large_file.py", "python")
        end_time = time.time()
        
        indexing_time = end_time - start_time
        
        # Verify performance
        assert result is True
        assert indexing_time < 5.0  # Should complete within 5 seconds
        
        print(f"Indexing 1000 lines took: {indexing_time:.2f}s")
        
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_query_performance(self, mock_ast_class, mock_openai_class, mock_qdrant_class):
        """Test query performance với multiple concurrent queries"""
        
        # Setup mocks
        mock_qdrant_client = Mock()
        mock_openai_client = Mock()
        mock_ast_parser = Mock()
        
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        # Mock responses
        mock_qdrant_client.collection_exists.return_value = True
        mock_qdrant_client.search.return_value = [
            Mock(id=i, score=0.9-i*0.1, payload={
                "content": f"def function_{i}(): return {i}",
                "filename": f"file_{i}.py",
                "language": "python"
            }) for i in range(10)
        ]
        mock_openai_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1] * 1536)]
        )
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        agent = RAGContextAgent()
        
        # Test concurrent queries
        queries = [
            "fibonacci function",
            "error handling",
            "database connection",
            "async operations",
            "data validation"
        ]
        
        start_time = time.time()
        
        # Run concurrent queries
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(agent.query, query, top_k=5) for query in queries]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        concurrent_time = end_time - start_time
        
        # Verify all queries succeeded
        assert len(results) == 5
        for result in results:
            assert "total_results" in result
            assert result["total_results"] > 0
        
        # Performance check
        assert concurrent_time < 3.0  # Should complete within 3 seconds
        
        print(f"5 concurrent queries took: {concurrent_time:.2f}s")
        
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_memory_usage(self, mock_ast_class, mock_openai_class, mock_qdrant_class):
        """Test memory usage với large datasets"""
        
        # Setup mocks
        mock_qdrant_client = Mock()
        mock_openai_client = Mock()
        mock_ast_parser = Mock()
        
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        # Mock responses
        mock_qdrant_client.collection_exists.return_value = True
        mock_qdrant_client.upsert.return_value = Mock(status="completed")
        mock_openai_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1] * 1536)]
        )
        mock_ast_parser.parse_code.return_value = {
            'functions': [{'name': f'func_{i}'} for i in range(100)],
            'classes': [{'name': f'Class_{i}'} for i in range(20)],
            'imports': ['os', 'sys'],
            'total_lines': 500,
            'complexity_score': 15
        }
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        agent = RAGContextAgent()
        
        # Index multiple files
        for i in range(10):
            code = self._generate_large_code_file(500)
            agent.index_code_file(code, f"file_{i}.py", "python")
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Memory usage should be reasonable
        assert memory_increase < 100  # Less than 100MB increase
        
        print(f"Memory increase after indexing 10 files: {memory_increase:.2f}MB")
        
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_batch_operations(self, mock_ast_class, mock_openai_class, mock_qdrant_class):
        """Test batch indexing operations"""
        
        # Setup mocks
        mock_qdrant_client = Mock()
        mock_openai_client = Mock()
        mock_ast_parser = Mock()
        
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        # Mock responses
        mock_qdrant_client.collection_exists.return_value = True
        mock_qdrant_client.upsert.return_value = Mock(status="completed")
        mock_openai_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1] * 1536)]
        )
        mock_ast_parser.parse_code.return_value = {
            'functions': [{'name': 'test_func'}],
            'classes': [],
            'imports': [],
            'total_lines': 10,
            'complexity_score': 1
        }
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        agent = RAGContextAgent()
        
        # Batch index multiple files
        files = [
            ("def func1(): pass", "file1.py", "python"),
            ("def func2(): pass", "file2.py", "python"),
            ("def func3(): pass", "file3.py", "python"),
            ("def func4(): pass", "file4.py", "python"),
            ("def func5(): pass", "file5.py", "python")
        ]
        
        start_time = time.time()
        
        # Sequential indexing
        results = []
        for code, filename, language in files:
            result = agent.index_code_file(code, filename, language)
            results.append(result)
        
        end_time = time.time()
        batch_time = end_time - start_time
        
        # Verify all succeeded
        assert all(results)
        assert len(results) == 5
        
        # Performance check
        assert batch_time < 2.0  # Should complete within 2 seconds
        
        print(f"Batch indexing 5 files took: {batch_time:.2f}s")
        
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_stress_testing(self, mock_ast_class, mock_openai_class, mock_qdrant_class):
        """Stress test với high load"""
        
        # Setup mocks
        mock_qdrant_client = Mock()
        mock_openai_client = Mock()
        mock_ast_parser = Mock()
        
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        # Mock responses
        mock_qdrant_client.collection_exists.return_value = True
        mock_qdrant_client.search.return_value = [
            Mock(id=1, score=0.9, payload={"content": "test", "filename": "test.py"})
        ]
        mock_openai_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1] * 1536)]
        )
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        agent = RAGContextAgent()
        
        # Stress test với many rapid queries
        num_queries = 50
        queries = [f"test query {i}" for i in range(num_queries)]
        
        start_time = time.time()
        
        # Rapid fire queries
        results = []
        for query in queries:
            result = agent.query(query, top_k=1)
            results.append(result)
        
        end_time = time.time()
        stress_time = end_time - start_time
        
        # Verify all queries succeeded
        assert len(results) == num_queries
        for result in results:
            assert "total_results" in result
        
        # Performance check
        avg_time_per_query = stress_time / num_queries
        assert avg_time_per_query < 0.1  # Less than 100ms per query on average
        
        print(f"Stress test: {num_queries} queries in {stress_time:.2f}s")
        print(f"Average time per query: {avg_time_per_query:.3f}s")
        
    def _generate_large_code_file(self, num_lines: int) -> str:
        """Generate a large code file for testing"""
        
        lines = [
            "#!/usr/bin/env python3",
            "\"\"\"Large test file for performance testing\"\"\"",
            "",
            "import os",
            "import sys",
            "import json",
            "from typing import List, Dict, Optional",
            ""
        ]
        
        # Generate functions
        for i in range(num_lines // 20):
            lines.extend([
                f"def function_{i}(param1, param2={i}):",
                f"    \"\"\"Function {i} documentation\"\"\"",
                f"    result = param1 + param2 + {i}",
                f"    if result > {i * 10}:",
                f"        return result * 2",
                f"    else:",
                f"        return result",
                ""
            ])
        
        # Generate classes
        for i in range(num_lines // 50):
            lines.extend([
                f"class TestClass{i}:",
                f"    \"\"\"Test class {i}\"\"\"",
                f"    ",
                f"    def __init__(self, value={i}):",
                f"        self.value = value",
                f"        self.data = []",
                f"    ",
                f"    def method_{i}(self, param):",
                f"        return self.value + param + {i}",
                f"    ",
                f"    def process_data(self, data_list):",
                f"        for item in data_list:",
                f"            self.data.append(item * {i})",
                f"        return self.data",
                ""
            ])
        
        # Pad to exact number of lines
        while len(lines) < num_lines:
            lines.append(f"# Padding line {len(lines)}")
        
        return "\n".join(lines[:num_lines])


class TestRAGIntegration:
    """Integration tests cho RAGContextAgent"""
    
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_end_to_end_workflow(self, mock_ast_class, mock_openai_class, mock_qdrant_class):
        """Test complete end-to-end workflow"""
        
        # Setup mocks
        mock_qdrant_client = Mock()
        mock_openai_client = Mock()
        mock_ast_parser = Mock()
        
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        # Mock collection setup
        mock_qdrant_client.collection_exists.return_value = False
        mock_qdrant_client.create_collection.return_value = True
        mock_qdrant_client.upsert.return_value = Mock(status="completed")
        
        # Mock search results
        mock_qdrant_client.search.return_value = [
            Mock(id=1, score=0.95, payload={
                "content": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "filename": "math_utils.py",
                "language": "python",
                "function_name": "fibonacci"
            })
        ]
        
        # Mock OpenAI responses
        mock_openai_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1] * 1536)]
        )
        mock_openai_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="The fibonacci function implements recursive calculation."))]
        )
        
        # Mock AST parsing
        mock_ast_parser.parse_code.return_value = {
            'functions': [{'name': 'fibonacci', 'line_start': 1, 'line_end': 3}],
            'classes': [],
            'imports': [],
            'total_lines': 3,
            'complexity_score': 2
        }
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        # 1. Initialize agent
        agent = RAGContextAgent()
        
        # 2. Index code
        code = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
        index_result = agent.index_code_file(code, "math_utils.py", "python")
        assert index_result is True
        
        # 3. Query code
        query_result = agent.query("fibonacci function", top_k=1)
        assert query_result["total_results"] == 1
        assert "fibonacci" in query_result["results"][0]["content_preview"]
        
        # 4. Generate context
        context_result = agent.query_with_context("Explain fibonacci", generate_response=True)
        assert "total_chunks" in context_result
        assert "response" in context_result
        assert "fibonacci" in context_result["response"]
        
        # 5. Get stats
        stats = agent.get_collection_stats()
        assert "total_points" in stats
        
        print("✅ End-to-end workflow completed successfully")
        
    @patch('deepcode_insight.agents.rag_context.QdrantClient')
    @patch('deepcode_insight.agents.rag_context.OpenAI')
    @patch('deepcode_insight.agents.rag_context.ASTParsingAgent')
    def test_error_recovery(self, mock_ast_class, mock_openai_class, mock_qdrant_class):
        """Test error recovery scenarios"""
        
        # Setup mocks
        mock_qdrant_client = Mock()
        mock_openai_client = Mock()
        mock_ast_parser = Mock()
        
        mock_qdrant_class.return_value = mock_qdrant_client
        mock_openai_class.return_value = mock_openai_client
        mock_ast_class.return_value = mock_ast_parser
        
        # Mock normal operations
        mock_qdrant_client.collection_exists.return_value = True
        mock_ast_parser.parse_code.return_value = {
            'functions': [], 'classes': [], 'imports': [],
            'total_lines': 1, 'complexity_score': 0
        }
        
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        agent = RAGContextAgent()
        
        # Test 1: OpenAI API failure during indexing
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")
        result = agent.index_code_file("def test(): pass", "test.py", "python")
        assert result is False
        
        # Test 2: Recovery after API failure
        mock_openai_client.embeddings.create.side_effect = None
        mock_openai_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1] * 1536)]
        )
        mock_qdrant_client.upsert.return_value = Mock(status="completed")
        
        result = agent.index_code_file("def test(): pass", "test.py", "python")
        assert result is True
        
        # Test 3: Query with API failure
        mock_openai_client.embeddings.create.side_effect = Exception("Rate limit")
        result = agent.query("test query")
        assert "error" in result
        
        print("✅ Error recovery scenarios tested successfully")


def run_performance_tests():
    """Run all performance tests"""
    
    print("🚀 === Running RAGContextAgent Performance Tests ===\n")
    
    # Create test instance
    perf_tests = TestRAGPerformance()
    integration_tests = TestRAGIntegration()
    
    test_methods = [
        (perf_tests, "test_indexing_performance"),
        (perf_tests, "test_query_performance"),
        (perf_tests, "test_memory_usage"),
        (perf_tests, "test_batch_operations"),
        (perf_tests, "test_stress_testing"),
        (integration_tests, "test_end_to_end_workflow"),
        (integration_tests, "test_error_recovery")
    ]
    
    results = []
    
    for test_instance, method_name in test_methods:
        print(f"\n🧪 Running {method_name}...")
        
        try:
            # Setup
            if hasattr(test_instance, 'setup_method'):
                test_instance.setup_method()
            
            # Run test
            method = getattr(test_instance, method_name)
            method()
            
            # Teardown
            if hasattr(test_instance, 'teardown_method'):
                test_instance.teardown_method()
            
            results.append((method_name, "PASSED"))
            print(f"✅ {method_name} PASSED")
            
        except Exception as e:
            results.append((method_name, f"FAILED: {e}"))
            print(f"❌ {method_name} FAILED: {e}")
    
    # Summary
    print(f"\n📊 Performance Test Results:")
    print(f"=" * 50)
    
    passed = sum(1 for _, status in results if status == "PASSED")
    total = len(results)
    
    for method_name, status in results:
        status_icon = "✅" if status == "PASSED" else "❌"
        print(f"{status_icon} {method_name}: {status}")
    
    print(f"\n🎯 Summary: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    print("🚀 RAGContextAgent Performance & Integration Testing\n")
    
    success = run_performance_tests()
    
    if success:
        print(f"\n🎉 All performance tests passed!")
    else:
        print(f"\n❌ Some performance tests failed!")
    
    print(f"\n📚 Performance Test Coverage:")
    print(f"  ✓ Indexing performance với large files")
    print(f"  ✓ Concurrent query performance")
    print(f"  ✓ Memory usage monitoring")
    print(f"  ✓ Batch operations")
    print(f"  ✓ Stress testing")
    print(f"  ✓ End-to-end workflow")
    print(f"  ✓ Error recovery scenarios")
    
    print(f"\n🔧 To run với pytest:")
    print(f"   python -m pytest tests/test_rag_performance.py -v")
    print(f"   python -m pytest tests/test_rag_performance.py::TestRAGPerformance::test_indexing_performance -v") 