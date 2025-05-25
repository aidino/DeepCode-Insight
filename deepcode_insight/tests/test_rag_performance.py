"""Performance tests cho RAGContextAgent"""

import pytest
import time
import psutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch
import os
import random
import string

from ..agents.rag_context import RAGContextAgent

def generate_random_code(size=1000):
    """Generate random Python code với specified size"""
    functions = []
    for i in range(size // 50):  # Assume average function is 50 lines
        func_name = f"test_func_{i}"
        func_body = "\n    ".join([
            f"var_{j} = {random.randint(1, 100)}" 
            for j in range(random.randint(3, 8))
        ])
        functions.append(f"""def {func_name}():
    {func_body}
    return True
""")
    return "\n\n".join(functions)

@pytest.fixture
def performance_agent():
    """RAGContextAgent với real Qdrant (assuming running locally)"""
    agent = RAGContextAgent(
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="performance_test"
    )
    yield agent
    # Cleanup after tests
    agent.clear_collection()

def test_indexing_performance(performance_agent):
    """Test indexing performance với large files"""
    # Generate test data
    file_sizes = [1000, 5000, 10000]  # Lines of code
    results = {}
    
    for size in file_sizes:
        code = generate_random_code(size)
        
        # Measure memory before
        mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # Measure time
        start_time = time.time()
        success = performance_agent.index_code_file(
            code=code,
            filename=f"test_{size}.py",
            language="python"
        )
        index_time = time.time() - start_time
        
        # Measure memory after
        mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        mem_used = mem_after - mem_before
        
        results[size] = {
            'index_time': index_time,
            'memory_used': mem_used,
            'success': success
        }
        
        # Performance assertions
        assert index_time < size * 0.1  # Should index 10 lines per second minimum
        assert mem_used < size * 0.5  # Should use less than 0.5MB per line
        assert success is True

async def concurrent_query(agent, query):
    """Helper for concurrent querying"""
    return await asyncio.to_thread(agent.query, query)

async def test_concurrent_query_performance(performance_agent):
    """Test concurrent query performance"""
    # Index some test data first
    code = generate_random_code(1000)
    performance_agent.index_code_file(code, "test_concurrent.py", "python")
    
    # Test concurrent queries
    num_concurrent = 10
    queries = [f"test query {i}" for i in range(num_concurrent)]
    
    start_time = time.time()
    results = await asyncio.gather(*[
        concurrent_query(performance_agent, query) 
        for query in queries
    ])
    total_time = time.time() - start_time
    
    # Performance assertions
    assert total_time < num_concurrent * 2  # Should handle each query within 2 seconds
    assert all(len(r["matches"]) > 0 for r in results)  # All queries should return results

def test_batch_operations(performance_agent):
    """Test batch indexing và querying performance"""
    # Generate batch of files
    batch_size = 5
    files = [
        (generate_random_code(500), f"batch_test_{i}.py")
        for i in range(batch_size)
    ]
    
    # Test batch indexing
    start_time = time.time()
    for code, filename in files:
        success = performance_agent.index_code_file(code, filename, "python")
        assert success is True
    batch_index_time = time.time() - start_time
    
    # Performance assertions
    assert batch_index_time < batch_size * 5  # Should index each file within 5 seconds
    
    # Test batch querying
    queries = [f"function test_func_{i}" for i in range(batch_size)]
    start_time = time.time()
    results = [performance_agent.query(q) for q in queries]
    batch_query_time = time.time() - start_time
    
    assert batch_query_time < batch_size * 2  # Should query each within 2 seconds
    assert all(len(r["matches"]) > 0 for r in results)

def test_memory_usage(performance_agent):
    """Test memory usage during operations"""
    large_code = generate_random_code(10000)
    process = psutil.Process(os.getpid())
    
    # Monitor memory during indexing
    mem_usage = []
    def monitor_memory():
        while True:
            mem_usage.append(process.memory_info().rss / 1024 / 1024)
            time.sleep(0.1)
    
    # Start memory monitoring in separate thread
    with ThreadPoolExecutor(max_workers=1) as executor:
        monitor_future = executor.submit(monitor_memory)
        
        try:
            # Perform indexing
            success = performance_agent.index_code_file(
                large_code, 
                "memory_test.py",
                "python"
            )
            assert success is True
            
            # Stop monitoring
            monitor_future.cancel()
            
            # Analyze memory usage
            max_mem = max(mem_usage)
            min_mem = min(mem_usage)
            mem_increase = max_mem - min_mem
            
            # Memory usage assertions
            assert mem_increase < 1000  # Should not increase more than 1GB
            assert max_mem < 2000  # Total memory should not exceed 2GB
            
        finally:
            monitor_future.cancel()

def test_stress_testing(performance_agent):
    """Stress testing với continuous operations"""
    num_operations = 50
    operation_times = []
    
    for i in range(num_operations):
        # Alternate between indexing và querying
        start_time = time.time()
        
        if i % 2 == 0:
            # Indexing operation
            code = generate_random_code(100)
            success = performance_agent.index_code_file(
                code,
                f"stress_test_{i}.py",
                "python"
            )
            assert success is True
        else:
            # Query operation
            results = performance_agent.query(f"test query {i}")
            assert len(results["matches"]) >= 0
        
        operation_time = time.time() - start_time
        operation_times.append(operation_time)
        
        # Performance assertions
        assert operation_time < 5  # Each operation should complete within 5 seconds
    
    # Analyze operation times
    avg_time = sum(operation_times) / len(operation_times)
    max_time = max(operation_times)
    
    assert avg_time < 2  # Average operation time should be under 2 seconds
    assert max_time < 5  # Maximum operation time should be under 5 seconds 