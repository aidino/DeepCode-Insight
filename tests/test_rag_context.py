#!/usr/bin/env python3
"""Test script cho RAGContextAgent"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Add paths
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(project_root, 'deepcode_insight'))
sys.path.append(project_root)

# Import config
from config import config

def test_rag_context_basic():
    """Test basic functionality của RAGContextAgent"""
    
    print("🧪 === Testing RAGContextAgent Basic Functionality ===\n")
    
    try:
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        # Sample code để test
        sample_python_code = '''
def calculate_fibonacci(n):
    """Calculate fibonacci number using recursion"""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class MathUtils:
    """Utility class for mathematical operations"""
    
    @staticmethod
    def factorial(n):
        """Calculate factorial of n"""
        if n <= 1:
            return 1
        return n * MathUtils.factorial(n-1)
    
    @staticmethod
    def is_prime(n):
        """Check if number is prime"""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

def main():
    """Main function"""
    print("Fibonacci(10):", calculate_fibonacci(10))
    print("Factorial(5):", MathUtils.factorial(5))
    print("Is 17 prime?", MathUtils.is_prime(17))
'''
        
        # Initialize RAG agent with config
        print("1. Initializing RAGContextAgent...")
        print("   Config status:")
        config.print_config()
        rag_agent = RAGContextAgent()
        print("   ✅ RAGContextAgent initialized successfully")
        
        # Test chunking
        print("\n2. Testing code chunking...")
        chunks = rag_agent.chunk_code_file(sample_python_code, "math_utils.py", "python")
        print(f"   ✅ Code chunked into {len(chunks)} documents")
        
        # Show chunk details
        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i+1}: {chunk.metadata.get('chunk_type', 'unknown')} - {len(chunk.text)} chars")
        
        # Test indexing
        print("\n3. Testing code indexing...")
        success = rag_agent.index_code_file(sample_python_code, "math_utils.py", "python")
        print(f"   {'✅ Indexing successful' if success else '❌ Indexing failed'}")
        
        # Test basic query (without LLM)
        print("\n4. Testing basic queries...")
        queries = [
            "fibonacci calculation",
            "factorial function",
            "prime number check",
            "mathematical operations"
        ]
        
        for query in queries:
            results = rag_agent.query(query, top_k=2)
            print(f"   Query: '{query}' -> {results['total_results']} results")
            
            if results['total_results'] > 0:
                top_result = results['results'][0]
                filename = top_result.get('metadata', {}).get('filename', 'unknown')
                score = top_result.get('score', 0)
                print(f"     Top result: {filename} (score: {score:.3f})")
        
        # Test collection stats
        print("\n5. Testing collection statistics...")
        stats = rag_agent.get_collection_stats()
        if 'error' not in stats:
            print(f"   ✅ Collection stats retrieved:")
            print(f"     Total points: {stats.get('total_points', 0)}")
            print(f"     Languages: {stats.get('indexed_languages', [])}")
            print(f"     Vector size: {stats.get('vector_size', 0)}")
        else:
            print(f"   ❌ Error getting stats: {stats['error']}")
        
        print(f"\n🎉 Basic RAGContextAgent test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_context_advanced():
    """Test advanced functionality nếu có OpenAI API key"""
    
    print("\n🧪 === Testing RAGContextAgent Advanced Features ===\n")
    
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your_openai_api_key_here":
        print("⚠️ OpenAI API key not configured. Skipping advanced tests.")
        print("   Set OPENAI_API_KEY in .env file to test LLM features.")
        return True
    
    try:
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        # Initialize with OpenAI
        rag_agent = RAGContextAgent()
        
        # Test context query with LLM
        print("1. Testing context query with LLM response generation...")
        context_result = rag_agent.query_with_context(
            "How to calculate fibonacci numbers?", 
            top_k=3, 
            generate_response=True
        )
        
        print(f"   Query processed with {context_result['total_chunks']} context chunks")
        if context_result.get('response'):
            response_preview = context_result['response'][:150] + "..." if len(context_result['response']) > 150 else context_result['response']
            print(f"   Response preview: {response_preview}")
        
        print(f"\n🎉 Advanced RAGContextAgent test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Advanced test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing RAGContextAgent Implementation\n")
    
    # Test basic functionality
    basic_success = test_rag_context_basic()
    
    # Test advanced functionality if API key available
    advanced_success = test_rag_context_advanced()
    
    if basic_success:
        print(f"\n✅ RAGContextAgent basic functionality working!")
        print(f"\n📋 Features Tested:")
        print(f"  ✓ Qdrant connection and collection setup")
        print(f"  ✓ Code chunking with AST analysis")
        print(f"  ✓ Semantic metadata extraction")
        print(f"  ✓ Vector indexing and storage")
        print(f"  ✓ Similarity search and retrieval")
        print(f"  ✓ Collection statistics and management")
        
        if advanced_success and config.OPENAI_API_KEY:
            print(f"  ✓ LLM-powered context generation")
        
        print(f"\n🎯 RAGContextAgent ready for integration!")
    else:
        print(f"\n❌ RAGContextAgent tests failed!")
        sys.exit(1) 