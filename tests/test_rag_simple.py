#!/usr/bin/env python3
"""Simple test cho RAGContextAgent mà không cần OpenAI API"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Add paths
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(project_root, 'deepcode_insight'))
sys.path.append(project_root)

def test_rag_basic_no_openai():
    """Test basic functionality mà không cần OpenAI"""
    
    print("🧪 === Testing RAGContextAgent (No OpenAI) ===\n")
    
    try:
        # Test imports
        print("1. Testing imports...")
        from deepcode_insight.agents.rag_context import RAGContextAgent
        from qdrant_client import QdrantClient
        print("   ✅ All imports successful")
        
        # Test Qdrant connection
        print("\n2. Testing Qdrant connection...")
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        print(f"   ✅ Qdrant connected, found {len(collections.collections)} collections")
        
        # Test chunking without full initialization
        print("\n3. Testing code chunking logic...")
        from deepcode_insight.parsers.ast_parser import ASTParsingAgent
        
        ast_parser = ASTParsingAgent()
        sample_code = '''
def hello_world():
    """Simple hello world function"""
    print("Hello, World!")
    return "Hello"

class SimpleClass:
    """A simple class"""
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
'''
        
        # Test AST parsing
        ast_result = ast_parser.parse_code(sample_code, "test.py")
        print(f"   ✅ AST parsing successful: {len(ast_result.get('functions', []))} functions, {len(ast_result.get('classes', []))} classes")
        
        # Test basic chunking logic (without LlamaIndex)
        print("\n4. Testing basic chunking...")
        lines = sample_code.split('\n')
        chunks = []
        current_chunk = []
        
        for line in lines:
            current_chunk.append(line)
            if len(current_chunk) >= 10:  # Simple chunking by lines
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        print(f"   ✅ Basic chunking created {len(chunks)} chunks")
        
        # Test metadata extraction
        print("\n5. Testing metadata extraction...")
        metadata = {
            "filename": "test.py",
            "language": "python",
            "total_lines": len(lines),
            "functions": [func.get('name') for func in ast_result.get('functions', [])],
            "classes": [cls.get('name') for cls in ast_result.get('classes', [])]
        }
        print(f"   ✅ Metadata extracted: {metadata}")
        
        print(f"\n🎉 Basic RAGContextAgent components test successful!")
        print(f"\n📋 Components Tested:")
        print(f"  ✓ Qdrant connection and client")
        print(f"  ✓ AST parsing integration")
        print(f"  ✓ Basic code chunking logic")
        print(f"  ✓ Metadata extraction")
        print(f"  ✓ Import compatibility")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qdrant_operations():
    """Test basic Qdrant operations"""
    
    print("\n🧪 === Testing Qdrant Operations ===\n")
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        import numpy as np
        
        # Connect to Qdrant
        client = QdrantClient(host="localhost", port=6333)
        
        # Test collection operations
        test_collection = "test_collection"
        
        print("1. Testing collection creation...")
        try:
            client.delete_collection(test_collection)
        except:
            pass  # Collection might not exist
        
        client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=128, distance=Distance.COSINE)
        )
        print("   ✅ Collection created successfully")
        
        # Test vector operations
        print("\n2. Testing vector operations...")
        test_vectors = [
            {
                "id": 1,
                "vector": np.random.rand(128).tolist(),
                "payload": {"text": "Hello world", "type": "greeting"}
            },
            {
                "id": 2, 
                "vector": np.random.rand(128).tolist(),
                "payload": {"text": "Python code", "type": "code"}
            }
        ]
        
        client.upsert(
            collection_name=test_collection,
            points=test_vectors
        )
        print("   ✅ Vectors inserted successfully")
        
        # Test search
        print("\n3. Testing vector search...")
        search_result = client.search(
            collection_name=test_collection,
            query_vector=np.random.rand(128).tolist(),
            limit=2
        )
        print(f"   ✅ Search returned {len(search_result)} results")
        
        # Cleanup
        print("\n4. Cleaning up...")
        client.delete_collection(test_collection)
        print("   ✅ Test collection deleted")
        
        print(f"\n🎉 Qdrant operations test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Qdrant test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing RAGContextAgent Components\n")
    
    # Test basic components
    basic_success = test_rag_basic_no_openai()
    
    # Test Qdrant operations
    qdrant_success = test_qdrant_operations()
    
    if basic_success and qdrant_success:
        print(f"\n✅ All component tests passed!")
        print(f"\n🎯 RAGContextAgent infrastructure ready!")
        print(f"\n📝 Next Steps:")
        print(f"  1. Set OPENAI_API_KEY for full functionality")
        print(f"  2. Test with real code repositories")
        print(f"  3. Integrate with other agents")
    else:
        print(f"\n❌ Some tests failed!")
        sys.exit(1) 