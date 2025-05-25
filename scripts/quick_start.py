#!/usr/bin/env python3
"""Quick start demo cho DeepCode-Insight RAGContextAgent"""

import sys
import os
from pathlib import Path

# Add paths
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(project_root, 'deepcode_insight'))
sys.path.append(project_root)

def main():
    """Quick start demo"""
    
    print("🚀 DeepCode-Insight Quick Start Demo")
    print("====================================\n")
    
    try:
        # Import config
        from deepcode_insight.config import config
        
        print("1. 🔧 Configuration Status:")
        config.print_config()
        
        # Check if we have API key
        has_openai = config.OPENAI_API_KEY and config.OPENAI_API_KEY != "your_openai_api_key_here"
        
        if not has_openai:
            print("\n⚠️ OpenAI API key not configured.")
            print("   📝 To add your API key:")
            print("   1. Edit file .env")
            print("   2. Replace 'your_openai_api_key_here' with your actual key")
            print("   3. See setup_api_keys.md for detailed instructions")
            print("\n   Running demo with mock embeddings for now...")
            
            # Run mock demo
            print("\n2. 🧪 Running Mock Demo:")
            sys.path.append(os.path.join(project_root, 'tests'))
            from demo_rag_context import demo_rag_context
            demo_rag_context()
            
        else:
            print("\n✅ OpenAI API key configured!")
            print("   Running real data demo...")
            
            # Check Qdrant
            try:
                from qdrant_client import QdrantClient
                client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
                collections = client.get_collections()
                print(f"   ✅ Qdrant connected ({len(collections.collections)} collections)")
                
                # Run real demo
                print("\n2. 🧪 Running Real Data Demo:")
                sys.path.append(os.path.join(project_root, 'tests'))
                from test_rag_real_data import test_real_rag_context
                test_real_rag_context()
                
            except Exception as e:
                print(f"   ❌ Qdrant connection failed: {e}")
                print("   Please start Qdrant: docker compose up -d")
                return False
        
        print("\n🎉 Quick Start Demo Completed!")
        print("\n📚 Next Steps:")
        if not has_openai:
            print("1. 🔑 Setup API Keys:")
            print("   - Edit .env file to add your OpenAI API key")
            print("   - See setup_api_keys.md for detailed instructions")
            print("   - Run 'python config.py' to verify")
            print("\n2. 🧪 Run Real Data Tests:")
            print("   python test_rag_real_data.py")
        else:
            print("1. 📚 Index your code repositories")
            print("2. 🔍 Use semantic search on your codebase")
            print("3. 🚀 Build RAG-powered applications")
        
        print("\n📖 Documentation:")
        print("   - docs/setup_api_keys.md - API keys setup guide")
        print("   - docs/README_SETUP.md - Complete setup guide")
        print("   - docs/README_RAG.md - RAG usage guide")
        print("   - docs/ENVIRONMENT_SETUP_SUMMARY.md - Technical details")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick start failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 