#!/usr/bin/env python3
"""
Final verification test cho DeepCode-Insight sau refactoring
"""

def main():
    print('🧪 Final Verification Test')
    print('=' * 50)

    # Test 1: Config import
    try:
        from deepcode_insight.config import config
        print('✅ Config import: OK')
    except Exception as e:
        print(f'❌ Config import: {e}')
        return False

    # Test 2: Core agents import
    try:
        from deepcode_insight.agents import RAGContextAgent, StaticAnalysisAgent
        print('✅ Core agents import: OK')
    except Exception as e:
        print(f'❌ Core agents import: {e}')
        return False

    # Test 3: RAG initialization
    try:
        from deepcode_insight.agents.rag_context import RAGContextAgent
        agent = RAGContextAgent()
        print('✅ RAG initialization: OK')
    except Exception as e:
        print(f'❌ RAG initialization: {e}')
        return False

    # Test 4: Code chunking
    try:
        code = 'def hello(): return "Hello World"'
        chunks = agent.chunk_code_file(code, 'test.py', 'python')
        print(f'✅ Code chunking: OK ({len(chunks)} chunks)')
    except Exception as e:
        print(f'❌ Code chunking: {e}')
        return False

    # Test 5: Indexing
    try:
        result = agent.index_code_file(code, 'test.py', 'python')
        print(f'✅ Code indexing: OK ({result})')
    except Exception as e:
        print(f'❌ Code indexing: {e}')
        return False

    # Test 6: Query (offline mode)
    try:
        query_result = agent.query("hello function")
        print(f'✅ Query execution: OK (offline mode)')
    except Exception as e:
        print(f'❌ Query execution: {e}')
        return False

    print('\n🎉 All verification tests passed!')
    print('\n📋 Summary:')
    print('  ✓ Configuration system working')
    print('  ✓ Agent imports working')
    print('  ✓ RAG initialization working')
    print('  ✓ Code chunking working')
    print('  ✓ Code indexing working')
    print('  ✓ Query system working')
    print('\n🚀 DeepCode-Insight refactoring completed successfully!')
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 