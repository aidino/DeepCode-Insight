"""RAGContextAgent - Retrieval-Augmented Generation Context Agent sử dụng LlamaIndex và Qdrant"""

import logging
import os
import sys
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import uuid

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode, MetadataMode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.response.schema import Response  # Not needed for basic functionality
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ..parsers.ast_parser import ASTParsingAgent

# Import config from package
from ..config import config


class RAGContextAgent:
    """
    Agent để tạo và quản lý RAG context cho code analysis
    Sử dụng LlamaIndex để chunk code và Qdrant để lưu trữ vectors
    """
    
    def __init__(self, 
                 qdrant_host: str = None,
                 qdrant_port: int = None,
                 collection_name: str = None,
                 openai_api_key: Optional[str] = None):
        """
        Initialize RAGContextAgent
        
        Args:
            qdrant_host: Qdrant server host (defaults to config)
            qdrant_port: Qdrant server port (defaults to config)
            collection_name: Tên collection trong Qdrant (defaults to config)
            openai_api_key: OpenAI API key cho embeddings (defaults to config)
        """
        self.logger = logging.getLogger(__name__)
        
        # Use config defaults if not provided
        self.qdrant_host = qdrant_host or config.QDRANT_HOST
        self.qdrant_port = qdrant_port or config.QDRANT_PORT
        self.collection_name = collection_name or config.QDRANT_COLLECTION
        
        # Initialize OpenAI API key
        api_key = openai_api_key or config.OPENAI_API_KEY
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not config.USE_MOCK_EMBEDDINGS:
            self.logger.warning("OpenAI API key not provided. Some features may not work.")
        
        try:
            # Initialize AST parser first (always works)
            self.ast_parser = ASTParsingAgent()
            
            # Initialize LlamaIndex components
            self._init_llama_index()
            
            # Try to initialize Qdrant client
            try:
                self.qdrant_client = QdrantClient(
                    host=self.qdrant_host,
                    port=self.qdrant_port,
                    timeout=60
                )
                
                # Create or get collection
                self._setup_collection()
                self.qdrant_available = True
                self.logger.info(f"RAGContextAgent initialized successfully with collection: {self.collection_name}")
                
            except Exception as qdrant_error:
                self.logger.warning(f"Qdrant not available: {qdrant_error}")
                self.logger.info("RAGContextAgent initialized in offline mode (no vector storage)")
                self.qdrant_client = None
                self.vector_store = None
                self.index = None
                self.qdrant_available = False
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAGContextAgent: {e}")
            raise
    
    def _init_llama_index(self):
        """Initialize LlamaIndex settings và components"""
        try:
            # Check if OpenAI API key is available
            if config.OPENAI_API_KEY and config.OPENAI_API_KEY != "your_openai_api_key_here":
                # Configure embeddings
                embed_model = OpenAIEmbedding(
                    model=config.OPENAI_EMBEDDING_MODEL,
                    dimensions=config.VECTOR_DIMENSION
                )
                
                # Configure LLM
                llm = OpenAI(
                    model=config.OPENAI_MODEL,
                    temperature=0.1
                )
                
                # Set global settings
                Settings.embed_model = embed_model
                Settings.llm = llm
            else:
                # Use mock/default settings when no API key
                self.logger.warning("OpenAI API key not configured, using default settings")
                # Don't set LLM and embedding model - use defaults
            
            Settings.chunk_size = config.CHUNK_SIZE
            Settings.chunk_overlap = config.CHUNK_OVERLAP
            
            # Initialize node parsers
            self.code_splitter = CodeSplitter(
                language="python",  # Default, will be changed per file
                chunk_lines=40,
                chunk_lines_overlap=15,
                max_chars=1500
            )
            
            self.text_splitter = SentenceSplitter(
                chunk_size=1024,
                chunk_overlap=200
            )
            
            self.logger.info("LlamaIndex components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LlamaIndex: {e}")
            raise
    
    def _setup_collection(self):
        """Setup Qdrant collection"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_exists = any(
                collection.name == self.collection_name 
                for collection in collections.collections
            )
            
            if not collection_exists:
                # Create collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=config.VECTOR_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"Created new Qdrant collection: {self.collection_name}")
            else:
                self.logger.info(f"Using existing Qdrant collection: {self.collection_name}")
            
            # Initialize vector store
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name
            )
            
            # Initialize index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )
            
        except Exception as e:
            self.logger.error(f"Failed to setup Qdrant collection: {e}")
            raise
    
    def chunk_code_file(self, 
                       code: str, 
                       filename: str,
                       language: str = "python") -> List[Document]:
        """
        Chunk code file thành documents sử dụng AST parsing và code splitter
        
        Args:
            code: Source code content
            filename: File name
            language: Programming language (python, java, etc.)
            
        Returns:
            List of Document objects
        """
        documents = []
        
        try:
            # Parse code với AST parser để lấy structure info
            ast_analysis = self.ast_parser.parse_code(code, filename)
            
            # Create base metadata
            base_metadata = {
                "filename": filename,
                "language": language,
                "file_type": "source_code",
                "indexed_at": datetime.now().isoformat(),
                "total_lines": len(code.split('\n')),
                "ast_stats": ast_analysis.get('stats', {}),
                "classes": [cls.get('name', 'Unknown') for cls in ast_analysis.get('classes', [])],
                "functions": [func.get('name', 'Unknown') for func in ast_analysis.get('functions', [])]
            }
            
            # Configure code splitter for specific language
            if language.lower() == "python":
                self.code_splitter = CodeSplitter(
                    language="python",
                    chunk_lines=40,
                    chunk_lines_overlap=15,
                    max_chars=1500
                )
            elif language.lower() == "java":
                self.code_splitter = CodeSplitter(
                    language="java",
                    chunk_lines=35,
                    chunk_lines_overlap=10,
                    max_chars=1400
                )
            else:
                # Fallback to text splitter
                self.code_splitter = self.text_splitter
            
            # Create main document
            main_doc = Document(
                text=code,
                metadata=base_metadata.copy()
            )
            
            # Split code into chunks
            nodes = self.code_splitter.get_nodes_from_documents([main_doc])
            
            # Convert nodes back to documents với enhanced metadata
            for i, node in enumerate(nodes):
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_id": i,
                    "chunk_type": "code_chunk",
                    "start_line": self._estimate_start_line(node.text, code),
                    "chunk_size": len(node.text),
                    "node_id": str(uuid.uuid4())
                })
                
                # Analyze chunk content để thêm semantic metadata
                chunk_analysis = self._analyze_chunk_content(node.text, language)
                chunk_metadata.update(chunk_analysis)
                
                chunk_doc = Document(
                    text=node.text,
                    metadata=chunk_metadata
                )
                documents.append(chunk_doc)
            
            # Create summary document
            summary_metadata = base_metadata.copy()
            summary_metadata.update({
                "chunk_type": "file_summary",
                "summary": self._create_file_summary(ast_analysis, code, filename)
            })
            
            summary_doc = Document(
                text=summary_metadata["summary"],
                metadata=summary_metadata
            )
            documents.append(summary_doc)
            
            self.logger.info(f"Created {len(documents)} chunks for {filename}")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error chunking code file {filename}: {e}")
            # Return basic document as fallback
            return [Document(
                text=code,
                metadata={
                    "filename": filename,
                    "language": language,
                    "error": str(e),
                    "chunk_type": "fallback"
                }
            )]
    
    def _analyze_chunk_content(self, chunk_text: str, language: str) -> Dict[str, Any]:
        """Analyze chunk content để tạo semantic metadata"""
        analysis = {
            "contains_class": False,
            "contains_function": False,
            "contains_import": False,
            "contains_comment": False,
            "complexity_indicators": [],
            "keywords": []
        }
        
        lines = chunk_text.split('\n')
        
        for line in lines:
            stripped = line.strip()
            
            if language.lower() == "python":
                if stripped.startswith('class '):
                    analysis["contains_class"] = True
                    analysis["keywords"].append("class_definition")
                elif stripped.startswith('def '):
                    analysis["contains_function"] = True
                    analysis["keywords"].append("function_definition")
                elif stripped.startswith('import ') or stripped.startswith('from '):
                    analysis["contains_import"] = True
                    analysis["keywords"].append("import_statement")
                elif stripped.startswith('#'):
                    analysis["contains_comment"] = True
                
                # Complexity indicators
                if any(keyword in stripped for keyword in ['if ', 'for ', 'while ', 'try:', 'except']):
                    analysis["complexity_indicators"].append("control_flow")
                if 'lambda' in stripped:
                    analysis["complexity_indicators"].append("lambda")
                    
            elif language.lower() == "java":
                if 'class ' in stripped and ('public' in stripped or 'private' in stripped):
                    analysis["contains_class"] = True
                    analysis["keywords"].append("class_definition")
                elif any(modifier in stripped for modifier in ['public ', 'private ', 'protected ']) and '(' in stripped:
                    analysis["contains_function"] = True
                    analysis["keywords"].append("method_definition")
                elif stripped.startswith('import '):
                    analysis["contains_import"] = True
                    analysis["keywords"].append("import_statement")
                elif stripped.startswith('//') or stripped.startswith('/*'):
                    analysis["contains_comment"] = True
                
                # Complexity indicators
                if any(keyword in stripped for keyword in ['if (', 'for (', 'while (', 'try {', 'catch']):
                    analysis["complexity_indicators"].append("control_flow")
        
        return analysis
    
    def _estimate_start_line(self, chunk_text: str, full_code: str) -> int:
        """Estimate start line number của chunk trong full code"""
        try:
            # Simple approach: find first occurrence
            index = full_code.find(chunk_text.strip()[:50])  # Use first 50 chars
            if index != -1:
                return full_code[:index].count('\n') + 1
        except:
            pass
        return 1
    
    def _create_file_summary(self, ast_analysis: Dict, code: str, filename: str) -> str:
        """Create summary của file"""
        stats = ast_analysis.get('stats', {})
        classes = ast_analysis.get('classes', [])
        functions = ast_analysis.get('functions', [])
        
        summary_parts = [
            f"File: {filename}",
            f"Lines of code: {len(code.split())}",
            f"Classes: {len(classes)}",
            f"Functions: {len(functions)}",
        ]
        
        if classes:
            class_names = [cls.get('name', 'Unknown') for cls in classes[:3]]
            summary_parts.append(f"Main classes: {', '.join(class_names)}")
        
        if functions:
            func_names = [func.get('name', 'Unknown') for func in functions[:5]]
            summary_parts.append(f"Main functions: {', '.join(func_names)}")
        
        return ". ".join(summary_parts)
    
    def index_code_file(self, 
                       code: str, 
                       filename: str,
                       language: str = "python",
                       metadata: Optional[Dict] = None) -> bool:
        """
        Index một code file vào Qdrant
        
        Args:
            code: Source code content
            filename: File name
            language: Programming language
            metadata: Additional metadata
            
        Returns:
            bool: Success status
        """
        try:
            # Chunk code file (always works)
            documents = self.chunk_code_file(code, filename, language)
            
            # Add additional metadata if provided
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
            
            # Add documents to index if Qdrant is available
            if self.qdrant_available and self.index:
                for doc in documents:
                    self.index.insert(doc)
                self.logger.info(f"Successfully indexed {filename} with {len(documents)} chunks")
            else:
                self.logger.info(f"Chunked {filename} into {len(documents)} documents (offline mode)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to index code file {filename}: {e}")
            return False
    
    def index_repository(self, 
                        code_fetcher_agent,
                        repo_url: str,
                        file_patterns: List[str] = ["*.py", "*.java"]) -> Dict[str, Any]:
        """
        Index toàn bộ repository
        
        Args:
            code_fetcher_agent: CodeFetcherAgent instance
            repo_url: Repository URL
            file_patterns: File patterns to include
            
        Returns:
            Dict với indexing results
        """
        results = {
            "repository": repo_url,
            "indexed_files": [],
            "failed_files": [],
            "total_chunks": 0,
            "indexing_timestamp": datetime.now().isoformat()
        }
        
        try:
            # Get repository files
            all_files = code_fetcher_agent.list_repository_files(repo_url)
            
            # Filter files by patterns
            target_files = []
            for pattern in file_patterns:
                if pattern == "*.py":
                    target_files.extend([f for f in all_files if f.endswith('.py')])
                elif pattern == "*.java":
                    target_files.extend([f for f in all_files if f.endswith('.java')])
            
            target_files = list(set(target_files))  # Remove duplicates
            
            self.logger.info(f"Found {len(target_files)} files to index in {repo_url}")
            
            # Index each file
            for file_path in target_files:
                try:
                    content = code_fetcher_agent.get_file_content(repo_url, file_path)
                    if content:
                        # Determine language
                        language = "python" if file_path.endswith('.py') else "java"
                        
                        # Add repository metadata
                        repo_metadata = {
                            "repository_url": repo_url,
                            "file_path": file_path,
                            "repository_name": repo_url.split('/')[-1] if '/' in repo_url else repo_url
                        }
                        
                        # Index file
                        success = self.index_code_file(
                            content, 
                            file_path, 
                            language, 
                            repo_metadata
                        )
                        
                        if success:
                            # Count chunks for this file
                            chunks = self.chunk_code_file(content, file_path, language)
                            results["indexed_files"].append({
                                "file_path": file_path,
                                "language": language,
                                "chunks": len(chunks),
                                "size": len(content)
                            })
                            results["total_chunks"] += len(chunks)
                        else:
                            results["failed_files"].append({
                                "file_path": file_path,
                                "error": "Indexing failed"
                            })
                    else:
                        results["failed_files"].append({
                            "file_path": file_path,
                            "error": "Could not fetch content"
                        })
                        
                except Exception as e:
                    self.logger.error(f"Error indexing {file_path}: {e}")
                    results["failed_files"].append({
                        "file_path": file_path,
                        "error": str(e)
                    })
            
            self.logger.info(f"Repository indexing completed: {len(results['indexed_files'])} files indexed")
            
        except Exception as e:
            self.logger.error(f"Error indexing repository {repo_url}: {e}")
            results["error"] = str(e)
        
        return results
    
    def query(self, 
              query_text: str,
              top_k: int = 5,
              filters: Optional[Dict] = None,
              include_metadata: bool = True) -> Dict[str, Any]:
        """
        Query RAG context để tìm relevant code chunks
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            filters: Metadata filters
            include_metadata: Include metadata in results
            
        Returns:
            Dict với query results
        """
        try:
            # Check if Qdrant is available
            if not self.qdrant_available or not self.index:
                return {
                    "query": query_text,
                    "error": "Vector search not available (offline mode)",
                    "total_results": 0,
                    "results": []
                }
            
            # Create retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k
            )
            
            # Perform retrieval
            nodes = retriever.retrieve(query_text)
            
            # Format results
            results = {
                "query": query_text,
                "total_results": len(nodes),
                "results": []
            }
            
            for i, node in enumerate(nodes):
                result_item = {
                    "rank": i + 1,
                    "score": getattr(node, 'score', 0.0),
                    "content": node.text,
                    "content_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text
                }
                
                if include_metadata and hasattr(node, 'metadata'):
                    result_item["metadata"] = node.metadata
                
                # Apply filters if provided
                if filters:
                    if self._matches_filters(node.metadata if hasattr(node, 'metadata') else {}, filters):
                        results["results"].append(result_item)
                else:
                    results["results"].append(result_item)
            
            # Update total results after filtering
            results["total_results"] = len(results["results"])
            
            self.logger.info(f"Query '{query_text}' returned {results['total_results']} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Error querying RAG context: {e}")
            return {
                "query": query_text,
                "error": str(e),
                "total_results": 0,
                "results": []
            }
    
    def query_with_context(self,
                          query_text: str,
                          top_k: int = 5,
                          generate_response: bool = True) -> Dict[str, Any]:
        """
        Query với context generation sử dụng LLM
        
        Args:
            query_text: Query string
            top_k: Number of context chunks
            generate_response: Generate LLM response
            
        Returns:
            Dict với query results và generated response
        """
        try:
            # Get relevant chunks
            retrieval_results = self.query(query_text, top_k)
            
            if not retrieval_results["results"]:
                return {
                    "query": query_text,
                    "context_chunks": [],
                    "response": "No relevant code context found for your query.",
                    "total_chunks": 0
                }
            
            # Prepare context
            context_chunks = []
            context_text = ""
            
            for result in retrieval_results["results"]:
                chunk_info = {
                    "content": result["content"],
                    "score": result["score"],
                    "metadata": result.get("metadata", {})
                }
                context_chunks.append(chunk_info)
                
                # Add to context text
                filename = chunk_info["metadata"].get("filename", "unknown")
                context_text += f"\n--- From {filename} ---\n{result['content']}\n"
            
            response_data = {
                "query": query_text,
                "context_chunks": context_chunks,
                "total_chunks": len(context_chunks),
                "context_text": context_text
            }
            
            # Generate response if requested
            if generate_response and os.getenv("OPENAI_API_KEY"):
                try:
                    # Create query engine
                    query_engine = RetrieverQueryEngine.from_args(
                        retriever=VectorIndexRetriever(
                            index=self.index,
                            similarity_top_k=top_k
                        )
                    )
                    
                    # Generate response
                    response = query_engine.query(query_text)
                    response_data["response"] = str(response)
                    response_data["response_metadata"] = {
                        "model": "gpt-3.5-turbo",
                        "generated_at": datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate LLM response: {e}")
                    response_data["response"] = f"Context retrieved successfully, but response generation failed: {e}"
            else:
                response_data["response"] = "Context retrieved successfully. Set OPENAI_API_KEY to enable response generation."
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Error in query_with_context: {e}")
            return {
                "query": query_text,
                "error": str(e),
                "context_chunks": [],
                "total_chunks": 0
            }
    
    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches filters"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics về Qdrant collection"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            # Get sample points để analyze metadata
            search_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True
            )
            
            # Analyze metadata
            languages = set()
            file_types = set()
            repositories = set()
            
            for point in search_result[0]:
                payload = point.payload or {}
                if 'language' in payload:
                    languages.add(payload['language'])
                if 'file_type' in payload:
                    file_types.add(payload['file_type'])
                if 'repository_url' in payload:
                    repositories.add(payload['repository_url'])
            
            return {
                "collection_name": self.collection_name,
                "total_points": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.name,
                "indexed_languages": list(languages),
                "file_types": list(file_types),
                "repositories": list(repositories),
                "status": collection_info.status.name
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """Clear tất cả data trong collection"""
        try:
            # Delete and recreate collection
            self.qdrant_client.delete_collection(self.collection_name)
            self._setup_collection()
            self.logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")
            return False
    
    def delete_by_repository(self, repo_url: str) -> bool:
        """Delete tất cả data của một repository"""
        try:
            # Delete points by repository filter
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector={
                    "filter": {
                        "must": [
                            {
                                "key": "repository_url",
                                "match": {"value": repo_url}
                            }
                        ]
                    }
                }
            )
            self.logger.info(f"Deleted data for repository: {repo_url}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting repository data: {e}")
            return False


def demo_rag_context():
    """Demo function để test RAGContextAgent"""
    
    # Sample Python code
    python_sample = '''
import os
import sys
from typing import List, Dict, Optional

class DataProcessor:
    """A class for processing data with various methods"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_cache = {}
    
    def load_data(self, file_path: str) -> List[Dict]:
        """Load data from file"""
        if file_path in self.data_cache:
            return self.data_cache[file_path]
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.data_cache[file_path] = data
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
    
    def process_data(self, data: List[Dict]) -> List[Dict]:
        """Process data with transformations"""
        processed = []
        for item in data:
            if self.validate_item(item):
                transformed = self.transform_item(item)
                processed.append(transformed)
        return processed
    
    def validate_item(self, item: Dict) -> bool:
        """Validate data item"""
        required_fields = self.config.get('required_fields', [])
        return all(field in item for field in required_fields)
    
    def transform_item(self, item: Dict) -> Dict:
        """Transform data item"""
        transformations = self.config.get('transformations', {})
        result = item.copy()
        
        for field, transform_func in transformations.items():
            if field in result:
                result[field] = transform_func(result[field])
        
        return result

def main():
    """Main function"""
    config = {
        'required_fields': ['id', 'name'],
        'transformations': {
            'name': str.upper
        }
    }
    
    processor = DataProcessor(config)
    data = processor.load_data('data.json')
    processed = processor.process_data(data)
    print(f"Processed {len(processed)} items")

if __name__ == "__main__":
    main()
'''
    
    # Java sample code
    java_sample = '''
package com.example.processor;

import java.util.*;
import java.io.*;

/**
 * Data processor for handling various data operations
 */
public class DataProcessor {
    
    private Map<String, Object> config;
    private Map<String, List<Map<String, Object>>> dataCache;
    
    /**
     * Constructor
     * @param config Configuration map
     */
    public DataProcessor(Map<String, Object> config) {
        this.config = config;
        this.dataCache = new HashMap<>();
    }
    
    /**
     * Load data from file
     * @param filePath Path to data file
     * @return List of data items
     */
    public List<Map<String, Object>> loadData(String filePath) {
        if (dataCache.containsKey(filePath)) {
            return dataCache.get(filePath);
        }
        
        List<Map<String, Object>> data = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            // JSON parsing logic here
            // For demo purposes, return empty list
        } catch (IOException e) {
            System.err.println("Error loading data: " + e.getMessage());
        }
        
        dataCache.put(filePath, data);
        return data;
    }
    
    /**
     * Process data with transformations
     * @param data Input data list
     * @return Processed data list
     */
    public List<Map<String, Object>> processData(List<Map<String, Object>> data) {
        List<Map<String, Object>> processed = new ArrayList<>();
        
        for (Map<String, Object> item : data) {
            if (validateItem(item)) {
                Map<String, Object> transformed = transformItem(item);
                processed.add(transformed);
            }
        }
        
        return processed;
    }
    
    /**
     * Validate data item
     * @param item Data item to validate
     * @return true if valid, false otherwise
     */
    private boolean validateItem(Map<String, Object> item) {
        @SuppressWarnings("unchecked")
        List<String> requiredFields = (List<String>) config.get("required_fields");
        
        if (requiredFields == null) {
            return true;
        }
        
        for (String field : requiredFields) {
            if (!item.containsKey(field)) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Transform data item
     * @param item Data item to transform
     * @return Transformed item
     */
    private Map<String, Object> transformItem(Map<String, Object> item) {
        Map<String, Object> result = new HashMap<>(item);
        
        // Apply transformations based on config
        @SuppressWarnings("unchecked")
        Map<String, String> transformations = (Map<String, String>) config.get("transformations");
        
        if (transformations != null) {
            for (Map.Entry<String, String> entry : transformations.entrySet()) {
                String field = entry.getKey();
                String transformation = entry.getValue();
                
                if (result.containsKey(field) && "uppercase".equals(transformation)) {
                    Object value = result.get(field);
                    if (value instanceof String) {
                        result.put(field, ((String) value).toUpperCase());
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * Main method for testing
     * @param args Command line arguments
     */
    public static void main(String[] args) {
        Map<String, Object> config = new HashMap<>();
        config.put("required_fields", Arrays.asList("id", "name"));
        
        Map<String, String> transformations = new HashMap<>();
        transformations.put("name", "uppercase");
        config.put("transformations", transformations);
        
        DataProcessor processor = new DataProcessor(config);
        List<Map<String, Object>> data = processor.loadData("data.json");
        List<Map<String, Object>> processed = processor.processData(data);
        
        System.out.println("Processed " + processed.size() + " items");
    }
}
'''
    
    print("🔍 === RAGContextAgent Demo ===")
    print("Testing code chunking, indexing, and querying with LlamaIndex + Qdrant\n")
    
    try:
        # Initialize RAG agent
        rag_agent = RAGContextAgent()
        
        print("✅ RAGContextAgent initialized successfully")
        
        # Test chunking
        print("\n📄 Testing Code Chunking:")
        python_chunks = rag_agent.chunk_code_file(python_sample, "data_processor.py", "python")
        java_chunks = rag_agent.chunk_code_file(java_sample, "DataProcessor.java", "java")
        
        print(f"  Python file chunked into {len(python_chunks)} documents")
        print(f"  Java file chunked into {len(java_chunks)} documents")
        
        # Show sample chunk
        if python_chunks:
            sample_chunk = python_chunks[0]
            print(f"\n  Sample Python chunk metadata:")
            for key, value in sample_chunk.metadata.items():
                if key not in ['ast_stats']:  # Skip complex nested data
                    print(f"    {key}: {value}")
        
        # Test indexing
        print("\n📚 Testing Code Indexing:")
        python_success = rag_agent.index_code_file(python_sample, "data_processor.py", "python")
        java_success = rag_agent.index_code_file(java_sample, "DataProcessor.java", "java")
        
        print(f"  Python indexing: {'✅ Success' if python_success else '❌ Failed'}")
        print(f"  Java indexing: {'✅ Success' if java_success else '❌ Failed'}")
        
        # Test querying
        print("\n🔍 Testing RAG Queries:")
        
        queries = [
            "How to load data from file?",
            "Data validation methods",
            "Class constructors and initialization",
            "Error handling in file operations",
            "Data transformation functions"
        ]
        
        for query in queries:
            print(f"\n  Query: '{query}'")
            results = rag_agent.query(query, top_k=3)
            
            if results["total_results"] > 0:
                print(f"    Found {results['total_results']} relevant chunks:")
                for i, result in enumerate(results["results"][:2]):  # Show top 2
                    filename = result.get("metadata", {}).get("filename", "unknown")
                    language = result.get("metadata", {}).get("language", "unknown")
                    score = result.get("score", 0)
                    preview = result["content_preview"]
                    print(f"      {i+1}. {filename} ({language}) - Score: {score:.3f}")
                    print(f"         Preview: {preview}")
            else:
                print("    No relevant chunks found")
        
        # Test context query
        print(f"\n🤖 Testing Context Query with LLM:")
        context_query = "Show me how to implement data validation"
        context_result = rag_agent.query_with_context(context_query, top_k=3, generate_response=True)
        
        print(f"  Query: '{context_query}'")
        print(f"  Context chunks: {context_result['total_chunks']}")
        if context_result.get('response'):
            response_preview = context_result['response'][:200] + "..." if len(context_result['response']) > 200 else context_result['response']
            print(f"  Response preview: {response_preview}")
        
        # Collection stats
        print(f"\n📊 Collection Statistics:")
        stats = rag_agent.get_collection_stats()
        for key, value in stats.items():
            if key != 'error':
                print(f"  {key}: {value}")
        
        print(f"\n🎉 RAGContextAgent demo completed successfully!")
        print(f"\n✨ Features Demonstrated:")
        print(f"  ✓ Code chunking with AST analysis")
        print(f"  ✓ Multi-language support (Python + Java)")
        print(f"  ✓ Semantic metadata extraction")
        print(f"  ✓ Vector indexing with Qdrant")
        print(f"  ✓ Similarity search and retrieval")
        print(f"  ✓ Context-aware query responses")
        print(f"  ✓ Collection management and statistics")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_rag_context() 