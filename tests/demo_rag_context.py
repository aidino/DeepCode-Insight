#!/usr/bin/env python3
"""Demo script cho RAGContextAgent với mock embeddings"""

import sys
import os
import logging
import numpy as np
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)

# Add path
sys.path.append(os.path.join(os.path.dirname(__file__), 'deepcode_insight'))

class MockRAGContextAgent:
    """Mock version của RAGContextAgent để demo mà không cần OpenAI API"""
    
    def __init__(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from deepcode_insight.parsers.ast_parser import ASTParsingAgent
        
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.ast_parser = ASTParsingAgent()
        self.collection_name = "demo_deepcode_context"
        
        # Setup collection
        self._setup_collection()
        
        # Mock embeddings cache
        self.embeddings_cache = {}
        
    def _setup_collection(self):
        """Setup demo collection"""
        from qdrant_client.models import Distance, VectorParams
        
        try:
            self.qdrant_client.delete_collection(self.collection_name)
        except:
            pass
        
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # Smaller dimension for demo
        )
        print(f"✅ Created demo collection: {self.collection_name}")
    
    def _mock_embedding(self, text: str) -> List[float]:
        """Generate mock embedding based on text content"""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        # Simple mock: use hash of text to generate consistent embeddings
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        np.random.seed(seed)
        
        # Generate embedding with some semantic meaning
        embedding = np.random.rand(384).astype(float)
        
        # Add some semantic features based on content
        if 'class' in text.lower():
            embedding[0:10] += 0.5  # Class indicator
        if 'def ' in text:
            embedding[10:20] += 0.5  # Function indicator
        if 'import' in text:
            embedding[20:30] += 0.5  # Import indicator
        if any(keyword in text.lower() for keyword in ['error', 'exception', 'try', 'catch']):
            embedding[30:40] += 0.5  # Error handling indicator
        if any(keyword in text.lower() for keyword in ['data', 'process', 'transform']):
            embedding[40:50] += 0.5  # Data processing indicator
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        self.embeddings_cache[text] = embedding.tolist()
        return embedding.tolist()
    
    def chunk_and_index_code(self, code: str, filename: str, language: str = "python") -> Dict[str, Any]:
        """Chunk và index code với mock embeddings"""
        
        # Parse với AST
        ast_analysis = self.ast_parser.parse_code(code, filename)
        
        # Simple chunking by functions and classes
        chunks = []
        lines = code.split('\n')
        current_chunk = []
        chunk_id = 0
        
        for i, line in enumerate(lines):
            current_chunk.append(line)
            
            # End chunk at function/class definitions or every 20 lines
            if (line.strip().startswith('def ') or 
                line.strip().startswith('class ') or 
                len(current_chunk) >= 20):
                
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    if chunk_text.strip():  # Only non-empty chunks
                        
                        # Create metadata
                        metadata = {
                            "filename": filename,
                            "language": language,
                            "chunk_id": chunk_id,
                            "start_line": i - len(current_chunk) + 1,
                            "end_line": i,
                            "chunk_size": len(chunk_text),
                            "contains_class": 'class ' in chunk_text,
                            "contains_function": 'def ' in chunk_text,
                            "contains_import": 'import ' in chunk_text,
                        }
                        
                        # Generate mock embedding
                        embedding = self._mock_embedding(chunk_text)
                        
                        # Store in Qdrant
                        self.qdrant_client.upsert(
                            collection_name=self.collection_name,
                            points=[{
                                "id": chunk_id,
                                "vector": embedding,
                                "payload": {
                                    "text": chunk_text,
                                    **metadata
                                }
                            }]
                        )
                        
                        chunks.append({
                            "id": chunk_id,
                            "text": chunk_text,
                            "metadata": metadata
                        })
                        
                        chunk_id += 1
                        current_chunk = []
        
        # Handle remaining lines
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if chunk_text.strip():
                metadata = {
                    "filename": filename,
                    "language": language,
                    "chunk_id": chunk_id,
                    "start_line": len(lines) - len(current_chunk),
                    "end_line": len(lines),
                    "chunk_size": len(chunk_text),
                    "contains_class": 'class ' in chunk_text,
                    "contains_function": 'def ' in chunk_text,
                    "contains_import": 'import ' in chunk_text,
                }
                
                embedding = self._mock_embedding(chunk_text)
                
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[{
                        "id": chunk_id,
                        "vector": embedding,
                        "payload": {
                            "text": chunk_text,
                            **metadata
                        }
                    }]
                )
                
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": metadata
                })
        
        return {
            "filename": filename,
            "language": language,
            "total_chunks": len(chunks),
            "chunks": chunks,
            "ast_analysis": ast_analysis
        }
    
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Query với mock embeddings"""
        
        # Generate query embedding
        query_embedding = self._mock_embedding(query_text)
        
        # Search in Qdrant
        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        # Format results
        results = []
        for result in search_results.points:
            results.append({
                "id": result.id,
                "score": result.score,
                "text": result.payload["text"],
                "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                "preview": result.payload["text"][:200] + "..." if len(result.payload["text"]) > 200 else result.payload["text"]
            })
        
        return {
            "query": query_text,
            "total_results": len(results),
            "results": results
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        
        # Get sample points for analysis
        sample_points = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            limit=100,
            with_payload=True
        )
        
        # Analyze content
        languages = set()
        filenames = set()
        total_functions = 0
        total_classes = 0
        
        for point in sample_points[0]:
            payload = point.payload
            if 'language' in payload:
                languages.add(payload['language'])
            if 'filename' in payload:
                filenames.add(payload['filename'])
            if payload.get('contains_function'):
                total_functions += 1
            if payload.get('contains_class'):
                total_classes += 1
        
        return {
            "collection_name": self.collection_name,
            "total_points": collection_info.points_count,
            "languages": list(languages),
            "files": list(filenames),
            "functions_chunks": total_functions,
            "class_chunks": total_classes,
            "vector_size": collection_info.config.params.vectors.size
        }

def demo_rag_context():
    """Demo RAGContextAgent functionality"""
    
    print("🔍 === RAGContextAgent Demo (Mock Embeddings) ===\n")
    
    # Sample code files
    python_sample = '''
import os
import json
from typing import List, Dict, Optional

class DataProcessor:
    """A comprehensive data processing class"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cache = {}
        self.processed_count = 0
    
    def load_data(self, file_path: str) -> List[Dict]:
        """Load data from JSON file with caching"""
        if file_path in self.cache:
            return self.cache[file_path]
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.cache[file_path] = data
            return data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return []
    
    def validate_item(self, item: Dict) -> bool:
        """Validate a single data item"""
        required_fields = self.config.get('required_fields', [])
        return all(field in item for field in required_fields)
    
    def transform_item(self, item: Dict) -> Dict:
        """Apply transformations to data item"""
        result = item.copy()
        transformations = self.config.get('transformations', {})
        
        for field, transform_type in transformations.items():
            if field in result:
                if transform_type == 'uppercase':
                    result[field] = str(result[field]).upper()
                elif transform_type == 'lowercase':
                    result[field] = str(result[field]).lower()
        
        return result
    
    def process_batch(self, data: List[Dict]) -> List[Dict]:
        """Process a batch of data items"""
        processed = []
        
        for item in data:
            try:
                if self.validate_item(item):
                    transformed = self.transform_item(item)
                    processed.append(transformed)
                    self.processed_count += 1
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
        
        return processed
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        return {
            'processed_count': self.processed_count,
            'cache_size': len(self.cache),
            'config': self.config
        }

def create_processor(config_file: str) -> DataProcessor:
    """Factory function to create DataProcessor"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return DataProcessor(config)

def main():
    """Main processing function"""
    config = {
        'required_fields': ['id', 'name', 'type'],
        'transformations': {
            'name': 'uppercase',
            'type': 'lowercase'
        }
    }
    
    processor = DataProcessor(config)
    data = processor.load_data('sample_data.json')
    processed = processor.process_batch(data)
    stats = processor.get_statistics()
    
    print(f"Processed {len(processed)} items")
    print(f"Statistics: {stats}")

if __name__ == "__main__":
    main()
'''
    
    java_sample = '''
package com.example.utils;

import java.util.*;
import java.io.*;
import java.nio.file.*;

/**
 * Utility class for file operations and data processing
 */
public class FileProcessor {
    
    private Map<String, Object> config;
    private List<String> processedFiles;
    private int errorCount;
    
    /**
     * Constructor with configuration
     * @param config Configuration map
     */
    public FileProcessor(Map<String, Object> config) {
        this.config = config;
        this.processedFiles = new ArrayList<>();
        this.errorCount = 0;
    }
    
    /**
     * Read file content as string
     * @param filePath Path to the file
     * @return File content as string
     * @throws IOException If file cannot be read
     */
    public String readFile(String filePath) throws IOException {
        try {
            Path path = Paths.get(filePath);
            String content = Files.readString(path);
            processedFiles.add(filePath);
            return content;
        } catch (IOException e) {
            errorCount++;
            throw new IOException("Failed to read file: " + filePath, e);
        }
    }
    
    /**
     * Write content to file
     * @param filePath Destination file path
     * @param content Content to write
     * @throws IOException If file cannot be written
     */
    public void writeFile(String filePath, String content) throws IOException {
        try {
            Path path = Paths.get(filePath);
            Files.writeString(path, content);
            processedFiles.add(filePath);
        } catch (IOException e) {
            errorCount++;
            throw new IOException("Failed to write file: " + filePath, e);
        }
    }
    
    /**
     * Process multiple files in a directory
     * @param directoryPath Directory containing files
     * @param fileExtension File extension filter
     * @return List of processed file paths
     */
    public List<String> processDirectory(String directoryPath, String fileExtension) {
        List<String> processed = new ArrayList<>();
        
        try {
            Path dir = Paths.get(directoryPath);
            Files.walk(dir)
                .filter(Files::isRegularFile)
                .filter(path -> path.toString().endsWith(fileExtension))
                .forEach(path -> {
                    try {
                        String content = readFile(path.toString());
                        // Process content here
                        processed.add(path.toString());
                    } catch (IOException e) {
                        System.err.println("Error processing file: " + path + " - " + e.getMessage());
                    }
                });
        } catch (IOException e) {
            System.err.println("Error walking directory: " + e.getMessage());
        }
        
        return processed;
    }
    
    /**
     * Get processing statistics
     * @return Statistics map
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("processedFiles", processedFiles.size());
        stats.put("errorCount", errorCount);
        stats.put("config", config);
        return stats;
    }
    
    /**
     * Validate file path
     * @param filePath Path to validate
     * @return true if valid, false otherwise
     */
    private boolean validatePath(String filePath) {
        if (filePath == null || filePath.trim().isEmpty()) {
            return false;
        }
        
        Path path = Paths.get(filePath);
        return Files.exists(path) && Files.isReadable(path);
    }
    
    /**
     * Main method for testing
     * @param args Command line arguments
     */
    public static void main(String[] args) {
        Map<String, Object> config = new HashMap<>();
        config.put("maxFileSize", 1024 * 1024); // 1MB
        config.put("allowedExtensions", Arrays.asList(".txt", ".json", ".xml"));
        
        FileProcessor processor = new FileProcessor(config);
        
        try {
            String content = processor.readFile("sample.txt");
            System.out.println("File content length: " + content.length());
            
            Map<String, Object> stats = processor.getStatistics();
            System.out.println("Processing statistics: " + stats);
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}
'''
    
    try:
        # Initialize mock RAG agent
        print("1. Initializing Mock RAGContextAgent...")
        rag_agent = MockRAGContextAgent()
        
        # Index Python code
        print("\n2. Indexing Python code...")
        python_result = rag_agent.chunk_and_index_code(python_sample, "data_processor.py", "python")
        print(f"   ✅ Python code indexed: {python_result['total_chunks']} chunks")
        
        # Index Java code
        print("\n3. Indexing Java code...")
        java_result = rag_agent.chunk_and_index_code(java_sample, "FileProcessor.java", "java")
        print(f"   ✅ Java code indexed: {java_result['total_chunks']} chunks")
        
        # Show chunk examples
        print(f"\n4. Sample chunks:")
        if python_result['chunks']:
            chunk = python_result['chunks'][0]
            print(f"   Python chunk example:")
            print(f"     Lines {chunk['metadata']['start_line']}-{chunk['metadata']['end_line']}")
            print(f"     Contains function: {chunk['metadata']['contains_function']}")
            print(f"     Preview: {chunk['text'][:100]}...")
        
        if java_result['chunks']:
            chunk = java_result['chunks'][0]
            print(f"   Java chunk example:")
            print(f"     Lines {chunk['metadata']['start_line']}-{chunk['metadata']['end_line']}")
            print(f"     Contains class: {chunk['metadata']['contains_class']}")
            print(f"     Preview: {chunk['text'][:100]}...")
        
        # Test queries
        print(f"\n5. Testing semantic queries:")
        queries = [
            "How to read files?",
            "Data validation methods",
            "Error handling in file operations",
            "Class constructors and initialization",
            "JSON processing and parsing",
            "Configuration management"
        ]
        
        for query in queries:
            print(f"\n   Query: '{query}'")
            results = rag_agent.query(query, top_k=3)
            
            if results['total_results'] > 0:
                print(f"   Found {results['total_results']} relevant chunks:")
                for i, result in enumerate(results['results'][:2]):  # Show top 2
                    filename = result['metadata']['filename']
                    language = result['metadata']['language']
                    score = result['score']
                    lines = f"{result['metadata']['start_line']}-{result['metadata']['end_line']}"
                    print(f"     {i+1}. {filename} ({language}) lines {lines} - Score: {score:.3f}")
                    print(f"        Preview: {result['preview'][:80]}...")
            else:
                print("   No relevant chunks found")
        
        # Show statistics
        print(f"\n6. Collection Statistics:")
        stats = rag_agent.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print(f"\n🎉 RAGContextAgent Demo Completed Successfully!")
        print(f"\n✨ Features Demonstrated:")
        print(f"  ✓ Code chunking with AST analysis")
        print(f"  ✓ Multi-language support (Python + Java)")
        print(f"  ✓ Semantic metadata extraction")
        print(f"  ✓ Vector indexing with Qdrant")
        print(f"  ✓ Similarity search and retrieval")
        print(f"  ✓ Mock embeddings for testing")
        print(f"  ✓ Collection management and statistics")
        
        print(f"\n🎯 Ready for Production:")
        print(f"  → Replace mock embeddings with OpenAI embeddings")
        print(f"  → Integrate with CodeFetcherAgent for repository indexing")
        print(f"  → Add LLM-powered response generation")
        print(f"  → Scale to large codebases")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_rag_context() 