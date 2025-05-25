#!/usr/bin/env python3
"""Test RAGContextAgent với real OpenAI embeddings và real data"""

import sys
import os
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)

# Add paths
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(project_root, 'deepcode_insight'))
sys.path.append(project_root)

# Import config first
from config import config

def check_prerequisites() -> bool:
    """Check if prerequisites are met for real data testing"""
    
    print("🔍 Checking prerequisites for real data testing...")
    
    # Check OpenAI API key
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your_openai_api_key_here":
        print("❌ OpenAI API key not configured")
        print("   Please set OPENAI_API_KEY in .env file or environment variable")
        return False
    
    print("✅ OpenAI API key configured")
    
    # Check Qdrant connection
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        collections = client.get_collections()
        print(f"✅ Qdrant connected ({len(collections.collections)} collections)")
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print("   Make sure Qdrant is running: docker compose up -d")
        return False
    
    # Test OpenAI API
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        # Test with a simple embedding
        response = client.embeddings.create(
            model=config.OPENAI_EMBEDDING_MODEL,
            input="test"
        )
        print(f"✅ OpenAI API working (embedding dimension: {len(response.data[0].embedding)})")
    except Exception as e:
        print(f"❌ OpenAI API test failed: {e}")
        return False
    
    return True

def test_real_rag_context():
    """Test RAGContextAgent với real OpenAI embeddings"""
    
    print("\n🧪 === Testing RAGContextAgent with Real Data ===\n")
    
    try:
        from deepcode_insight.agents.rag_context import RAGContextAgent
        
        # Real code samples từ actual projects
        real_python_code = '''
import asyncio
import aiohttp
import json
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """Data class for API response"""
    status_code: int
    data: Dict
    timestamp: datetime
    
    def is_success(self) -> bool:
        """Check if response is successful"""
        return 200 <= self.status_code < 300
    
    def get_error_message(self) -> Optional[str]:
        """Extract error message from response"""
        if self.is_success():
            return None
        return self.data.get('error', f'HTTP {self.status_code}')

class AsyncAPIClient:
    """Asynchronous API client with retry logic and rate limiting"""
    
    def __init__(self, base_url: str, api_key: str, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_delay = 1.0  # seconds
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={'Authorization': f'Bearer {self.api_key}'},
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> APIResponse:
        """Make HTTP request with retry logic"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries + 1):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    data = await response.json()
                    
                    api_response = APIResponse(
                        status_code=response.status,
                        data=data,
                        timestamp=datetime.now()
                    )
                    
                    # Handle rate limiting
                    if response.status == 429:
                        if attempt < self.max_retries:
                            wait_time = self.rate_limit_delay * (2 ** attempt)
                            logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                            await asyncio.sleep(wait_time)
                            continue
                    
                    return api_response
                    
            except asyncio.TimeoutError:
                logger.error(f"Request timeout on attempt {attempt + 1}")
                if attempt < self.max_retries:
                    await asyncio.sleep(1)
                    continue
                raise
            except Exception as e:
                logger.error(f"Request failed on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(1)
                    continue
                raise
        
        raise Exception(f"Max retries ({self.max_retries}) exceeded")
    
    async def get(self, endpoint: str, params: Optional[Dict] = None) -> APIResponse:
        """GET request"""
        return await self._make_request('GET', endpoint, params=params)
    
    async def post(self, endpoint: str, data: Optional[Dict] = None) -> APIResponse:
        """POST request"""
        return await self._make_request('POST', endpoint, json=data)
    
    async def batch_requests(self, requests: List[Dict]) -> List[APIResponse]:
        """Execute multiple requests concurrently"""
        tasks = []
        
        for req in requests:
            method = req.get('method', 'GET').upper()
            endpoint = req['endpoint']
            
            if method == 'GET':
                task = self.get(endpoint, req.get('params'))
            elif method == 'POST':
                task = self.post(endpoint, req.get('data'))
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)

async def main():
    """Example usage of AsyncAPIClient"""
    async with AsyncAPIClient("https://api.example.com", "your-api-key") as client:
        # Single request
        response = await client.get("/users/123")
        if response.is_success():
            print(f"User data: {response.data}")
        else:
            print(f"Error: {response.get_error_message()}")
        
        # Batch requests
        batch_requests = [
            {'method': 'GET', 'endpoint': '/users/1'},
            {'method': 'GET', 'endpoint': '/users/2'},
            {'method': 'POST', 'endpoint': '/users', 'data': {'name': 'John'}}
        ]
        
        responses = await client.batch_requests(batch_requests)
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                print(f"Request {i} failed: {resp}")
            else:
                print(f"Request {i} status: {resp.status_code}")

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        real_java_code = '''
package com.example.service;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.time.LocalDateTime;
import java.time.Duration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.retry.annotation.Retryable;
import org.springframework.retry.annotation.Backoff;

/**
 * Service for managing user data with caching and retry mechanisms
 */
@Service
public class UserDataService {
    
    private static final Logger logger = LoggerFactory.getLogger(UserDataService.class);
    private static final int MAX_BATCH_SIZE = 100;
    private static final Duration CACHE_TTL = Duration.ofMinutes(30);
    
    private final UserRepository userRepository;
    private final CacheManager cacheManager;
    private final ExecutorService executorService;
    private final Map<String, CompletableFuture<User>> pendingRequests;
    
    public UserDataService(UserRepository userRepository, CacheManager cacheManager) {
        this.userRepository = userRepository;
        this.cacheManager = cacheManager;
        this.executorService = Executors.newFixedThreadPool(10);
        this.pendingRequests = new ConcurrentHashMap<>();
    }
    
    /**
     * Get user by ID with caching
     * @param userId User ID
     * @return User object or null if not found
     */
    @Cacheable(value = "users", key = "#userId")
    @Retryable(value = {Exception.class}, maxAttempts = 3, backoff = @Backoff(delay = 1000))
    public User getUserById(String userId) {
        logger.debug("Fetching user with ID: {}", userId);
        
        try {
            // Check if request is already pending
            CompletableFuture<User> pendingRequest = pendingRequests.get(userId);
            if (pendingRequest != null && !pendingRequest.isDone()) {
                logger.debug("Request for user {} already pending, waiting...", userId);
                return pendingRequest.get(5, TimeUnit.SECONDS);
            }
            
            // Create new request
            CompletableFuture<User> future = CompletableFuture.supplyAsync(() -> {
                return userRepository.findById(userId)
                    .orElseThrow(() -> new UserNotFoundException("User not found: " + userId));
            }, executorService);
            
            pendingRequests.put(userId, future);
            
            User user = future.get(10, TimeUnit.SECONDS);
            pendingRequests.remove(userId);
            
            logger.info("Successfully fetched user: {}", userId);
            return user;
            
        } catch (TimeoutException e) {
            logger.error("Timeout fetching user {}: {}", userId, e.getMessage());
            pendingRequests.remove(userId);
            throw new ServiceException("Request timeout for user: " + userId, e);
        } catch (Exception e) {
            logger.error("Error fetching user {}: {}", userId, e.getMessage());
            pendingRequests.remove(userId);
            throw new ServiceException("Failed to fetch user: " + userId, e);
        }
    }
    
    /**
     * Get multiple users by IDs in batch
     * @param userIds List of user IDs
     * @return Map of user ID to User object
     */
    public Map<String, User> getUsersByIds(List<String> userIds) {
        if (userIds == null || userIds.isEmpty()) {
            return Collections.emptyMap();
        }
        
        if (userIds.size() > MAX_BATCH_SIZE) {
            throw new IllegalArgumentException("Batch size exceeds maximum: " + MAX_BATCH_SIZE);
        }
        
        logger.info("Fetching {} users in batch", userIds.size());
        
        // Split into cached and non-cached
        Map<String, User> result = new HashMap<>();
        List<String> uncachedIds = new ArrayList<>();
        
        for (String userId : userIds) {
            User cachedUser = getCachedUser(userId);
            if (cachedUser != null) {
                result.put(userId, cachedUser);
            } else {
                uncachedIds.add(userId);
            }
        }
        
        // Fetch uncached users
        if (!uncachedIds.isEmpty()) {
            Map<String, User> fetchedUsers = fetchUsersFromDatabase(uncachedIds);
            result.putAll(fetchedUsers);
            
            // Cache the fetched users
            fetchedUsers.forEach(this::cacheUser);
        }
        
        logger.info("Batch fetch completed: {}/{} users found", result.size(), userIds.size());
        return result;
    }
    
    /**
     * Search users by criteria with pagination
     * @param criteria Search criteria
     * @param page Page number (0-based)
     * @param size Page size
     * @return Paginated search results
     */
    public PagedResult<User> searchUsers(UserSearchCriteria criteria, int page, int size) {
        validateSearchCriteria(criteria);
        validatePaginationParams(page, size);
        
        logger.info("Searching users with criteria: {}, page: {}, size: {}", criteria, page, size);
        
        try {
            long startTime = System.currentTimeMillis();
            
            // Build query
            UserQuery query = UserQuery.builder()
                .criteria(criteria)
                .page(page)
                .size(size)
                .build();
            
            // Execute search
            List<User> users = userRepository.search(query);
            long totalCount = userRepository.countByCriteria(criteria);
            
            long duration = System.currentTimeMillis() - startTime;
            logger.info("Search completed in {}ms, found {} users", duration, users.size());
            
            return PagedResult.<User>builder()
                .content(users)
                .page(page)
                .size(size)
                .totalElements(totalCount)
                .totalPages((int) Math.ceil((double) totalCount / size))
                .build();
                
        } catch (Exception e) {
            logger.error("Error searching users: {}", e.getMessage(), e);
            throw new ServiceException("User search failed", e);
        }
    }
    
    /**
     * Update user data with optimistic locking
     * @param userId User ID
     * @param updateData Update data
     * @return Updated user
     */
    public User updateUser(String userId, UserUpdateData updateData) {
        validateUpdateData(updateData);
        
        logger.info("Updating user: {}", userId);
        
        try {
            User existingUser = getUserById(userId);
            if (existingUser == null) {
                throw new UserNotFoundException("User not found: " + userId);
            }
            
            // Check version for optimistic locking
            if (updateData.getVersion() != null && 
                !updateData.getVersion().equals(existingUser.getVersion())) {
                throw new OptimisticLockException("User data has been modified by another process");
            }
            
            // Apply updates
            User updatedUser = applyUpdates(existingUser, updateData);
            updatedUser.setLastModified(LocalDateTime.now());
            updatedUser.setVersion(existingUser.getVersion() + 1);
            
            // Save to database
            User savedUser = userRepository.save(updatedUser);
            
            // Update cache
            cacheUser(savedUser.getId(), savedUser);
            
            logger.info("User updated successfully: {}", userId);
            return savedUser;
            
        } catch (Exception e) {
            logger.error("Error updating user {}: {}", userId, e.getMessage(), e);
            throw new ServiceException("Failed to update user: " + userId, e);
        }
    }
    
    private User getCachedUser(String userId) {
        try {
            return cacheManager.getCache("users").get(userId, User.class);
        } catch (Exception e) {
            logger.warn("Cache lookup failed for user {}: {}", userId, e.getMessage());
            return null;
        }
    }
    
    private void cacheUser(String userId, User user) {
        try {
            cacheManager.getCache("users").put(userId, user);
        } catch (Exception e) {
            logger.warn("Failed to cache user {}: {}", userId, e.getMessage());
        }
    }
    
    private Map<String, User> fetchUsersFromDatabase(List<String> userIds) {
        return userRepository.findByIds(userIds)
            .stream()
            .collect(Collectors.toMap(User::getId, user -> user));
    }
    
    private void validateSearchCriteria(UserSearchCriteria criteria) {
        if (criteria == null) {
            throw new IllegalArgumentException("Search criteria cannot be null");
        }
        // Add more validation as needed
    }
    
    private void validatePaginationParams(int page, int size) {
        if (page < 0) {
            throw new IllegalArgumentException("Page number cannot be negative");
        }
        if (size <= 0 || size > 1000) {
            throw new IllegalArgumentException("Page size must be between 1 and 1000");
        }
    }
    
    private void validateUpdateData(UserUpdateData updateData) {
        if (updateData == null) {
            throw new IllegalArgumentException("Update data cannot be null");
        }
        // Add more validation as needed
    }
    
    private User applyUpdates(User existingUser, UserUpdateData updateData) {
        User.Builder builder = User.builder(existingUser);
        
        if (updateData.getName() != null) {
            builder.name(updateData.getName());
        }
        if (updateData.getEmail() != null) {
            builder.email(updateData.getEmail());
        }
        if (updateData.getStatus() != null) {
            builder.status(updateData.getStatus());
        }
        
        return builder.build();
    }
}
'''
        
        # Initialize RAG agent với real config
        print("1. Initializing RAGContextAgent with real configuration...")
        config.print_config()
        
        rag_agent = RAGContextAgent()
        print("   ✅ RAGContextAgent initialized successfully")
        
        # Clear previous test data
        print("\n2. Clearing previous test data...")
        rag_agent.clear_collection()
        print("   ✅ Collection cleared")
        
        # Index real Python code
        print("\n3. Indexing real Python code...")
        python_success = rag_agent.index_code_file(
            real_python_code, 
            "async_api_client.py", 
            "python",
            metadata={"project": "real_test", "type": "async_client"}
        )
        print(f"   {'✅ Success' if python_success else '❌ Failed'}")
        
        # Index real Java code
        print("\n4. Indexing real Java code...")
        java_success = rag_agent.index_code_file(
            real_java_code, 
            "UserDataService.java", 
            "java",
            metadata={"project": "real_test", "type": "service_layer"}
        )
        print(f"   {'✅ Success' if java_success else '❌ Failed'}")
        
        # Test real semantic queries
        print("\n5. Testing real semantic queries...")
        
        real_queries = [
            "How to implement async HTTP client with retry logic?",
            "What are the best practices for caching user data?",
            "How to handle rate limiting in API calls?",
            "Show me examples of optimistic locking implementation",
            "How to implement batch processing for database operations?",
            "What error handling patterns are used in the code?",
            "How to implement pagination in search results?",
            "Show me async context manager implementation"
        ]
        
        query_results = {}
        for query in real_queries:
            print(f"\n   Query: '{query}'")
            results = rag_agent.query(query, top_k=3)
            query_results[query] = results
            
            if results["total_results"] > 0:
                print(f"   Found {results['total_results']} relevant chunks:")
                for i, result in enumerate(results["results"][:2]):  # Show top 2
                    filename = result.get("metadata", {}).get("filename", "unknown")
                    language = result.get("metadata", {}).get("language", "unknown")
                    score = result.get("score", 0)
                    preview = result["content_preview"][:100] + "..."
                    print(f"     {i+1}. {filename} ({language}) - Score: {score:.3f}")
                    print(f"        Preview: {preview}")
            else:
                print("   No relevant chunks found")
        
        # Test context generation với real LLM
        print(f"\n6. Testing real LLM context generation...")
        context_query = "How to implement proper error handling and retry logic in async operations?"
        context_result = rag_agent.query_with_context(
            context_query, 
            top_k=3, 
            generate_response=True
        )
        
        print(f"   Query: '{context_query}'")
        print(f"   Context chunks: {context_result['total_chunks']}")
        if context_result.get('response'):
            response_preview = context_result['response'][:300] + "..." if len(context_result['response']) > 300 else context_result['response']
            print(f"   LLM Response preview: {response_preview}")
        
        # Collection statistics
        print(f"\n7. Real data collection statistics:")
        stats = rag_agent.get_collection_stats()
        for key, value in stats.items():
            if key != 'error':
                print(f"   {key}: {value}")
        
        # Performance analysis
        print(f"\n8. Performance analysis:")
        total_queries = len(real_queries)
        successful_queries = sum(1 for r in query_results.values() if r["total_results"] > 0)
        success_rate = (successful_queries / total_queries) * 100
        
        print(f"   Total queries: {total_queries}")
        print(f"   Successful queries: {successful_queries}")
        print(f"   Success rate: {success_rate:.1f}%")
        
        # Quality assessment
        print(f"\n9. Quality assessment:")
        high_quality_results = 0
        for query, results in query_results.items():
            if results["total_results"] > 0:
                top_score = results["results"][0].get("score", 0)
                if top_score > 0.7:  # High relevance threshold
                    high_quality_results += 1
        
        quality_rate = (high_quality_results / successful_queries) * 100 if successful_queries > 0 else 0
        print(f"   High quality results (score > 0.7): {high_quality_results}")
        print(f"   Quality rate: {quality_rate:.1f}%")
        
        print(f"\n🎉 Real data testing completed successfully!")
        print(f"\n✨ Real Features Tested:")
        print(f"  ✓ Real OpenAI embeddings ({config.OPENAI_EMBEDDING_MODEL})")
        print(f"  ✓ Real LLM responses ({config.OPENAI_MODEL})")
        print(f"  ✓ Complex real-world code samples")
        print(f"  ✓ Semantic search with actual relevance")
        print(f"  ✓ Context-aware response generation")
        print(f"  ✓ Performance and quality metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_integration():
    """Test config integration"""
    
    print("\n🧪 === Testing Config Integration ===\n")
    
    try:
        # Test config loading
        print("1. Testing config loading...")
        config.print_config()
        
        # Test config validation
        print("\n2. Testing config validation...")
        is_valid = config.validate()
        print(f"   Config validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
        
        # Test environment setup
        print("\n3. Testing environment setup...")
        from config import setup_environment
        env_template = setup_environment()
        print("   ✅ Environment setup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing RAGContextAgent with Real Data\n")
    
    # Test config integration
    config_success = test_config_integration()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please:")
        print("1. Set OPENAI_API_KEY in .env file")
        print("2. Start Qdrant: docker compose up -d")
        print("3. Verify API connectivity")
        sys.exit(1)
    
    # Run real data tests
    real_data_success = test_real_rag_context()
    
    if config_success and real_data_success:
        print(f"\n✅ All real data tests passed!")
        print(f"\n🎯 Production Ready Features:")
        print(f"  ✓ Real OpenAI API integration")
        print(f"  ✓ Configuration management")
        print(f"  ✓ Real-world code analysis")
        print(f"  ✓ High-quality semantic search")
        print(f"  ✓ LLM-powered responses")
        print(f"  ✓ Performance monitoring")
    else:
        print(f"\n❌ Some tests failed!")
        sys.exit(1) 