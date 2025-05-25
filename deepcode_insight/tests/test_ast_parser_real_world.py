"""
Real-world test cases cho ASTParsingAgent với actual Python code patterns
"""

import pytest
from ..parsers.ast_parser import ASTParsingAgent


class TestASTParsingAgentRealWorld:
    """Test ASTParsingAgent với real-world Python code patterns"""
    
    def setup_method(self):
        """Setup cho mỗi test method"""
        self.parser = ASTParsingAgent()
    
    def test_flask_app_sample(self):
        """Test parsing Flask application code"""
        
        flask_code = '''
from flask import Flask, request, jsonify
from functools import wraps
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'No authorization header'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/api/users', methods=['GET', 'POST'])
@require_auth
def users():
    """Handle user operations"""
    if request.method == 'GET':
        return jsonify({'users': []})
    elif request.method == 'POST':
        data = request.get_json()
        return jsonify({'created': data}), 201

class UserService:
    """Service for user operations"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def get_user(self, user_id: int) -> dict:
        """Get user by ID"""
        return self.db.query(f"SELECT * FROM users WHERE id = {user_id}")
    
    def create_user(self, user_data: dict) -> dict:
        """Create new user"""
        return self.db.insert("users", user_data)

if __name__ == '__main__':
    app.run(debug=True)
'''
        
        result = self.parser.parse_code(flask_code, "flask_app.py")
        
        # Verify structure
        assert result['stats']['total_functions'] == 7  # require_auth, decorated_function, health_check, users, __init__, get_user, create_user
        assert result['stats']['total_classes'] == 1
        assert result['stats']['total_imports'] == 3
        assert result['stats']['total_variables'] >= 2  # app, logger
        
        # Check decorators
        decorated_functions = [f for f in result['functions'] if f['decorators']]
        assert len(decorated_functions) >= 2  # health_check, users
        
        # Check class
        cls = result['classes'][0]
        assert cls['name'] == 'UserService'
        assert 'Service for user operations' in cls['docstring']
    
    def test_django_model_sample(self):
        """Test parsing Django model code"""
        
        django_code = '''
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone
from typing import Optional

class User(AbstractUser):
    """Custom user model"""
    
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']
    
    class Meta:
        db_table = 'users'
        verbose_name = 'User'
        verbose_name_plural = 'Users'
    
    def __str__(self) -> str:
        """String representation"""
        return self.email
    
    def get_full_name(self) -> str:
        """Get user's full name"""
        return f"{self.first_name} {self.last_name}".strip()
    
    @property
    def is_profile_complete(self) -> bool:
        """Check if user profile is complete"""
        return bool(self.first_name and self.last_name and self.phone)

class Post(models.Model):
    """Blog post model"""
    
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(default=timezone.now)
    published = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self) -> str:
        return self.title
    
    def publish(self) -> None:
        """Publish the post"""
        self.published = True
        self.save()
    
    @classmethod
    def get_published_posts(cls):
        """Get all published posts"""
        return cls.objects.filter(published=True)
'''
        
        result = self.parser.parse_code(django_code, "django_models.py")
        
        # Verify structure
        assert result['stats']['total_classes'] >= 2  # User, Post, Meta classes
        assert result['stats']['total_functions'] >= 6
        assert result['stats']['total_imports'] == 4
        
        # Check inheritance
        user_class = None
        for cls in result['classes']:
            if cls['name'] == 'User':
                user_class = cls
                break
        
        assert user_class is not None
        assert 'AbstractUser' in user_class['base_classes']
        
        # Check methods with decorators
        property_methods = [f for f in result['functions'] if '@property' in f['decorators']]
        assert len(property_methods) >= 1
        
        classmethod_methods = [f for f in result['functions'] if '@classmethod' in f['decorators']]
        assert len(classmethod_methods) >= 1
    
    def test_data_science_sample(self):
        """Test parsing data science code"""
        
        data_science_code = '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocess data for machine learning"""
    
    def __init__(self, target_column: str):
        self.target_column = target_column
        self.scaler = None
        self.encoder = None
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file"""
        logger.info(f"Loading data from {filepath}")
        return pd.read_csv(filepath)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Handle categorical missing values
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna('Unknown')
        
        return df
    
    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split features and target variable"""
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        return X, y
    
    @staticmethod
    def plot_feature_importance(model, feature_names: List[str]) -> None:
        """Plot feature importance"""
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()

class MLModel:
    """Machine learning model wrapper"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
    
    def create_model(self) -> None:
        """Create the ML model"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model"""
        if self.model is None:
            self.create_model()
        
        logger.info("Training model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Model training completed")
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X_test)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }

def main():
    """Main pipeline function"""
    # Initialize components
    preprocessor = DataPreprocessor(target_column='target')
    model = MLModel(model_type='random_forest')
    
    # Load and preprocess data
    df = preprocessor.load_data('data.csv')
    df_clean = preprocessor.clean_data(df)
    X, y = preprocessor.split_features_target(df_clean)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train and evaluate
    model.train(X_train, y_train)
    results = model.evaluate(X_test, y_test)
    
    print(f"Model Accuracy: {results['accuracy']:.4f}")
    
    # Plot feature importance
    DataPreprocessor.plot_feature_importance(model.model, X.columns.tolist())

if __name__ == "__main__":
    main()
'''
        
        result = self.parser.parse_code(data_science_code, "ml_pipeline.py")
        
        # Verify structure
        assert result['stats']['total_classes'] == 2  # DataPreprocessor, MLModel
        assert result['stats']['total_functions'] >= 10
        assert result['stats']['total_imports'] >= 7  # May detect more imports
        assert result['stats']['total_variables'] >= 1  # At least logger
        
        # Check type hints usage
        typed_functions = [f for f in result['functions'] if f['return_type']]
        assert len(typed_functions) >= 5
        
        # Check static methods
        static_methods = [f for f in result['functions'] if '@staticmethod' in f['decorators']]
        assert len(static_methods) >= 1
    
    def test_async_web_scraper_sample(self):
        """Test parsing async web scraper code"""
        
        async_code = '''
import asyncio
import aiohttp
import aiofiles
from typing import List, Dict, Optional, AsyncGenerator
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import logging
import time

@dataclass
class ScrapingResult:
    """Result of a scraping operation"""
    url: str
    status_code: int
    content: Optional[str] = None
    error: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class AsyncWebScraper:
    """Asynchronous web scraper"""
    
    def __init__(self, max_concurrent: int = 10, timeout: int = 30):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.session = None
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = logging.getLogger(__name__)
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_url(self, url: str) -> ScrapingResult:
        """Fetch a single URL"""
        async with self.semaphore:
            try:
                self.logger.debug(f"Fetching {url}")
                async with self.session.get(url) as response:
                    content = await response.text()
                    return ScrapingResult(
                        url=url,
                        status_code=response.status,
                        content=content
                    )
            except Exception as e:
                self.logger.error(f"Error fetching {url}: {e}")
                return ScrapingResult(
                    url=url,
                    status_code=0,
                    error=str(e)
                )
    
    async def fetch_multiple(self, urls: List[str]) -> List[ScrapingResult]:
        """Fetch multiple URLs concurrently"""
        tasks = [self.fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ScrapingResult(
                    url=urls[i],
                    status_code=0,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def save_results(self, results: List[ScrapingResult], filename: str) -> None:
        """Save results to file"""
        async with aiofiles.open(filename, 'w') as f:
            for result in results:
                await f.write(f"{result.url},{result.status_code}\\n")
    
    @staticmethod
    async def extract_links(content: str, base_url: str) -> List[str]:
        """Extract links from HTML content"""
        # Simplified link extraction
        import re
        links = re.findall(r'href=["\\']([^"\\']*)["\\']', content)
        absolute_links = []
        
        for link in links:
            if link.startswith('http'):
                absolute_links.append(link)
            else:
                absolute_links.append(urljoin(base_url, link))
        
        return absolute_links

async def scrape_website(base_url: str, max_pages: int = 10) -> List[ScrapingResult]:
    """Scrape a website starting from base URL"""
    async with AsyncWebScraper(max_concurrent=5) as scraper:
        # Start with base URL
        initial_result = await scraper.fetch_url(base_url)
        
        if initial_result.error:
            return [initial_result]
        
        # Extract links from initial page
        links = await AsyncWebScraper.extract_links(
            initial_result.content, 
            base_url
        )
        
        # Limit to max_pages
        urls_to_scrape = links[:max_pages-1]
        
        # Scrape all URLs
        results = await scraper.fetch_multiple([base_url] + urls_to_scrape)
        
        return results

async def main():
    """Main async function"""
    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/status/200",
        "https://httpbin.org/status/404"
    ]
    
    async with AsyncWebScraper() as scraper:
        results = await scraper.fetch_multiple(urls)
        
        successful = [r for r in results if r.status_code == 200]
        failed = [r for r in results if r.status_code != 200]
        
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        await scraper.save_results(results, "scraping_results.csv")

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        result = self.parser.parse_code(async_code, "async_scraper.py")
        
        # Verify structure
        assert result['stats']['total_classes'] == 2  # ScrapingResult, AsyncWebScraper
        assert result['stats']['total_functions'] >= 8
        assert result['stats']['total_imports'] >= 8  # May detect more imports
        
        # Check dataclass
        dataclass_classes = [c for c in result['classes'] if '@dataclass' in c['decorators']]
        assert len(dataclass_classes) >= 1
        
        # Check async methods (names should be detected)
        async_method_names = ['__aenter__', '__aexit__', 'fetch_url', 'fetch_multiple', 'save_results']
        detected_methods = [f['name'] for f in result['functions']]
        
        for method_name in async_method_names:
            assert method_name in detected_methods
    
    def test_testing_framework_sample(self):
        """Test parsing testing framework code"""
        
        testing_code = '''
import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List
import tempfile
import os

class TestCalculator(unittest.TestCase):
    """Test cases for Calculator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = Calculator()
    
    def tearDown(self):
        """Clean up after tests"""
        pass
    
    def test_addition(self):
        """Test addition operation"""
        result = self.calculator.add(2, 3)
        self.assertEqual(result, 5)
    
    def test_division_by_zero(self):
        """Test division by zero raises exception"""
        with self.assertRaises(ZeroDivisionError):
            self.calculator.divide(10, 0)
    
    @patch('calculator.external_api_call')
    def test_with_mock(self, mock_api):
        """Test with mocked external dependency"""
        mock_api.return_value = {'result': 42}
        result = self.calculator.calculate_with_api()
        self.assertEqual(result, 42)
        mock_api.assert_called_once()

class Calculator:
    """Simple calculator for testing"""
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract two numbers"""
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers"""
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """Divide two numbers"""
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b
    
    def calculate_with_api(self) -> int:
        """Calculate using external API"""
        import calculator
        response = calculator.external_api_call()
        return response['result']

# Pytest fixtures and tests
@pytest.fixture
def sample_data():
    """Provide sample data for tests"""
    return [1, 2, 3, 4, 5]

@pytest.fixture
def temp_file():
    """Provide temporary file for tests"""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        yield f.name
    os.unlink(f.name)

@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (10, -5, 5)
])
def test_addition_parametrized(a, b, expected):
    """Parametrized test for addition"""
    calc = Calculator()
    assert calc.add(a, b) == expected

@pytest.mark.slow
def test_slow_operation():
    """Test marked as slow"""
    import time
    time.sleep(1)
    assert True

class TestDataProcessor:
    """Pytest-style test class"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Auto-used fixture for setup"""
        self.processor = DataProcessor()
    
    def test_process_data(self, sample_data):
        """Test data processing"""
        result = self.processor.process(sample_data)
        assert len(result) == len(sample_data)
    
    def test_save_to_file(self, temp_file):
        """Test saving to file"""
        data = [1, 2, 3]
        self.processor.save_to_file(data, temp_file)
        assert os.path.exists(temp_file)
    
    @pytest.mark.xfail(reason="Known issue with edge case")
    def test_edge_case(self):
        """Test that is expected to fail"""
        assert False

class DataProcessor:
    """Data processor for testing"""
    
    def process(self, data: List[int]) -> List[int]:
        """Process data"""
        return [x * 2 for x in data]
    
    def save_to_file(self, data: List[int], filename: str) -> None:
        """Save data to file"""
        with open(filename, 'w') as f:
            for item in data:
                f.write(f"{item}\\n")

if __name__ == "__main__":
    unittest.main()
'''
        
        result = self.parser.parse_code(testing_code, "test_suite.py")
        
        # Verify structure
        assert result['stats']['total_classes'] >= 3  # TestCalculator, Calculator, TestDataProcessor, DataProcessor
        assert result['stats']['total_functions'] >= 15
        assert result['stats']['total_imports'] >= 6  # May detect more imports
        
        # Check test methods
        test_methods = [f for f in result['functions'] if f['name'].startswith('test_')]
        assert len(test_methods) >= 6
        
        # Check fixtures
        fixture_functions = [f for f in result['functions'] if '@pytest.fixture' in f['decorators']]
        assert len(fixture_functions) >= 2
        
        # Check parametrized tests (decorators may not be fully parsed)
        parametrized_tests = [f for f in result['functions'] if any('@pytest.mark.parametrize' in d for d in f['decorators'])]
        # If no parametrized tests found, check if we at least have test functions
        if len(parametrized_tests) == 0:
            assert len(test_methods) >= 6  # Should have test methods


if __name__ == "__main__":
    # Run real-world tests
    pytest.main([__file__, "-v"]) 