# Makefile cho LangGraph Demo
# Cung cấp các shortcuts để chạy tests và demo

.PHONY: help install test test-unit test-integration test-edge demo clean

# Default target
help:
	@echo "LangGraph Demo - Available Commands:"
	@echo ""
	@echo "  install           - Cài đặt dependencies với Poetry"
	@echo "  demo             - Chạy LangGraph demo"
	@echo "  test             - Chạy tất cả tests"
	@echo "  test-unit        - Chỉ chạy unit tests"
	@echo "  test-integration - Chỉ chạy integration tests"
	@echo "  test-edge        - Chỉ chạy edge case tests"
	@echo "  test-verbose     - Chạy tests với verbose output"
	@echo "  test-coverage    - Chạy tests với coverage report (cần cài pytest-cov)"
	@echo "  clean            - Xóa cache và temporary files"
	@echo ""

# Cài đặt dependencies
install:
	poetry install --no-root

# Chạy demo
demo:
	poetry run python main.py

# Chạy tất cả tests
test:
	poetry run pytest

# Chạy unit tests
test-unit:
	poetry run pytest -m unit -v

# Chạy integration tests  
test-integration:
	poetry run pytest -m integration -v

# Chạy edge case tests
test-edge:
	poetry run pytest -m edge_case -v

# Chạy tests với verbose output
test-verbose:
	poetry run pytest -v

# Chạy tests với coverage (cần cài pytest-cov trước)
test-coverage:
	@echo "Lưu ý: Cần cài pytest-cov: poetry add --group dev pytest-cov"
	poetry run pytest --cov=src --cov-report=html --cov-report=term

# Xóa cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage 