# Makefile cho LangGraph Demo
# Cung cấp các shortcuts để chạy tests và demo

.PHONY: help install test test-unit test-integration test-edge demo clean cli-help cli-validate

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
	@echo "  test-cli-edge    - Chạy CLI edge case tests"
	@echo "  test-cli-performance - Chạy CLI performance tests"
	@echo "  test-cli-all     - Chạy tất cả CLI tests"
	@echo "  test-code-fetcher - Chạy CodeFetcherAgent basic tests"
	@echo "  test-code-fetcher-unit - Chạy CodeFetcherAgent unit tests với mocks"
	@echo "  test-code-fetcher-advanced - Chạy CodeFetcherAgent advanced tests"
	@echo "  test-code-fetcher-all - Chạy tất cả CodeFetcherAgent tests"
	@echo "  cli-help         - Hiển thị CLI help"
	@echo "  cli-validate     - Test CLI validation functions"
	@echo "  cli-fetch-info   - Test CodeFetcherAgent repository info"
	@echo "  cli-fetch-files  - Test CodeFetcherAgent file listing"
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

# Chạy CLI edge case tests
test-cli-edge:
	poetry run pytest tests/test_cli_edge_cases.py -v

# Chạy CLI performance tests
test-cli-performance:
	poetry run pytest tests/test_cli_performance.py -v

# Chạy tất cả CLI tests
test-cli-all:
	poetry run pytest tests/test_cli*.py -v

# CodeFetcherAgent tests
test-code-fetcher:
	poetry run pytest tests/test_code_fetcher.py -v

test-code-fetcher-unit:
	poetry run pytest tests/test_code_fetcher_unit.py -v

test-code-fetcher-advanced:
	poetry run pytest tests/test_code_fetcher_advanced.py -v

test-code-fetcher-all:
	poetry run pytest tests/test_code_fetcher*.py -v

# Chạy tests với coverage (cần cài pytest-cov trước)
test-coverage:
	@echo "Lưu ý: Cần cài pytest-cov: poetry add --group dev pytest-cov"
	poetry run pytest --cov=src --cov-report=html --cov-report=term

# CLI commands
cli-help:
	poetry run python cli.py --help

cli-validate:
	poetry run python cli.py validate

# CodeFetcherAgent commands
cli-fetch-info:
	poetry run python cli.py fetch --repo-url https://github.com/octocat/Hello-World --get-info

cli-fetch-files:
	poetry run python cli.py fetch --repo-url https://github.com/octocat/Hello-World --list-files

# Xóa cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage 