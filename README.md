# LangGraph Demo - Hai Agents Trò Chuyện

Demo đơn giản sử dụng LangGraph để tạo workflow với hai agents truyền message cho nhau.

## 🚀 Cài Đặt

### Yêu Cầu
- Python >= 3.9
- Poetry

### Cài Đặt Dependencies

```bash
# Cài đặt dependencies
poetry install --no-root

# Hoặc sử dụng pip (không khuyến khích)
pip install langgraph langchain langchain-openai
```

## 📋 Cấu Trúc Dự Án

```
├── src/
│   ├── __init__.py         # Package init
│   ├── state.py           # Định nghĩa state cho LangGraph
│   └── graph.py           # Logic chính của graph và agents
├── agents/
│   ├── __init__.py         # Agents package init
│   └── code_fetcher.py     # CodeFetcherAgent - Git operations
├── tests/
│   ├── test_graph.py       # Core LangGraph tests
│   ├── test_cli.py         # CLI functionality tests
│   ├── test_code_fetcher.py # CodeFetcherAgent tests
│   └── ...                 # Other test files
├── cli.py                 # Command Line Interface
├── main.py                # Entry point để chạy demo
├── pyproject.toml         # Poetry configuration
└── README.md              # Tài liệu này
```

## 🎯 Mô Tả

Dự án này demo cách sử dụng LangGraph để:

1. **Định nghĩa State**: Sử dụng `TypedDict` để định nghĩa structure của state được truyền giữa các nodes
2. **Tạo Agents**: Hai agents đơn giản truyền message cho nhau
3. **Workflow Logic**: Sử dụng conditional edges để điều khiển luồng thực thi
4. **State Management**: Quản lý trạng thái và quyết định khi nào dừng

### Agents

- **Agent 1** 🤖: Khởi tạo cuộc trò chuyện và gửi message
- **Agent 2** 🦾: Nhận và phản hồi message từ Agent 1
- **CodeFetcherAgent** 🔄: Clone repositories, fetch PR diffs, và analyze code changes

## 🏃 Chạy Demo

```bash
# Sử dụng Poetry (khuyến khích)
poetry run python main.py

# Hoặc sử dụng Makefile
make demo

# Hoặc trực tiếp với Python
python main.py
```

## 🖥️ Command Line Interface (CLI)

Dự án cung cấp CLI mạnh mẽ sử dụng Click để phân tích Pull Requests:

### Cài Đặt CLI
```bash
# CLI được cài đặt cùng với dependencies
poetry install --no-root
```

### Sử dụng CLI

```bash
# Hiển thị help
poetry run python cli.py --help
# hoặc: make cli-help

# Phân tích Pull Request
poetry run python cli.py analyze \
  --repo-url https://github.com/owner/repo \
  --pr-id 123 \
  --output-format text

# Chạy demo với repository context
poetry run python cli.py demo \
  --repo-url https://github.com/microsoft/vscode \
  --pr-id 42

# Test validation functions
poetry run python cli.py validate
# hoặc: make cli-validate
```

### CLI Commands

#### `analyze` - Phân Tích Pull Request
```bash
poetry run python cli.py analyze [OPTIONS]

Options:
  --repo-url TEXT        Repository URL (GitHub, GitLab, Bitbucket) [required]
  --pr-id INTEGER        Pull Request ID (positive integer) [required]  
  --output-format CHOICE Output format: text, json, markdown [default: text]
  --verbose, -v          Enable verbose output
```

#### `demo` - Chạy Demo Workflow  
```bash
poetry run python cli.py demo [OPTIONS]

Options:
  --repo-url TEXT    Repository URL [required]
  --pr-id INTEGER    PR ID [required]
```

#### `validate` - Test Validation Functions
```bash
poetry run python cli.py validate
```

#### `fetch` - Repository Data Fetching
```bash
poetry run python cli.py fetch [OPTIONS]

Options:
  --repo-url TEXT    Repository URL [required]
  --pr-id INTEGER    PR ID để fetch diff (optional)
  --list-files       List files trong repository
  --get-info         Get repository information
```

**Examples:**
```bash
# Get repository information
poetry run python cli.py fetch --repo-url https://github.com/microsoft/vscode --get-info

# List repository files
poetry run python cli.py fetch --repo-url https://github.com/microsoft/vscode --list-files

# Get PR diff (GitHub only)
poetry run python cli.py fetch --repo-url https://github.com/microsoft/vscode --pr-id 123

# Combine multiple operations
poetry run python cli.py fetch --repo-url https://github.com/microsoft/vscode --get-info --list-files --pr-id 123
```

### Validation Rules

**Repository URL:**
- Phải là HTTPS URL hợp lệ
- Hỗ trợ: GitHub, GitLab, Bitbucket
- Format: `https://platform.com/owner/repo`

**PR ID:**
- Phải là số nguyên dương (> 0)
- Ví dụ: 1, 123, 9999

### Output Formats

**Text (default):**
```
📊 PR Analysis Results:
─────────────────────────
Repository: https://github.com/owner/repo
PR ID: 123
...
```

**JSON:**
```json
{
  "repository": "https://github.com/owner/repo",
  "pr_id": 123,
  "analysis": {...}
}
```

**Markdown:**
```markdown
# PR Analysis Report
**Repository:** https://github.com/owner/repo
**PR ID:** 123
...
```

## 🧪 Testing

Dự án bao gồm comprehensive test suite sử dụng pytest:

```bash
# Chạy tất cả tests
make test
# hoặc: poetry run pytest

# Chạy unit tests
make test-unit
# hoặc: poetry run pytest -m unit

# Chạy integration tests
make test-integration
# hoặc: poetry run pytest -m integration

# Chạy edge case tests
make test-edge
# hoặc: poetry run pytest -m edge_case

# Chạy tests với verbose output
make test-verbose

# Xem tất cả commands có sẵn
make help
```

### Test Coverage

#### Core Tests (tests/test_graph.py, test_graph_with_markers.py)
- **Unit Tests**: Individual agent functions và logic components
- **Integration Tests**: Full workflow end-to-end testing
- **Edge Cases**: Error handling và boundary conditions
- **State Validation**: Đảm bảo state được update correctly qua các steps

#### CLI Tests (tests/test_cli.py)
- **Validation Functions**: URL và PR ID validation
- **Command Testing**: All CLI commands với CliRunner
- **Output Formats**: JSON, Markdown, Text formatting
- **Interactive Mode**: Prompt-based input testing

#### CLI Edge Cases (tests/test_cli_edge_cases.py)
- **Missing Arguments**: Test thiếu required parameters
- **Invalid Combinations**: Multiple invalid arguments
- **Validation Edge Cases**: Special characters, unicode, malformed URLs
- **Error Handling**: Exception handling và proper cleanup
- **Interactive Scenarios**: Complex prompt testing

#### CLI Performance (tests/test_cli_performance.py)
- **Performance Testing**: Validation speed với large datasets
- **Stress Testing**: Large PR IDs, long URLs, rapid commands
- **Concurrency**: Concurrent command execution
- **Memory Usage**: Large output handling
- **Resource Cleanup**: Proper shutdown và cleanup

#### CodeFetcherAgent Tests (tests/test_code_fetcher*.py)
- **Basic Tests** (test_code_fetcher.py): Core functionality với real Git operations
- **Unit Tests** (test_code_fetcher_unit.py): Mocked GitPython calls, diff extraction, error handling
- **Advanced Tests** (test_code_fetcher_advanced.py): Complex scenarios, edge cases, performance
- **Total Coverage**: 62 comprehensive test cases covering:
  - Repository cloning và caching
  - PR diff extraction với detailed mocking
  - File content retrieval và encoding handling
  - Error scenarios và graceful degradation
  - Large repository handling (10,000+ files)
  - Unicode và binary file support
  - Multiple platform support (GitHub, GitLab, Bitbucket)
  - Workspace management và cleanup

### Test Commands

```bash
# Core functionality tests
make test                    # Tất cả tests
make test-unit              # Unit tests only  
make test-integration       # Integration tests only

# CLI-specific tests
make test-cli-all           # Tất cả CLI tests
make test-cli-edge          # CLI edge cases
make test-cli-performance   # CLI performance tests

# CodeFetcherAgent tests
make test-code-fetcher      # Basic functionality tests
make test-code-fetcher-unit # Unit tests với mocks
make test-code-fetcher-advanced # Advanced scenarios
make test-code-fetcher-all  # Tất cả CodeFetcherAgent tests

# Performance và detailed testing
poetry run pytest tests/test_cli_performance.py -v -s  # Với output
poetry run pytest -k "performance" -v                # Performance tests only
```

### Output Mẫu

```
🚀 Bắt đầu LangGraph Demo - Hai Agents Trò Chuyện
==================================================
🤖 Agent 1 đang xử lý message số 1
📝 Agent 1 nói: Xin chào từ Agent 1! (Lần thứ 1)
➡️  Chuyển đến agent_2
🦾 Agent 2 đang xử lý message số 2
📝 Agent 2 trả lời: Chào Agent 1! Tôi đã nhận được tin nhắn của bạn. (Lần thứ 2)
...
🏁 Kết thúc cuộc trò chuyện!

==================================================
📋 Tóm tắt cuộc trò chuyện:
1. Xin chào từ Agent 1! (Lần thứ 1)
2. Chào Agent 1! Tôi đã nhận được tin nhắn của bạn. (Lần thứ 2)
...

📊 Tổng số message: 5
✅ Demo hoàn thành!
```

## 🔄 CodeFetcherAgent

CodeFetcherAgent là một powerful agent để work với Git repositories:

### Features
- **Repository Cloning**: Clone từ GitHub, GitLab, Bitbucket
- **PR Diff Analysis**: Fetch và analyze Pull Request changes
- **File Operations**: List files, get file content
- **Error Handling**: Graceful handling của Git errors
- **Workspace Management**: Automatic cleanup của temporary files

### Usage Examples

```python
from agents.code_fetcher import CodeFetcherAgent

# Initialize agent
agent = CodeFetcherAgent()

# Get repository information
info = agent.get_repository_info("https://github.com/microsoft/vscode")
print(f"Repository: {info['full_name']}")
print(f"Latest commit: {info['latest_commit']['message']}")

# Get PR diff
pr_diff = agent.get_pr_diff("https://github.com/microsoft/vscode", 123)
print(f"Files changed: {len(pr_diff['files_changed'])}")
print(f"Additions: {pr_diff['stats']['additions']}")

# List repository files
files = agent.list_repository_files("https://github.com/microsoft/vscode")
print(f"Total files: {len(files)}")

# Always cleanup
agent.cleanup()
```

### CLI Integration
CodeFetcherAgent được integrate vào CLI:

```bash
# Test repository operations
make cli-fetch-info    # Get repository info
make cli-fetch-files   # List repository files

# Or directly
poetry run python cli.py fetch --repo-url https://github.com/microsoft/vscode --get-info --list-files
```

## 🔧 Tùy Chỉnh

Bạn có thể tùy chỉnh:

- **Số lượng message**: Thay đổi điều kiện `message_count >= 4` trong `agent_1` và `agent_2`
- **Nội dung message**: Sửa đổi logic trong các agent functions
- **Thêm agents**: Tạo thêm nodes và conditional edges
- **State structure**: Thêm fields vào `AgentState` trong `state.py`
- **CodeFetcherAgent**: Extend functionality để support more Git operations

## 📚 Tài Liệu

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)

## 🤝 Đóng Góp

Mọi đóng góp đều được chào đón! Hãy tạo issue hoặc pull request.
