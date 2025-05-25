# 🚀 End-to-End Test Guide cho DeepCode-Insight LangGraph Workflow

## Tổng quan

End-to-end test script được thiết kế để kiểm tra toàn bộ LangGraph workflow từ input PR URL đến việc tạo ra Markdown report. Test này đảm bảo rằng tất cả các agents hoạt động đúng cách và state được quản lý chính xác.

## 📁 Files Test

### 1. `tests/test_end_to_end_simple.py`
**Test đơn giản và nhanh chóng**
- ✅ Test workflow với PR URL (mocked)
- ✅ Test workflow với local code content
- ✅ Verification đầy đủ của state và output
- ✅ Mock tất cả external dependencies

### 2. `tests/test_end_to_end_workflow.py`
**Test comprehensive với nhiều scenarios**
- ✅ Test complete workflow với PR URL
- ✅ Test error handling scenarios
- ✅ Test performance metrics
- ✅ Detailed verification và reporting

### 3. `run_end_to_end_test.py`
**CLI runner script**
- ✅ Command-line interface để chạy tests
- ✅ Multiple test modes (basic, all, performance)
- ✅ Verbose output options

## 🎯 Cách Chạy Tests

### Option 1: Test Đơn Giản (Khuyến nghị)

```bash
# Chạy test đơn giản với pytest
python -m pytest tests/test_end_to_end_simple.py -v

# Hoặc chạy trực tiếp
python tests/test_end_to_end_simple.py
```

### Option 2: Test Comprehensive

```bash
# Chạy test comprehensive với pytest
python -m pytest tests/test_end_to_end_workflow.py -v -s

# Chạy specific test
python -m pytest tests/test_end_to_end_workflow.py::TestEndToEndWorkflow::test_complete_end_to_end_workflow_with_pr_url -v -s
```

### Option 3: CLI Runner

```bash
# Basic end-to-end test
python run_end_to_end_test.py

# Verbose mode
python run_end_to_end_test.py --verbose

# All test scenarios
python run_end_to_end_test.py --all-tests

# Performance test only
python run_end_to_end_test.py --performance

# Custom output directory
python run_end_to_end_test.py --output-dir custom_reports
```

## 🔍 Test Scenarios

### 1. PR URL Workflow Test
**Mô tả:** Test complete workflow từ PR URL đến report generation

**Input:**
- Repository URL: `https://github.com/test-user/sample-repo`
- PR ID: `123`
- Target file: `src/calculator.py`

**Mocked Components:**
- CodeFetcherAgent (repository operations)
- LLMOrchestratorAgent (LLM analysis)

**Verification:**
- ✅ Workflow completion status
- ✅ State progression through all agents
- ✅ Input data preservation
- ✅ Code fetching results
- ✅ Static analysis results
- ✅ Report generation và file creation
- ✅ Report content structure
- ✅ Agent interactions

### 2. Local Code Analysis Test
**Mô tả:** Test workflow với local code content (không cần repository)

**Input:**
- Code content trực tiếp
- Filename: `test_local.py`

**Verification:**
- ✅ Workflow completion
- ✅ Static analysis
- ✅ Report generation

### 3. Error Handling Tests
**Mô tả:** Test các error scenarios

**Scenarios:**
- Invalid repository URL
- Empty repository (no Python files)
- LLM service unavailable (graceful degradation)

### 4. Performance Test
**Mô tả:** Test performance metrics

**Metrics:**
- Execution time < 10 seconds
- Report size > 500 characters
- Memory usage tracking

## 📊 Expected Output

### Successful Test Run
```
🚀 Starting End-to-End Workflow Test
============================================================
📂 Repository: https://github.com/test-user/sample-repo
🔀 PR ID: 123
📄 Target file: src/calculator.py

🔄 Running complete workflow...
🤖 UserInteractionAgent: Processing input...
✅ UserInteractionAgent: Input validated successfully
🔄 CodeFetcherAgent: Fetching code...
✅ CodeFetcherAgent: Successfully fetched code from src/calculator.py
🔍 StaticAnalysisAgent: Analyzing code...
✅ StaticAnalysisAgent: Analysis completed
🤖 LLMOrchestratorAgent: Processing with LLM...
⚠️ Warning: LLM service not available, skipping LLM analysis
📊 ReportingAgent: Generating report...
✅ ReportingAgent: Report generated successfully
📄 Report saved to: analysis_reports/report_calculator_20250525_205822.md
⏱️ Workflow completed

📋 Verifying Results:
------------------------------
✅ Workflow marked as finished
✅ Processing status: report_generated
✅ Code content preserved
✅ Static analysis completed
✅ Report file created: report_calculator_20250525_205822.md
✅ Report content valid (1007 characters)
✅ All agent interactions verified

🎉 End-to-End Test PASSED!
📁 Report saved to: /tmp/tmpXXXXXX/report_calculator_20250525_205822.md
```

### Generated Report Structure
```markdown
# 📊 Code Analysis Report

**File:** `src/calculator.py`  
**Generated:** 2025-05-25 20:58:22  
**Analysis Tool:** DeepCode-Insight  

---

## 🔍 Static Analysis Results

### 📈 Code Metrics
| Metric | Value |
|--------|-------|
| Cyclomatic Complexity | 0 |
| Maintainability Index | 66.03 |
| Code Quality Score | 66.03 |
| Lines Of Code | 19 |
| Comment Ratio | 0.00 |
| Function To Class Ratio | 4.00 |

### ⚠️ Issues Found
#### Missing Docstrings
- **missing_function_docstring** in `calculate_average` (line 17) - Function 'calculate_average' thiếu docstring

### 💡 Suggestions
- Thêm docstrings cho 1 functions/classes để cải thiện documentation
- Thêm comments để giải thích logic phức tạp

---

## 📝 Report Information
- **Generated by:** DeepCode-Insight ReportingAgent
- **Analysis Date:** 2025-05-25 20:58:22
- **File Analyzed:** `src/calculator.py`

*This report was automatically generated. Please review findings and recommendations carefully.*
```

## 🔧 Test Configuration

### Mock Configuration
```python
# Sample PR data for testing
sample_pr_data = {
    'repo_url': 'https://github.com/test-user/sample-repo',
    'pr_id': '123',
    'target_file': 'src/calculator.py',
    'repository_info': {
        'full_name': 'test-user/sample-repo',
        'platform': 'github',
        'owner': 'test-user',
        'repo_name': 'sample-repo'
    },
    'pr_diff': {
        'files_changed': ['src/calculator.py'],
        'stats': {'additions': 45, 'deletions': 12, 'files': 1}
    },
    'code_content': '''
class Calculator:
    """A simple calculator class."""
    
    def add(self, a, b):
        """Add two numbers."""
        return a + b
    
    def subtract(self, a, b):
        """Subtract second number from first."""
        return a - b

def very_long_function_name_that_exceeds_the_recommended_line_length_limit():
    """This function name is too long."""
    pass

def calculate_average(numbers):
    return sum(numbers) / len(numbers) if numbers else 0
'''
}
```

### Agent State Configuration
```python
initial_state: AgentState = {
    **DEFAULT_AGENT_STATE,
    'repo_url': sample_pr_data['repo_url'],
    'pr_id': sample_pr_data['pr_id'],
    'target_file': sample_pr_data['target_file'],
    'config': {
        'output_dir': temp_dir,
        'test_mode': True
    }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure project root is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Missing Dependencies**
   ```bash
   pip install langgraph pytest tree_sitter
   ```

3. **Mock Not Working**
   - Kiểm tra import paths trong patch statements
   - Đảm bảo mock được apply đúng scope

4. **Test Timeout**
   - Kiểm tra LLM service availability
   - Sử dụng mock để skip LLM calls

### Debug Mode
```bash
# Run với debug output
python -m pytest tests/test_end_to_end_simple.py -v -s --tb=long

# Run với pdb debugger
python -m pytest tests/test_end_to_end_simple.py --pdb
```

## 📈 Test Metrics

### Coverage Areas
- ✅ UserInteractionAgent: Input validation
- ✅ CodeFetcherAgent: Repository operations (mocked)
- ✅ StaticAnalysisAgent: Code analysis
- ✅ LLMOrchestratorAgent: LLM integration (mocked)
- ✅ ReportingAgent: Report generation
- ✅ State management: Immutable updates
- ✅ Error handling: Graceful degradation
- ✅ File I/O: Report file creation

### Performance Benchmarks
- Workflow execution: < 10 seconds
- Report generation: < 1 second
- Memory usage: < 100MB
- File size: > 500 characters

## 🎯 Next Steps

1. **Extend Test Coverage**
   - Add more code samples
   - Test different programming languages
   - Test larger repositories

2. **Integration Testing**
   - Test with real repositories
   - Test with actual LLM services
   - Test with different configurations

3. **Performance Optimization**
   - Benchmark different scenarios
   - Optimize agent interactions
   - Improve state management

4. **CI/CD Integration**
   - Add to GitHub Actions
   - Automated testing on PRs
   - Performance regression detection

---

**Lưu ý:** End-to-end tests sử dụng mocking để đảm bảo tests chạy nhanh và không phụ thuộc vào external services. Để test với real data, sử dụng integration tests riêng biệt. 