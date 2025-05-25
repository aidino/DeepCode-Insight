
## Tổng quan Quy trình làm việc với Cursor AI

Trước khi đi vào chi tiết, đây là cách chúng ta sẽ sử dụng Cursor trong suốt dự án:

1.  **Bắt đầu Ngày/Nhiệm vụ:**
    * Mở Cursor, mở dự án.
    * **Chat với Cursor:** "Cursor, hôm nay chúng ta sẽ [Mục tiêu nhiệm vụ, ví dụ: triển khai `ASTParsingAgent` cho Java]. Hãy xem lại các tệp `@ASTParsingAgent` và `@tree_sitter_wrapper.py`."
2.  **Viết Code:**
    * **Yêu cầu Tạo Mã:** `Cursor: Generate the basic structure for the ASTParsingAgent's Java parsing method.`
    * **Hoàn thiện Tự động (AI Autocomplete):** Tận dụng gợi ý mã thông minh của Cursor.
    * **Sử dụng Snippets/Prompts Tùy chỉnh:** Gọi các prompt đã lưu để tạo các cấu trúc lặp lại.
3.  **Hiểu Code:**
    * **Chọn đoạn mã & Chat:** "Cursor, giải thích đoạn mã này làm gì và tại sao nó sử dụng `tree-sitter` queries."
    * **Hỏi về Tệp/Thư mục:** `Cursor: Tệp @agents/state.py đóng vai trò gì trong luồng LangGraph?`
4.  **Gỡ lỗi:**
    * **Dán Lỗi & Chat:** `Cursor: Tôi gặp lỗi [Dán lỗi] khi chạy test cho @CodeFetcherAgent. Hãy xem mã nguồn và gợi ý cách sửa.`
5.  **Tái cấu trúc (Refactor):**
    * **Chọn đoạn mã & Chat:** `Cursor: Refactor this function to be more modular and add error handling.`
6.  **Viết Test:**
    * **Yêu cầu Tạo Test:** `Cursor: Generate pytest unit tests for the @DiagramGenerationAgent class, mocking PlantUML calls.`
7.  **Sử dụng `@` liên tục:** Đây là "quy tắc" quan trọng nhất để giữ cho Cursor luôn "đồng bộ" với suy nghĩ của bạn.

---

## Roadmap Chi tiết theo Từng Giai đoạn (Tập trung vào Cursor AI)

### Giai đoạn 1: POC - Bộ máy Cốt lõi & Python (3-4 tháng)

* **Mục tiêu:** Xây dựng nền tảng, xác thực kiến trúc với Python, và thành thạo quy trình làm việc cơ bản với Cursor.

| Tính năng / Tác tử | Quy trình Làm việc với Cursor AI | Kiểm thử (với sự hỗ trợ của Cursor) |
| :--- | :--- | :--- |
| **Setup & LangGraph** | `Cursor: Initialize a new Python project using Poetry. Install LangGraph. Generate a basic LangGraph graph (state.py, graph.py) showing how two agents pass a message.` | `Cursor: Generate a simple pytest script to run the basic graph and assert that the state is updated correctly.` |
| **UserInteractionAgent (CLI)** | `Cursor: Using Python's 'click', create a CLI (cli.py) that accepts 'repo_url' and 'pr_id'. Add validation to ensure inputs are provided.` | `Cursor: Generate pytest tests for cli.py using 'CliRunner', testing valid inputs and error cases (missing/invalid arguments).` |
| **CodeFetcherAgent** | `Cursor: Create agents/code_fetcher.py. Implement a class 'CodeFetcherAgent' using 'GitPython' to clone/fetch and get the diff for a PR ID. Use @cli.py as input reference. Handle potential Git errors.` | `Cursor: Generate unit tests for @CodeFetcherAgent. Mock 'GitPython' calls to test diff extraction and error handling without actual network calls.` |
| **ASTParsingAgent (Python)** | `Cursor: Create parsers/ast_parser.py. Implement 'ASTParsingAgent' using 'tree-sitter' and 'tree-sitter-python'. It needs a method to parse Python code string into an AST. Use @CodeFetcherAgent output as reference.` | `Cursor: Generate tests for @ASTParsingAgent. Provide sample Python code strings (valid, invalid) and assert that the AST is generated or errors are handled.` |
| **StaticAnalysisAgent (Python)** | `Cursor: Create agents/static_analyzer.py. Implement 'StaticAnalysisAgent'. Generate 2-3 Tree-sitter queries for Python (e.g., missing docstring) and write Python code within the agent to run these queries on an AST from @ASTParsingAgent.` | `Cursor: Generate tests for @StaticAnalysisAgent. Provide sample ASTs and assert that the correct findings (or lack thereof) are reported.` |
| **LLM Integration (Ollama)** | `Cursor: Create utils/llm_caller.py. Write a function to call a local Ollama (CodeLlama) API via HTTP, sending a prompt and code snippet. Handle API keys (env vars) and errors.` | `Cursor: Generate tests for @llm_caller.py. Use 'pytest-mock' to mock the HTTP request/response and test prompt formatting and error handling.` |
| **LLMOrchestratorAgent** | `Cursor: Create agents/llm_orchestrator.py. Implement 'LLMOrchestratorAgent' as a LangGraph node. It should take findings from @StaticAnalysisAgent, format a summary prompt, and call @llm_caller.py.` | `Cursor: Generate tests for @LLMOrchestratorAgent. Mock the LLM call and test if it correctly processes input state and formats the prompt.` |
| **ReportingAgent** | `Cursor: Create agents/reporter.py. Implement 'ReportingAgent' to take a list of findings and LLM summaries, then generate a simple Markdown report file.` | `Cursor: Generate tests for @ReporterAgent. Provide sample findings and check if the generated Markdown file has the correct structure and content.` |
| **Tích hợp LangGraph** | `Cursor: In graph.py, define the Pydantic state model (using @agents/state.py). Connect all agents (@UserInteractionAgent -> @CodeFetcherAgent -> ... -> @ReporterAgent) into a LangGraph workflow. Explain how state is managed.` | **Kiểm thử Tích hợp:** `Cursor: Help me write an end-to-end test script that runs the entire LangGraph with a sample PR URL, and checks if a Markdown report is generated.` |

---

### Giai đoạn 2: Phân tích Nâng cao & Sơ đồ Lớp (4-5 tháng)

* **Mục tiêu:** Nâng cao chất lượng phân tích, thêm giải pháp, tạo sơ đồ lớp và hỗ trợ Java cơ bản.

| Tính năng / Tác tử | Quy trình Làm việc với Cursor AI | Kiểm thử (với sự hỗ trợ của Cursor) |
| :--- | :--- | :--- |
| **StaticAnalysisAgent (Mở rộng)** | `Cursor: Help me add more Python rules to @StaticAnalysisAgent based on Google Style Guide. Also, generate the setup to handle Java code using 'tree-sitter-java' and add 2 basic Java rules.` | `Cursor: Generate new test cases for @StaticAnalysisAgent covering the new Python and Java rules.` |
| **RAGContextAgent** | `Cursor: Set up Qdrant using Docker. Create agents/rag_context.py. Implement 'RAGContextAgent' using 'LlamaIndex' to chunk code (@ASTParsingAgent) and index it into Qdrant. Add a query method.` | `Cursor: Generate tests for @RAGContextAgent. Mock 'LlamaIndex' and 'Qdrant' clients to test indexing and querying logic.` |
| **LLMOrchestratorAgent (Nâng cao)** | `Cursor: Refactor @LLMOrchestratorAgent to (1) call @RAGContextAgent for context, (2) build a Chain-of-Thought prompt for solution generation, (3) add support for OpenAI/Gemini APIs via an abstract interface.` | `Cursor: Generate tests for the new RAG integration and multi-LLM support in @LLMOrchestratorAgent, mocking RAG and LLM calls.` |
| **SolutionSuggestionAgent** | `Cursor: Create agents/solution_suggester.py. This agent takes raw LLM solutions and uses a 'Refine this suggestion' prompt to make them clearer and more actionable.` | `Cursor: Generate tests for @SolutionSuggestionAgent. Provide raw LLM outputs and expected refined outputs (can be subjective, focus on structure).` |
| **DiagramGenerationAgent (Sơ đồ Lớp)** | `Cursor: Create agents/diagram_generator.py. Implement a method using @ASTParsingAgent (Python/Java) to extract class names, fields, methods, and inheritance. Generate PlantUML text for a class diagram.` | `Cursor: Generate tests for @DiagramGenerationAgent. Provide sample ASTs and assert that the generated PlantUML text matches the expected output.` |
| **Hỗ trợ Java Cơ bản** | `Cursor: Ensure @ASTParsingAgent, @StaticAnalysisAgent, and @DiagramGenerationAgent now correctly handle Java code based on the 'tree-sitter-java' integration.` | **Kiểm thử Tích hợp (Java):** `Cursor: Help me create an end-to-end test using a sample Java PR to verify basic analysis and class diagram generation.` |

---

### Giai đoạn 3: Quét Toàn bộ Dự án & Tính năng Nâng cao (5-6 tháng)

* **Mục tiêu:** Quét toàn bộ dự án, tạo sơ đồ tuần tự, hỗ trợ Kotlin/Android, và dự đoán rủi ro.

| Tính năng / Tác tử | Quy trình Làm việc với Cursor AI | Kiểm thử (với sự hỗ trợ của Cursor) |
| :--- | :--- | :--- |
| **ProjectScanningAgent** | `Cursor: Create agents/project_scanner.py. Design its LangGraph flow. It needs to list files, manage context across many files (perhaps using hierarchical summaries via @LLMOrchestratorAgent), and aggregate findings.` | `Cursor: How can I effectively test @ProjectScanningAgent on a medium-sized project? Suggest strategies for mocking file systems and managing test data.` |
| **ImpactAnalysisAgent** | `Cursor: Create agents/impact_analyzer.py. Using diffs and ASTs (before/after), it should find changed elements. Use Tree-sitter queries to find references and identify affected components.` | `Cursor: Generate tests for @ImpactAnalysisAgent. Provide sample diffs/ASTs and a small project structure, then assert the identified impacts.` |
| **DiagramGenerationAgent (Sơ đồ Tuần tự & Đánh dấu)** | `Cursor: Enhance @DiagramGenerationAgent. Add a method to trace calls from modified functions (using AST) and generate PlantUML sequence diagrams. Modify output to add colors/tags based on @ImpactAnalysisAgent data.` | `Cursor: Generate tests to verify sequence diagram generation and that changes/impacts are correctly highlighted in both class and sequence diagrams.` |
| **Hỗ trợ Kotlin/Android** | `Cursor: Integrate 'tree-sitter-kotlin' and 'tree-sitter-xml'. Add logic to @ASTParsingAgent. Add 2-3 specific Android rules to @StaticAnalysisAgent (e.g., manifest issues, layout warnings).` | `Cursor: Generate tests for Kotlin/Android parsing and rule checking using sample Android project files.` |
| **Mô hình Dự đoán Rủi ro** | `Cursor: Integrate 'radon' into a new agent or @ProjectScanningAgent. Combine complexity metrics with static findings and LLM assessments to calculate a simple risk score.` | `Cursor: Generate tests for the risk score calculation logic using various input metrics.` |
| **Tối ưu Ngữ cảnh Lớn** | `Cursor: How can I implement hierarchical summarization effectively using @LLMOrchestratorAgent to manage context for full project scans? Show an example prompt structure.` | **Kiểm thử Hiệu năng:** `Cursor: Suggest ways to benchmark the performance and memory usage of the full project scan.` |

---

### Giai đoạn 4: Mở rộng, Tối ưu & Phản hồi (Liên tục)

* **Mục tiêu:** Hỗ trợ nhiều ngôn ngữ hơn, cải thiện hiệu suất, và tạo vòng lặp cải tiến.

| Tính năng / Tác tử | Quy trình Làm việc với Cursor AI | Kiểm thử (với sự hỗ trợ của Cursor) |
| :--- | :--- | :--- |
| **Hỗ trợ Ngôn ngữ Mới (ví dụ: JS)** | `Cursor: Integrate 'tree-sitter-javascript'. Add basic JS rules to @StaticAnalysisAgent based on popular linters.` | `Cursor: Generate tests for JavaScript parsing and analysis.` |
| **Tối ưu hóa Hiệu suất** | `Cursor: Implement Redis caching for @LLMOrchestratorAgent. Refactor the LangGraph to allow batching parallel API calls.` | `Cursor: Help me set up benchmarking tests using 'pytest-benchmark' to measure performance before and after optimization.` |
| **Cơ chế Phản hồi** | `Cursor: Generate a simple Flask/FastAPI endpoint (api.py) to receive user feedback (ratings/comments) from reports and save it to a PostgreSQL database.` | `Cursor: Generate tests for the @api.py endpoint.` |
| **Cải thiện Báo cáo (XAI)** | `Cursor: Enhance the HTML report using Mermaid.js. Modify @LLMOrchestratorAgent prompts to ask for 'Chain-of-Thought' explanations and display them in the report.` | `Cursor: How can I test the rendering and interactivity of the new HTML/Mermaid.js report? Suggest tools or strategies.` |
| **MLOps & Bảo trì** | `Cursor: Implement structured logging (e.g., using 'loguru') across all agents. Add utilities to track LLM costs.` | `Cursor: Help me design a regression testing strategy for when I update LLM models or Tree-sitter grammars.` |

.