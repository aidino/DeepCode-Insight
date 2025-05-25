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

## 🏃 Chạy Demo

```bash
# Sử dụng Poetry (khuyến khích)
poetry run python main.py

# Hoặc sử dụng Makefile
make demo

# Hoặc trực tiếp với Python
python main.py
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

Tests cover:
- **Unit Tests**: Individual agent functions và logic components
- **Integration Tests**: Full workflow end-to-end testing
- **Edge Cases**: Error handling và boundary conditions
- **State Validation**: Đảm bảo state được update correctly qua các steps

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

## 🔧 Tùy Chỉnh

Bạn có thể tùy chỉnh:

- **Số lượng message**: Thay đổi điều kiện `message_count >= 4` trong `agent_1` và `agent_2`
- **Nội dung message**: Sửa đổi logic trong các agent functions
- **Thêm agents**: Tạo thêm nodes và conditional edges
- **State structure**: Thêm fields vào `AgentState` trong `state.py`

## 📚 Tài Liệu

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)

## 🤝 Đóng Góp

Mọi đóng góp đều được chào đón! Hãy tạo issue hoặc pull request.
