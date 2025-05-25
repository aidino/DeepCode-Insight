# 🔑 API Keys Setup Guide

## 📋 Tổng quan

File `.env` đã được tạo với template. Bạn cần thêm API keys thực tế để sử dụng full functionality của RAGContextAgent.

## 🔧 Cách thêm API Keys

### 1. Edit file .env

```bash
# Mở file .env với editor yêu thích
nano .env
# hoặc
code .env
# hoặc
vim .env
```

### 2. Thay thế placeholder values

#### OpenAI API Key (Required)
```bash
# Thay thế dòng này:
OPENAI_API_KEY=your_openai_api_key_here

# Bằng API key thực tế:
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Cách lấy OpenAI API Key:**
1. Đi tới https://platform.openai.com/api-keys
2. Đăng nhập vào tài khoản OpenAI
3. Click "Create new secret key"
4. Copy key và paste vào file .env

#### Google AI API Key (Optional)
```bash
# Thay thế dòng này:
GOOGLE_API_KEY=your_google_api_key_here

# Bằng API key thực tế:
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Cách lấy Google AI API Key:**
1. Đi tới https://makersuite.google.com/app/apikey
2. Đăng nhập vào Google account
3. Click "Create API key"
4. Copy key và paste vào file .env

## 🧪 Test API Keys

### 1. Test Configuration
```bash
python config.py
```

Kết quả mong đợi:
```
🔧 DeepCode-Insight Configuration:
  OpenAI API Key: ✅ Set
  Google API Key: ✅ Set
  ...
✅ Configuration is valid
```

### 2. Test OpenAI Connection
```bash
python -c "
from config import config
from openai import OpenAI
client = OpenAI(api_key=config.OPENAI_API_KEY)
response = client.embeddings.create(model='text-embedding-3-small', input='test')
print(f'✅ OpenAI API working! Embedding dimension: {len(response.data[0].embedding)}')
"
```

### 3. Test Google AI Connection (Optional)
```bash
python -c "
from config import config
import google.generativeai as genai
genai.configure(api_key=config.GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content('Hello')
print('✅ Google AI API working!')
"
```

## 🚀 Run Real Data Tests

Sau khi setup API keys, bạn có thể chạy real data tests:

```bash
# Test với real OpenAI embeddings
python test_rag_real_data.py

# Quick start demo
python quick_start.py

# Interactive setup (nếu muốn thay đổi config)
python setup_env.py
```

## 🔒 Security Notes

### File .env đã được bảo vệ:
- ✅ Đã thêm vào `.gitignore` - không bị commit lên Git
- ✅ Chỉ readable bởi user hiện tại
- ✅ Config system mask sensitive data khi print

### Best Practices:
- **Không share** file `.env` với ai khác
- **Không commit** API keys lên Git
- **Regenerate keys** nếu bị lộ
- **Use environment variables** trong production

## 🐛 Troubleshooting

### API Key không được nhận diện:
```bash
# Kiểm tra file .env có tồn tại
ls -la .env

# Kiểm tra format của API key
cat .env | grep API_KEY

# Test config loading
python -c "from config import config; print(config.OPENAI_API_KEY[:10] + '...')"
```

### OpenAI API Errors:
- **Invalid API key**: Kiểm tra key có đúng format `sk-proj-...`
- **Rate limit**: Đợi một chút rồi thử lại
- **Quota exceeded**: Kiểm tra billing trong OpenAI dashboard

### Google AI API Errors:
- **Invalid API key**: Kiểm tra key có đúng format `AIzaSy...`
- **API not enabled**: Enable Generative AI API trong Google Cloud Console

## 📊 Expected Results

Sau khi setup thành công, bạn sẽ thấy:

### Config Status:
```
🔧 DeepCode-Insight Configuration:
  OpenAI API Key: ✅ Set
  Google API Key: ✅ Set
  ...
✅ Configuration is valid
```

### Real Data Test Results:
```
🧪 === Testing RAGContextAgent with Real Data ===

✅ OpenAI API working (embedding dimension: 1536)
✅ Real embeddings: 1536 dimensions
✅ Complex code indexing: Python + Java
✅ Semantic queries: 8 test queries
✅ LLM responses: Context-aware generation
✅ Performance: Sub-second queries
✅ Quality: High relevance scores

🎉 Real data testing completed successfully!
```

## 🎯 Next Steps

1. **✅ Setup API Keys** - Thêm OpenAI và Google AI keys
2. **🧪 Run Tests** - Verify everything works
3. **📚 Index Code** - Start indexing your repositories
4. **🔍 Query & Explore** - Use semantic search on your codebase
5. **🚀 Build Applications** - Create your own RAG-powered tools

Happy coding! 🎉 