# RAG 知识库问答系统

上传 TXT 文件，AI 自动建知识库，基于文档内容回答问题。

## 功能

- 上传多个 TXT 文件
- 自动切分、向量化、存入知识库
- 基于文档内容进行问答
- 显示引用来源
- 可调分块大小（chunk_size）

## 技术栈

- Python 3.12
- Streamlit（网页界面）
- LangChain（AI 流程编排）
- ChromaDB（向量数据库）
- HuggingFace Embeddings（文本向量化）
- DeepSeek API（大模型）

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 配置环境变量

在项目根目录创建 `.env` 文件：

```
API_KEY=你的DeepSeek API Key
BASE_URL=https://api.deepseek.com
```

3. 启动应用

```bash
streamlit run app.py
```

4. 在浏览器中上传 TXT 文件开始提问

## 项目结构

```
├── app.py              # Streamlit 主程序
├── requirements.txt    # 依赖列表
├── .env                # 环境变量
├── .gitignore          # Git 忽略规则
├── docs/               # 测试文档
└── chroma_db/          # 向量数据库
```
