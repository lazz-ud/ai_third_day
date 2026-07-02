import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ── 1. 加载向量数据库 ──
print("正在加载向量数据库……")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
print("✅ 向量数据库加载完成\n")

# ── 2. 创建检索器 ──
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── 3. 创建大模型 ──
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model="deepseek-v4-pro",
    temperature=0.3
)

# ── 4. 创建 Prompt 模板 ──
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识库问答助手。请根据以下资料回答用户的问题。"
               "如果资料里没有相关信息，就说'资料中没有提及'。"),
    ("user", "资料：\n{context}\n\n问题：{question}")
])

# ── 5. 问答函数 ──
def ask(question):
    """接收问题 → 检索 → 拼 Prompt → 调 AI → 返回答案"""
    # R：检索
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    # A：增强（拼 Prompt）
    messages = prompt.format_messages(context=context, question=question)

    # G：生成
    result = llm.invoke(messages)
    return result.content

# ── 6. 开始问答 ──
print("知识库已就绪！你可以提问了（输入 q 退出）\n")

while True:
    question = input("\n💬 你的问题：")
    if question.lower() == "q":
        break

    print("🤔 正在检索并思考……")
    answer = ask(question)
    print(f"\n🤖 回答：{answer}")