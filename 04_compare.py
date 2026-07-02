import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv

load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model="deepseek-v4-pro",
    temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", "根据资料回答问题，没有就说没有。"),
    ("user", "资料：\n{context}\n\n问题：{question}")
])

question = "Python 是什么？"

# 对比 k=1, k=3, k=5 的效果
for k in [1, 3, 5]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    messages = prompt.format_messages(context=context, question=question)
    result = llm.invoke(messages)

    print(f"\n{'=' * 50}")
    print(f"k={k}（检索了 {len(docs)} 条）")
    print(f"{'=' * 50}")
    print(f"回答：{result.content[:200]}")