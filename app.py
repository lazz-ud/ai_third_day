import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="RAG 知识库问答", page_icon="📚")
st.title("📚 RAG 知识库问答系统")
st.caption("上传 TXT 文件 → AI 自动建知识库 → 向 AI 提问")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

with st.sidebar:
    st.header("📤 上传文档")
    uploaded_files = st.file_uploader("选择 TXT 文件", type="txt", accept_multiple_files=True)

    # ── chunk_size 滑动条 ──
    chunk_size = st.slider(
        "分块大小（chunk_size）",
        min_value=100, max_value=1000, value=500, step=100,
        help="每块的字数。越小检索越精确，越大上下文越完整"
    )

    if uploaded_files:
        with st.spinner("⏳ 正在建知识库……"):
            all_docs = []
            for uploaded_file in uploaded_files:
                content = uploaded_file.read().decode("utf-8")
                temp_path = f"./temp_{uploaded_file.name}"
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(content)
                loader = TextLoader(temp_path, encoding="utf-8")
                docs = loader.load()
                all_docs.extend(docs)
                os.remove(temp_path)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=50,
                separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
            )
            chunks = splitter.split_documents(all_docs)

            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
            st.session_state.vectorstore = Chroma.from_documents(
                documents=chunks, embedding=embeddings
            )
            st.session_state.db_ready = True
            st.session_state.messages = []
            st.success(f"✅ 建库完成！共 {len(chunks)} 个片段（chunk_size={chunk_size}）")

    if st.session_state.db_ready:
        if st.button("🔄 重新开始"):
            st.session_state.vectorstore = None
            st.session_state.messages = []
            st.session_state.db_ready = False
            st.rerun()

if not st.session_state.db_ready:
    st.info("👈 请在左侧上传 TXT 文件开始")
    st.markdown("""
    **使用说明：**
    1. 点击左侧 **"选择 TXT 文件"** 上传一个或多个文本文件
    2. 拖动 **chunk_size 滑动条** 调节分块大小
    3. 等待系统自动建知识库
    4. 在底部输入框提问
    """)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.db_ready:
    if query := st.chat_input("💬 请输入你的问题……"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("🤔 正在检索并思考……"):
                try:
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    )
                    docs = retriever.invoke(query)
                    context = "\n\n".join([d.page_content for d in docs])

                    llm = ChatOpenAI(
                        api_key=os.getenv("API_KEY"),
                        base_url=os.getenv("BASE_URL"),
                        model="deepseek-v4-pro",
                        temperature=0.3
                    )

                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "根据以下资料回答用户的问题。如果没有相关信息，就说'资料中没有提及'。"),
                        ("user", "资料：\n{context}\n\n问题：{question}")
                    ])
                    messages = prompt.format_messages(context=context, question=query)

                    result = llm.invoke(messages)
                    answer = result.content

                    st.markdown(answer)

                    with st.expander("📎 查看参考来源"):
                        for i, doc in enumerate(docs):
                            st.markdown(f"**来源 {i + 1}：**")
                            st.markdown(doc.page_content)
                            st.markdown("---")

                except Exception as e:
                    answer = f"❌ 出错了：{str(e)}"
                    st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
