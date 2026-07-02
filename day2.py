# ============================================================
# app.py
# 作用：把 RAG 问答包装成网页界面
# 用户上传文档 → 建知识库 → 提问 → 看到回答
#
# 运行：streamlit run app.py
# ============================================================

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ── streamlit：用来做网页 ──
import streamlit as st

# ── 文档加载和切分 ──
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 向量化 + 向量库 ──
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── 大模型 ──
from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ========== 页面设置 ==========
st.set_page_config(page_title="RAG 知识库问答", page_icon="📚")
st.title("📚 RAG 知识库问答系统")
st.caption("上传 TXT 文件 → AI 自动建知识库 → 向 AI 提问")

# ========== 初始化状态 ==========
# st.session_state 是 Streamlit 的"记忆仓库"
# 每次交互网页都会重新运行，但 session_state 里的东西不会丢
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

# ========== 侧边栏：上传文件 ==========
with st.sidebar:
    st.header("📤 上传文档")

    # st.file_uploader：文件上传组件
    # accept_multiple_files=True：允许一次上传多个文件
    # type：限制只接受 txt 文件
    uploaded_files = st.file_uploader(
        "选择 TXT 文件",
        type="txt",
        accept_multiple_files=True
    )

    # 如果用户上传了文件
    if uploaded_files:
        with st.spinner("⏳ 正在建知识库……"):
            # ── 1. 读取所有文件 ──
            all_docs = []
            for uploaded_file in uploaded_files:
                # 把上传的文件内容读成字符串
                content = uploaded_file.read().decode("utf-8")
                # 存成临时文件让 TextLoader 能读
                temp_path = f"./temp_{uploaded_file.name}"
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write(content)

                loader = TextLoader(temp_path, encoding="utf-8")
                docs = loader.load()
                all_docs.extend(docs)

                # 删掉临时文件
                os.remove(temp_path)

            # ── 2. 切分 ──
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
            )
            chunks = splitter.split_documents(all_docs)

            # ── 3. 向量化 + 存库 ──
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
            st.session_state.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            # 更新状态
            st.session_state.db_ready = True
            st.session_state.messages = []

            # 显示成功消息
            st.success(f"✅ 建库完成！共 {len(chunks)} 个片段")

    # 重置按钮
    if st.session_state.db_ready:
        if st.button("🔄 重新开始"):
            st.session_state.vectorstore = None
            st.session_state.messages = []
            st.session_state.db_ready = False
            st.rerun()

# ========== 欢迎信息 ==========
if not st.session_state.db_ready:
    st.info("👈 请在左侧上传 TXT 文件开始")
    st.markdown("""
    **使用说明：**
    1. 点击左侧 **"选择 TXT 文件"** 上传一个或多个文本文件
    2. 等待系统自动建知识库
    3. 在底部输入框提问
    """)

# ========== 聊天区域 ==========
# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ========== 提问区域 ==========
if st.session_state.db_ready:
    if query := st.chat_input("💬 请输入你的问题……"):

        # 显示用户的问题
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # AI 回答
        with st.chat_message("assistant"):
            with st.spinner("🤔 正在检索并思考……"):
                try:
                    # R：检索
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    )
                    docs = retriever.invoke(query)
                    context = "\n\n".join([d.page_content for d in docs])

                    # A：增强
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

                    # G：生成
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