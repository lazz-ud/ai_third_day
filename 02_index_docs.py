# ============================================================
# 02_index_docs.py
#
# 作用：读取 docs 文件夹里的所有 TXT 文件
#       切分成小块 → 向量化 → 存入 ChromaDB
#
# 运行后：你就有了一套从本地文档构建的知识库
# ============================================================

import os

# 设置 HuggingFace 镜像（防止模型下载失败）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ── TextLoader：读取文本文件 ──
# LangChain 的文档加载器，专门用来读 .txt 文件
from langchain_community.document_loaders import TextLoader

# ── RecursiveCharacterTextSplitter：文本切分器 ──
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── HuggingFaceEmbeddings：把文字转成向量 ──
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Chroma：向量数据库 ──
from langchain_community.vectorstores import Chroma

# ── 第一步：找到 docs 文件夹里的所有 txt 文件 ──
docs_folder = "./docs"

# os.listdir：列出文件夹里的所有文件名
# 用列表推导式只保留以 .txt 结尾的文件
txt_files = [f for f in os.listdir(docs_folder) if f.endswith(".txt")]

print(f"找到 {len(txt_files)} 个文件：")
for f in txt_files:
    print(f"  - {f}")

# ── 第二步：读取所有文件 ──
# 用 TextLoader 逐个读取 .txt 文件
# loader.load() 返回一个 Document 对象列表
# 把所有文件的内容合并到 all_docs 列表里
all_docs = []
for filename in txt_files:
    filepath = os.path.join(docs_folder, filename)   # 拼接完整路径
    loader = TextLoader(filepath, encoding="utf-8")   # 创建加载器
    docs = loader.load()                             # 读取文件内容
    all_docs.extend(docs)                            # 合并到总列表

print(f"\n✅ 读取完成，共 {len(all_docs)} 个文档")

# ── 第三步：切分成小块 ──
# chunk_size=500：每块 500 字（中文场景常用值）
# chunk_overlap=50：块与块重叠 50 字，保证边界信息不丢
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks = splitter.split_documents(all_docs)

print(f"✅ 切分完成，共 {len(chunks)} 个小块\n")

# 打印前 3 块看看效果
print("前 3 块预览：")
for i, chunk in enumerate(chunks[:3]):
    print(f"  [{i+1}] (来源：{chunk.metadata.get('source', '未知')})")
    print(f"      {chunk.page_content[:80]}...")
    print()

# ── 第四步：加载 Embedding 模型 ──
# 把文字转成向量
# BAAI/bge-small-zh-v1.5：专门优化中文的嵌入模型
# 第一次运行会下载模型（约 133MB），之后直接使用缓存
print("正在加载 Embedding 模型……")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
print("✅ 模型加载完成")

# ── 第五步：向量化 + 存入 ChromaDB ──
# Chroma.from_documents 一步完成：
#   ① 把 chunks 里的文字取出来
#   ② 传给 embeddings 模型转成向量
#   ③ 把向量 + 原文 + 元数据 一起存入 ChromaDB
# persist_directory：数据存在这个文件夹里，程序关闭后不丢失
print("正在向量化并存入数据库……")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"     # 存在 ./chroma_db 文件夹
)
vectorstore.persist()                   # 写入硬盘

print(f"\n✅ 全部完成！数据已存入 ./chroma_db 文件夹")
print(f"   共 {len(chunks)} 个小块，来自 {len(txt_files)} 个文件")