from langchain_text_splitters import RecursiveCharacterTextSplitter

long_text =  """
人工智能（AI）是计算机科学的一个分支。
它试图理解智能的本质，并创造出能够以人类智能相似方式做出反应的智能机器。
AI的研究领域包括机器人、语言识别、图像识别和自然语言处理等。
机器学习是实现AI的一种方法，它让计算机从数据中学习，而不是通过明确的编程指令。
深度学习是机器学习的一个子集，使用多层神经网络来处理复杂的问题。
"""
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks = splitter.split_text(long_text)
print(f"原始长度：{len(long_text)} 字")
print(f"切分成 {len(chunks)} 块：\n")
for i, chunk in enumerate(chunks):
    print(f"  第 {i+1} 块（{len(chunk)} 字）：{chunk}")

# 2. 向量化（把文字转成数字）
print("\n" + "=" * 60)
print("现在把每块文字转成向量（数字）看看")
print("=" * 60)




import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.DefaultEmbeddingFunction()
vectors = ef(chunks)
for i,(chunk,vec) in enumerate(zip(chunks,vectors)):
    print(f"\n第{i+1} 块：{chunk}")
    print(f"  向量维度：{len(vec)} 个数字")
    print(f"  前 10 个数字：{vec[:10]}")
print("\n" + "=" * 60)
print("算一下第 1 块和第 2 块、第 1 块和第 4 块的相似度")
print("=" * 60)
import numpy as np

def cosine_similarity(a, b):
        """计算两个向量的余弦相似度（1 表示完全一样，0 表示完全不相关）"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_1_2 = cosine_similarity(vectors[0], vectors[1])
sim_1_4 = cosine_similarity(vectors[0], vectors[3])

print(f"\n第 1 块 vs 第 2 块（AI vs 机器学习）：相似度 = {sim_1_2:.4f}")
print(f"第 1 块 vs 第 4 块（AI vs 深度学习）：相似度 = {sim_1_4:.4f}")