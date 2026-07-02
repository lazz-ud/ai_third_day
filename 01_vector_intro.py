import  chromadb

client = chromadb.PersistentClient(path='./chroma_db')
collection = client.get_or_create_collection(name='test_collection')

collection.add(
    documents=["苹果是一种水果", "西瓜是夏天最受欢迎的水果", "Python是一种编程语言"],
    ids=['doc1','doc2','doc3']

)
results = collection.query(
    query_texts=['水果'],
    n_results=2
)
print(f'查询结果：{results}')
print(f"\n命中 {len(results['documents'][0])} 条记录：")
for i, doc in enumerate(results['documents'][0]):
    print(f"  {i+1}. {doc}")