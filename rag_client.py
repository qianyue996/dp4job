from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class RAG_SEARCH():
    def __init__(self):
        # 定义 Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="../models/bge-large-zh-v1.5",
                                           model_kwargs={"device": "cpu"})

        # 向量数据库持久化路径
        persist_directory = 'data_base/vector_db/chroma'

        # 加载数据库
        vectordb = Chroma(
            persist_directory=persist_directory, 
            embedding_function=embeddings
        )
        self.retriever=vectordb.as_retriever()

    def __call__(self, query):
        docs = self.retriever.invoke(query)
        return self.format_docs(docs)

    # 文档格式化（把文档列表转换成字符串）
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)