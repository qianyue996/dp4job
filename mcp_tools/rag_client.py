from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class RAG_SEARCH():
    def __init__(self):
        # 定义 Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="models/bge-large-zh-v1.5",
            model_kwargs={"device": "cpu"}
        )

        # 定义reranker
        self.reranker = HuggingFaceCrossEncoder(
            model_name="models/bge-reranker-base",
            model_kwargs={"device": "cpu"}
        )

        # 向量数据库持久化路径
        persist_directory = 'data_base/vector_db/chroma'

        # 加载数据库
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        self.retriever = vectordb.as_retriever(search_kwargs={"k": 20}) # 可以适当增加检索数量，以便reranker有更多文档可供选择

    def __call__(self, query):
        # 1. 初步检索文档
        initial_docs = self.retriever.invoke(query)

        # 2. 提取文档内容和reranker的输入格式
        # reranker通常需要一个列表，其中每个元素是 [query, document_content]
        reranker_input = [[query, doc.page_content] for doc in initial_docs]

        # 3. 进行rerank
        # scores = self.reranker.compute_score(reranker_input) # compute_score 方法返回得分
        # FlagReranker 也提供了 `rerank` 方法，直接返回排序后的结果
        reranked_results = self.reranker.score(reranker_input)

        # 4. 根据reranked_results重新排序原始文档
        # reranked_results 是一个列表，每个元素包含 'query', 'corpus', 'score', 'rank'
        # 我们需要根据 'rank' 或者 'score' 来重新组织 initial_docs
        # 为了方便，我们可以创建一个字典，将原始文档的page_content映射到原始文档对象
        sorted_results = sorted(zip(reranked_results, [doc.page_content for doc in initial_docs]))
        sorted_results = [doc for _, doc in sorted_results][:5]

        return "\n".join([doc for doc in sorted_results])

# 示例用法 (假设你已经有了数据和模型)
if __name__ == "__main__":
    # 假设你已经通过其他方式填充了 chroma 数据库
    # 例如：
    # from langchain_community.document_loaders import TextLoader
    # from langchain.text_splitter import CharacterTextSplitter
    #
    # loader = TextLoader("your_document.txt", encoding="utf-8")
    # documents = loader.load()
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # docs = text_splitter.split_documents(documents)
    #
    # # 初始化 Embeddings (和 RAG_SEARCH 中一样)
    # embeddings = HuggingFaceBgeEmbeddings(
    #     model_name="models/bge-large-zh-v1.5",
    #     model_kwargs={"device": "cpu"},
    #     encode_kwargs={'normalize_embeddings': True}
    # )
    #
    # # 创建并持久化向量数据库
    # persist_directory = 'data_base/vector_db/chroma'
    # vectordb = Chroma.from_documents(
    #     documents=docs,
    #     embedding=embeddings,
    #     persist_directory=persist_directory
    # )
    # vectordb.persist()


    rag_search = RAG_SEARCH()
    query = "上海有什么工作？"
    results = rag_search(query)
    print(results)