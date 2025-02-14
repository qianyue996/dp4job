from langchain.memory import ChatMessageHistory
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate,MessagesPlaceholder
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

import os 

# 定义 Embeddings
embeddings = HuggingFaceEmbeddings(model_name="remote_models/embedding_model")

# 向量数据库持久化路径
persist_directory = 'data_base/vector_db/chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embeddings
)
retriever=vectordb.as_retriever()

chat=ChatOpenAI(
    model="",
    openai_api_key="EMPTY",
    openai_api_base='http://localhost:8000/v1',
    stream_options={"include_usage": True}
)

# Prompt模板
system_prompt=SystemMessagePromptTemplate.from_template('You are a helpful assistant.')
user_prompt=HumanMessagePromptTemplate.from_template("""你是问答任务助手。使用以下检索到的上下文片段来回答问题。如果你不知道答案，就说你不知道。
Question: {query} 
Context: {context} 
Answer:
""")
full_chat_prompt=ChatPromptTemplate.from_messages([system_prompt,MessagesPlaceholder(variable_name="chat_history"),user_prompt])

chat_history=[]
# 构建 RAG 链
chain = (
    {
        "context": itemgetter("query") | retriever,
        "query": itemgetter("query"),
        "chat_history": itemgetter("chat_history")
    }
    | full_chat_prompt
    | chat
)

while True:
    query=input('问题：')
    response=chain.invoke({'query':query, 'chat_history':chat_history})
    chat_history.extend((HumanMessage(content=query),response))
    print(response.content)
    chat_history=chat_history[-20:] # 最新10轮对话