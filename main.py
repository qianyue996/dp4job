from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate,MessagesPlaceholder
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_openai import ChatOpenAI
from operator import itemgetter
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

# 加载faiss向量库，用于知识召回
retriever=vectordb.as_retriever(search_kwargs={"k":5})

# 用vllm部署openai兼容的服务端接口，然后走ChatOpenAI客户端调用
os.environ['VLLM_USE_MODELSCOPE']='True'
chat=ChatOpenAI(
    model="remote_models/Qwen-7B-Chat-Int4",
    openai_api_key="EMPTY",
    openai_api_base='http://localhost:8000/v1',
    stop=['<|im_end|>'],
    streaming=True
)

# Prompt模板
system_prompt=SystemMessagePromptTemplate.from_template('You are a helpful assistant.')
user_prompt=HumanMessagePromptTemplate.from_template('''
Answer the question based only on the following context:

{context}

Question: {query}
''')
full_chat_prompt=ChatPromptTemplate.from_messages([system_prompt,MessagesPlaceholder(variable_name="chat_history"),user_prompt])

'''
<|im_start|>system
You are a helpful assistant.
<|im_end|>
...
<|im_start|>user
Answer the question based only on the following context:

{context}

Question: {query}
<|im_end|>
<|im_start|>assitant
......
<|im_end|>
'''

# Chat chain
chat_chain={
        "context": itemgetter("query") | retriever,
        "query": itemgetter("query"),
        "chat_history":itemgetter("chat_history"),
    }|full_chat_prompt|chat

# 开始对话
chat_history=[]
while True:
    query=input('query:')
    response=chat_chain.invoke({'query':query,'chat_history':chat_history})
    chat_history.extend((HumanMessage(content=query),response))
    print(response.content)
    chat_history=chat_history[-20:] # 最新10轮对话