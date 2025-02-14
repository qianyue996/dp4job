from langchain_openai import ChatOpenAI
from langchain.schema import (
    HumanMessage,
)
import asyncio
import os


chat=ChatOpenAI(
    model="qwen/Qwen-7B-Chat-Int4",
    openai_api_key="EMPTY",
    openai_api_base='http://localhost:8000/v1',
    stream_options={"include_usage": True}
)

resp = chat.stream([HumanMessage(content="今天天气怎么样")])
print(resp)