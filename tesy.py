from fastapi import FastAPI
from sse_starlette import EventSourceResponse
import asyncio
import json

app = FastAPI()

# 模拟增量生成多个部分（tokens）
async def results_generator():
    for i in range(5):  # 假设生成 5 个增量数据（每个为一个 JSON 对象）
        await asyncio.sleep(1)  # 模拟处理延迟
        choice = {
            "id": f"chatcmpl-{i}",
            "object": "chat.completion.chunk",
            "created": 1677858241 + i,  # 假设每次返回的时间戳递增
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": f"Generated content part {i + 1}"  # 模拟每个 token 或消息片段
                    },
                    "index": 0
                }
            ]
        }
        yield choice  # 每次返回一个完整的 JSON 格式响应字符串

    # 完成后，发送 "DONE" 消息以标志流的结束
    yield {'data': '[DONE]'}

@app.post("/v1/chat/completions")
async def stream_response(stream: bool = True):
    async def event_generator():
        async for message in results_generator():
            # 每次返回一个完整的 JSON 对象作为一个增量数据块
            yield json.dumps(message)
    
    # 使用 EventSourceResponse 返回逐个生成的 JSON 格式字节流
    return EventSourceResponse(event_generator(), media_type="text/event-stream")

# 启动服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
