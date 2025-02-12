from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
import asyncio
import uvicorn

app = FastAPI()

# 模拟一个生成器
async def results_generator():
    for i in range(5):
        await asyncio.sleep(1)  # 模拟等待时间
        yield {"outputs": [{"text": f"Generated part {i + 1}."}]}  # 模拟文本输出

@app.post("/v1/chat/completions")
async def stream_response(stream: bool = True):
    if stream:
        async def stream_results():
            async for request_output in results_generator():
                text_outputs = request_output["outputs"][0]["text"]
                ret = {
                    "model": "gpt-3.5-turbo",  # 示例模型名称
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": text_outputs
                            }
                        }
                    ]
                }
                # 逐步返回 JSON 格式的数据
                yield json.dumps(ret) + "\n"  # 每次输出一个完整的 JSON 对象并加上换行符

        return StreamingResponse(stream_results(), media_type="text/event-stream")
