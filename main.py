from fastapi import FastAPI, Request

# 创建 FastAPI 实例
app = FastAPI()

# 创建 POST 路由
@app.post("/v1/chat/completions")
async def chat(request:Request):
    # 你可以在这里进行一些处理，比如保存数据、进行计算等
    return print(request.text)