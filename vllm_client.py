import requests

url = "http://localhost:8000/chat"
data = {
    "query": "今天天气怎么样？"
}

# 流式请求
with requests.post(url, json=data, params={"stream": True}, stream=True) as response:
    for line in response.iter_lines():
        if line:
            print(f"Received: {line.decode('utf-8')}")
