import requests
import json

url = "http://localhost:8000/chat"  # 假设你的 API URL
data = {'query': '周杰伦是谁？用500字作答'}

response = requests.post(url, json=data, stream=False)

# 流式接收
for chunk in response.iter_content(chunk_size=1024):
    if chunk:
        print(chunk.decode('utf-8'))
