import requests
import json

url = "http://localhost:8000/chat"
data={
    'query': '周杰伦？'
}

# response = requests.post(url, json=data)
# print(response.text)


with requests.post(url, json=data) as response:
    for line in response.iter_lines():
        print(f"Received: {line.decode('utf-8')}")
