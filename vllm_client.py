# -*- coding:utf-8 -*-
import json
import requests

request_param = {
    'query': "请你写一篇日记，内容不限，自己创作，不少于300字，记住300字",
    'stream': False
}

response = requests.post(url="http://localhost:8000/chat", json=request_param)

print(response.json()['text'][0])