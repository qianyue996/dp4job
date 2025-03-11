import gradio as gr
import json
import requests
from rag_client import RAG_SEARCH

# 初始化客户端（推荐从环境变量读取API密钥）
url = "http://127.0.0.1:8000/v1/chat/completions"

model = "Qwen-7B-Chat-Int4"

MAX_HISTORY_LEN=50

rag_search = RAG_SEARCH()

def model_chat(messages: list):
    # bug1 我的后端不支持处理gradio产生的messages格式，只允许messages列表中的字典存在两个键，role和content，额外的'metadata'和'options'会不被解析导致报错
    for _dict in messages:
        if 'metadata' in _dict.keys():
            del _dict['metadata']
        if 'options' in _dict.keys():
            del _dict['options']

    try:
        response = requests.post(
            url=url,
            json={"model": model, "messages": messages, "stream": True},
            stream=True
        )
    except Exception as e:
        yield f"请求错误：{e}"

    for chunk in response.iter_lines():
        if chunk:
            try:
                line = chunk.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]
                data = json.loads(line)
                text = data["choices"][0]["message"]["content"]
                yield text
            except Exception as e:
                print("解析数据错误：", e)
                continue

def chat(user_prompt: str, chat_history: list, mode: bool):
    print(chat_history)
    # 给模型指定角色类型
    if chat_history == []:
        chat_history.append({'role': 'system', 'content': '你是一个高级人力资源助手，帮助用户匹配能力相当的工作岗位。'})
    
    if mode:
        # rag检索内容
        rag_content = rag_search(user_prompt)
        # 用户+RAG
        chat_history.append({'role': 'user', 'content': user_prompt + '\n' + rag_content})
        
        # 流式更新助手回复，并实时更新 chat_history
        for chunk in model_chat(list(chat_history)):
        # yield 返回时，清空输入框，并传递更新后的消息列表
            yield "", chat_history + [{'role': 'assistant', 'content': chunk}]
    else:
        # 用户问题
        chat_history.append({'role': 'user', 'content': user_prompt})
        # 流式更新助手回复，并实时更新 chat_history
        for chunk in model_chat(list(chat_history)):
            # yield 返回时，清空输入框，并传递更新后的消息列表
            yield "", chat_history + [{'role': 'assistant', 'content': chunk}]
    chat_history.append({'role': 'assistant', 'content': chunk})
    while len(chat_history)>MAX_HISTORY_LEN:
        chat_history.pop(0)

# 主程序
with gr.Blocks() as app:
    with gr.Row():
        gr.Markdown("""<h1><center>基于深度学习的就业指导系统模型</center></h1>""")
    with gr.Row():
        chatbot = gr.Chatbot(type="messages", label='聊天区')
    with gr.Row():
        msg = gr.Textbox(label='输入框')
    with gr.Row():
        submit = gr.Button("发送")
        switch_rag = gr.Checkbox(label="RAG模式", value=False)  # 开关
        clear = gr.ClearButton([msg, chatbot])

    # 存储发送按钮状态，普通请求 or rag模式
    rag_state = gr.State(False)
    # 更新 state 状态
    def update_state(state_value):
        return state_value  # 返回新状态
    # 当开关切换时，更新状态
    switch_rag.change(update_state, inputs=switch_rag, outputs=rag_state)

    submit.click(chat, [msg, chatbot, rag_state], [msg, chatbot])
    msg.submit(chat, [msg, chatbot, rag_state], [msg, chatbot])

if __name__ == "__main__":
    app.queue(200)  # 请求队列
    app.launch(server_name="0.0.0.0", server_port=7860, max_threads=500) # 线程池