import os 
import json
from vllm import AsyncEngineArgs,AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from modelscope import AutoTokenizer, GenerationConfig
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse
import uvicorn
import uuid
from utils.prompt_utils import _build_prompt

# http接口服务
app=FastAPI()

# vLLM参数
model_dir="../models/Qwen-7B-Chat-Int4"
tensor_parallel_size=1
gpu_memory_utilization=0.90
quantization='gptq'
dtype='float16'
max_model_len=2048
# vLLM模型加载
def load_vllm():
    global generation_config,tokenizer,stop_tokens_ids,engine
    # 模型基础配置
    generation_config=GenerationConfig.from_pretrained(model_dir,trust_remote_code=True)
    # 加载分词器
    tokenizer=AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
    tokenizer.eos_token_id=generation_config.eos_token_id
    # 推理终止词
    stop_tokens_ids=[tokenizer.im_start_id,tokenizer.im_end_id,tokenizer.eos_token_id]
    # vLLM基础配置
    args=AsyncEngineArgs(model_dir)
    args.device='cpu'
    args.tokenizer=model_dir
    args.tensor_parallel_size=tensor_parallel_size
    args.trust_remote_code=True
    args.quantization=quantization
    args.gpu_memory_utilization=gpu_memory_utilization
    args.dtype=dtype
    args.max_num_seqs=20   # batch最大20条样本
    args.max_model_len=max_model_len
    # 加载模型
    os.environ['VLLM_USE_MODELSCOPE']='True'
    engine=AsyncLLMEngine.from_engine_args(args)
    return generation_config,tokenizer,stop_tokens_ids,engine

generation_config,tokenizer,stop_tokens_ids,engine=load_vllm()

@app.post("/v1/chat/completions")
async def chat(request: Request):
    request_data = await request.json()
    stream = request_data.get("stream")
    messages = request_data.get("messages", None)
    
    # user发言字典列表
    user_messages = [message for message in messages if message['role'] == 'user']
    """
    [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "user", "content": "What is the population of France?"}
    ]
    """
    # assistant发言字典列表
    assistant_messages = [message for message in messages if message['role'] == 'assistant']
    """
    [
    {"role": "assistant", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "What is the population of France?"}
    ]
    """
    # system类型列表
    system_messages = [message for message in messages if message['role'] == 'system']
    """
    [
    {"role": "system", "content": "What is the capital of France?"},
    {"role": "system", "content": "What is the population of France?"}
    ]
    """
    # 取最新用户发言: Str
    latest_user_message = user_messages[-1:][0]['content']
    # 历史对话列表
    history = messages[:-1]
    if messages[0]['role'] != "system":
        system = "you are a helpful assistant"
    else:
        system = system_messages[-1:][0]['content']
    request_id = str(uuid.uuid4().hex)

    # 构造prompt
    prompt_text, prompt_tokens = _build_prompt(generation_config, tokenizer, latest_user_message, history=history, system=system)
    prompt = {"prompt_token_ids": prompt_tokens}

    # vLLM请求配置
    sampling_params = SamplingParams(
        stop_token_ids=stop_tokens_ids,
        top_p=generation_config.top_p,
        top_k=-1 if generation_config.top_k == 0 else generation_config.top_k,
        temperature=generation_config.temperature,
        repetition_penalty=generation_config.repetition_penalty,
        max_tokens=generation_config.max_new_tokens
    )

    # 获取生成器
    results_generator = engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)

    # 流式响应
    async def stream_results():
        async for request_output in results_generator:
            text_outputs = request_output.outputs[0].text
            ret = { "model": model_dir,
                    "choices": [{
                    "message": {
                    "role": "assistant",
                    "content": text_outputs}}]}
            yield json.dumps(ret, ensure_ascii=False)
    if stream:
        return EventSourceResponse(stream_results())

    # 非流式响应
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    text_outputs = final_output.outputs[0].text
    response = {
        "model": model_dir,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": text_outputs
                }
            }
        ]
    }
    return JSONResponse(response)
"""
response = {
        "id": "chatcmpl-xyz",
        "object": "chat.completion",
        "created": 1613422357,
        "model": request.model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": assistant_reply
                },
                "finish_reason": "stop",
                "index": 0
            }
        ]
    }
"""

@app.get("/v1/models")
async def get_json():
    # 定义你想要返回的 JSON 数据
    data = {
            "object": "list",
            "data": [
            {
                "id": model_dir,
                "object": "model",
                "created": 2333333,
                "owned_by": "淺月"
            },
        ]
    }
    # 返回 JSON 格式的响应
    return JSONResponse(data)

if __name__ == '__main__':
    uvicorn.run(app,host=None,port=8000,log_level="debug")
