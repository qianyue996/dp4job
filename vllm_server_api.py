import os 
import asyncio
import json
from typing import AsyncGenerator

from vllm import AsyncEngineArgs,AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from modelscope import AutoTokenizer, GenerationConfig
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import uuid

from prompt_utils import _build_prompt

# http接口服务
app=FastAPI()

# vLLM参数
model_dir="remote_models/Qwen-7B-Chat-Int4"
tensor_parallel_size=1
gpu_memory_utilization=0.9
quantization='gptq'
dtype='float16'

# vLLM模型加载
def load_vllm():
    global generation_config,tokenizer,stop_words_ids,engine
    # 模型基础配置
    generation_config=GenerationConfig.from_pretrained(model_dir,trust_remote_code=True)
    # 加载分词器
    tokenizer=AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
    tokenizer.eos_token_id=generation_config.eos_token_id
    # 推理终止词
    stop_words_ids=[tokenizer.im_start_id,tokenizer.im_end_id,tokenizer.eos_token_id]
    # vLLM基础配置
    args=AsyncEngineArgs(model_dir)
    args.tokenizer=model_dir
    args.tensor_parallel_size=tensor_parallel_size
    args.trust_remote_code=True
    args.quantization=quantization
    args.gpu_memory_utilization=gpu_memory_utilization
    args.dtype=dtype
    args.max_num_seqs=20   # batch最大20条样本
    # 加载模型
    os.environ['VLLM_USE_MODELSCOPE']='True'
    engine=AsyncLLMEngine.from_engine_args(args)
    return generation_config,tokenizer,stop_words_ids,engine

generation_config,tokenizer,stop_words_ids,engine=load_vllm()

@app.post("/chat")
async def chat(request: Request, stream: bool = False):
    try:
        request_data = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    
    query = request_data.get("query", None)
    if not query:
        return JSONResponse({"error": "Missing 'query' field"}, status_code=400)
    
    system = request_data.get('system', 'You are a helpful assistant.')
    history = request_data.get('history', [])
    request_id = str(uuid.uuid4().hex)

    # 构造prompt
    prompt_text, prompt_tokens = _build_prompt(generation_config, tokenizer, query, history=history, system=system)
    inputs = {"prompt_token_ids": prompt_tokens}

    # vLLM请求配置
    sampling_params = SamplingParams(
        stop_token_ids=stop_words_ids,
        early_stopping=False,
        top_p=generation_config.top_p,
        top_k=-1 if generation_config.top_k == 0 else generation_config.top_k,
        temperature=generation_config.temperature,
        repetition_penalty=generation_config.repetition_penalty,
        max_tokens=generation_config.max_new_tokens
    )

    # 获取生成器
    results_generator = engine.generate(inputs=inputs, sampling_params=sampling_params, request_id=request_id)

    # 流式响应
    if stream:
        try:
            return StreamingResponse(stream_results(results_generator), media_type="application/json")
        except asyncio.CancelledError:
            return JSONResponse({"error": "Request cancelled"}, status_code=499)

    # 非流式响应
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None:
        return JSONResponse({"error": "No response generated"}, status_code=500)

    prompt = final_output.prompt or ""
    text_outputs = [prompt + (output.text or "") for output in final_output.outputs]
    return JSONResponse({"text": text_outputs})

# 流式响应
async def stream_results(results_generator) -> AsyncGenerator[bytes, None]:
    try:
        async for request_output in results_generator:
            prompt = request_output.prompt or ""
            text_outputs = [prompt + (output.text or "") for output in request_output.outputs]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\n").encode("utf-8")
    except asyncio.CancelledError:
        return


if __name__ == '__main__':
    uvicorn.run(app,host=None,port=8000,log_level="debug")