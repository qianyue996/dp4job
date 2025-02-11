import os
import subprocess

subprocess.run("export VLLM_USE_MODELSCOPE=True", shell=True)
subprocess.run("python -m vllm.entrypoints.openai.api_server --device auto --model 'remote_models/Qwen-7B-Chat-Int4' -q gptq --chat-template template_chatml.jinja --trust-remote-code --max_model_len 6144 --dtype float16 --gpu-memory-utilization 0.90", shell=True)
