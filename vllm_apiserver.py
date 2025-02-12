import os
import subprocess

subprocess.run("export VLLM_USE_MODELSCOPE=True", shell=True)
subprocess.run("python -m vllm.entrypoints.openai.api_server --device auto --model 'remote_models/Qwen-1_8B-Chat' --chat-template template_chatml.jinja --trust-remote-code --max_model_len 4096 --dtype float16 --gpu-memory-utilization 0.60", shell=True)
