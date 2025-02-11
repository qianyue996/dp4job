from vllm_wrapper import vLLMWrapper
from vllm import LLM

model = "remote_models/Qwen-1_8B-Chat"

vllm_model = vLLMWrapper(model_dir=model,
                        dtype="float16",
                        tensor_parallel_size=1,
                        gpu_memory_utilization=0.6)

history=None 
while True:
    Q=input('提问:')
    response, history = vllm_model.chat(query=Q,
                                        history=history)
    print(response)
    history=history[:20]