### Auto-GPTQ
在使用GPTQ量化时，它的校验数据集一定不能太大，随便一点就行了，否则爆内存。


`pip install "peft<0.8.0" deepspeed`

合并并存储模型需要非量化的原模型，并修改lora模型的adapter.config里面的模型路径，指向非量化模型

量化微调后模型
这一小节用于量化全参/LoRA微调后的模型。（注意：你不需要量化Q-LoRA模型因为它本身就是量化过的。） 如果你需要量化LoRA微调后的模型，请先根据上方说明去合并你的模型权重。

GPTQ
我们提供了基于AutoGPTQ的量化方案，并开源了Int4和Int8量化模型。量化模型的效果损失很小，但能显著降低显存占用并提升推理速度。

以下我们提供示例说明如何使用Int4量化模型。在开始使用前，请先保证满足要求（如torch 2.0及以上，transformers版本为4.32.0及以上，等等），并安装所需安装包：

pip install auto-gptq optimum
如安装auto-gptq遇到问题，我们建议您到官方repo搜索合适的wheel。

注意：预编译的auto-gptq版本对torch版本及其CUDA版本要求严格。同时，由于 其近期更新，你可能会遇到transformers、optimum或peft抛出的版本错误。 我们建议使用符合以下要求的最新版本：

torch==2.1 auto-gptq>=0.5.1 transformers>=4.35.0 optimum>=1.14.0 peft>=0.6.1
torch>=2.0,<2.1 auto-gptq<0.5.0 transformers<4.35.0 optimum<1.14.0 peft>=0.5.0,<0.6.0
随后即可使用和上述一致的用法调用量化模型：

# 可选模型包括："Qwen/Qwen-7B-Chat-Int4", "Qwen/Qwen-14B-Chat-Int4"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat-Int4",
    device_map="auto",
    trust_remote_code=True
).eval()
response, history = model.chat(tokenizer, "Hi", history=None)
