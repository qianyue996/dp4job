`pip install "peft<0.8.0" deepspeed`

合并并存储模型需要非量化的原模型，并修改lora模型的adapter.config里面的模型路径，指向非量化模型

量化微调后模型
这一小节用于量化全参/LoRA微调后的模型。（注意：你不需要量化Q-LoRA模型因为它本身就是量化过的。） 如果你需要量化LoRA微调后的模型，请先根据上方说明去合并你的模型权重。

我们推荐使用auto_gptq去量化你的模型。

pip install auto-gptq optimum
注意: 当前AutoGPTQ有个bug，可以在该issue查看。这里有个修改PR，你可以使用该分支从代码进行安装。

首先，准备校准集。你可以重用微调你的数据，或者按照微调相同的方式准备其他数据。

第二步，运行以下命令：

python run_gptq.py \
    --model_name_or_path $YOUR_LORA_MODEL_PATH \
    --data_path $DATA \
    --out_path $OUTPUT_PATH \
    --bits 4 # 4 for int4; 8 for int8
这一步需要使用GPU，根据你的校准集大小和模型大小，可能会消耗数个小时。