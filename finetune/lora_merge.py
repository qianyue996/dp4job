from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

def main(path_to_adapter, new_model_directory):
    # 模型保存
    model = AutoPeftModelForCausalLM.from_pretrained(
        path_to_adapter, # path to the output directory
        device_map="auto",
        trust_remote_code=True
    ).eval()

    merged_model = model.merge_and_unload()
    # max_shard_size and safe serialization are not necessary.
    # They respectively work for sharding checkpoint and save the model to safetensors
    merged_model.save_pretrained(new_model_directory,
                                 max_shard_size="2048MB",
                                 safe_serialization=True)

    #2 分词器保存
    tokenizer = AutoTokenizer.from_pretrained(
        path_to_adapter, # path to the output directory
        trust_remote_code=True
    )
    tokenizer.save_pretrained(new_model_directory)

if __name__ == '__main__':
    # 微调后模型路径
    path_to_adapter="/path/to/Qwen-/output_qwen/checkpoint-2000"
    # 合并后模型路径
    new_model_directory="/path/to/Qwen"
    main(path_to_adapter, new_model_directory)