from datasets import load_dataset
import json
import random

format_train_data = []

# 格式化、模板化数据集
def format_dataset(id, input, output):
    template = {
    "id": "identity_0",
    "conversations": [
            {
            "from": "user",
            "value": "你好"
            },
            {
            "from": "assistant",
            "value": "我是一个语言模型，我叫通义千问。"
            }
        ]
    }
    template['conversations'][0]['value'] = input
    template['conversations'][1]['value'] = output
    template['id'] = id

    return template

def main(hf_datasets_path, custom_datasets_path):
    user_text_messages = []
    assistant_text_messages = []

    # 操作hf数据集
    # 加载数据集
    _datasets = load_dataset(hf_datasets_path)
    # 过滤出只属于advice-seeking类型的数据集
    filter_datasets = _datasets['train'].filter(lambda x: x['system_prompt_key'] == 'advice-seeking')
    list_filter_datasets = [i for i in filter_datasets]

    # 数据集太长，训练简直要命，故取随机元素集合
    list_filter_datasets = random.sample(list_filter_datasets, 2000)

    for dataset in list_filter_datasets:
        # 分离user和assistant的对话
        user_messages = [message for message in dataset['conversations'] if message['role'] == 'user']
        assistant_messages = [message for message in dataset['conversations'] if message['role'] == 'assistant']
        for i in range(len(user_messages)):
            user_text = user_messages[i]['content']
            assistant_text = assistant_messages[i]['content']
            user_text_messages.append(user_text)
            assistant_text_messages.append(assistant_text)

    # 操作自定义数据集
    with open(custom_datasets_path, 'r', encoding='utf-8')as f:
        custom_datasets = json.load(f)
    
    for i in range(len(custom_datasets)):
        user_text_messages.append(custom_datasets[i]['user'])
        assistant_text_messages.append(custom_datasets[i]['assistant'])

    print('#'*20)
    print(f"训练数据集总长度为：{len(user_text_messages)}")
    print('#'*20)

    for index in range(len(user_text_messages)):
        id = 'identity_' + str(index)
        input = user_text_messages[index]
        output = assistant_text_messages[index]
        one_data = format_dataset(id, input, output)
        format_train_data.append(one_data)

if __name__ == '__main__':
    hf_datasets_path = '../smoltalk-chinese/data'
    custom_datasets_path = 'finetune/custom_datasets.json'

    output_path = 'finetune/train.json'

    main(hf_datasets_path, custom_datasets_path)
    with open(output_path, 'w', encoding='utf-8')as f:
        json.dump(format_train_data, f, ensure_ascii=False, indent=2)
