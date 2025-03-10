### How to use

#### 制作向量数据库

运行`make_database.py`

#### 前后端分别运行
运行`vllm_server_api.py`启动后端，监听`127.0.0.1:8000`

运行`vllm_webui.py`启动前端，监听`0.0.0.0:7860`

### 复现过程
python版本：3.10.14
<br>
Pytorch版本: 2.1.0 cu118
<br>
依赖项:
```
vllm==0.7.0
auto-gptq==0.5.1    #pip install auto-gptq==0.5.1 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

```


参考自以下项目，在此感谢。
<br>
[QwenLM/Qwen](https://github.com/QwenLM/Qwen)
<br>
[owenliang/rag-retrieval](https://github.com/owenliang/rag-retrieval)
<br>
[owenliang/qwen-vllm](https://github.com/owenliang/qwen-vllm)
