from langchain_community.document_loaders import CSVLoader
import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os

# 获取文件路径函数
def get_files(dir_path):
    # args：dir_path，目标文件夹路径
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        # os.walk 函数将递归遍历指定文件夹
        for filename in filenames:
            # 通过后缀名判断文件类型是否满足要求
            if filename.endswith(".md"):
                # 如果满足要求，将其绝对路径加入到结果列表
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".txt"):
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".csv"):
                file_list.append(os.path.join(filepath, filename))
    return file_list



# 加载文件函数
def get_text(dir_path):
    # args：dir_path，目标文件夹路径
    # 首先调用上文定义的函数得到目标文件路径列表
    file_lst = get_files(dir_path)
    # docs 存放加载之后的document对象
    docs = []
    # 遍历所有目标文件
    for one_file in tqdm(file_lst):
        file_type = one_file.split('.')[-1]
        if file_type == 'csv':
            # 分批大小
            batch_size = 1000
            # 使用 Pandas 读取 CSV 文件并分批处理
            chunk_iterable = pd.read_csv(one_file, encoding="utf-8", chunksize=batch_size)
            for i, chunk_df in enumerate(chunk_iterable):
                for index, row in chunk_df.iterrows():
                    content = "\n".join([f"{col}: {val}" for col, val in zip(chunk_df.columns, row)])
                    metadata = {"row_number": index}
                    document = Document(page_content=content, metadata=metadata)
                    docs.append(document)
        else:
            # 如果是不符合条件的文件，直接跳过
            continue
    return docs

if __name__ == '__main__':
    # 目标文件夹
    tar_dir = [
        "raw_data"
    ]

    # 加载目标文件
    docs = []
    for dir_path in tar_dir:
        docs.extend(get_text(dir_path))

    # # 对文本进行分块
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=500, chunk_overlap=150)
    # split_docs = text_splitter.split_documents(docs)

    # 加载开源词向量模型
    embeddings = HuggingFaceEmbeddings(model_name="../models/bge-large-zh-v1.5")

    # 构建向量数据库
    # 定义持久化路径
    persist_directory = 'data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )