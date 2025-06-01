import pandas as pd
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os

job_info = """招聘信息：
发布时间：2023-{post_time}
公司名称： {company_name}
公司规模： {company_size}
公司类型： {company_type}
公司描述： {company_desc}
岗位类型： {job_type}
岗位名称： {job_name}
学历： {education}
薪资： {salary}
招聘人数：{need}
福利： {walfare}
城市:  {city}
经验要求： {experience}
工作地点： {job_area}
岗位描述： {job_desc}
"""

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
                    content = job_info.format(
                        post_time = row[1],
                        company_name = row[2],
                        company_size = row[3],
                        company_type = row[4],
                        job_type = row[5],
                        job_name = row[6],
                        education = row[7],
                        salary = row[8],
                        need = row[9],
                        walfare = row[10],
                        city = row[11],
                        experience = row[12],
                        company_desc = row[13],
                        job_area = row[14],
                        job_desc = row[15]
                    )
                    metadata = {
                        "row_number": index
                        # "original_data" : {
                        #     "post_time" : row[1],
                        #     "company_name" : row[2],
                        #     "company_size" : row[3],
                        #     "company_type" : row[4],
                        #     "job_type" : row[5],
                        #     "job_name" : row[6],
                        #     "education" : row[7],
                        #     "salary" : row[8],
                        #     "need" : row[9],
                        #     "walfare" : row[10],
                        #     "city" : row[11],
                        #     "experience" : row[12],
                        #     "company_desc" : row[13],
                        #     "job_area" : row[14],
                        #     "job_desc" : row[15]
                        # }
                    }
                    document = Document(page_content=content, metadata=metadata)
                    docs.append(document)
        else:
            # 如果是不符合条件的文件，直接跳过
            continue
    return docs

if __name__ == '__main__':
    tar_dir = [
        "raw_data"
    ]

    # 加载目标文件
    docs = []
    for dir_path in tar_dir:
        docs.extend(get_text(dir_path))

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=512, chunk_overlap=256)
    # split_docs = text_splitter.split_documents(docs)

    # 加载开源词向量模型
    embeddings = HuggingFaceEmbeddings(
        model_name="models/bge-large-zh-v1.5",
        model_kwargs={"device": "cpu"}
    )

    # 构建向量数据库
    persist_directory = 'data_base/vector_db/chroma'
    os.makedirs(persist_directory, exist_ok=True)
    # 加载数据库
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )