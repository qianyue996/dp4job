#!/bin/bash

# 检查必要工具是否可用
check_git_lfs_installed() {
    if command -v git-lfs &> /dev/null; then
        return 0
    else
        echo "Git LFS is not installed."
        return 1
    fi
}

if check_git_lfs_installed; then
    echo " "
else
    exit 1
fi

# 定义目标目录路径
MODEL_DIR="./models"
EMBEDDING_DIR="$MODEL_DIR/embedding_model"

# 检查父目录（a）是否存在，如果不存在则创建它
if [ ! -d "$MODEL_DIR" ]; then
    mkdir "$MODEL_DIR"
fi

# 进入到父目录并检查子目录（b）是否存在，如果不存在则执行 c 命令
cd "$MODEL_DIR" || { echo "Failed to change into $MODEL_DIR"; exit 1; }

if [ ! -d "$EMBEDDING_DIR" ]; then
    echo "embedding模型不存在, 自动为您下载..."
    git clone https://www.modelscope.cn/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.git embedding_model
fi