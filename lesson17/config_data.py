md5_path = "./md5.txt"

#chroma
collection_name = "rag"
persist_directory = "./chroma_db"

#splitter
chunk_size = 1000
chunk_overlap = 100
separators = ["。", "？", "！", "；", "，", "、", "。", "？", "！", "；", "，", "、","\n","\n\n"," "]
max_split_char_number = 1000 # 文本分割阈值

#相似度检索阈值
similarity_threshold = 2#返回满足的向量数量

embedding_model = "qwen3-embedding:0.6b"
chat_model="tencent/Hunyuan-MT-7B"

# session config
session_config = {
    "configurable": {
        "session_id": "user_001"
    }
}

