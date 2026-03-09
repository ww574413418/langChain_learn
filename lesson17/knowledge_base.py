'''
知识库
基于md5检查传入的文件是否已经被存储了
'''
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

def chcek_md5(md5_str:str) -> bool:
    '''检查传入的文件知否已经被处理过了,True 处理过,False没有处理过'''
    if not os.path.exists(config.md5_path):
        # 文件不存在
        open(config.md5_path,"w",encoding="utf-8").close()
        return False
    else:
        with open(config.md5_path,"r",encoding="utf-8") as f:
            md5_list = f.readlines()
            for md5 in md5_list:
                if md5.strip() == md5_str:
                    return True
            return False

def save_md5(md5_str:str) -> None:
    '''将传入的md5字符串,记录到文件内保存'''
    with open(config.md5_path,"a",encoding="utf-8") as f:
        f.write(md5_str + "\n")


def get_string_md5(input_str:str,encoding="utf-8"):
    '''将传入的字符串转成md5字符串'''
    # 将字符串转成字节数组
    str_bytes = input_str.encode(encoding)
    # 创建md5对象
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    md5_hex = md5_obj.hexdigest() # 获取md5的16进制字符串
    return md5_hex

class KnowledgeBaseService(object):
    def __init__(self):
        # 确保文件夹存在
        os.makedirs(config.persist_directory,exist_ok=True)
        # 向量存储实例 chroma数据库
        self.chroma = Chroma(
            collection_name=config.collection_name,#数据库表ming
            embedding_function=OllamaEmbeddings(model="qwen3-embedding:0.6b"),
            persist_directory=config.persist_directory
        )
        # 分词器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len
        )

    def upload_by_str(self,data,filename):
        '''将传入的字符串,进行向量化,存入向量数据库'''
        # 将传入的字符串md5
        md5_str = get_string_md5(data)
        if chcek_md5(md5_str):
            print("文件已经处理过了,已在知识库中")
            return

        print("开始处理文件")
        if len( data) > config.max_split_char_number:
            # 开始文本分割
            knowledge_chunk:list[str] = self.splitter.split_text(data)
        else:
            knowledge_chunk: list[str] = [data]

        metadata = {
            "source": filename,
            "create time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"zc"
        }
        self.chroma.add_texts(
            texts=knowledge_chunk,
            metadatas=[metadata for _ in knowledge_chunk]
        )

        # 标记处理过的文件
        save_md5(md5_str)
        return "处理完成"


if __name__ == '__main__':
    r1 = get_string_md5("jay")
    r2 = get_string_md5("jay")
    r3 = get_string_md5("周杰伦")

    print(r1,r2)
    print(r1,r3)

    save_md5("baba327d241746ee0829e7e88117d4d5")
    print(chcek_md5("baba327d241746ee0829e7e88117d4d5"))

    service = KnowledgeBaseService()
    r = service.upload_by_str("周杰伦是一个歌手", "test file")
    print(r)