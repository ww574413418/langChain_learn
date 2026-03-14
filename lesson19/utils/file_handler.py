import hashlib
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from .logger_handler import logger as log

def get_file_md5_hex(file_path:str)->str:
    '''
    获取文件的md5的16位值,用于去重
    :return:
    '''
    # 判断文件是否存在
    if not os.path.exists(file_path):
        log.error(f"the file is not exist,file path={file_path}")

    if not os.path.isfile(file_path):
        log.error(f"the path is not a file,file path={file_path}")

    md5_obj = hashlib.md5()

    # 定义chunk 防止文件过大
    chunk_size = 1024 * 1024
    try:
        # rb 二进制读取
        with open(file_path, "rb") as f:
            '''
                while chunk := f.read(chunk_size): 等价与:
                chunk = f.read(chunk_size)
                while chunk:
                    md5_obj.update(chunk)
                    chunk = f.read(chunk_size)
            '''
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        log.error(f"get file md5 error,file path={file_path},error={e}")


def listdir_with_allowed_type(file_path:str,allowed_type:tuple[str])-> tuple[str] | tuple[str, ...]:
    '''
    返回文件夹中允许的文件类型
    :return:
    '''
    #需要文件的路径
    files = []
    #判断是否是文件夹
    if not os.path.isdir(file_path):
        log.error(f"the path is not a directory,file path={file_path}")
        return allowed_type

    for f in os.listdir(file_path):
        if f.endswith(allowed_type):
            files.append(os.path.join(file_path,f))
    #tuple 让files不能在修改
    return tuple(files)


def pdf_loader(file_path:str,password:str=None)->list[Document]:
    return PyPDFLoader(file_path=file_path,password=password).load()

def text_loader(file_path:str,encoding:str="utf-8")->list[Document]:
    return TextLoader(file_path,encoding=encoding).load()