import os.path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.logger_handler import logger as log
from utils.config_handler import chroma_config
from model.model_factory import embedding_model_silicon
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import text_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex


class VectorStoreService:

    def __init__(self):
        self.vect_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embedding_model_silicon,
            persist_directory=get_abs_path(chroma_config["persist_directory"])
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len
        )

    def get_retriever(self):
        return self.vect_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k":chroma_config["k"]}
        )


    def check_md5_hex(self,md5_for_check:str) -> bool:
        '''
        using md5 to check weather the file has been resolved
        :param md5_for_check:
        :return:
        '''
        # the file is not exist
        if not os.path.exists(get_abs_path(chroma_config["md5_hex_store"])):
            #create the file
            with open(get_abs_path(chroma_config["md5_hex_store"]),"w",encoding="utf-8") as f:
                f.write(md5_for_check + "\n")
            return False

        with open(get_abs_path(chroma_config["md5_hex_store"]),"r") as f:
            for line in f.readlines():
                line = line.strip()
                if line == md5_for_check:
                    return True
            return False

    def save_md5(self,md5_for_check):
        '''
        save md5 code in file
        :param md5_for_check:
        :return:
        '''
        with open(get_abs_path(chroma_config["md5_hex_store"]),"a",encoding="utf-8") as f:
            f.write(md5_for_check + "\n")



    def get_file_documents(self,files_path:str) -> list:
        '''
        load file from dir and count the md5
        :param files: file path list
        :return:
        '''
        if files_path.endswith("txt"):
            return text_loader(files_path,encoding="utf-8")
        elif files_path.endswith("pdf"):
            return pdf_loader(files_path)
        return []


    def load_documents(self):
        # get allowed type file path list
        allowed_file_path:list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_config["data_path"]),
            tuple(chroma_config["allow_knowledage_file_type"]))

        for path in allowed_file_path:
            # get md5 hex
            md5_hex = get_file_md5_hex(path)

            if self.check_md5_hex(md5_hex):
                log.info(f"{path} has been resolve in knowledge base,skip resolve")
                continue
            try:
                document:list[Document] = self.get_file_documents(path)

                if not document:
                    log.error("there are no matched file")
                    continue

                split_documents = self.splitter.split_documents(document)
                if not split_documents:
                    log.warning(f"there are no split file,file path={path}")
                    continue

                # save vector to chroma db
                self.vect_store.add_documents(split_documents)

                # save the md5 value of vector in txt
                self.save_md5(md5_hex)

                log.info("allowed type documents have been loaded")

            except Exception as e:
                # exc_info is true means recording detail error stack info,
                # and the another value means recording only error message
                log.error(f"load file error,file path={path},error={e}",exc_info=True)
                continue


vector_store = VectorStoreService()

if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_documents()

    retriever = vs.get_retriever()

    res = retriever.invoke("机器人找不到充电座怎么处理？")

    for s in res:
        print(s.page_content)
