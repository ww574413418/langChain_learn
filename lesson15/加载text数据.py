from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_loader = TextLoader(
    file_path="./data/Python基础语法.txt",
    encoding="utf-8"
)

document = text_loader.load()

spliter = RecursiveCharacterTextSplitter(
    chunk_size=500, #每个分段的最大字符数
    chunk_overlap=50, #分段之间允许的重叠自负
    separators=["\n\n","\n","\t"," "], #文本分割符号
    length_function=len,
)
split_doc = spliter.split_documents(document)
print(len(split_doc))

for doc in split_doc:
    print("="*20,doc.page_content,"="*20)