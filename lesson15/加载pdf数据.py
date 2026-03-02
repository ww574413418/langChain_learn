from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path="./data/pdf1.pdf",
    mode="page", #page按页面划分不同的document,single当个document
    #password="123456",
 )

pdy_doc = loader.load()
print(pdy_doc)
print(len(pdy_doc))