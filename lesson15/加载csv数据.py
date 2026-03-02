from langchain_community.document_loaders import CSVLoader


loader = CSVLoader(
    file_path="./data/stu.csv",
    encoding="utf-8",
    csv_args={
        "delimiter": ",",#指定分隔符
        "quotechar": '"',#指定带有双引号的字段
        "fieldnames": ["name","age","sex","hobby"],#如果没有表头可以自定义表头/有表头不要使用
    },
)

documents = loader.load()
for document in documents:
    print(type( document),document)

documents = loader.lazy_load() # 懒加载
for document in documents:
    print(type( document),document)



