from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="./data/stu.json",
    jq_schema=".",
    text_content=False,#谁知读取的内容默认不是字符串,这样才可以直接读取json对象,否则都是按字符串读取
)

json = loader.load()
print(json)

loader = JSONLoader(
    file_path="./data/stus.json",
    jq_schema=".[].name",
    text_content=False,#谁知读取的内容默认不是字符串,这样才可以直接读取json对象,否则都是按字符串读取
)

json = loader.load()
print(json)

loader = JSONLoader(
    file_path="./data/stu_json_lines.json",
    jq_schema=".name",
    text_content=False,#谁知读取的内容默认不是字符串,这样才可以直接读取json对象,否则都是按字符串读取
    json_lines=True,#每一行是一个json
)

json = loader.load()
print(json)