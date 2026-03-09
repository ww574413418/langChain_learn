'''
基于streamlit工具实现web网页上传文件
每次更改代码都会重新执行一次代码,导致无法保存状态,需要用seesion_state(字典)来保存
'''
import time
import streamlit as st
from knowledge_base import KnowledgeBaseService

# 添加网页标题
st.title("知识库更新服务")
# 文件上传
upload_file= st.file_uploader(
    label="请上传文件",
    type=["txt","csv"],
    accept_multiple_files=False,#多文件上传
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()


if upload_file is not None:
    file_name = upload_file.name
    file_tpye = upload_file.type
    file_size = upload_file.size/1024

    st.subheader(f"文件名:{file_name}")
    st.write(f"格式{file_tpye} | 大小:{file_size:.2f}kb")#write 普通文本

    text = upload_file.getvalue().decode("utf-8")
    with st.spinner("正在处理文件..."): # 添加一个处理转圈
        time.sleep(1)
        service = st.session_state["service"]
        result = service.upload_by_str(text,file_name)
        st.write(result)

