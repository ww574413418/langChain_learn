import time
import uuid
import streamlit as st
from agent.react_agent import agent

# title
st.set_page_config(
    page_title="我的智能客服",  # 浏览器标签栏文字
    page_icon="🤖",            # 标签栏图标
)
st.title("智能客服")
st.divider()

#第一次聊天记录id
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

if "agent" not in st.session_state:
    st.session_state["agent"] = agent

if "history_messages" not in st.session_state:
    st.session_state["history_messages"] = []

for message in st.session_state["history_messages"]:
    st.chat_message(message["role"]).write(message["content"])



# user input prompt
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["history_messages"].append({"role": "user", "content": prompt})

    response_message = []
    with st.spinner(text="thinking..."):
        result = st.session_state["agent"].execute_stream(prompt,st.session_state["thread_id"])

        def stream_result(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                for char in chunk:
                    time.sleep(0.01)
                    # 一个一个字符的流式输出
                    yield char


        st.chat_message("assistant").write_stream(stream_result(result,response_message))
        # response_message[-1] 只保存结果,不保存思考过程
        st.session_state["history_messages"].append({"role": "assistant", "content": response_message[-1]})