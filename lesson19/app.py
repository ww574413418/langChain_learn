import streamlit as st
from agent.react_agent import ReactAgent

# title
st.set_page_config(
    page_title="我的智能客服",  # 浏览器标签栏文字
    page_icon="🤖",            # 标签栏图标
)
st.title("智能客服")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent

if "history_messages" not in st.session_state:
    st.session_state["history_messages"] = []

for message in st.session_state["history_messages"]:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# user input prompt
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["history_messages"].append({"role": "user", "content": prompt})

    response_message = []
    with st.spinner(text="thinking..."):
        result = st.session_state["agent"].execute_stream(prompt)

        def stream_result(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        st.chat_message("assistant").write_stream(stream_result(result,response_message))
        st.session_state["history_messages"].append({"role": "assistant", "content": response_message[-1]})