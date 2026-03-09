import streamlit as st
import config_data as config
from rag import RagService

st.set_page_config(
    page_title="我的智能客服",  # 浏览器标签栏文字
    page_icon="🤖",            # 标签栏图标
)
st.title("智能客服")
st.divider()

if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好,有什么可以帮您?"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()



for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])


prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    with st.spinner("思考中..."):
        res = st.session_state["rag"].chain.stream(
            {"input": prompt}, config.session_config
        )

        def text_stream():
            for chunk in res:
                if hasattr(chunk, "content"):
                    text = chunk.content
                else:
                    text = str(chunk)
                if text:
                    yield text

        assistant_text = st.chat_message("assistant").write_stream(text_stream())
        st.session_state["message"].append(
            {"role": "assistant", "content": assistant_text}
        )
