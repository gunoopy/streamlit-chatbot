import streamlit as st
import os

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler


## 환경설정
st.set_page_config(page_title="Chatbot", page_icon="💬")
MODEL = ["gpt-3.5-turbo", "gpt-4"][1] # 사용할 모델

## 스트리밍 답변을 위한 콜백
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)


## 제목
st.title("🐔 Chat with LangChain 🔗")
st.caption(f"> {MODEL}, ConversationChain")


## API key 입력
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not openai_api_key: # key 입력 없을 시(초기 상태)
    st.info("Please add your OpenAI API key to continue.")
    st.stop()


## 모델 생성
if "model" not in st.session_state :
    llm = ChatOpenAI(temperature=0.7,
                     model=MODEL,
                     streaming=True,
                     api_key=openai_api_key,
                     openai_api_key=openai_api_key)
    st.session_state["model"] = ConversationChain(llm=llm)



## 메시지 표출
for m in st.session_state.model.memory.chat_memory.messages :
    if type(m) == HumanMessage :
        with st.chat_message("user") :
            st.write(m.content)
    else :
        with st.chat_message("assistant") :
            st.write(m.content)



## 사용자 프롬프트 입력
if prompt := st.chat_input("Ask me anything!") :
    try :
        with st.chat_message("user") :
            st.markdown(prompt)
        with st.chat_message("assistant") :
            st_cb = StreamHandler(st.empty())
            response = st.session_state.model.run(prompt,
                                                  callbacks=[st_cb])
    except Exception :
        st.warning('Please enter your valid OpenAI API key!', icon="⚠")
