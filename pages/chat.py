import streamlit as st
import random
import time
from sidebar import menu
from style import global_style
from st_copy_to_clipboard import st_copy_to_clipboard
from rag.generate_langgraph_chat import get_graph_response

def reset_conversation():
  st.session_state.message_list = []
  del st.session_state["message_list"]

def stream_data(value):
  for word in value.split(" "):
    yield word + " "
    time.sleep(0.02)

st.set_page_config(page_title="AI Playground - Chat", page_icon="🤖", layout="centered")
menu()
global_style()
st.title("Chat", help="'clear'를 입력하시면 대화가 초기화됩니다", anchor=False)

if 'message_list' not in st.session_state:
  st.session_state.message_list = []

for message in st.session_state.message_list:
  role = message["role"]
  content = message["content"]
  with st.chat_message(role):
    st.write(content)

def test():
  if st.session_state.user_input == "clear" and 'message_list' in st.session_state and st.session_state.message_list != []:
    reset_conversation()

if user_question := st.chat_input(placeholder="사내 자료, 회사 정보, 문체비 규정 등 궁금한 내용들을 질문해주세요", on_submit=test, key="user_input"):
  if user_question != "clear":
    with st.chat_message("user"):
      st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.chat_message("ai"):
      with st.container():
        col_message, col_copy = st.columns([0.9, 0.1], vertical_alignment="center")
        with col_message:
          with st.spinner(" "):
            ai_response = get_graph_response(user_question)
            ai_message = st.write_stream(stream_data(ai_response["answer"]))
        with col_copy:
          col1, col2 = st.columns(2, vertical_alignment="center")
          with col1:
            if 'source' in ai_response:
              st.text("", help=ai_response["source"])
          with col2:
            st_copy_to_clipboard(text=ai_message, before_copy_label="✨", key=random.random())

    st.session_state.message_list.append({"role": "ai", "content": ai_message})