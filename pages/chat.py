import streamlit as st
from sidebar import menu
from style import global_style
from rag.generate import get_ai_response
from st_copy_to_clipboard import st_copy_to_clipboard
import random

def reset_conversation():
  st.session_state.message_list = []
  del st.session_state["message_list"]

st.set_page_config(page_title="AI Playground - Chat", page_icon="🤖", layout="centered")
menu()
global_style()

col_title, col_button = st.columns([0.9, 0.1], vertical_alignment="bottom")
with col_title:
  st.title("Chat")
with col_button:
  if 'message_list' in st.session_state:
    if st.session_state.message_list != []:
      st.button('clear', on_click=reset_conversation)

if 'message_list' not in st.session_state:
  st.session_state.message_list = []

for message in st.session_state.message_list:
  role = message["role"]
  content = message["content"]
  with st.chat_message(role):
    st.write(content)

if user_question := st.chat_input(placeholder="사내 자료, 회사 정보, 문체비 규정 등 궁금한 내용들을 질문해주세요"):
  with st.chat_message("user"):
    st.write(user_question)
  st.session_state.message_list.append({"role": "user", "content": user_question})

  with st.chat_message("ai"):
    with st.container():
      col_message, col_copy = st.columns([0.9, 0.1], vertical_alignment="center")
      with col_message:
        with st.spinner(" "):
          ai_message = st.write_stream(get_ai_response(user_question))
      with col_copy:
        st_copy_to_clipboard(text=ai_message, before_copy_label="✨", key=random.random())
  st.session_state.message_list.append({"role": "ai", "content": ai_message})