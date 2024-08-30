import streamlit as st
from sidebar import menu
from rag.llm import get_ai_response
from style import half_wide

half_wide()
menu()

st.title("AI Assistant")
st.caption("회사 내규 등 사내정보에 대해 답변해드립니다!")

if 'message_list' not in st.session_state:
  st.session_state.message_list = []

for message in st.session_state.message_list:
  with st.chat_message(message["role"]):
    st.write(message["content"])

if user_question := st.chat_input(placeholder="사내 자료, 회사 정보 등 궁금한 내용들을 말씀해 주세요!"):
  with st.chat_message("user"):
    st.write(user_question)
  st.session_state.message_list.append({"role": "user", "content": user_question})

  ai_response = get_ai_response(user_question)
  with st.chat_message("ai"):
    # ai_message = st.write_stream(ai_response)
    ai_message = st.write(ai_response)
  st.session_state.message_list.append({"role": "ai", "content": ai_message})