import streamlit as st

def menu():
  st.sidebar.page_link("app.py", label="Home")
  st.sidebar.page_link("pages/account.py", label="Data Preprocessing")
  st.sidebar.page_link("pages/chat.py", label="Chat")
  st.sidebar.page_link("pages/ocr.py", label="Document OCR")