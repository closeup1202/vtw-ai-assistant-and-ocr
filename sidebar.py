import streamlit as st

def menu():
  st.sidebar.page_link("app.py", label="Home")
  st.sidebar.page_link("pages/account.py", label="Account")
  st.sidebar.page_link("pages/chat.py", label="Chat")
  st.sidebar.page_link("pages/document_ocr.py", label="Document OCR")