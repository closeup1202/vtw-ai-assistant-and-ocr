import streamlit as st

def menu():
  st.sidebar.page_link("app.py", label="Home")
  st.sidebar.page_link("pages/crawling.py", label="Crawling & Summary")
  st.sidebar.page_link("pages/chat.py", label="Chat")
  st.sidebar.page_link("pages/ocr.py", label="Document OCR")