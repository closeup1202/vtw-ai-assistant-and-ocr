import streamlit as st

def half_wide():
  margins_css = """
  <style>
    .main {
      padding: 0 50px
    }
    </style>
  """
  st.markdown(margins_css, unsafe_allow_html=True)