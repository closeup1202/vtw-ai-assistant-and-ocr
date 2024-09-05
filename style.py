import streamlit as st

@st.cache_resource
def global_style(
  middle_frame: bool = False
):
  css = "";
  if middle_frame == True:
    css = """
    <style>
      .main {
        padding: 0 55px
      }
      .main, body {
        letter-spacing: -0.5px
      }
      .stChatInput textarea, 
      .stChatInput button {
        padding: 15px 20px;
      }
    </style>
    """
  else:
    css = """
      <style>
        .main, body {
          letter-spacing: -0.5px
        }

        .stChatInput textarea, 
        .stChatInput button {
          padding: 15px 20px;
        }
      </style>
      """
  st.markdown(css, unsafe_allow_html=True)
