import streamlit as st
from sidebar import menu
from style import global_style

st.set_page_config(page_title="Ai Playground - Data Preprocessing", page_icon="🤖", layout="wide")
menu()
global_style()