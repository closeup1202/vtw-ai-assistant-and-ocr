import streamlit as st

st.set_page_config(page_title="VTW AI ASSISTANT", page_icon="🤖")

pages = {
    "Home" : [
        st.Page(page="home.py", title="Introduce", default=True)
    ],
    "Account": [
        st.Page("pages/account.py", title="Manage your account"),
    ],
    "Playground": [
        st.Page("pages/chat.py", title="Chat"),
        st.Page("pages/document_ocr.py", title="Document OCR"),
    ],
}

pg = st.navigation(pages)
pg.run()