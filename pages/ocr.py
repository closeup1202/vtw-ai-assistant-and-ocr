import streamlit as st
from sidebar import menu
from style import global_style
from ocr.easyocr import image_save, get_ocr

st.set_page_config(page_title="AI Playground - Document OCR", page_icon="🤖", layout="wide")
global_style(middle_frame=True)
menu()

st.title("OCR")

uploaded_file = st.file_uploader(
  label="Upload a document to get OCR results",
  accept_multiple_files=False,
  type=['jpg', 'png']
)

if uploaded_file is not None:
  saved_image = image_save(uploaded_file)
  if saved_image is not None:
    st.html("<hr style='margin: -8px 0' />")
    with st.spinner():
      (img_bounded_box, extracted_text) = get_ocr(saved_image)

    col_left, col_right = st.columns(2, gap="large", vertical_alignment="top")

    with col_left:
      st.html("<h4>Image with text bounding box</h4>")
      st.image(img_bounded_box, caption=f"OCR Result", channels="BGR", output_format="auto")

    with col_right:
      st.html("<h4>Extracted characters</h4>")
      for text in extracted_text:
        st.write(text)