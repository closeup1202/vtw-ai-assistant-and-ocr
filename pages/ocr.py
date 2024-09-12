import streamlit as st
import easyocr
from PIL import ImageFont, ImageDraw, Image
import cv2 
from sidebar import menu
from style import global_style
import numpy as np

def image_save(uploaded_fille):
  byte_value = uploaded_fille.getvalue()
  saved_path = f"ocr/images/{uploaded_file.name}"
  with open(saved_path, "wb") as f:
    f.write(byte_value)
  return saved_path

@st.cache_data(show_spinner=False)
def get_ocr(saved_image):
  reader = easyocr.Reader(['ko','en'], model_storage_directory="ocr/.EasyOCR/model", gpu=False) 
  saved_image = cv2.imread(saved_image)
  # TODO: 파일명이 한글일 경우 분기 처리
  saved_image = saved_image.astype(np.uint8)
  result = reader.readtext(saved_image)
  img_bounded_box = Image.fromarray(saved_image)
  font = ImageFont.truetype("fonts/KoPub Dotum Medium.ttf", 15)
  draw = ImageDraw.Draw(img_bounded_box)
  extracted_text = []
  for index, value in enumerate(result) :
      x = value[0][0][0] 
      y = value[0][0][1] 
      w = value[0][1][0] - x
      h = value[0][2][1] - value[0][1][1]
      extracted_text.append(str(index+1) + ". " + value[1])
      draw_text_position = (int(x), y-20)
      draw.rectangle(((x, y), (x+w, y+h)), outline=tuple([57, 245, 54]), width=2)
      draw.text(draw_text_position, str(index+1), font=font, fill=tuple([0, 0, 0]))
  return (img_bounded_box, extracted_text)

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