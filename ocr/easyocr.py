import streamlit as st
import easyocr
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import tempfile


def image_save(uploaded_file):
  # byte_value = uploaded_file.getvalue()
  # saved_path = f"tmp/{uploaded_file.name}"
  with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp_file:
    temp_file.write(uploaded_file.read())
    return temp_file.name
  # with open(saved_path, "wb") as f:
  #   f.write(byte_value)
  # return saved_path

@st.cache_data(show_spinner=False)
def get_ocr(saved_image):
  reader = easyocr.Reader(['ko', 'en'], 
                          model_storage_directory="ocr/.EasyOCR/model", 
                          # user_network_directory="ocr/.EasyOCR/user_network",
                          # recog_network="custom",
                          gpu=True) 
  saved_image = np.array(Image.open(saved_image))
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