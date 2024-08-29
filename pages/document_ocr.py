import streamlit as st
import easyocr
from PIL import ImageFont, ImageDraw, Image
import cv2 

with st.spinner('Wait for it...'):
  reader = easyocr.Reader(['ko','en'], gpu=False, model_storage_directory="ocr/.EasyOCR/model") 
  result = reader.readtext('ocr/sample/business_certificate.jpeg')

  img = cv2.imread("ocr/sample/business_certificate.jpeg")
  img = Image.fromarray(img)
  font = ImageFont.truetype("fonts/KoPub Dotum Medium.ttf", 20)
  draw = ImageDraw.Draw(img)
  for i in result :
      x = i[0][0][0] 
      y = i[0][0][1] 
      w = i[0][1][0] - i[0][0][0] 
      h = i[0][2][1] - i[0][1][1]
      draw_text_position = (int((2*x + w) / 2), y-20)
      color = [0, 0, 0]
      draw.rectangle(((x, y), (x+w, y+h)), outline=tuple(color), width=2)
      draw.text(draw_text_position, str(i[1]), font=font, fill=tuple(color),)
st.image(img, caption="certificate for buisiness registration", output_format="auto")