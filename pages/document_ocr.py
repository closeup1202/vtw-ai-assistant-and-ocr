import streamlit as st
import easyocr
from PIL import ImageFont, ImageDraw, Image
import cv2 
import base64
from sidebar import menu

def image_save(uploaded_fille):
  byte_value = uploaded_fille.getvalue()
  saved_path = f"ocr/images/{uploaded_file.name}"
  with open(saved_path, "wb") as f:
    f.write(byte_value)
  return saved_path

def get_ocr(saved_image):
  st.divider()
  with st.spinner():
    reader = easyocr.Reader(['ko','en'], gpu=False, model_storage_directory="ocr/.EasyOCR/model") 
    result = reader.readtext(saved_image)
    img = cv2.imread(saved_image)
    img = Image.fromarray(img)
    font = ImageFont.truetype("fonts/KoPub Dotum Medium.ttf", 20)
    draw = ImageDraw.Draw(img)
    textList = []
    for i in result :
        x = i[0][0][0] 
        y = i[0][0][1] 
        w = i[0][1][0] - i[0][0][0] 
        h = i[0][2][1] - i[0][1][1]
        textList.append(i[1])
        draw_text_position = (int((2*x + w) / 2), y-20)
        draw.rectangle(((x, y), (x+w, y+h)), outline=tuple([57, 245, 54]), width=2)
        draw.text(draw_text_position, str(i[1]), font=font, fill=tuple([0, 0, 0]))

  col_left, col_right = st.columns(2, gap="large", vertical_alignment="top")

  with col_left:
    st.html("<h4>image with boxing</h4>")
    st.image(img, caption=f"OCR Result", channels="BGR", output_format="auto")

  with col_right:
    st.html("<h4>extracted text</h4>")
    for text in textList:
      st.write(text)

menu()

st.header("Upload a document to get OCR results")

uploaded_file = st.file_uploader(
  label="please upload a image",
  accept_multiple_files=False,
  type=['jpg', 'pdf', 'png']
)

if uploaded_file is not None:
  if uploaded_file.type == "application/pdf": # application/pdf, image/jpeg
    bytes_data = uploaded_file.read()
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

  saved_image = image_save(uploaded_file)

  if saved_image is not None:
    get_ocr(saved_image)