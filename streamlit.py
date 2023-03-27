import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
import numpy as np
from PIL import Image,ImageOps

st.title('Sen klasslardagi istalgan rasmingni chiz!Men esa uni bashorat qilaman')
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 10)
choose = st.sidebar.selectbox(
    "Classes:", class_names
)
if choose:
    class_img = Image.open('class_images/'+choose+'.png')
    gray_img = ImageOps.grayscale(class_img)
    st.sidebar.image(gray_img)

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color='#FFFFFF',
    background_color='#000000',
    update_streamlit=True,
    width = 250,
    height=250,
    drawing_mode='freedraw',
    point_display_radius= 0,
    key="canvas",
)

if st.button(label='Predict') and canvas_result.image_data is not None:
    img = canvas_result.image_data
    img = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    img = cv2.resize(img,(28,28))
    img = img.astype('float32')
    img /= 255
    img = np.array([img])
    model = keras.models.load_model('model\cnn_model.h5')
    st.text(class_names[np.argmax(model.predict(img))])

