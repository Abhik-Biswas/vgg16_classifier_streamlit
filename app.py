import numpy as np
import streamlit as st
from keras.applications.vgg16 import VGG16
from PIL import Image
import keras
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

model = VGG16()

from keras.preprocessing.image import load_img

import streamlit as st

from tempfile import NamedTemporaryFile

st.set_option('deprecation.showfileUploaderEncoding', False)

buffer = st.file_uploader("Image here pl0x")
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    image = keras.preprocessing.image.load_img(temp_file.name, target_size=(224,224))
    #st.write(load_img(temp_file.name))

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image(image, width=250, caption='Uploaded Image')

with col3:
    st.write(' ')

#image = keras.preprocessing.image.load_img(image, target_size=(224, 224))
image = keras.preprocessing.img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
y_hat = model.predict(image)
label = decode_predictions(y_hat)[0][0]

out = f'The uploaded image is that of {label[1]}, predicted with {label[2]*100}% certainty.'

st.write(out)
