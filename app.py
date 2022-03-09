import numpy as np
import streamlit as st
from keras.applications.vgg16 import VGG16
from PIL import Image
import keras
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

model = VGG16()

img_file_buffer = st.file_uploader('Upload image to be predicted: ', type=['png','jpg','jpeg'])
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    image = np.array(image)

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image(image, width=250, caption='Uploaded Image')

with col3:
    st.write(' ')

#image = keras.preprocessing.image.load_img(image, target_size=(224, 224))
#image = keras.preprocessing.img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
y_hat = model.predict(image)
label = decode_predictions(y_hat)[0][0]

out = f'The uploaded image is that of {label[1]}, predicted with {label[2]*100}% certainty.'

st.write(out)
