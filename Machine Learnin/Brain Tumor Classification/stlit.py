#importing libraries
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import base64


#markdown
st.markdown('<h1 style="color:black;">Brain Tumor Detection</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="color:black;">The tumor detection model classifies image into following categories:</h2>', unsafe_allow_html=True)
st.markdown('<ul style="color:black;"> <li>Glioma</li><li>Meningioma</li><li>Pituitary</li></ul>', unsafe_allow_html=True)


# background image to streamlit

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('./canvas.png')


##increasing brightness
def increase_bright(img):
    #getting the hue, saturation, and value of the image
    h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    
    #value to increase the brightness by
    value=30

    lim = 255 - value
    #stop overflow
    v[v > lim] = 255
    #increase brightness
    v[v <= lim] += value
    
    #making the final image
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


#uploading the file
upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
c1, c2= st.columns(2)
if upload is not None:
  file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
  im = cv2.imdecode(file_bytes, 1)
  img= np.asarray(im)
  image= cv2.resize(img,(512, 512))
  img= np.expand_dims(img, 0)
  c1.header('Input Image')
  c1.image(im)
  c1.write(img.shape)

  image = increase_bright(image)


  ##region filling
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
  image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


  #loading the model
  classes = dict(zip([0,1,2],['glioma', 'pituitary', 'meningioma']))
  model = tf.keras.saving.load_model('bright_region_filling.keras')
  pred = model.predict(image.reshape(1,512,512,3))
  c2.header('Output')
  c2.subheader('Predicted tumor category: ')
  c2.write(f'Category - {classes[pred.argmax(axis=-1)[0]].title()}')
  c2.write(f'Confidence - {np.max(pred)*100:.2f}%')
  



