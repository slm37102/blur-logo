import streamlit as st
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from google.oauth2 import service_account
from google.cloud import vision
import io
import pandas as pd

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = vision.ImageAnnotatorClient(credentials=credentials)

def detect_logos(path):
    """Detects logos in the file."""

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.logo_detection(image=image)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    
    return response

def blur_logos(path, option):
    response = detect_logos(path)
    logos = response.logo_annotations
    n_logo = len(logos)
    # Read in image
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    temp = image.copy()
    descriptions = []
    scores = []
    for logo in logos:
        descriptions.append(logo.description)
        scores.append(logo.score)
        
        # Create ROI coordinates
        x = logo.bounding_poly.vertices[0].x
        y = logo.bounding_poly.vertices[0].y
        w = logo.bounding_poly.vertices[2].x - logo.bounding_poly.vertices[0].x
        h = logo.bounding_poly.vertices[2].y - logo.bounding_poly.vertices[0].y

        # Grab ROI with Numpy slicing and blur
        ROI = image[y:y+h, x:x+w]
        blur = None 
        if option == 'Gaussian Blur':
          blur = cv2.GaussianBlur(ROI, (151,151), 0)
        elif option == 'Pixel Blur':
          # Resize input to "pixelated" size
          temp = cv2.resize(ROI, (8, 8), interpolation=cv2.INTER_LINEAR)

          # Initialize output image
          blur = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        # Insert ROI back into image
        image[y:y+h, x:x+w] = blur
        cv2.rectangle(temp,(x,y),(x+w,y+h),(0,255,0),2)
    if showBB:
        st.image(temp)
    return image, n_logo, descriptions, scores

st.title('Blur logo')
image = st.file_uploader("Upload Image")
filepath = 'image.png'
option = st.selectbox(
    'Blur Method',
    ('Gaussian Blur', 'Pixel Blur'))
showBB = st.checkbox('Show Location of Logo')
button_pressed = st.button('Blur Logo')

if button_pressed and image:
    image = Image.open(image)
    image.save(filepath)
    col1, col2 = st.columns(2)
    with col1:
        st.header("Before Blur")
        if not showBB:
            st.image(image)
        image, n_logo, descriptions, scores = blur_logos(filepath, option)
    with col2:
        st.header("After Blur")
        st.image(image)
    cv2.imwrite(filepath, image)
    with open(filepath, "rb") as file:
        st.download_button(
            label="Save Image",
            data=file,
            file_name=filepath,
            mime='image/png')
    st.success(f'Number of logos founded: {n_logo}')
    df = pd.DataFrame(data={
        'Logo Name': descriptions,
        'Confidence': scores
    })
    st.dataframe(df)

elif button_pressed:
    st.error('Image not uploaded')