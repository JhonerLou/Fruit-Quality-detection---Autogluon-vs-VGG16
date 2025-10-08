import streamlit as st
from PIL import Image
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

# Load the pre-trained AutoGluon model
predictor = MultiModalPredictor.load(r"c:\Users\acer\Downloads\AutogluonModels\ag-20241215_160144")

# Streamlit app layout
st.title("FreshAI")

st.write("Upload an image, and it will predict the quality for you.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# When an image is uploaded
if uploaded_file is not None:
    # Open the image and display it
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Save the image to a temporary file
    temp_file_path = "temp_image.jpg"
    image.save(temp_file_path)

    # Create a DataFrame for the prediction
    data = pd.DataFrame({"image": [temp_file_path]})
    
    # Perform prediction
    prediction = predictor.predict(data)
    
    # Display the prediction result
    st.write("Prediction:", prediction)
