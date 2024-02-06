# helper.py

# Import necessary libraries
import cv2
import streamlit as st

def detect_objects_in_image(conf, model):
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Perform object detection
        result = model.predict(image)

        # Display the result
        st.image(result, caption='Detected Image', use_column_width=True)

# Add other helper functions like play_stored_video, play_youtube_video if needed
