import cv2
import streamlit as st
def detect_objects_in_image(conf, model):
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        result = model.predict(image)
        st.image(result, caption='Detected Image', use_column_width=True)

