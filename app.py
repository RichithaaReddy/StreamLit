import PIL
import streamlit as st
from ultralytics import YOLO
import requests
import cv2
import numpy as np
import tempfile
import os
import io
# class_label_to_disease = {
#     0: "Alternaria",
#     1: "Anthracnose",
#     2: "Bacterial Blight",
#     3: "Cercospora",
#     5: "Healthy",
# }
class_info = {
    0: {
        'name': 'Alternaria',
        'description': 'This class represents a unhealthy pomegranate.',
        'translation_key': 'disease0_translation_key',
    },
    1: {
        'name': 'Anthracnose',
        'description': 'This class represents the first type of pomegranate disease.',
        'translation_key': 'disease1_translation_key',
    },
    2: {
        'name': 'Bacterial_Blight',
        'description': 'This class represents the second type of pomegranate disease.',
        'translation_key': 'disease2_translation_key',
    },
    3: {
        'name': 'Cercospora',
        'description': 'This class represents the second type of pomegranate disease.',
        'translation_key': 'disease14_translation_key',
    },
    4: {
        'name': 'Healthy',
        'description': 'This class represents the first type of pomegranate disease.',
        'translation_key': 'healthy_translation_key',
    },
}
def translate_text_rapidapi(text_to_translate, target_language):
    if target_language == 'en':
        return text_to_translate
        
    url = "https://google-translation-unlimited.p.rapidapi.com/translate"
    headers = {
        'content-type': 'application/x-www-form-urlencoded',
        'X-RapidAPI-Key': '4db7d01892msh1a35513855d61b8p176a71jsn3c4545830608',
        'X-RapidAPI-Host': 'google-translation-unlimited.p.rapidapi.com'
    }
    data = {
        'texte': text_to_translate,
        'to_lang': target_language
    }

    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        translation_data = response.json().get('translation_data', {})
        if translation_data:
            translated_text = translation_data.get('translation', '')
            return translated_text
        else:
            st.write("Unexpected response format:")
            st.write(response.text)
            return "Error: Unexpected response format"
    else:
        return f"Error: {response.status_code}"
model_path = 'weights/best.pt'
languages = {
    'en':'English',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'te': 'Telugu',
    'ta': 'Tamil',
    'fa': 'Persian',
    'tr': 'Turkish',
    'ar': 'Arabic',
    'es': 'Spanish',
    'fr': 'French',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ur': 'Urdu',
    'ka': 'Georgian',
    'it': 'Italian',
    'el': 'Greek',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'or': 'Oriya',
    'pa': 'Punjabi'
}

st.set_page_config(
    page_title="Pomegranate Disease Detection", 
    page_icon="🤖",     
    layout="wide",     
    initial_sidebar_state="expanded",  
)
selected_language_value = st.sidebar.selectbox("Select Language:", list(languages.values()))
selected_language = next(key for key, value in languages.items() if value == selected_language_value)


with st.sidebar:
    st.header("Image Config")     
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100
st.title("Pomegranate Disease Detection")
st.caption('Then click the :blue[Detect Objects] button and check the result.')
col1, col2 = st.columns(2)
with col1:
    if source_img:
        uploaded_image = PIL.Image.open(source_img)
        st.image(source_img,
                 caption="Uploaded Image",
                 use_column_width=True
                 )

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if st.sidebar.button('Detect Objects'):
    res = model.predict(uploaded_image, conf=confidence)
    boxes = res[0].boxes
    confidences = boxes.conf
    class_labels = boxes.cls

    # Display the detected image with bounding boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    with col2:
        st.image(res_plotted, caption='Detected Image', use_column_width=True)

        try:
            st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
            with st.expander("Detection Results"):
                detected_classes = set()  # Initialize detected_classes set to avoid UndefinedVariable error
                for i, box in enumerate(boxes):
                    class_label = int(class_labels[i])  # Convert to int if needed
                    confidence = float(confidences[i])  # Convert to float if needed
                    detected_class_info = class_info.get(class_label, {})

                    # Check if the detected class has already been displayed
                    if detected_class_info['name'] not in detected_classes:
                        detected_classes.add(detected_class_info['name'])

                        # Display detected class name, description, and confidence
                        st.markdown(f"<span style='color: blue;'>{detected_class_info.get('name', 'Unknown')}</span>", unsafe_allow_html=True)
                        # st.write(f"Description: {detected_class_info.get('description', '')}")
                        # st.write(f"Confidence: {confidence}")

                        # Check if translation is required and show translated description
                        if selected_language != 'en':
                            translation_key = detected_class_info.get('translation_key', '')
                            # translated_class = translate_text_rapidapi(detected_class_info.get('name', ''), selected_language)
                            translated_description = translate_text_rapidapi(detected_class_info.get('description', ''), selected_language)
                            # st.write(f"Translated Class: {translated_class}")
                            st.write(f" {translated_description}")
                        else:
                            # translation_key = detected_class_info.get('translation_key', '')  
                            # st.write(f"Class: {detected_class_info.get('name', '')}")
                            st.write(f" {detected_class_info.get('description', '')}")


        except Exception as ex:
            st.write("")


# /////////////////////////////////////////////////////////////////////////////////////////////////////
