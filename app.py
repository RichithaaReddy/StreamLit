import PIL
import streamlit as st
from ultralytics import YOLO
import requests
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
    res = model.predict(uploaded_image,
                        conf=confidence
                        )
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    with col2:
        st.image(res_plotted,
                 caption='Detected Image',
                 use_column_width=True
                 )
        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.xywh)
        except Exception as ex:
            st.write("No image is uploaded yet!")
            
text_to_translate = "It is fruit disease "
target_language = languages[selected_language]

# Use st.spinner to indicate loading
with st.spinner("Translating..."):
    if target_language != 'en':  # Only translate if the selected language is not English
        translated_text = translate_text_rapidapi(text_to_translate, target_language)
        st.write(f"{translated_text}")
        

    else:
        translated_text = text_to_translate
        st.write(f"{text_to_translate}")
        
