import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model


# Load model
model = load_model(r"C:\Users\kaios\benh_la_tra\model_netnet1.keras", compile=False)
# Class names
class_names = [
    "Anthracnose", 
    "Algal leaf", 
    "Bird eye spot", 
    "brouwn blight", 
    "gray light", 
    "healthy", 
    "red leaf spot", 
    "white spot"
]

IMG_SIZE = (256, 256)

def preprocess_image(image: Image.Image):
    if image.size != IMG_SIZE:
        image = image.resize(IMG_SIZE)

    img_array = img_to_array(image)

    img_norm = img_array / 255.0

    img_input = np.expand_dims(img_norm, axis=0)

    return img_input

def classify_image(image):
    img_input = preprocess_image(image)
    preds = model.predict(img_input)
    pred_idx = np.argmax(preds)
    pred_score = float(np.max(preds))

    return class_names[pred_idx], pred_score


st.set_page_config(page_title="Mimi AI", layout="wide")

st.markdown("<h1 style='text-align:center;'>Chương trình demo phân loại bệnh trên lá cây chè!</h1>", unsafe_allow_html=True)

tabs = st.tabs(["Phân loại bệnh trên lá cây chè"])

with tabs[0]:

    st.write("Hãy tải ảnh lên và nhấn **Xử Lý** để phân loại tình trạng.")

    uploaded_file = st.file_uploader("Xin Nhập Ảnh Vào", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        # Chia màn hình thành 3 cột: [Trống] - [Ảnh] - [Trống]
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown("### 🖼 Ảnh Gốc")
            st.image(image, use_container_width=True, width='content')

        if st.button("Xử Lý"):
            class_name, score = classify_image(image)

            with col3:
                st.markdown("### 🧠 Kết Quả Phân Loại")
                st.image(image, use_container_width=True, width='content')

                st.success(f"**Dự đoán:** {class_name}")
                st.info(f"**Confidence:** {score:.4f}")

    else:
        with col1:
            st.info("Vui lòng upload ảnh để bắt đầu.")
