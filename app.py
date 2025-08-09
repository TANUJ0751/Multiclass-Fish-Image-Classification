import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import pandas as pd
import os

# === CONFIG ===
IMG_SIZE = (224, 224)

# 🔸 Define available models and their paths
MODEL_OPTIONS = {
    "Custom CNN": "./models/custom_cnn_model.h5",
    "VGG16": "./models/VGG16.h5",
    "ResNet50": "./models/ResNet50.h5",
    "MobileNetV2": "./models/MobileNetV2.h5",
    "InceptionV3": "./models/InceptionV3.h5",
    "EfficientNetB0": "./models/EfficientNetB0.h5"
}
# 🔸 Define corresponding preprocess_input functions
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnetb0_preprocess

PREPROCESS_FUNCS = {
    "Custom CNN": lambda x: x / 255.0,  # Normalization for custom CNN
    "VGG16": vgg16_preprocess,
    "ResNet50": resnet50_preprocess,
    "MobileNetV2": mobilenetv2_preprocess,
    "InceptionV3": inceptionv3_preprocess,
    "EfficientNetB0": efficientnetb0_preprocess
}
CLASS_NAMES = sorted(os.listdir("./data/train"))

# === STREAMLIT PAGE CONFIG ===
st.set_page_config(page_title="Fish Image Classifier", layout="centered")
st.title("🐟 Multiclass Fish Image Classification")
st.markdown("Upload a fish image and select a model to predict the species.")

# === MODEL SELECTION ===
selected_model_name = st.selectbox("Choose a model for prediction:", list(MODEL_OPTIONS.keys()))

# === Select preprocess function based on model ===
preprocess_input_fn = PREPROCESS_FUNCS[selected_model_name]

@st.cache_resource
def load_selected_model(model_path):
    return tf.keras.models.load_model(
        model_path
    )
model = load_selected_model(MODEL_OPTIONS[selected_model_name])

# === UPLOAD IMAGE ===
uploaded_file = st.file_uploader("Upload a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    if selected_model_name=="Custom CNN":
        img_array = img_array / 255.0  # Normalize
    else:
        img_array = preprocess_input_fn(img_array)
    # Predict
    predictions = model.predict(img_array)[0]
    top_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[top_idx]
    confidence = predictions[top_idx]

    # Layout: columns
    col1, col2 = st.columns([2, 3])

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)


    with col2:
        st.markdown("### Prediction:")
        st.success(f"**{predicted_class}**")
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")
        st.markdown(f"**Model Used:** {selected_model_name}")

    # Create confidence DataFrame
    confidence_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Confidence": predictions
    })
    confidence_df = confidence_df.sort_values(by="Confidence", ascending=False)
    confidence_df["Confidence (%)"] = confidence_df["Confidence"].apply(lambda x: f"{x * 100:.2f}%")
    confidence_df = confidence_df[["Class", "Confidence (%)"]]
    confidence_df.index = range(1, len(confidence_df) + 1)
    confidence_df.index.name = "Rank"

    # Display table below columns
    st.markdown("### Confidence Scores (Ranked):")
    st.table(confidence_df)
