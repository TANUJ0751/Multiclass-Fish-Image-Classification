import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# =====================
# GPU CONFIG (optional but speeds up training/inference)
# =====================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        st.write("✅ GPU memory growth enabled")
    except RuntimeError as e:
        st.write(e)

# =====================
# CUSTOM OBJECTS (replace with yours if needed)
# =====================
# Example: if you used any custom layers or metrics during training, define them here
def my_preprocess(x):
    return x / 255.0

# Example metric (delete if not used)
def top_3_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

# =====================
# LOAD MODEL
# =====================
@st.cache_resource
def load_selected_model(model_path):
    return load_model(model_path, custom_objects={
        "my_preprocess": my_preprocess,
        "top_3_accuracy": top_3_accuracy
    })

model_path = "models/custom_cnn_model.h5"  # Change to your model path
model = load_selected_model(model_path)

# =====================
# APP UI
# =====================
st.title("🧠 Custom CNN Image Classifier")
st.write("Upload an image and let the model classify it.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# =====================
# PREDICTION FUNCTION
# =====================
def preprocess_image(image):
    img = image.resize((224, 224))  # Change to your input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds) * 100

    st.write(f"**Predicted Class:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

