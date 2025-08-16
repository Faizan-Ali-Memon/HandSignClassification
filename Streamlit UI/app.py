# app.py (Compact UI)
import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

DEFAULT_MODEL_PATH = r"C:\Users\mahmo\Desktop\Hand Sign Classification\model\resnet50_model.h5"
CLASS_LABELS = {i: str(i) for i in range(6)}

st.set_page_config(page_title="Hand Sign Classifier (0–5)", layout="wide")

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at:\n{path}")
    return tf.keras.models.load_model(path)

def get_target_size(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    try:
        return (shape[2] or 224, shape[1] or 224)
    except Exception:
        return (224, 224)

def preprocess_image(pil_img: Image.Image, target_size):
    img = pil_img.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

# Load model
model = load_model(DEFAULT_MODEL_PATH)
target_size = get_target_size(model)

st.title("🤚 Hand Sign Classifier (0–5)")
st.caption("Upload an image → instantly see predicted class + confidence.")

uploaded = st.file_uploader("Upload a hand sign image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    x = preprocess_image(image, target_size)
    preds = model.predict(x, verbose=0)
    probs = preds[0] if preds.ndim == 2 else preds.squeeze()

    if not np.isclose(probs.sum(), 1.0, atol=1e-3):
        exp = np.exp(probs - np.max(probs))
        probs = exp / exp.sum()

    predicted_idx = int(np.argmax(probs))
    predicted_label = CLASS_LABELS[predicted_idx]
    confidence = float(probs[predicted_idx] * 100.0)

    df = pd.DataFrame({
        "class_label": [CLASS_LABELS[i] for i in CLASS_LABELS],
        "confidence_%": (probs * 100.0).round(2)
    }).sort_values("confidence_%", ascending=False)

    # --- First row: image, metrics, top-3 ---
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", width=220)
    with col2:
        st.metric("Predicted Class", predicted_label)
        st.metric("Confidence", f"{confidence:.2f} %")
    with col3:
        st.write("**Top-3 Classes**")
        for i, row in df.head(3).iterrows():
            st.write(f"{row['class_label']}: {row['confidence_%']}%")

    # --- Second row: chart & table ---
    col4, col5 = st.columns([1, 1])
    with col4:
        st.bar_chart(df.set_index("class_label"))
    with col5:
        st.dataframe(df.reset_index(drop=True), use_container_width=True)
else:
    st.info("Upload a .jpg/.jpeg/.png image to get a prediction.")
