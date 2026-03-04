import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ---------------- CONFIG ----------------
IMG_SIZE = 224
MODEL_PATH = "model/currency_mobilenet_model.h5"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# Correct MobileNetV2 preprocessing
def preprocess(img):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# ---------------- UI DESIGN ----------------
st.set_page_config(page_title="Fake Currency Detector", layout="centered")

st.markdown("""
<style>
body { 
    background: linear-gradient(135deg, #0f0f0f, #2c2c2c);
    color: white;
}
.block-container {
    backdrop-filter: blur(20px);
    background: rgba(255,255,255,0.05);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.1);
}
h1 { text-align:center; color:#00eaff; }
</style>
""", unsafe_allow_html=True)

st.title("₹ Fake Currency Detector")

# ---------------- LOAD MODEL ----------------
model = load_model()
if not model:
    st.error("⚠️ Model not found. Please place 'currency_mobilenet_model.h5' inside /model/")
    st.stop()

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)
img = None

with col1:
    st.subheader("📤 Upload / Capture Note")

    uploaded = st.file_uploader("Upload Note Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Note", use_container_width=True)

with col2:
    st.subheader("🔍 Prediction")

    # PREDICTION BUTTON
    predict_btn = st.button("🔎 Predict Note Authenticity", use_container_width=True)

    if predict_btn:
        if img is None:
            st.warning("⚠️ Please upload an image first.")
        else:
            with st.spinner("Analyzing Note..."):
                x = preprocess(img)
                prob = float(model.predict(x)[0][0])

                # Prediction logic
                label = "REAL" if prob > 0.15 else "FAKE"

            # ---- BIG RESULT TEXT ----
            st.markdown(
                f"""
                <h2 style="color:white; font-size:40px; font-weight:800; margin-bottom:10px;">
                    Result: {label}
                </h2>
                <h3 style="color:#00eaff; font-size:30px; margin-top:-10px;">
                    Conformed
                </h3>
                """,
                unsafe_allow_html=True
            )

            # ---- STATUS MESSAGE ----
            if label == "REAL":
                st.markdown(
                    """
                    <div style="background-color:#0f5132; padding:18px; border-radius:12px;">
                        <h3 style="color:#d1e7dd; font-size:32px;">✔ This note appears REAL.</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style="background-color:#5c1e1e; padding:18px; border-radius:12px;">
                        <h3 style="color:#f8d7da; font-size:32px;">❌ FAKE NOTE DETECTED!</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

st.markdown("---")
st.caption("Presented by SJCIT")
