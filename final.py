import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# =========================
# CONFIG
# =========================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

st.set_page_config(
    page_title="OCT Image Classification",
    layout="centered"
)

# =========================
# TITLE
# =========================
st.title("OCT Image Classification")
st.write("Upload an OCT scan image and the model will predict its class.")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_oct_model():
    return load_model("oct_model_python312.h5")

model = load_oct_model()

# =========================
# CLASS NAMES
# ⚠️ MUST MATCH MODEL OUTPUT ORDER
# =========================
CLASS_NAMES = ["AMD", "CNV", "CSR", "DME","DR","DRUSEN","MH","NORMAL",]

# =========================
# CHECK MODEL OUTPUT
# =========================
NUM_CLASSES = model.output_shape[-1]

if NUM_CLASSES != len(CLASS_NAMES):
    st.error(
        f"Model outputs {NUM_CLASSES} classes, "
        f"but CLASS_NAMES has {len(CLASS_NAMES)} labels."
    )
    st.stop()

# =========================
# IMAGE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Choose an OCT image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    # -------------------------
    # LOAD IMAGE (FORCE RGB)
    # -------------------------
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # -------------------------
    # PREPROCESS
    # -------------------------
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Debug (optional)
    st.write("Image shape:", img_array.shape)

    # -------------------------
    # PREDICTION
    # -------------------------
    pred = model.predict(img_array)
    class_index = int(np.argmax(pred))
    confidence = float(np.max(pred))

    # -------------------------
    # DISPLAY RESULT
    # -------------------------
    st.success(f"Prediction: {CLASS_NAMES[class_index]}")
    st.info(f"Confidence: {confidence * 100:.2f}%")

    # -------------------------
    # ALL CLASS PROBABILITIES
    # -------------------------
    st.write("### Class Probabilities")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}: {pred[0][i] * 100:.2f}%")
