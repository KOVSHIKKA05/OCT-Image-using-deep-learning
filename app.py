import streamlit as st
import numpy as np
import json
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="OCT Classification", layout="centered")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

USER_FILE = "users.json"
UPLOAD_DIR = "uploads"
MODEL_PATH = "oct_model_python312.h5"

# =========================
# CREATE FILES / FOLDERS
# =========================
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def load_users():
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, "w") as f:
            json.dump({}, f)
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

# =========================
# SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_oct_model():
    return load_model(MODEL_PATH)

model = load_oct_model()

# ⚠️ MUST MATCH TRAINING ORDER
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

# Safety check (prevents IndexError)
MODEL_CLASSES = model.output_shape[-1]
if MODEL_CLASSES != len(CLASS_NAMES):
    st.error(
        f"Model outputs {MODEL_CLASSES} classes "
        f"but CLASS_NAMES has {len(CLASS_NAMES)} labels.\n"
        f"Fix CLASS_NAMES to match your trained model."
    )
    st.stop()

# =========================
# LOGIN / REGISTER
# =========================
if not st.session_state.logged_in:
    st.title("🔐 Login & Register")

    tab1, tab2 = st.tabs(["Login", "Register"])

    # ---------- LOGIN ----------
    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            users = load_users()
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    # ---------- REGISTER ----------
    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Register"):
            users = load_users()
            if new_user == "" or new_pass == "":
                st.warning("Please fill all fields")
            elif new_user in users:
                st.error("User already exists")
            else:
                users[new_user] = new_pass
                save_users(users)
                st.success("Registration successful! Please login.")

# =========================
# AFTER LOGIN → PREDICTION
# =========================
else:
    st.sidebar.success(f"Logged in as: {st.session_state.username}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    st.title("🩺 OCT Image Classification")

    uploaded_image = st.file_uploader(
        "Upload OCT Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        # Save image
        image_path = os.path.join(
            UPLOAD_DIR,
            f"{st.session_state.username}_{uploaded_image.name}"
        )

        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        # Display image
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded OCT Image", use_column_width=True)

        # Preprocess
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)
        class_index = int(np.argmax(preds))
        confidence = float(np.max(preds))

        prediction = CLASS_NAMES[class_index]

        # Display result
        st.success(f"Prediction: {prediction}")
        st.info(f"Confidence: {confidence * 100:.2f}%")
