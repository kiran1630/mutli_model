# ---- Imports (Cleaned and Organized) ----
import streamlit as st
import time
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from image_classification import ImageClassifier
from image_segmentation import segmentation
from landslide_detection import load_model,  predict,preprocess_image
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow import keras
import cv2
from ultralytics import YOLO
import yaml
import matplotlib.pyplot as plt
import base64
import io
from io import BytesIO
from collections import defaultdict
import seaborn as sn
import random
import home
import os



# ---- Function to Encode Local Image ----
@st.cache_data # Cache the encoded image for efficiency
def get_img_as_base64(file):
    """ Reads an image file and returns its base64 encoded string and mime type """
    try:
        with open(file, "rb") as f:
            data = f.read()
        extension = os.path.splitext(file)[1].lower()
        if extension == ".png": mime_type = "png"
        elif extension in [".jpg", ".jpeg"]: mime_type = "jpeg"
        elif extension == ".gif": mime_type = "gif"
        else: mime_type = "png" # Default fallback
        return base64.b64encode(data).decode(), mime_type
    except FileNotFoundError:
        # Log error instead of showing in UI during background load
        print(f"ERROR: Background image file not found at {file}")
        return None, None
    except Exception as e:
        print(f"ERROR: Could not read background file {file}: {e}")
        return None, None

# ---- Function to Load External CSS ----
def load_css(file_path):
    """ Reads a CSS file and returns its content as a string """
    try:
        with open(file_path) as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Error: CSS file not found at '{file_path}'. Make sure it's in the same directory as the script.")
        return "" # Return empty string on error

# ---- Apply Styling ----
# 1. Load static CSS styles from the external file
css_file_path = "style.css" # Assumes style.css is in the same directory
static_css = load_css(css_file_path)

# 2. Try to load the background image
# !!! IMPORTANT: Replace this path with the ACTUAL path to your background image !!!
local_image_path = r"C:\Downloads\pic1.png" # << CHANGE THIS PATH !!
# local_image_path = "pic1.png" # Use this if image is in the same folder

img_base64 = None
image_type = "png" # Default
if os.path.exists(local_image_path):
    img_base64, image_type = get_img_as_base64(local_image_path)

# Warn if background image wasn't loaded, but still apply base CSS
if not img_base64:
     st.warning(f"Background image not found or failed to load from: '{local_image_path}'. Applying styles without custom background.")

# 3. Generate dynamic CSS rule for the background (only if image loaded)
dynamic_bg_css = ""
if img_base64:
    dynamic_bg_css = f"""
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                    url("data:image/{image_type};base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    """

# 4. Combine static CSS (from file) and dynamic CSS (background rule) and inject
# Always inject the static CSS part for consistent font/layout
# Only add the dynamic background rule if the image loaded
full_css = f"<style>\n{static_css}\n{dynamic_bg_css}\n</style>"
st.markdown(full_css, unsafe_allow_html=True)


# ---- Sidebar Navigation ----
st.sidebar.title("🛰️ Analysis Tools")
page = st.sidebar.selectbox(
    "Select Analysis",
    [
        "🏠 Home",
        "🏷 Image Classification",
        "🎭 Segmentation",
        "🏔️ Landslide Detection",
        
    ],
    label_visibility="collapsed",
    key="select_bySideBar"

)

# ---- Page Content Routing ----

# ----- Home Page -----
if page == "🏠 Home":
    st.markdown("<h1 class='home-title'>🛰️ Multimodal Satellite Image Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<div class='home-subtitle'>Unlock insights from satellite imagery with cutting-edge AI</div>", unsafe_allow_html=True)
    st.markdown(
        """
        Welcome to the **Multimodal Satellite Image Analysis App**! This innovative tool enables you to leverage advanced AI for various geospatial tasks. Select a tool from the sidebar to begin.

        <ul class="feature-list">
            <li>🏷️ <strong>Classify</strong> land cover or objects.</li>
            <li>🎭 <strong>Segment</strong> regions like water bodies or buildings.</li>
            <li>🏔️ <strong>Detect</strong> potential landslide areas.</li>

        </ul>

        **Built with**:
        <ul>
            <li>Streamlit 🧪</li>
            <li>PyTorch & TensorFlow 🧠</li>
            <li>OpenCV & YOLO 🐍</li>
            <li>Vision Transformers (ViT) ⚙️</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

# ----- Image Classification Page -----
elif page == "🏷 Image Classification":
    st.header("🏷 Image Classification")
    st.info("Upload a satellite image to classify its contents.")
    try:
        classifier = ImageClassifier() # Instantiate your class
        classifier.run() # Run its method
    except NameError:
        st.error("ImageClassifier functionality is not available (check imports).")
    except Exception as e:
        st.error(f"An error occurred during Image Classification setup: {e}")


# ----- Segmentation Page -----
elif page == "🎭 Segmentation":
    st.header(" Image Segmentation using YOLO ")
    st.info("Upload a satellite image for segmentation.")
    try:
        segmentation() # Call your segmentation function
    except NameError:
        st.error("Segmentation functionality is not available (check imports).")
    except Exception as e:
        st.error(f"An error occurred during Segmentation setup: {e}")


# ----- Landslide Detection Page -----
elif page == "🏔️ Landslide Detection":
    st.header("🏔️ Landslide Detection")
    st.info("Upload an image to classify it as landslide or non-landslide using a ViT model.")
    try:
        
        landslide_model_path =  r"vit_small_model.pth"
        if not os.path.exists(landslide_model_path):
             st.error(f"Landslide model not found at {landslide_model_path}")
             model = None
        else:
             model = load_model(landslide_model_path)

        if model: #
            uploaded_file_ls = st.file_uploader("📤 Choose an image...", type=["jpg", "png", "jpeg"])

            if uploaded_file_ls is not None:
                with st.spinner("🔄 Processing Landslide Detection..."):
                    image = Image.open(uploaded_file_ls) # Ensure RGB
                    processed_image = preprocess_image(image) 
                    prediction = predict(processed_image, model)
                    time.sleep(1)
                   
                st.image(image, caption="📌 Uploaded Image", use_container_width=True)
                st.success(f"### ✅ Prediction: {prediction}")
            
        else:
            st.warning("Landslide detection model could not be loaded.")

    except NameError:
        st.error("Landslide Detection functionality is not available (check imports).")
    except Exception as e:
        st.error(f"An error occurred during Landslide Detection: {e}")

