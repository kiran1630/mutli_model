import streamlit as st
import time
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from image_classification import ImageClassifier
from image_segmentation import segment_image, run_yolo, read_uploaded_image, load_yolo_model
from landslide_detection import load_model, preprocess_image, predict
from flood_mapping import load_flood_model, preprocess_image, predict_flood
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



st.markdown(
    """
    <style>
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #2C3E50;
        }
        /* Titles */
        h1, h2, h3 {
            color: #1ABC9C;
        }
        /* Uploaded image styling */
        img {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 📌 Sidebar Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏷 Image Classification", "🎭 Segmentation", "🌍 Landslide Detection", "🌊 Flood Mapping"])

# 📸 Image Classification
if page == "🏷 Image Classification":
    st.title("🔎 Image Classification")
    st.write("Upload an image to classify objects.")

    classifier = ImageClassifier()
    classifier.run()

# 🎭 Segmentation
elif page == "🎭 Segmentation":
    st.title("🎭 Image Segmentation using YOLO")
    st.write("Upload an image to perform segmentation")

    uploaded_file = st.file_uploader("📤 Choose an image for segmentation...", type=["jpg", "png", "jpeg"])
    conf_threshold = st.slider("🎯 Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

    if uploaded_file is not None:
        with st.spinner("🔄 Processing..."):
            image = read_uploaded_image(uploaded_file)
            segmented_image_path = segment_image(load_yolo_model(), image)
            time.sleep(1)

        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="📌 Uploaded Image", use_container_width=True)
        with col2:
            if segmented_image_path:
                st.image(segmented_image_path, caption="✅ Segmented Output", use_container_width=True)
            else:
                st.warning("⚠️ No segmentation detected.")

        st.write("### 🏷 Object Detection Results")
        run_yolo(image, conf_threshold)

# 🌍 Landslide Detection
elif page == "🌍 Landslide Detection":
    st.title("🌍 Landslide Detection using ViT Model")
    st.write("Upload an image to classify it as a landslide or non-landslide.")

    model = load_model(r"C:\Users\Realme\Downloads\vit_small_model.pth")

    uploaded_file = st.file_uploader("📤 Choose an image for landslide detection...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        with st.spinner("🔄 Processing..."):
            image = Image.open(uploaded_file)
            processed_image = preprocess_image(image)
            prediction = predict(processed_image, model)
            time.sleep(1)

        st.image(image, caption="📌 Uploaded Image", use_container_width=True)
        st.success(f"### ✅ Prediction: {prediction}")

# 🌊 Flood Mapping
elif page == "🌊 Flood Mapping":
    st.title("🌊 Flood Mapping using U-Net")
    st.write("Upload an image to detect flooded areas.")

    model_flood = load_flood_model(r"D:\download\unet_model.h5")

    uploaded_file = st.file_uploader("📤 Choose an image for flood mapping...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        with st.spinner("🔄 Processing..."):
            image = Image.open(uploaded_file)
            processed_image = preprocess_image(image)
            mask_colored = predict_flood(processed_image, model_flood)

            # Convert PIL image to OpenCV format for processing
            original_image = np.array(image)
            original_image = cv2.resize(original_image, (128, 128))
            
            # Blend the mask with the original image
            overlay = cv2.addWeighted(original_image, 0.7, mask_colored, 0.3, 0)

            # Check if any flood pixels (yellow) are detected
            flood_detected = np.any(mask_colored[:, :, 0] == 255)
            time.sleep(1)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original_image, caption="📌 Original Image", use_container_width=True)
        with col2:
            st.image(mask_colored, caption="🌊 Flood Mask (Blue-Yellow)", use_container_width=True)
        with col3:
            st.image(overlay, caption="🛑 Overlayed Flood Map", use_container_width=True)

        # Display flood detection message
        if flood_detected:
            st.warning("🚨 **Flood is detected!**")
        else:
            st.success("✅ No flood detected.")
