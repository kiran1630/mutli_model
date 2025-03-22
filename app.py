# app.py

import streamlit as st
from image_classification import ImageClassifier
from image_segmentation import segment_image, run_yolo, read_uploaded_image, load_yolo_model
import streamlit as st
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



st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Image Classification", "Segmentation"])

if page == "Image Classification":
    classifier = ImageClassifier()
    classifier.run()

elif page == "Segmentation":
    # st.title("Image Segmentation using YOLO")
    # st.write("Upload an image to perform segmentation")

    # uploaded_file = st.file_uploader("Choose an image for segmentation...", type=["jpg", "png", "jpeg"])
    # conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)  # Dynamic confidence slider

    # if uploaded_file is not None: # Check if uploaded_file is not None
    #     image = read_uploaded_image(uploaded_file)

    #     segmented_image_path = segment_image(load_yolo_model(), image)

    #     col1, col2 = st.columns([1, 1])
    #     with col1:
    #         st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    #     with col2:
    #         if segmented_image_path:
    #             st.image(segmented_image_path, caption="Segmented Output", use_container_width=True)
    #         else:
    #             st.write("No segmentation detected.")

    #     st.write("### Object Detection Results")
    #     run_yolo(image, conf_threshold)

    st.title("Image Segmentation using YOLO")
    st.write("Upload an image to perform segmentation")

    uploaded_file = st.file_uploader("Choose an image for segmentation...", type=["jpg", "png", "jpeg"])
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)  # Dynamic confidence slider

    if uploaded_file is not None:
        image = read_uploaded_image(uploaded_file)

        segmented_image_path = segment_image(load_yolo_model(), image)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        with col2:
            if segmented_image_path:
                st.image(segmented_image_path, caption="Segmented Output", use_container_width=True)
            else:
                st.write("No segmentation detected.")

        st.write("### Object Detection Results")
        run_yolo(image, conf_threshold)
    else:
        st.write("Please upload an image for segmentation.") #inform the user that he need to upload an image.