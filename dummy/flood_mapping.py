import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

# Load the U-Net model
@st.cache_resource
def load_flood_model(model_path):
    return load_model(model_path)

# Preprocess the image for model input
def preprocess_image(image, target_size=(128, 128)):
    image = np.array(image)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Perform flood segmentation
def predict_flood(image, model, threshold=0.3):
    prediction = model.predict(image)[0]  # Remove batch dimension
    binary_mask = (prediction > threshold).astype(np.uint8)  # Convert to binary mask

    # Create a color mask (Blue for background, Yellow for flood)
    mask_colored = np.zeros((128, 128, 3), dtype=np.uint8)
    mask_colored[binary_mask.squeeze() == 0] = [0, 0, 255]  # Background - Blue
    mask_colored[binary_mask.squeeze() == 1] = [255, 255, 0]  # Flood - Yellow

    return mask_colored
