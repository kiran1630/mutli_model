# image_segmentation.py

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
import matplotlib.pyplot as plt
from collections import defaultdict
from io import BytesIO
from PIL import Image

# Cache YOLO model
@st.cache_resource
def load_yolo_model():
    return YOLO(r"C:\Users\Realme\Desktop\yolo\best.pt")

# Cache class names
@st.cache_resource
def load_class_names():
    with open(r"C:\Users\Realme\Desktop\image\data.yaml", "r") as f:
        return yaml.safe_load(f)["names"]

# Read uploaded image and convert to OpenCV format
def read_uploaded_image(uploaded_file):
    if uploaded_file is not None: # check if uploaded_file is not none.
        bytes_data = uploaded_file.read()
        image = np.array(Image.open(BytesIO(bytes_data)))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to OpenCV format
    else:
        return None

# Function to segment image using YOLO
def segment_image(model, image):
    if image is None: # check if the image is none.
        return None;

    # If image is a path, read it using OpenCV
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            print("Error: Image not found!")
            return None

    results = model(image)

    # Check if segmentation masks exist
    if results[0].masks is None:
        print("No segmentation detected.")
        return None

    # Extract and process mask
    mask = results[0].masks.data.cpu().numpy().astype(np.uint8) * 255
    mask = np.max(mask, axis=0)  # Combine all masks

    # Ensure mask and image have the same dimensions
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Ensure mask has 3 channels (same as image)
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Blend image with mask
    segmented_image = cv2.addWeighted(image, 0.6, mask, 0.4, 0)

    # Save and return segmented image
    segmented_image_path = "segmented_output.jpg"
    cv2.imwrite(segmented_image_path, segmented_image)

    return segmented_image_path

def run_yolo(image, conf_threshold):
    if image is None: #check if the image is none.
        return;

    model = load_yolo_model()
    class_names = load_class_names()

    results = model(image, conf=conf_threshold, iou=0.1)

    if len(results) == 0 or len(results[0].boxes) == 0:
        st.write("No objects detected.")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Dictionary to store class counts
    class_counts = defaultdict(int)

    # Loop through results and count occurrences
    for result in results:
        detections = result.boxes
        for box in detections:
            cls_id = int(box.cls[0])
            class_name = class_names[cls_id]
            class_counts[class_name] += 1  # Increment count

    # Visualization (bounding boxes on the image)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = class_names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw rectangle
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

            # Add label
            label = f"{class_name} {conf:.2f}"
            ax.text(x1, y1 - 5, label, color='red', fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

    ax.set_xticks([])
    ax.set_yticks([])

    # Display the image with bounding boxes
    st.pyplot(fig)
    st.write("### Detected Object Counts:")
    for class_name, count in class_counts.items():
        st.write(f"{class_name} : {count}")