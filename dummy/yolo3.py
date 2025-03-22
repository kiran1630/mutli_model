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
import plotly.graph_objects as go

# Cache classification models
@st.cache_resource
def load_classification_model(model_name):
    model_paths = {
        "ViT": r"C:\Users\Realme\Desktop\model\vit_entire_model.pth",
        "ResNet": r"C:\Users\Realme\Desktop\model\model_100_epoch.h5",
        "VGG": r"C:\Users\Realme\Desktop\model\eurosat_rgb_model.h5"
    }

    model_path = model_paths[model_name]

    if model_name == "ViT":
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
    else:
        model = keras.models.load_model(model_path)

    return model

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
    bytes_data = uploaded_file.read()
    image = np.array(Image.open(BytesIO(bytes_data)))
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to OpenCV format

# Classification labels
class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

# Function to get classification prediction
def get_prediction(model, model_name, image_path, class_names):
    if model_name == "ViT":
        from torchvision import transforms
        device = "cuda" if torch.cuda.is_available() else "cpu"

        img = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transformed_image = transform(img).unsqueeze(dim=0)

        model.to(device)
        model.eval()

        with torch.no_grad():
            output = model(transformed_image.to(device))
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_class = class_names[pred_idx]
            pred_prob = probs[0][pred_idx].item()
    else:
        img = load_img(image_path, target_size=(64, 64))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        return predicted_class, confidence

    return pred_class, pred_prob

# Function to segment image using YOLO
def segment_image(model, image):
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

    # Display object counts
    
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



# Streamlit UI
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Image Classification", "Segmentation"])

if page == "Image Classification":
    st.title("Multi-Model Image Classification")
    st.write("Select a model and upload an image to classify")

    model_option = st.selectbox("Choose a Model:", ("ViT", "ResNet", "VGG"))
    model = load_classification_model(model_option)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_path = "temp.jpg"
        image.save(image_path)

        predicted_class, predicted_prob = get_prediction(model, model_option, image_path, class_names)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.write(f"**Model Selected:** {model_option}")
            st.write(f"**Predicted Class:** {predicted_class}")
            st.write(f"**Confidence:** {predicted_prob:.3f}")

elif page == "Segmentation":
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
