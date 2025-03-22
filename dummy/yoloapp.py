import streamlit as st
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array
import os
import cv2
from ultralytics import YOLO
import yaml
import matplotlib.pyplot as plt



# Load classification models
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

# Load YOLO segmentation model
@st.cache_resource
def load_yolo_model():
    yolo_model_path = r"C:\Users\Realme\Desktop\yolo\best.pt"  
    return YOLO(yolo_model_path)

# Classification class names
class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

# Classification prediction function
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
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        return predicted_class, confidence

    return pred_class, pred_prob

# Segmentation function
def segment_image(model, image_path):
    image = cv2.imread(image_path)
    results = model(image)
    
    # Get segmentation mask
    mask = results[0].masks.data.cpu().numpy().astype(np.uint8) * 255  # Convert to binary mask

    # Overlay mask on image
    segmented_image = cv2.addWeighted(image, 0.6, mask[0], 0.4, 0)

    # Save output image
    segmented_image_path = "segmented_output.jpg"
    cv2.imwrite(segmented_image_path, segmented_image)

    return segmented_image_path

@st.cache_resource
def load_yolo_model():
    path = r"C:\Users\Realme\Desktop\yolo\best.pt"
    return YOLO(path)

@st.cache_resource
def load_class_names():
    with open(r"C:\Users\Realme\Desktop\image\data.yaml", "r") as f:
        return yaml.safe_load(f)["names"]

# Function to run YOLO and display results
def run_yolo(image_path):
    model = load_yolo_model()
    class_names = load_class_names()

    # Run inference
    results = model(image_path, conf=0.4, iou=0.1)

    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)

    # Loop through results
    for result in results:
        detections = result.boxes
        num_objects = len(detections)
        st.write(f"Total objects detected: {num_objects}")

        # Draw bounding boxes
        for box in detections:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = class_names[cls_id]

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw rectangle
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

            # Add label
            label = f"{class_name} {conf:.2f}"
            ax.text(x1, y1 - 5, label, color='red', fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))

            st.write(f"Class: {class_name}, Confidence: {conf:.2f}")

    # Hide axes
    ax.set_xticks([])
    ax.set_yticks([])

    # Display the output
    st.pyplot(fig)


# Streamlit Navigation
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
            st.write(f"**Probability:** {predicted_prob:.3f}")

elif page == "Segmentation":
    st.title("Image Segmentation using YOLO")
    st.write("Upload an image to perform segmentation")

    yolo_model = load_yolo_model()
    uploaded_file = st.file_uploader("Choose an image for segmentation...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_path = "input_segmentation.jpg"
        image.save(image_path)


        segmented_image_path = segment_image(yolo_model, image_path)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.image(segmented_image_path, caption="Segmented Output", use_container_width=True)
