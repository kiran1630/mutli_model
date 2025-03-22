# image_classification.py

import streamlit as st
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow import keras
from io import BytesIO

# Cache classification models
@st.cache_resource
def load_classification_model(model_name):
    model_paths = {
        "ViT": r"C:\Users\Realme\Desktop\model\vit_entire_model.pth",
        "ResNet": r"C:\Users\Realme\Desktop\model\RES_NET_50_.h5",
        "VGG": r"C:\Users\Realme\Desktop\model\VGG_NET.h5"
    }

    model_path = model_paths[model_name]

    if model_name == "ViT":
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
    else:
        model = keras.models.load_model(model_path)

    return model

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

class ImageClassifier:
    def __init__(self):
        self.page_title = "Multi-Model Image Classification"

    def run(self):
        st.title(self.page_title)
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