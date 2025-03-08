import streamlit as st
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image 
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array

import os

# Load model function
@st.cache_resource
def load_model(model_name):
    model_paths = {
        "ViT": r"C:\Users\Realme\Desktop\image\model\vit_entire_model.pth",
        "ResNet": r"C:\Users\Realme\Desktop\image\model\model_100_epoch.h5",
        "VGG": r"C:\Users\Realme\Desktop\image\model\eurosat_rgb_model.h5"
    }

    model_path = model_paths[model_name]

    if model_name == "ViT":  
        model = torch.load(model_path, map_location=torch.device('cpu'))  # PyTorch model
        model.eval()
    else:  
        model = keras.models.load_model(model_path)  # TensorFlow model

    return model

# Class names
class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

# Prediction function
def get_prediction(model, model_name, image_path, class_names):
    if model_name == "ViT":  # PyTorch Model Prediction
        from torchvision import transforms
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load and transform image
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
    
    else:  # TensorFlow/Keras Model Prediction
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # img = image.load_img(image_path, target_size=(64, 64))
        # img_array = image.img_to_array(img)
        # img_array = np.expand_dims(img_array, axis=0)
        # img_array = img_array / 255.0  

        img = load_img(image_path, target_size=(64, 64))  # Correct function
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        # img = image.load_img(image_path, target_size=(64, 64))
        # img_array = tf_image.img_to_array(img)
        # img_array = np.expand_dims(img_array, axis=0)
        # img_array = img_array / 255.0  

        # predictions = model.predict(img_array)
        # predicted_class = class_names[np.argmax(predictions)]
        # confidence = np.max(predictions)

        return predicted_class, confidence

    return pred_class, pred_prob

# Stream
st.title("Multi-Model Image Classification")
st.write("Select a model and upload an image to classify")

# Model selection
model_option = st.selectbox("Choose a Model:", ("ViT", "ResNet", "VGG"))
model = load_model(model_option)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_path = "temp.jpg"
    image.save(image_path)
    
    # Predict class
    predicted_class, predicted_prob = get_prediction(model, model_option, image_path, class_names)
    
    # Display image and prediction in a single row
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.write(f"**Model Selected:** {model_option}")
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Probability:** {predicted_prob:.3f}")
