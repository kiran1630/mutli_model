import streamlit as st
import torch
import tensorflow as tf
import numpy as np
import folium
import os
import time
from streamlit_folium import st_folium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow import keras

# ✅ Move the model-loading function OUTSIDE the class and cache it properly
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

class ImageClassifier:
    def __init__(self):
        self.class_names = [
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
            'River', 'SeaLake'
        ]

    def capture_map_screenshot(self, lat, lon, zoom):
        updated_map = folium.Map(location=[lat, lon], zoom_start=zoom, control_scale=False)

        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google",
            name="Google Satellite Hybrid",
        ).add_to(updated_map)

        updated_map.save("updated_map.html")

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        service = Service("C:/chromedriver/chromedriver.exe")
        driver = webdriver.Chrome(service=service, options=options)

        driver.set_window_size(800, 500)
        driver.get("file://" + os.path.abspath("updated_map.html"))
        time.sleep(5)

        screenshot_path = "captured_map.png"
        driver.save_screenshot(screenshot_path)
        driver.quit()

        return screenshot_path

    def get_prediction(self, model, model_name, image_path):
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
                pred_class = self.class_names[pred_idx]
                pred_prob = probs[0][pred_idx].item()

            return pred_class, pred_prob

        else:
            img = load_img(image_path, target_size=(64, 64))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            predictions = model.predict(img_array)
            predicted_class = self.class_names[np.argmax(predictions)]
            confidence = np.max(predictions)

            return predicted_class, confidence

    def run(self):
        st.title("🌍 Satellite Image Classification with AI")

        model_option = st.selectbox("Choose a Model:", ("ViT", "ResNet", "VGG"))
        
        # ✅ Call the cached function outside the class
        model = load_classification_model(model_option)

        st.write("📍 Click on the map to select a location")

        m = folium.Map(location=[20, 78], zoom_start=6, control_scale=False)
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", 
            attr="Google",
            name="Google Satellite",
        ).add_to(m)

        map_data = st_folium(m, height=500, width=800, returned_objects=["last_clicked", "zoom"])

        if st.button("📸 Capture & Classify Map Image"):
            if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
                lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
                zoom = map_data.get("zoom", 6)

                screenshot_path = self.capture_map_screenshot(lat, lon, zoom)
                
                if os.path.exists(screenshot_path):
                    image = Image.open(screenshot_path)
                    predicted_class, predicted_prob = self.get_prediction(model, model_option, screenshot_path)

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.image(image, caption="Captured Satellite Image", use_container_width=True)
                    with col2:
                        st.write(f"**Model Selected:** {model_option}")
                        st.write(f"**Predicted Class:** {predicted_class}")
                        st.write(f"**Confidence:** {predicted_prob:.3f}")

                    with open(screenshot_path, "rb") as img_file:
                        st.download_button(
                            label="📥 Download Captured Image",
                            data=img_file,
                            file_name="classified_map.png",
                            mime="image/png"
                        )
                else:
                    st.error("❌ Failed to capture map. Try again.")
            else:
                st.warning("⚠️ Please select a location on the map before capturing.")
