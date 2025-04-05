import streamlit as st
import folium
import os
import time
import cv2
import numpy as np
from streamlit_folium import st_folium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from PIL import Image
from ultralytics import YOLO
import yaml
import matplotlib.pyplot as plt
from collections import defaultdict
from io import BytesIO

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
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        image = np.array(Image.open(BytesIO(bytes_data)))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return None

# Function to segment image using YOLO
def segment_image(model, image):
    if image is None:
        return None
    
    results = model(image)
    if results[0].masks is None:
        return None
    
    mask = results[0].masks.data.cpu().numpy().astype(np.uint8) * 255
    mask = np.max(mask, axis=0)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    segmented_image = cv2.addWeighted(image, 0.6, mask, 0.4, 0)
    segmented_image_path = "segmented_output.jpg"
    cv2.imwrite(segmented_image_path, segmented_image)
    return segmented_image_path

# Function to run YOLO on image
def run_yolo(image, conf_threshold):
    if image is None:
        return
    st.write("🟡 Passed to Model... Processing...")
    model = load_yolo_model()
    class_names = load_class_names()
    results = model(image, conf=conf_threshold, iou=0.1)
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        st.write("No objects detected.")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    class_counts = defaultdict(int)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = class_names[cls_id]
            class_counts[class_name] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            rect = plt.Rectangle((x1, y1), x2 - y1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            label = f"{class_name} {conf:.2f}"
            ax.text(x1, y1 - 5, label, color='red', fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))
    
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)
    st.write("✅ Detection Completed")
    st.write("### Detected Object Counts:")
    for class_name, count in class_counts.items():
        st.write(f"{class_name} : {count}")

# Function to capture map screenshot
def capture_map_screenshot(lat, lon, zoom):
    updated_map = folium.Map(location=[lat, lon], zoom_start=zoom, control_scale=False)
    folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", attr="Google").add_to(updated_map)
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
    image = Image.open(screenshot_path)
    image = image.convert("RGB")  # Convert to RGB to avoid broken PNG issue
    image = image.crop((0, 0, 800, 500))
    image.save(screenshot_path)
    return screenshot_path
def segmentation():
    # Streamlit UI
    st.title("🎭 Image Segmentation using YOLO")
    st.write("Select an image from the map or upload one manually.")

    # User selection
    mode = st.pills("Choose Mode:", ["Manual Upload", "Interactive Map"])
    selected_image = None

    if mode == "Interactive Map":
        m = folium.Map(location=[20, 78], zoom_start=6, control_scale=False)
        folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", attr="Google").add_to(m)
        map_data = st_folium(m, height=500, width=800, returned_objects=["last_clicked", "zoom"])
        st.write("📍 Click on the map to select an area")
        
        if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
            lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
            zoom = map_data.get("zoom", 6)
            screenshot_path = capture_map_screenshot(lat, lon, zoom)
            st.image(screenshot_path, caption="📌 Selected Map Image", use_container_width=True)
            selected_image = cv2.imread(screenshot_path)
    else:
        uploaded_file = st.file_uploader("📤 Choose an image for segmentation...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            selected_image = read_uploaded_image(uploaded_file)

    conf_threshold = st.slider("🎯 Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

    if selected_image is not None:
        st.write("🟢 Image Selected")
        process_button = st.button("▶ Process Image")
        
        if process_button:
            st.write("🔄 Image Sent to Model...")
            with st.spinner("Processing..."):
                segmented_image_path = segment_image(load_yolo_model(), selected_image)
                time.sleep(1)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(selected_image, caption="📌 Input Image", use_container_width=True)
            with col2:
                if segmented_image_path:
                    st.image(segmented_image_path, caption="✅ Segmented Output", use_container_width=True)
                else:
                    st.warning("⚠️ No segmentation detected.")
            run_yolo(selected_image, conf_threshold)
