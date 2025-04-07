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



MAP_DEFAULT_LOCATION = [20, 78]
MAP_DEFAULT_ZOOM = 6
CONF_THRESHOLD_DEFAULT = 0.4
CONF_THRESHOLD_STEP = 0.05
IMAGE_TYPES = ["jpg", "png", "jpeg"]
SEGMENTATION_ALPHA = 0.4
DETECTION_BOX_COLOR = 'lime'
DETECTION_TEXT_COLOR = 'red'
DETECTION_TEXT_BG_COLOR = 'white'
YOLO_MODEL_PATH = r"C:\Users\Realme\Desktop\yolo\best (1).pt"
DATA_YAML_PATH = r"C:\Users\Realme\Desktop\image\data.yaml"

# Initialize session state for storing values across reruns
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "lat" not in st.session_state:
    st.session_state.lat = None
if "lon" not in st.session_state:
    st.session_state.lon = None
if "zoom" not in st.session_state:
    st.session_state.zoom = 6
if "segmented_image_path" not in st.session_state:
    st.session_state.segmented_image_path = None

# Cache YOLO model
@st.cache_resource
def load_yolo_model():
    return YOLO(r"C:\Users\Realme\Desktop\yolo\best.pt")

# Cache class names
@st.cache_resource
def load_class_names():
    with open(r"C:\Users\Realme\Desktop\image\data.yaml", "r") as f:
        return yaml.safe_load(f)["names"]

# Function to segment image using YOLO
def segment_image(model, image):
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


def run_yolo(model, image, conf_threshold):
    if image is None:
        return
    st.write("🟡 Processing Image...")
    results = model(image, conf=conf_threshold, iou=0.1)
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        st.warning("⚠️ No objects detected.")
        return
    
    class_counts = defaultdict(int)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = load_class_names()[cls_id]
            class_counts[class_name] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            rect = plt.Rectangle((x1, y1), x2 - y1, y2 - y1, linewidth=2, edgecolor=DETECTION_BOX_COLOR, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{class_name} {conf:.2f}", color=DETECTION_TEXT_COLOR, fontsize=10, fontweight='bold', bbox=dict(facecolor=DETECTION_TEXT_BG_COLOR, alpha=0.5))

    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)

    st.success("✅ Detection Completed")
    st.write("### 📊 Object Count:")
    for class_name, count in class_counts.items():
        st.write(f"- **{class_name}** : {count}")

# Function to capture map screenshot
def capture_map_screenshot(lat, lon, zoom):
    updated_map = folium.Map(location=[lat, lon], zoom_start=zoom, control_scale=False,zoom_control=False)
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
    
    return screenshot_path
def segmentation():
    st.title("Image Segmentation using YOLO")
    st.write("Select an image from the map or upload one manually.")

    mode = st.selectbox("Choose Mode:", ["Manual Upload", "Interactive Map", "Manual Coordinates"])

    selected_image = None
    button_label = "📸 Capture & Segment Map Image" if mode == "Interactive Map" else "▶ Process Image"

    if mode == "Interactive Map":
        m = folium.Map(location=[20, 78], zoom_start=6, control_scale=False,zoom_control=False)
        folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", attr="Google").add_to(m)
        map_data = st_folium(m, height=500, width=800, returned_objects=["last_clicked", "zoom"])
        st.write("📍 Click on the map to select an area")

        if st.button(button_label):
            print(f"{button_label} button clicked")
            if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
                lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
                
                st.markdown(f"### Selected Location:", unsafe_allow_html=True)
                st.markdown(f"### **Latitude : `{lat}`**", unsafe_allow_html=True)
                st.markdown(f"### **Longitude: `{lon}`**", unsafe_allow_html=True)
                

                #zoom = map_data.get("zoom", 6)
                zoom = map_data["zoom"] if "zoom" in map_data else 10 
                screenshot_path = capture_map_screenshot(lat, lon, zoom)

                if screenshot_path:
                    model = load_yolo_model()
                    selected_image = cv2.imread(screenshot_path)
                    segmented_image_path = segment_image(model, selected_image)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(screenshot_path, caption="📌 Captured Map Image", use_container_width=True)
                    with col2:
                        if segmented_image_path:
                            st.image(segmented_image_path, caption="✅ Segmented Output", use_container_width=True)
                        else:
                            st.warning("⚠️ No segmentation detected.")

                    if segmented_image_path:
                        with open(segmented_image_path, "rb") as img_file:
                            st.download_button("📥 Download Segmented Image", img_file, file_name="segmented_map.png", mime="image/png")

                    conf_threshold = st.slider("🎯 Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
                    run_yolo(model,selected_image, conf_threshold)  # Ensure OpenCV format

                else:
                    st.error("❌ Failed to capture map. Try again.")
            else:
                st.warning("⚠️ Please select a location on the map before capturing.")
    elif mode == "Manual Coordinates":
            lat = st.number_input("Enter Latitude", value=20.0, format="%.6f")
            lon = st.number_input("Enter Longitude", value=78.0, format="%.6f")
            zoom = st.slider("Zoom Level", min_value=1, max_value=20, value=10)

            if st.button("📍 Fetch Image from Coordinates"):
                screenshot_path = capture_map_screenshot(lat, lon, zoom)

                if screenshot_path:
                    model = load_yolo_model()
                    selected_image = cv2.imread(screenshot_path)
                    segmented_image_path = segment_image(model, selected_image)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(screenshot_path, caption="🌍 Captured Map Image", use_container_width=True)
                    with col2:
                        if segmented_image_path:
                            st.image(segmented_image_path, caption="✅ Segmented Output", use_container_width=True)
                        else:
                            st.warning("⚠️ No segmentation detected.")

                    if segmented_image_path:
                        with open(segmented_image_path, "rb") as img_file:
                            st.download_button("📥 Download Segmented Image", img_file, file_name="segmented_map.png", mime="image/png")

                    conf_threshold = st.slider("🎯 Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
                    run_yolo(model, selected_image, conf_threshold)
                else:
                    st.error("❌ Failed to fetch map image.")

    else:
        uploaded_file = st.file_uploader("📤 Choose an image for segmentation...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            selected_image = Image.open(uploaded_file)

        conf_threshold = st.slider("🎯 Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

        if selected_image and st.button(button_label):
            print(f"{button_label} button clicked")
            st.write("🔄 Image Sent to Model...")
            with st.spinner("Processing..."):
                selected_image_cv = np.array(selected_image.convert('RGB'))
                selected_image_cv = cv2.cvtColor(selected_image_cv, cv2.COLOR_RGB2BGR)
                model = load_yolo_model()
                segmented_image_path = segment_image(model, selected_image_cv)

                if segmented_image_path:
                    run_yolo(model,selected_image_cv, conf_threshold)
                else:
                    print("Segmentation failed, YOLO not called")



