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
import cv2



if "lat" not in st.session_state:
    st.session_state.lat = 20  # default
if "lon" not in st.session_state:
    st.session_state.lon = 78  # default
if "zoom" not in st.session_state:
    st.session_state.zoom = 6  # default

# ✅ Move the model-loading function OUTSIDE the class and cache it properly
@st.cache_resource
def load_classification_model(model_name):
    model_paths = {
        "ViT": "vit_entire_model.pth",
        "ResNet": "resnet50.h5",
        
        "VGG": "VGG_NET.h5"
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

 
    def capture_map_screenshot(self,lat, lon, zoom):
        """
        Captures a map screenshot using Selenium, centers it on lat/lon,
        and crops it to exactly 640x640 pixels.
        """
        target_size = 340
        # Capture a larger area to ensure the 640x640 center is available
        # Increased height buffer slightly more for potential browser chrome
        capture_width, capture_height = 440, 440

        map_html_path = "updated_map.html"
        screenshot_full_path = "captured_map_full.png"  # Temporary path for the large screenshot
        final_cropped_path = "captured_map_640.png"    # Final 640x640 output path

        print(f"Generating map for {lat}, {lon} at zoom {zoom}...")
        try:
            updated_map = folium.Map(location=[lat, lon], zoom_start=zoom, control_scale=False, zoom_control=False)
            folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", attr="Google").add_to(updated_map)
            updated_map.save(map_html_path)
        except Exception as e:
            st.error(f"❌ Failed to generate map HTML: {e}")
            # Clean up html file if it exists
            if os.path.exists(map_html_path):
                try: os.remove(map_html_path)
                except OSError: pass
            return None # Indicate failure

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        # Set window size hint for the browser
        options.add_argument(f"--window-size={capture_width},{capture_height}")

        # Ensure this path is correct for your system
        chrome_driver_path = "C:/chromedriver/chromedriver.exe"
        service = Service(chrome_driver_path)
        driver = None
        print("Launching headless browser...")
        try:
            driver = webdriver.Chrome(service=service, options=options)
            # Setting window size again can sometimes be necessary
            driver.set_window_size(capture_width, capture_height)
            driver.get("file://" + os.path.abspath(map_html_path))
            print("Waiting for map tiles to load...")
            time.sleep(6) # Allow sufficient time for tiles to render

            print("Capturing full screenshot...")
            success = driver.save_screenshot(screenshot_full_path)
            if not success or not os.path.exists(screenshot_full_path):
                st.error("❌ Failed to save screenshot using Selenium.")
                return None # Indicate failure

            # --- Cropping Logic ---
            print("Loading captured image for cropping...")
            img = cv2.imread(screenshot_full_path)
            if img is None:
                st.error(f"❌ Failed to read the captured screenshot file: {screenshot_full_path}")
                return None # Indicate failure

            h, w = img.shape[:2]
            print(f"  Full screenshot size: {w}x{h}")

            # Check if captured image is large enough for the target crop
            if h < target_size or w < target_size:
                st.warning(f"⚠️ Captured image ({w}x{h}) is smaller than target {target_size}x{target_size}. Cannot crop accurately. Please check Selenium window size settings or increase capture buffer.")
                # Decide how to handle: return None, return the small image, or pad (returning None is safest)
                # Clean up the small screenshot before returning None
                if os.path.exists(screenshot_full_path):
                    try: os.remove(screenshot_full_path)
                    except OSError: pass
                return None # Indicate failure

            # Calculate top-left corner for the center crop
            start_x = (w - target_size) // 2
            start_y = (h - target_size) // 2

            print(f"  Cropping to {target_size}x{target_size} from ({start_x}, {start_y})...")
            # Perform the crop using NumPy slicing
            cropped_img = img[start_y : start_y + target_size, start_x : start_x + target_size]

            # Save the final 640x640 cropped image
            print(f"  Saving cropped image to {final_cropped_path}...")
            save_success = cv2.imwrite(final_cropped_path, cropped_img)
            if not save_success:
                st.error(f"❌ Failed to save the cropped {target_size}x{target_size} image.")
                # Clean up the full screenshot even if cropped save fails
                if os.path.exists(screenshot_full_path):
                    try: os.remove(screenshot_full_path)
                    except OSError: pass
                return None # Indicate failure

            print("  Cropping successful.")
            return final_cropped_path # Return the path to the successfully cropped 640x640 image
            
        except Exception as e:
            st.error(f"❌ An error occurred during map capture or cropping: {e}")
            import traceback
            print(traceback.format_exc()) # Print detailed error for debugging
            return None # Indicate failure
        finally:
            if driver:
                driver.quit()
                print("Browser closed.")
            # Clean up temporary files
            if os.path.exists(map_html_path):
                try:
                    os.remove(map_html_path)
                    print(f"  Removed {map_html_path}")
                except OSError: pass
            # Remove the large temporary screenshot *only if* the final cropped one exists
            if os.path.exists(final_cropped_path) and os.path.exists(screenshot_full_path):
                try:
                    os.remove(screenshot_full_path)
                    print(f"  Removed {screenshot_full_path}")
                except OSError: pass
            # If the final doesn't exist but the full one does (e.g., error after capture but before crop save)
            elif not os.path.exists(final_cropped_path) and os.path.exists(screenshot_full_path):
                try:
                    os.remove(screenshot_full_path)
                    print(f"  Removed {screenshot_full_path} (cropping likely failed)")
                except OSError: pass

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

        model_option = st.selectbox("Choose a Model:", ("ViT", "ResNet", "VGG"))
        
        model = load_classification_model(model_option)

        st.markdown("### 🔍 Select Input Mode")
        input_mode = st.pills(
            "MODE",
            ["🌍 Interactive Map", "🖼️ Manual Upload", "📌 Lat & Long Input"],
           
        )

        if input_mode == "🌍 Interactive Map":
            st.header("🌍 Interactive Map Selection")

            # Define default map settings
            MAP_DEFAULT_LOCATION = [20, 78]
            MAP_DEFAULT_ZOOM = 6

            # Create Folium map with satellite tiles
            m = folium.Map(location=MAP_DEFAULT_LOCATION, zoom_start=MAP_DEFAULT_ZOOM, control_scale=True, zoom_control=True)
            folium.TileLayer(
                tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                attr="Google Satellite",
                name="Google Satellite"
            ).add_to(m)
            folium.LatLngPopup().add_to(m)

            # Layout: Map and Controls
            map_col, controls_col = st.columns([3, 1])

            with map_col:
                map_data = st_folium(m, height=500, width=700, returned_objects=["last_clicked", "zoom", "center"])

            with controls_col:
                st.markdown("**Instructions:** Click on the map to select a location. Adjust zoom if needed.")

                lat = map_data.get("last_clicked", {}).get("lat") if map_data.get("last_clicked") else map_data.get("center", {}).get("lat", MAP_DEFAULT_LOCATION[0])
                lon = map_data.get("last_clicked", {}).get("lng") if map_data.get("last_clicked") else map_data.get("center", {}).get("lng", MAP_DEFAULT_LOCATION[1])
                zoom = map_data.get("zoom", MAP_DEFAULT_ZOOM)

                st.metric("Selected Latitude", f"{lat:.4f}")
                st.metric("Selected Longitude", f"{lon:.4f}")
                st.metric("Current Zoom", zoom)

                show_result = False

                if st.button("📸 Capture & Classify Map Image", key="capture_interactive_for_Classify"):
                    if lat and lon:
                        with st.spinner(f"Capturing map at {lat:.4f}, {lon:.4f} (Zoom: {zoom})..."):
                            screenshot_path = self.capture_map_screenshot(lat, lon, zoom)

                        if os.path.exists(screenshot_path):
                            image = Image.open(screenshot_path)
                            predicted_class, predicted_prob = self.get_prediction(model, model_option, screenshot_path)
                            show_result = True
                        else:
                            st.error("❌ Failed to capture map. Try again.")
                    else:
                        st.warning("⚠️ Please click on the map to select a location.")

            # Show prediction result below the map and controls
            if 'show_result' in locals() and show_result:
                st.divider()
                st.subheader("🧠 Prediction Result")
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.image(image, caption="📷 Captured Satellite Image", use_container_width=True)
                    with open(screenshot_path, "rb") as img_file:
                        st.download_button(
                            label="📥 Download Captured Image",
                            data=img_file,
                            file_name="classified_map.png",
                            mime="image/png"
                        )

                with col2:
                    st.markdown(f"**Model Selected:** {model_option}")
                    st.markdown(f"**Predicted Class:** <span style='color:limegreen;font-weight:bold'>{predicted_class}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** <span style='color:lightgreen;font-weight:bold'>{predicted_prob:.3f}</span>", unsafe_allow_html=True)



        elif input_mode == "🖼️ Manual Upload":
            uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

                with st.spinner("Classifying..."):
                    image_path = "uploaded_temp_image.png"
                    image.save(image_path)

                    predicted_class, predicted_prob = self.get_prediction(model, model_option, image_path)

                    st.write(f"**Model Selected:** {model_option}")
                    st.write(f"**Predicted Class:** {predicted_class}")
                    st.write(f"**Confidence:** {predicted_prob:.3f}")
                    
                    st.download_button(
                        label="📥 Download Uploaded Image",
                        data=uploaded_file,
                        file_name="classified_uploaded_image.png",
                        mime="image/png"
                    )

        elif input_mode == "📌 Lat & Long Input":
            lat = st.number_input("Enter Latitude", min_value=-90.0, max_value=90.0, format="%.6f")
            lon = st.number_input("Enter Longitude", min_value=-180.0, max_value=180.0, format="%.6f")
            zoom = st.slider("Zoom Level", min_value=1, max_value=20, value=15)

            if st.button("🔍 Capture from Coordinates"):
                st.markdown(f"### **Latitude : `{lat}`**")
                st.markdown(f"### **Longitude: `{lon}`**")

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

