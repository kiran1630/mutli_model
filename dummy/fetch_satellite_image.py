import streamlit as st
import folium
import os
import time
from streamlit_folium import st_folium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from PIL import Image

st.title("🛰️ Satellite Image Viewer & Downloader")

# Initialize Map
m = folium.Map(location=[20, 78], zoom_start=6, control_scale=False)

folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",  # 's' removes labels & roads
    attr="Google",
    name="Google Satellite (No Labels)",
).add_to(m)


st.write("📍 Click on the map to select a location (No markers or labels)")

# Render Map & Capture Click Data
map_data = st_folium(m, height=500, width=800, returned_objects=["last_clicked", "zoom"])

st.write("📌 Debug Info:", map_data)  # Debugging: Show Click Data

def capture_map_screenshot(lat, lon, zoom):
    updated_map = folium.Map(location=[lat, lon], zoom_start=zoom, control_scale=False)

    # Add ONLY the satellite layer (NO MARKERS, NO LABELS)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite Hybrid",
    ).add_to(updated_map)

    updated_map.save("updated_map.html")

    # Configure Selenium
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service("C:/chromedriver/chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=options)

    # Set window size to match map size
    width, height = 800, 500
    driver.set_window_size(width, height)

    # Load Map
    driver.get("file://" + os.path.abspath("updated_map.html"))
    time.sleep(5)  # Allow time for map to render

    # Take Screenshot
    screenshot_path = "captured_map.png"
    driver.save_screenshot(screenshot_path)
    driver.quit()

    # Crop the exact required area using PIL
    image = Image.open(screenshot_path)
    image = image.crop((0, 0, width, height))  # Crop to match map size
    image.save(screenshot_path)  # Overwrite with cropped version

    return screenshot_path

# Capture & Download Button
if st.button("📸 Capture & Download Map"):
    if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        zoom = map_data.get("zoom", 6)

        st.write(f"✅ Captured Click: Latitude: {lat}, Longitude: {lon}, Zoom: {zoom}")  # Debugging

        screenshot_path = capture_map_screenshot(lat, lon, zoom)
        
        if os.path.exists(screenshot_path):  # Ensure file exists before download
            with open(screenshot_path, "rb") as img_file:
                st.download_button(
                    label="📥 Download Captured Map",
                    data=img_file,
                    file_name="captured_map.png",
                    mime="image/png"
                )
        else:
            st.error("❌ Failed to capture screenshot. Please try again.")
    else:
        st.warning("⚠️ Please click on the map to select a location before capturing.")
