import folium
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from PIL import Image

def fetch_satellite_image(lat, lon, zoom=6, output_path="captured_map.png"):
    """Fetches a satellite image from Google Maps based on coordinates."""
    
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

    width, height = 800, 500
    driver.set_window_size(width, height)

    # Load Map
    driver.get("file://" + os.path.abspath("updated_map.html"))
    time.sleep(5)  # Allow time for map to render

    # Take Screenshot
    driver.save_screenshot(output_path)
    driver.quit()

    # Crop and save the image
    image = Image.open(output_path)
    image = image.crop((0, 0, width, height))  
    image.save(output_path)

    return output_path  # Return the path of the saved image
