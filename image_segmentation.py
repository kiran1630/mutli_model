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



if "selected_image" not in st.session_state:  # Keeping your original ones too
    st.session_state.selected_image = None
if "lat" not in st.session_state:
    st.session_state.lat = None
if "lon" not in st.session_state:
    st.session_state.lon = None
if "zoom" not in st.session_state:
    st.session_state.zoom = 6 # Or your desired default zoom

if "image_path" not in st.session_state:
    st.session_state["image_path"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "segmentation_model" not in st.session_state:
    st.session_state["segmentation_model"] = None
if "conf_threshold" not in st.session_state:
    st.session_state["conf_threshold"] = 0.5



# --- ADD/ENSURE THESE INITIALIZATIONS ARE PRESENT ---
if "image_path" not in st.session_state:
    st.session_state.image_path = None        # <--- Crucial fix
if "segmented_image_path" not in st.session_state:
    st.session_state.segmented_image_path = None # <--- Initialize this too
if 'conf_threshold' not in st.session_state:     # Initialize the slider value state
     st.session_state.conf_threshold = CONF_THRESHOLD_DEFAULT



@st.cache_resource
def load_yolo_model():
    try:
        return YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None  # Return None on error

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
# === Replace your existing capture_map_screenshot function with this one ===
# Make sure you have 'import cv2' and 'import os' at the top of your script.

def capture_map_screenshot(lat, lon, zoom):
    """
    Captures a map screenshot using Selenium, centers it on lat/lon,
    and crops it to exactly 640x640 pixels.
    """
    target_size = 540
    # Capture a larger area to ensure the 640x640 center is available
    # Increased height buffer slightly more for potential browser chrome
    capture_width, capture_height = 640, 640

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



def segmentation():
    st.title("Image Segmentation using YOLO")
    st.write("Select an image from the map or upload one manually.")

    # Load model once at the beginning
    model = load_yolo_model()
    if model is None:
        st.error("Failed to load YOLO Model. Cannot proceed.")
        st.stop() # Stop execution if model isn't loaded

    mode = st.selectbox("Choose Mode:", ["Manual Upload", "Interactive Map", "Manual Coordinates"])

    # --- Image Selection/Capture Section ---
    # This section focuses *only* on getting the image path into st.session_state.image_path

    reset_button_pressed = False # Flag to check if reset is clicked

   
    if mode == "Interactive Map":
        st.header("🌍 Interactive Map Selection")

        # Define default map settings
        MAP_DEFAULT_LOCATION = [20, 78]
        MAP_DEFAULT_ZOOM = 6

        # Create map with Google Satellite tiles
        m = folium.Map(location=MAP_DEFAULT_LOCATION, zoom_start=MAP_DEFAULT_ZOOM, control_scale=True, zoom_control=True)
        folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", attr="Google Satellite").add_to(m)
        folium.LatLngPopup().add_to(m)  # Show lat/lon on click

        # Use columns for layout
        map_col, controls_col = st.columns([3, 1])

        with map_col:
            map_output = st_folium(m, height=500, width=700, returned_objects=["last_clicked", "zoom", "center"])

        with controls_col:
            st.write("**Instructions:** Click map to select center. Adjust zoom if needed.")

            # Determine lat, lon, zoom based on click or center fallback
            current_lat = map_output.get("last_clicked", {}).get("lat") if map_output.get("last_clicked") else map_output.get("center", {}).get("lat", MAP_DEFAULT_LOCATION[0])
            current_lon = map_output.get("last_clicked", {}).get("lng") if map_output.get("last_clicked") else map_output.get("center", {}).get("lng", MAP_DEFAULT_LOCATION[1])
            current_zoom = map_output.get("zoom", MAP_DEFAULT_ZOOM)

            # Display selected values
            st.metric("Selected Latitude", f"{current_lat:.4f}")
            st.metric("Selected Longitude", f"{current_lon:.4f}")
            st.metric("Current Zoom", current_zoom)

            # Capture button
            if st.button("📸 Capture 640x640 Map Area", key="capture_interactive"):
                if current_lat and current_lon:
                    st.session_state.lat = current_lat
                    st.session_state.lon = current_lon
                    st.session_state.zoom = current_zoom

                    with st.spinner(f"Capturing and cropping map at {current_lat:.4f}, {current_lon:.4f} (Zoom: {current_zoom})..."):
                        screenshot_path = capture_map_screenshot(current_lat, current_lon, current_zoom)

                    if screenshot_path:
                        st.session_state.image_path = screenshot_path
                        st.session_state.segmented_image_path = None  # Reset previous result
                        st.rerun()  # Rerun to show new state
                    else:
                        st.error("❌ Failed to capture map. Try again.")
                        st.session_state.image_path = None
                else:
                    st.warning("⚠️ Please click on the map first to select a location.")



    elif mode == "Manual Coordinates":
        lat = st.number_input("Enter Latitude", value=st.session_state.lat or 20.0, format="%.6f")
        lon = st.number_input("Enter Longitude", value=st.session_state.lon or 78.0, format="%.6f")
        zoom = st.slider("Zoom Level", min_value=1, max_value=20, value=st.session_state.zoom or 10)

        if st.button("📍 Fetch Image from Coordinates"):
            print("Capture button clicked (Coords)")
            st.session_state.lat = lat # Store coords
            st.session_state.lon = lon
            st.session_state.zoom = zoom
            with st.spinner("Capturing and cropping map..."):
                screenshot_path = capture_map_screenshot(lat, lon, zoom)
            if screenshot_path:
                st.session_state.image_path = screenshot_path
                st.session_state.segmented_image_path = None # Reset previous segmentation result
                st.rerun() # Rerun immediately to show the processing block
            else:
                st.error("❌ Failed to fetch map image.")
                st.session_state.image_path = None # Ensure path is None on failure

    elif mode == "Manual Upload": # Changed from else to elif
        uploaded_file = st.file_uploader("📤 Choose an image for segmentation...", type=["jpg", "png", "jpeg"], key="file_uploader")
        if uploaded_file is not None:
             # Save the uploaded file to a temporary location for consistent access
             # Using a simple name here, consider more robust temp file handling if needed
             temp_save_path = f"temp_uploaded_image.{uploaded_file.name.split('.')[-1]}"
             try:
                 img_pil = Image.open(uploaded_file)
                 img_pil.save(temp_save_path)
                 print(f"Uploaded file saved to {temp_save_path}")
                 # Update session state only if the path changes or is not set
                 if st.session_state.image_path != temp_save_path:
                      st.session_state.image_path = temp_save_path
                      st.session_state.segmented_image_path = None # Reset previous segmentation result
                      # No rerun here needed, allow script flow to continue to processing block naturally
             except Exception as e:
                 st.error(f"Failed to process uploaded file: {e}")
                 st.session_state.image_path = None


    # --- Processing Section ---
    # This section runs only if an image path is stored in session state

    if st.session_state.image_path is not None:
        current_image_path = st.session_state.image_path
        print(f"Processing image path from session state: {current_image_path}")

        # Check if the image file actually exists
        if not os.path.exists(current_image_path):
             st.error(f"Image file not found at {current_image_path}. It might have been temporary. Please select/capture again.")
             st.session_state.image_path = None # Reset state
             st.session_state.segmented_image_path = None
             # Add a small delay and rerun
             time.sleep(3)
             st.rerun()

        else:
            # Load the image using OpenCV
            selected_image_cv = cv2.imread(current_image_path)

            if selected_image_cv is None:
                st.error(f"Failed to load image from path: {current_image_path}. Please try again.")
                st.session_state.image_path = None # Reset state
                st.session_state.segmented_image_path = None
                time.sleep(3)
                st.rerun()
            else:
                st.success(f"Image Loaded: {os.path.basename(current_image_path)} ({selected_image_cv.shape[1]}x{selected_image_cv.shape[0]})")
                st.write("---") # Separator

                # --- Display Original and Segmented Images ---
                col1, col2 = st.columns(2)
                with col1:
                     # Display the original image loaded via CV2
                     st.image(cv2.cvtColor(selected_image_cv, cv2.COLOR_BGR2RGB), caption="📌 Input Image", use_container_width=True)

                # --- Run Segmentation (Only if needed) ---
                # Check if segmentation was already done for this specific image path
                if 'segmented_image_path' not in st.session_state or st.session_state.segmented_image_path is None:
                    print(f"Running segmentation for {current_image_path}")
                    with st.spinner("Running Segmentation..."):
                         # Use copy() to avoid modifying the original array if segment_image does inplace changes
                         st.session_state.segmented_image_path = segment_image(model, selected_image_cv.copy())

                with col2:
                    if st.session_state.segmented_image_path and os.path.exists(st.session_state.segmented_image_path):
                        st.image(st.session_state.segmented_image_path, caption="✅ Segmented Output", use_container_width=True)
                        try:
                             with open(st.session_state.segmented_image_path, "rb") as img_file:
                                 st.download_button("📥 Download Segmented Image", img_file, file_name="segmented_output.png", mime="image/png")
                        except Exception as e:
                             st.warning(f"Could not prepare download for segmented image: {e}")
                    else:
                        st.info("Segmentation output not available or failed.")


                # --- Confidence Slider (Now it won't reset the image) ---
                st.write("---")
                st.subheader("⚙️ Detection Controls")
                # Use st.session_state to make slider value persistent if needed elsewhere,
                # or just read its value directly for run_yolo
                conf_threshold = st.slider(
                    "🎯 Confidence Threshold",
                    min_value=0.1,
                    max_value=1.0,
                    value=st.session_state.conf_threshold, # Use session state value
                    step=CONF_THRESHOLD_STEP,
                    key="conf_slider_main" # Give it a unique key
                )
                # Update session state if slider changes (optional, but good practice)
                st.session_state.conf_threshold = conf_threshold


                # --- Run Detection (This will rerun when slider changes) ---
                st.write("---")
                st.subheader("🔍 Object Detection Results")
                print(f"Running detection with threshold: {conf_threshold}")
                # Pass a copy of the image to run_yolo
                run_yolo(model, selected_image_cv.copy(), conf_threshold)

                # --- Add a Reset Button ---
                st.write("---")
                if st.button("🔄 Select New Image / Reset"):
                      print("Reset button clicked")
                      # Clean up temporary uploaded file if necessary
                      if mode == "Manual Upload" and st.session_state.image_path and "temp_uploaded_image" in st.session_state.image_path:
                           if os.path.exists(st.session_state.image_path):
                               try: os.remove(st.session_state.image_path)
                               except OSError as e: print(f"Error removing temp file: {e}")

                      # Reset all relevant session state variables
                      st.session_state.image_path = None
                      st.session_state.segmented_image_path = None
                      st.session_state.lat = None
                      st.session_state.lon = None
                      st.session_state.zoom = None
                      st.session_state.conf_threshold = CONF_THRESHOLD_DEFAULT # Reset slider value too
                      # Clear file uploader state explicitly if needed
                      # st.session_state.file_uploader = None # Requires key on file_uploader
                      st.rerun() # Rerun to go back to the initial selection state


# --- Make sure to call the main function ---
