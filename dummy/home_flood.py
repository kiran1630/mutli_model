
# # ----- Flood Mapping Page -----
# elif page == "🌊 Flood Mapping":
#     st.header("🌊 Flood Mapping")
#     st.info("Upload a satellite image (e.g., Sentinel-1 SAR) to map flooded areas using a U-Net model.")
#     try:
#         # !!! IMPORTANT: Replace with the ACTUAL path to YOUR flood model !!!
#         flood_model_path = r"D:\download\unet_model.h5" # << CHANGE THIS PATH !!
#         if not os.path.exists(flood_model_path):
#              st.error(f"Flood model not found at {flood_model_path}")
#              model_flood = None
#         else:
#              model_flood = load_flood_model(flood_model_path) # Assuming Keras/TF model

#         if model_flood: # Only proceed if model loaded
#             uploaded_file_fm = st.file_uploader("📤 Choose an image for Flood Mapping...", type=["jpg", "png", "jpeg", "tif", "tiff"])

#             if uploaded_file_fm is not None:
#                 with st.spinner("🔄 Processing Flood Mapping..."):
#                     image = Image.open(uploaded_file_fm,use_container_width=True)
#                     # Ensure image is RGB for blending if needed later
#                     if image.mode != 'RGB':
#                         image = image.convert('RGB')

#                     processed_image_for_model = preprocess_flood(image.copy()) # Use specific preprocess function
#                     # Assuming predict_flood returns a mask (e.g., single channel or colored)
#                     mask_result = predict_flood(processed_image_for_model, model_flood)

#                     # --- Visualization ---
#                     # Resize original image for display consistency (e.g., to mask size if known, or fixed size)
#                     display_size = (256, 256) # Or use mask_result.shape[:2] if consistent
#                     original_display = np.array(image.resize(display_size))

#                     # Ensure mask is suitable for display (e.g., 3-channel BGR if colored, or convert grayscale)
#                     if isinstance(mask_result, np.ndarray):
#                         if mask_result.ndim == 2: # Grayscale mask
#                             # Convert grayscale to BGR for display/overlay
#                             mask_display_bgr = cv2.cvtColor(mask_result, cv2.COLOR_GRAY2BGR)
#                         elif mask_result.shape[2] == 3: # Already 3-channel
#                              mask_display_bgr = cv2.resize(mask_result, display_size) # Ensure size matches
#                         else:
#                              st.error("Unexpected mask format from predict_flood")
#                              mask_display_bgr = np.zeros_like(original_display) # Fallback

#                         # Simple check for flood pixels (assuming non-black pixels indicate flood/water)
#                         # Adjust this logic based on your mask's actual meaning
#                         flood_detected = np.any(mask_result > 0)

#                         # Blend the mask with the original image
#                         # Ensure both are same size and type
#                         overlay = cv2.addWeighted(original_display, 0.6, mask_display_bgr, 0.4, 0)
#                     else:
#                         st.error("Prediction result is not a valid image format.")
#                         mask_display_bgr = np.zeros_like(original_display)
#                         overlay = original_display.copy()
#                         flood_detected = False
#                     # time.sleep(1) # Optional artificial delay


#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.image(original_display, caption="📌 Original Image (Resized)", use_column_width=True)
#                 with col2:
#                     st.image(mask_display_bgr, caption="🌊 Predicted Mask", use_column_width=True)
#                 with col3:
#                     st.image(overlay, caption="🗺️ Overlayed Map", use_column_width=True)

#                 # Display flood detection message
#                 st.write("---")
#                 if flood_detected:
#                     st.warning("🚨 **Potential Flood Area Detected!** (Based on mask presence)")
#                 else:
#                     st.success("✅ No significant flood area detected in the mask.")
#         else:
#              st.warning("Flood mapping model could not be loaded.")

#     except NameError:
#         st.error("Flood Mapping functionality is not available (check imports).")
#     except Exception as e:
#         st.error(f"An error occurred during Flood Mapping: {e}")
#         st.exception(e) # Print traceback for debugging