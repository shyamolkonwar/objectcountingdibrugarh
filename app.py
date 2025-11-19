import streamlit as st
from object_counter import ObjectCounter
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os

st.title("Smart Object Counter")

st.markdown("""
Upload an image to count objects using computer vision.
Choose between threshold-based counting or color-based detection.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_np = np.array(image)

    # Convert RGB to BGR for OpenCV
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Save to temporary file for processing
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, image_cv)

    # Display original image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Method selection
    method = st.selectbox("Counting Method", ["threshold", "color"])

    color = None
    if method == "color":
        color = st.selectbox("Color to detect", ["red", "green", "blue", "yellow", "orange"])

    # Minimum area parameter
    min_area = st.slider("Minimum Contour Area", 100, 2000, 500, help="Minimum area for object detection")

    if st.button("Count Objects"):
        with st.spinner("Processing image..."):
            counter = ObjectCounter(min_area=min_area)

            try:
                if method == "threshold":
                    count, result_image, _ = counter.count_objects_threshold(tmp_path)
                else:
                    count, result_image, _ = counter.count_objects_color(tmp_path, color=color)

                # Convert result back to RGB for display
                result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

                st.image(result_rgb, caption=f"Processed Image - {count} objects detected", use_column_width=True)
                st.success(f"Total objects detected: {count}")

            except Exception as e:
                st.error(f"Error processing image: {e}")

        # Clean up temporary file
        os.unlink(tmp_path)

st.markdown("---")
st.markdown("Built with Streamlit and OpenCV")
