import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# Page Configuration
st.set_page_config(
    page_title="Automated Welding Defect Inspector",
    page_icon="⚡",
    layout="wide"
)

# App Title & Description
st.title("⚡ Automated Welding Defect Inspector")
st.markdown("An AI-driven Computer Vision solution for real-time industrial quality inspection. Detects **Cracks**, **Porosity**, **Spatters**, and **Excess Reinforcement** with automatic bounding-box localization.")

# Sidebar Configuration
st.sidebar.header("Control Panel")
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

# Load Model Weights using Linux-compatible path format
@st.cache_resource
def load_model():
    # Use forward slashes for cross-platform compatibility
    model_path = "runs/detect/runs/welding/v1/weights/best.pt"
    
    # Fallback path if the primary path is not found
    if not os.path.exists(model_path):
        model_path = "results/weights/best.pt"
        
    if not os.path.exists(model_path):
        st.error(f"Could not load weights from path. Ensure training finished and best.pt exists.")
        return None
        
    return YOLO(model_path)

model = load_model()

# Image Upload Section
st.sidebar.subheader("Input Source")
uploaded_file = st.sidebar.file_uploader("Upload a Weld Inspection Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV / PIL image format
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
    if model is not None:
        with st.spinner("Analyzing welding defects..."):
            # Run YOLOv8 inference
            results = model.predict(source=img_array, conf=confidence_threshold)
            res_plotted = results[0].plot()
            
            with col2:
                st.subheader("Defect Detection Output")
                st.image(res_plotted, use_column_width=True)
                
            # Display Detection Details
            st.markdown("---")
            st.subheader("Inspection Summary")
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.write(f"**Total Defects Detected:** {len(boxes)}")
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls_id]
                    st.write(f"- **{class_name.capitalize()}** (Confidence: {conf:.2f})")
            else:
                st.success("No defects detected above the specified confidence threshold.")
else:
    st.info("👈 Upload a weld image from the sidebar to begin automated defect analysis.")