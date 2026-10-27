import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Welding Defect Inspector AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS injection for card layouts and modern design
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .metric-card {
        background-color: #1E222D;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 10px;
    }
    .metric-title {
        color: #A3A8B8;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 26px;
        font-weight: bold;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Model Loading (Cached for fast performance)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO(r'runs\detect\runs\welding\v1\weights\best.pt')

try:
    model = load_yolo_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Could not load weights from path. Ensure training finished: {e}")

# Class label mapping
CLASS_LABELS = {
    0: 'Crack',
    1: 'Excess Reinforcement',
    2: 'Porosity',
    3: 'Spatters'
}

# -----------------------------------------------------------------------------
# 3. Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/welder.png", width=70)
st.sidebar.title("Control Panel")
st.sidebar.markdown("---")

conf_threshold = st.sidebar.slider(
    "🎯 Confidence Threshold",
    min_value=0.05,
    max_value=1.00,
    value=0.25,
    step=0.05,
    help="Adjust minimum probability score required for defect detection."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Input Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload a Weld Inspection Image",
    type=['jpg', 'jpeg', 'png']
)

st.sidebar.markdown("---")
st.sidebar.caption("⚡ Powered by YOLOv8 | Industrial Quality Control System")

# -----------------------------------------------------------------------------
# 4. Main Header Section
# -----------------------------------------------------------------------------
st.title("⚡ Automated Welding Defect Inspector")
st.markdown("""
An AI-driven Computer Vision solution for real-time industrial quality inspection. 
Detects **Cracks**, **Porosity**, **Spatters**, and **Excess Reinforcement** with automatic bounding-box localization.
""")
st.markdown("---")

# -----------------------------------------------------------------------------
# 5. Execution & Visual Dashboard
# -----------------------------------------------------------------------------
if uploaded_file is not None and model_loaded:
    # Load and preprocess image
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    # Execute YOLO Inference
    with st.spinner("🔍 Running automated defect analysis..."):
        results = model.predict(source=img_np, conf=conf_threshold)
        result = results[0]
        
        # Plot visual detections
        annotated_frame = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Calculate defect statistics
        boxes = result.boxes
        total_defects = len(boxes)

    # Display Summary KPI Cards
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    with col_kpi1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Defects Found</div>
                <div class="metric-value">{total_defects}</div>
            </div>
        """, unsafe_allow_html=True)

    with col_kpi2:
        status_color = "🔴 DEFECTS DETECTED" if total_defects > 0 else "🟢 WELD PASSED"
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: {'#FF4B4B' if total_defects > 0 else '#00C853'};">
                <div class="metric-title">Inspection Status</div>
                <div class="metric-value" style="font-size: 20px;">{status_color}</div>
            </div>
        """, unsafe_allow_html=True)

    with col_kpi3:
        conf_display = f"{conf_threshold * 100:.0f}%"
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #29B6F6;">
                <div class="metric-title">Selected Confidence</div>
                <div class="metric-value">{conf_display}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📷 Visual Inspection View")
    
    # Dual-Column Image View
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col_right:
        st.subheader("AI Detection Result")
        st.image(annotated_rgb, use_container_width=True)

    # Detailed Log Breakdown
    st.markdown("---")
    with st.expander("📋 Detailed Inspection Log & Bounding Boxes", expanded=True):
        if total_defects > 0:
            log_data = []
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                label = CLASS_LABELS.get(cls_id, result.names[cls_id])
                confidence_score = float(box.conf[0].item())
                coords = [round(c, 1) for c in box.xyxy[0].tolist()]
                
                log_data.append({
                    "Defect #": idx + 1,
                    "Defect Type": label,
                    "Confidence": f"{confidence_score * 100:.2f}%",
                    "Bounding Box [x1, y1, x2, y2]": str(coords)
                })
            
            st.table(log_data)
        else:
            st.success("✅ No defects identified above the confidence threshold.")

elif uploaded_file is None:
    st.info("👈 Upload a weld image from the sidebar to begin automated defect analysis.")