import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile

# Load your trained YOLOv8 model
model = YOLO("weights/best.pt")  # Correct: path to the actual weights file

st.set_page_config(page_title="Welding Defect Detector", layout="centered")
st.title("🛠️ Welding Defect Detector")
st.markdown("Upload a welding image to detect and classify defects like porosity, pinhole, crack, etc.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Save image to a temporary file for YOLO inference
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        temp_path = tmp_file.name
        image.save(temp_path)

    # Run model inference
    results = model(temp_path)[0]

    # Show annotated image with bounding boxes
    annotated_img = Image.fromarray(results.plot())
    st.image(annotated_img, caption="🔍 Defects Detected", use_container_width=True)

