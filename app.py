import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="AI VISUAL INSPECTOR", layout="centered")
st.title("🛠️ Welding Defect Detector")
st.markdown("Upload a welding image to detect and classify defects like porosity, pinhole, crack, etc.")

# Load trained model (make sure 'best.pt' is in the same directory)
model_path = "best.pt"
if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please upload best.pt to the app directory.")
else:
    model = YOLO(model_path)

    uploaded_file = st.file_uploader("📤 Upload a welding image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            results = model(tmp.name)[0]
            annotated_image = Image.fromarray(results.plot())

        st.image(annotated_image, caption="🔍 Detected Defects", use_container_width=True)

        st.subheader("📋 Defect Types Detected:")
        if results.boxes and results.boxes.cls.numel() > 0:
            class_ids = results.boxes.cls.int().tolist()
            class_names = [results.names[int(cls_id)] for cls_id in class_ids]
            unique_names = sorted(set(class_names))
            for name in unique_names:
                st.success(f"✅ {name}")
        else:
            st.info("No defects detected.")

