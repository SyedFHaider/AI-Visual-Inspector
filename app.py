import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

pip install -r requirements.txt
streamlit run app.py


st.set_page_config(page_title="AI VISUAL INSPECTOR", page_icon="⚡", layout="centered")

# App title with custom style
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>⚡ AI VISUAL INSPECTOR</h1>", unsafe_allow_html=True)
st.markdown("---")

MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file '{MODEL_PATH}' not found. Please upload it to the app directory.")
else:
    model = YOLO(MODEL_PATH)

    uploaded_file = st.file_uploader("📤 Upload a Welding Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        # Show uploaded image
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            image.save(tmp.name)
            results = model(tmp.name)[0]

        # Annotated result
        annotated_img = Image.fromarray(results.plot())
        st.image(annotated_img, caption="🔍 Detection Results", use_container_width=True)

        # Show detected classes
        if results.boxes and len(results.boxes.cls) > 0:
            detected = [results.names[int(cls)] for cls in results.boxes.cls.tolist()]
            detected_unique = sorted(set(detected))

            st.markdown("### 🛠️ Detected Defect Types")
            cols = st.columns(len(detected_unique))
            for i, defect in enumerate(detected_unique):
                cols[i].markdown(f"<div style='background-color: #28a745; color: white; padding: 0.5em; border-radius: 8px; text-align: center;'>{defect}</div>", unsafe_allow_html=True)
        else:
            st.info("✅ No defects detected. Great weld!")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>🔧 Built with Ultralytics YOLOv8 and Streamlit</p>", unsafe_allow_html=True)

