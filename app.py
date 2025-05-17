from ultralytics import YOLO
import streamlit as st
from PIL import Image
import tempfile

# Load trained model file (ensure best.pt is in the same folder or give full path)
model = YOLO("best.pt")  

st.title("AI VISUAL INSPECTOR")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image")

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        img.save(tmp.name)
        results = model(tmp.name)[0]  # Run inference

    annotated_img = Image.fromarray(results.plot())
    st.image(annotated_img, caption="Detection Results")

    if results.boxes:
        detected_classes = [results.names[int(cls)] for cls in results.boxes.cls.tolist()]
        st.write("Detected Defects:", set(detected_classes))
    else:
        st.write("No defects detected.")

