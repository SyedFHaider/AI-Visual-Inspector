import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from fpdf import FPDF
import datetime

# Suggestions for known defects
DEFECT_SUGGESTIONS = {
    "crack": "Ensure proper cooling rates and avoid high residual stresses.",
    "porosity": "Use dry electrodes and maintain clean work surfaces.",
    "incomplete_fusion": "Increase heat input or adjust torch angle.",
    "slag_inclusion": "Clean slag between passes and use proper technique.",
    "undercut": "Reduce travel speed and maintain correct angle.",
    "burn_through": "Lower current and avoid excessive heat input.",
    "spatter": "Adjust voltage or use anti-spatter spray."
}

# Streamlit App Config
st.set_page_config(page_title="AI VISUAL INSPECTOR", page_icon="⚡", layout="centered")
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>⚡ AI VISUAL INSPECTOR</h1>", unsafe_allow_html=True)
st.markdown("---")

MODEL_PATH = "best.pt"

def generate_pdf_report(original_img, annotated_img, detected_defects, filename="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Welding Inspection Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    # Add original image
    orig_path = "original_temp.jpg"
    original_img.save(orig_path)
    pdf.image(orig_path, w=150)
    os.remove(orig_path)

    # Add annotated image
    annotated_path = "annotated_temp.jpg"
    annotated_img.save(annotated_path)
    pdf.image(annotated_path, w=150)
    os.remove(annotated_path)

    # Detected defects and suggestions
    if detected_defects:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "Detected Defect Types & Recommendations:", ln=True)
        pdf.set_font("Arial", size=12)
        for defect in detected_defects:
            pdf.cell(200, 10, f"- {defect.capitalize()}: {DEFECT_SUGGESTIONS.get(defect, 'No suggestion available')}", ln=True)
    else:
        pdf.cell(200, 10, "✅ No defects detected. Great weld!", ln=True)

    pdf.output(filename)
    return filename

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file '{MODEL_PATH}' not found. Please upload it to the app directory.")
else:
    model = YOLO(MODEL_PATH)

    uploaded_file = st.file_uploader("📤 Upload a Welding Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            image.save(tmp.name)
            results = model(tmp.name)[0]

        annotated_img = Image.fromarray(results.plot())
        st.image(annotated_img, caption="🔍 Detection Results", use_container_width=True)

        # Show detected classes
        if results.boxes and len(results.boxes.cls) > 0:
            detected = [results.names[int(cls)] for cls in results.boxes.cls.tolist()]
            detected_unique = sorted(set(detected))

            st.markdown("### 🛠️ Detected Defect Types")
            cols = st.columns(len(detected_unique))
            for i, defect in enumerate(detected_unique):
                cols[i].markdown(
                    f"<div style='background-color: #28a745; color: white; padding: 0.5em; border-radius: 8px; text-align: center;'>{defect}</div>",
                    unsafe_allow_html=True
                )

            if st.button("📄 Generate Report (PDF)"):
                report_path = generate_pdf_report(image, annotated_img, detected_unique)
                with open(report_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Report",
                        data=file,
                        file_name="Welding_Inspection_Report.pdf",
                        mime="application/pdf"
                    )

        else:
            st.info("✅ No defects detected. Great weld!")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>🔧 Built with Ultralytics YOLOv8 and Streamlit</p>", unsafe_allow_html=True)
