import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Define the label map
label_map = {
    'Flowering': 'Flowering',
    'Vegetative': 'Vegetative',
    'Germination': 'Germination',
    'Harvesting': 'Harvesting'
}

# Streamlit app
def main():
    st.set_page_config(page_title="Lumina Flora", layout="centered")
    
    st.markdown("""
        <style>
            .title {text-align: center; font-size: 36px; font-weight: bold; color: #2C3E50;}
            .subtitle {text-align: center; font-size: 20px; color: #34495E;}
            .section {margin-top: 20px; padding: 10px; border-radius: 10px; background-color: #ECF0F1;}
            .result-box {padding: 10px; background-color: #D5F5E3; border-radius: 10px;}
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="title">🌱 Lumina Flora: Plant Growth Stage Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Capture an image to detect the plant growth stage</p>', unsafe_allow_html=True)
    
    image_file = st.camera_input("📸 Take a picture")
    
    if image_file:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Read as BGR
        
        # Convert BGR to RGB before displaying
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="📷 Captured Image", use_column_width=True)
        
        with st.spinner("🔍 Processing Image..."):
            model = YOLO("model_50_1024_8_0_4.pt")
            results = model.predict(source=image, save=False, conf=0.25)

        annotated_image = image.copy()
        detection_results = []

        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result[0], result[1], result[2], result[3], result[4], result[5]
            confidence = round(confidence, 2)
            class_name = model.names[class_id]
            class_name = label_map.get(class_name, class_name)
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 255, 0), 4)
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            detection_results.append({'Label': class_name, 'Confidence': confidence})
        
        # Convert annotated image to RGB before displaying
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        st.image(annotated_image_rgb, caption="✅ Detected Growth Stages", use_column_width=True)
        
        if detection_results:
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.subheader("📊 Detection Results")
            for result in detection_results:
                st.markdown(f'<div class="result-box">🌱 <b>Growth Stage:</b> {result["Label"]} <br> 🔍 <b>Confidence:</b> {result["Confidence"]}</div>', unsafe_allow_html=True)
                st.write("----")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
