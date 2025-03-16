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
    st.title("Lumina Flora: Plant Growth Stage Detection")

    image_file = st.camera_input("Take a picture")
    
    if image_file:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Captured Image", use_column_width=True)
        
        with st.spinner("Processing Image..."):
            model = YOLO("40 Epoch Plant Growth Stage YOLOv8 Model.pt")
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
        
        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Detected Growth Stages", use_column_width=True)
        
        if detection_results:
            st.subheader("Detection Results")
            for result in detection_results:
                st.write(f"🌱 **Growth Stage:** {result['Label']}")
                st.write(f"🔍 **Confidence:** {result['Confidence']}")
                st.write("----")

if __name__ == '__main__':
    main()
