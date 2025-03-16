import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Define the label map
label_map = {
    'Flowering': 'Growing',
    'Vegetative': 'Growing',
    'Germination': 'Germination',
    'Harvesting': 'Harvesting'
}

# Function to preprocess image for Thurinus Lettuce
def preprocess_thurinus(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_img[:, :, 0] = 60  # Change hue to a greenish value
    green_hued_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return green_hued_img

# Function to provide system response based on growth stage and plant type
def get_system_response(plant_type, growth_stage):
    responses = {
        "Olmetie Lettuce": {
            "Germination": {
                "light_color": "Blue",
                "light_intensity": "5,000-10,000 lux",
                "temperature": "16-20°C"
            },
            "Growing": {
                "light_color": "Red + Blue",
                "light_intensity": "15,000-20,000 lux",
                "temperature": "18-22°C"
            },
            "Harvesting": {
                "light_color": "Red + Blue",
                "light_intensity": "15,000-20,000 lux",
                "temperature": "18-22°C"
            }
        },
        "Thurinus Lettuce": {
            "Germination": {
                "light_color": "Blue",
                "light_intensity": "5,000-8,000 lux",
                "temperature": "15-19°C"
            },
            "Growing": {
                "light_color": "Red + Blue",
                "light_intensity": "12,000-18,000 lux",
                "temperature": "18-24°C"
            },
            "Harvesting": {
                "light_color": "Red + Blue",
                "light_intensity": "12,000-18,000 lux",
                "temperature": "18-24°C"
            }
        }
    }
    return responses.get(plant_type, {}).get(growth_stage, {})

# Streamlit app
def main():
    st.set_page_config(layout="wide", page_title="Lumina Flora")

    st.markdown("""
        <style>
        .header {font-size:52px; font-weight:bold; text-align:center; color: #117A65;}
        .subheader {font-size:28px; font-weight:bold; text-align:left; color: #117A65;}
        .settings {font-size:20px; font-weight:bold; color: #34495E;}
        .sidebar-names li {margin-bottom: 0px;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="header">Lumina Flora: Plant Growth Stage Detection</p>', unsafe_allow_html=True)

    # Sidebar for navigation
    st.sidebar.title("Lumina Flora: Options")
    option = st.sidebar.selectbox("Select Plant Type:", ["None", "Olmetie Lettuce", "Thurinus Lettuce"])

    if option != "None":
        st.sidebar.markdown(f"**Plant Type:** {option}")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("<p class='subheader'>Input Phase</p>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if option == "Thurinus Lettuce":
                    image = preprocess_thurinus(original_image)
                else:
                    image = original_image

                if st.button("Detect Growth Stage"):
                    with st.spinner("Processing..."):
                        model = YOLO("40 Epoch Plant Growth Stage YOLOv8 Model.pt")
                        results = model.predict(source=image, save=False, conf=0.25)

                    detection_results = []
                    annotated_image = original_image.copy()

                    for result in results[0].boxes.data.tolist():
                        x1, y1, x2, y2, confidence, class_id = result[0], result[1], result[2], result[3], result[4], result[5]
                        confidence = round(confidence, 2)
                        class_name = model.names[class_id]
                        class_name = label_map.get(class_name, class_name)
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 255, 0), 4)
                        label = f"{class_name} ({confidence:.2f})"
                        font_scale, font_thickness = 1.0, 2
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                        text_x, text_y = x1, y1 - 10
                        label_bg_x2 = text_x + text_size[0] + 4
                        label_bg_y2 = text_y + text_size[1] + 4
                        cv2.rectangle(annotated_image, (text_x - 2, text_y - text_size[1] - 4), 
                                      (label_bg_x2, label_bg_y2), (255, 255, 0), -1)
                        cv2.putText(annotated_image, label, (text_x, text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

                        detection_results.append({
                            'Label': class_name,
                            'Confidence': confidence,
                            'Bounding Box': (x1, y1, x2, y2)
                        })

                    if detection_results:
                        st.markdown("<p class='subheader'>Device Configuration</p>", unsafe_allow_html=True)
                        for result in detection_results:  # Iterate through the list of results
                            response = get_system_response(option, result['Label'])
                            if response:
                                st.markdown(f"<p class='settings'>🌱 <b>Growth Stage:</b> {result['Label']}</p>", unsafe_allow_html=True)
                                st.markdown(f"<p class='settings'>🔍 <b>Confidence:</b> {result['Confidence']}</p>", unsafe_allow_html=True)
                                st.markdown(f"<p class='settings'>🌈 <b>Light Color:</b> {response['light_color']}</p>", unsafe_allow_html=True)
                                st.markdown(f"<p class='settings'>💡 <b>Light Intensity:</b> {response['light_intensity']}</p>", unsafe_allow_html=True)
                                st.markdown(f"<p class='settings'>🌡️ <b>Temperature:</b> {response['temperature']}</p>", unsafe_allow_html=True)
                            st.write("----")

                    with col2:
                        st.markdown("<p class='subheader'>Detection Results</p>", unsafe_allow_html=True)
                        st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=300)
                        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Detected Growth Stages", width=300)

    st.sidebar.markdown("""
        <ul class="sidebar-names">
            Made by Team 45
            <li>Dela Cruz, Akio Gavin</li>
            <li>Mendoza, John Renz</li>
            <li>Moreno, Remar</li>
            <li>Villanueva, Mark John</li>
        </ul>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
