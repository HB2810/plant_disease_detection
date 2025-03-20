import streamlit as st
import torch
from utils import load_model, predict, preprocess_image
import os
from PIL import Image

# Load Model
MODEL_PATH = 'models/plant_disease_model.pth'
NUM_CLASSES = 38
CLASS_NAMES = os.listdir('class_names.txt')

# Disease Information (Descriptions & Treatments)
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial spot causes dark, water-soaked lesions on leaves.",
        "treatment": "Use copper-based fungicides and avoid overhead watering."
    },
    "Tomato__Leaf_Mold": {
        "description": "Leaf mold leads to yellow spots and fuzzy growth on leaves.",
        "treatment": "Improve air circulation and apply organic fungicides."
    },
    "Healthy": {
        "description": "The plant appears healthy with no visible signs of disease.",
        "treatment": "No treatment needed. Maintain good care practices."
    }
}

# Load trained model
model = load_model(MODEL_PATH, NUM_CLASSES)

def main():
    st.set_page_config(page_title="Plant Disease Detection", layout="wide")
    st.title("🌿 Plant Disease Detection System")
    st.write("Upload an image of a plant leaf to detect the disease.")

    # File Upload Section
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display Uploaded Image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=600)

        # Prediction Logic
        if st.button("Predict Disease"):
            result = predict(uploaded_file, model, CLASS_NAMES)
            confidence_score = torch.nn.functional.softmax(model(preprocess_image(uploaded_file)), dim=1).max().item() * 100

            st.success(f"**Predicted Disease:** {result}")
            st.info(f"**Confidence Score:** {confidence_score:.2f}%")

            # Show Description & Treatment
            if result in disease_info:
                st.subheader("🩺 Disease Information")
                st.write(f"**Description:** {disease_info[result]['description']}")
                st.write(f"**Treatment:** {disease_info[result]['treatment']}")
            else:
                st.warning("No detailed information available for this disease.")

if __name__ == "__main__":
    main()
