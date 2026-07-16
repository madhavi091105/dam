import streamlit as st
import torch
from PIL import Image

from classifier import load_classifier, predict

st.set_page_config(page_title="Damage Classifier", layout="centered")
st.title("🏗️ Damage Classification Model — Tester")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/best_model.pth"


@st.cache_resource
def get_model():
    return load_classifier(MODEL_PATH, DEVICE)


model, num_classes = get_model()
st.caption(f"Loaded model on **{DEVICE}** — detected **{num_classes} classes**")

uploaded_file = st.file_uploader("Upload a test image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input Image", use_container_width=True)

    if st.button("Run Classification"):
        with st.spinner("Running inference..."):
            result = predict(model, image, DEVICE)

        st.success(f"Prediction: **{result['label']}** ({result['confidence']*100:.2f}% confidence)")
        st.bar_chart(result["probs"])
else:
    st.info("Upload an image to test the classifier.")