import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import SegformerForSemanticSegmentation

# 1. Model Loading
@st.cache_resource
def load_model():
    # Architecture load karo
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", num_labels=3, ignore_mismatched_sizes=True)
    # Weights load karo (Strict=False se error nahi aayega)
    checkpoint = torch.load("model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

model = load_model()

# 2. UI
st.title("Dam Damage Segmentation")
uploaded_file = st.file_uploader("Image upload karo...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image")
    
    # 3. Prediction
    img_tensor = torch.tensor(np.array(image.resize((512, 512)))).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    with torch.no_grad():
        output = model(pixel_values=img_tensor)
        prediction = torch.argmax(output.logits, dim=1).squeeze().numpy()
        
    # Result mask ko visual banao
    st.image(prediction * 100, caption="Segmentation Mask") # *100 kiya hai taaki mask dikhe