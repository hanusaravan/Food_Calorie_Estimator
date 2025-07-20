import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import numpy as np

# Model Definition (must match training exactly)
class ResNetCalories(nn.Module):
    def __init__(self):
        super(ResNetCalories, self).__init__()
        base_model = models.resnet18(weights=None)
        num_features = base_model.fc.in_features
        base_model.fc = nn.Linear(num_features, 4)
        self.model = base_model

    def forward(self, x):
        return self.model(x)

# Load model with proper checkpoint handling
@st.cache_resource
def load_model():
    try:
        model = ResNetCalories()
        checkpoint = torch.load("best_model.pth", map_location=torch.device('cpu'))
        
        # Handle both full checkpoint and state_dict-only saves
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.error("Please ensure:")
        st.error("1. The model file exists in this directory")
        st.error("2. The model architecture matches your training code")
        st.error("3. You're loading the correct file (should be .pth)")
        return None

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

st.title("🍽️ Food Nutrition Estimator")
uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(input_tensor)[0].cpu().numpy()
        
        prediction = np.maximum(prediction, 0)
        calories, carbs, protein, fat = prediction
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Energy")
            st.metric("Calories", f"{calories:.0f} kcal")
        with col2:
            st.subheader("Macronutrients")
            st.metric("Carbs", f"{carbs:.1f} g")
            st.metric("Protein", f"{protein:.1f} g")
            st.metric("Fat", f"{fat:.1f} g")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")