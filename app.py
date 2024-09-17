import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import get_model
from config import NUM_CLASSES
from huggingface_hub import hf_hub_download
import io

# Load the model
@st.cache_resource
def load_model():
    model = get_model(NUM_CLASSES)
    
    # Download the model file directly into memory
    model_file = hf_hub_download(repo_id="bcallander/dog_breed_classifier", filename="dog_breed_classifier.pth")
    
    with open(model_file, "rb") as f:
        model_bytes = io.BytesIO(f.read())
    
    model.load_state_dict(torch.load(model_bytes, map_location=torch.device('cpu')))
    model.eval()
    return model

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(model, image, class_names, threshold=0.6):
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    confidence = confidence.item()
    predicted_class = class_names[predicted.item()]

    if confidence < threshold:
        return "other", None
    else:
        return predicted_class, confidence

# Streamlit app
def main():
    st.title("Dog Breed Classifier")

    # Load the model
    model = load_model()

    # Define class names (make sure this matches your training data)
    class_names = ['French Bulldog', 'German Shepherd', 'Golden Retriever', 'Poodle']

    # File uploader
    uploaded_file = st.file_uploader("Choose an image of a dog...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')

        # Make prediction
        predicted_class, confidence = predict_image(model, image, class_names, threshold=0.6)

        # Display results
        if predicted_class == "other":
            st.write("The image doesn't match any of the known dog breeds with sufficient confidence.")
        else:
            st.write(f"Predicted breed: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}")

        st.image(image, caption='Uploaded Image', use_column_width=True)

if __name__ == "__main__":
    main()