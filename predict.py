import torch
from torchvision import transforms
from PIL import Image
import argparse
from model import get_model
from config import NUM_CLASSES

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path):
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(model, image_path, class_names, threshold=0.7):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    confidence = confidence.item()
    predicted_class = class_names[predicted.item()]

    if confidence < threshold:
        return "other", confidence
    else:
        return predicted_class, confidence

def main():
    parser = argparse.ArgumentParser(description="Predict dog breed from an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--model_path", type=str, default="dog_breed_classifier.pth", help="Path to the trained model")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold for prediction")
    args = parser.parse_args()

    # Define class names (make sure this matches your training data)
    class_names = ['french_bulldog', 'german_shepherd', 'golden_retriever', 'poodle']

    model = load_model(args.model_path)
    predicted_class, confidence = predict_image(model, args.image_path, class_names, args.threshold)

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()