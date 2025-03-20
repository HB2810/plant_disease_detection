import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import PlantDiseaseModel
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from models.model import PlantDiseaseModel # Import the model class for inference
import os

# Load the trained model
def load_model(model_path, num_classes=38):
    from models.model import PlantDiseaseModel  # Correct import for PlantDiseaseModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Updated torch.load with weights_only=False
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = PlantDiseaseModel(num_classes).to(device)
    model.load_state_dict(checkpoint['model'].state_dict())
    model.eval()
    
    return model


# Image Preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Prediction Function
def predict(image_path, model, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    return class_names[predicted_class.item()]

# Example Usage (for testing purpose)
if __name__ == "__main__":
    model_path = 'models/plant_disease_model.pth'
    num_classes = 38  # Updated for 38 classes
    class_names = os.listdir('D:/plant_disease_detection/dataset/train')

    model = load_model(model_path, num_classes)
    test_image_path = 'D:\\plant_disease_detection\\dataset\\test\\Pepper__bell___Bacterial_spot\\0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG'  # Sample image path

    result = predict(test_image_path, model, class_names)
    print(f'Predicted Class: {result}')
