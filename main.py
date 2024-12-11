from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import io
import json

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load the model on the CPU
device = torch.device('cpu')  # Force CPU usage
model = models.resnet34(pretrained=False)
num_classes = 6  # Replace with the actual number of classes
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("best_resnet18_model.pth", map_location=device))  # Ensure model loads on CPU
model.to(device)  # Move model to CPU
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels (replace with your actual class labels)
class_labels = [
    'Hair Loss Photos Alopecia and other Hair Diseases',
    'Scabies Lyme Disease and other Infestations and Bites',
    'Melanoma Skin Cancer Nevi and Moles',
    'Eczema Photos',
    'Acne and Rosacea Photos',
    'Psoriasis pictures Lichen Planus and related diseases',
]


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = transform(image).unsqueeze(0).to(device)  # Send image to CPU

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_labels[predicted.item()]
            scores = {label: score for label, score in zip(class_labels, outputs[0].tolist())}

        return jsonify({'predicted_class': predicted_class, 'scores': scores})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
