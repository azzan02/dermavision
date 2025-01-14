from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import io

# Create Flask app and set the static folder to 'public'
app = Flask(__name__, static_folder='public')
CORS(app)  # Enable CORS for all routes

# Load the model on the CPU
device = torch.device('cpu')
model = models.resnet34(pretrained=False)
num_classes = 4
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("skin_model_new.pth", map_location=device))
model.to(device)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels and descriptions
class_labels = [
    'Acne and Rosacea',
    'Atopic Dermatitis',
    'Melanoma Skin Cancer Nevi and Moles',
    'Scabies Lyme Disease and other Infestations and Bites'
]

class_descriptions = {
    'Acne and Rosacea': 
        "Acne is a common skin condition that causes pimples, blackheads, and whiteheads, usually on the face, back, and chest. Rosacea is a chronic condition that results in redness, visible blood vessels, and small, red, pus-filled bumps, typically on the face.",
    
    'Atopic Dermatitis': 
        "Atopic Dermatitis (eczema) is a chronic skin condition that causes red, inflamed, and itchy skin. It often appears on the hands, face, and inside of the elbows or knees, and it can flare up due to allergens, irritants, or stress.",
    
    'Melanoma Skin Cancer Nevi and Moles': 
        "Melanoma is a serious form of skin cancer that develops in the cells that produce melanin (pigment) in the skin. It can appear as an unusual mole, spot, or dark patch on the skin and requires early detection for successful treatment.",
    
    'Scabies Lyme Disease and other Infestations and Bites': 
        "Scabies is a skin infestation caused by mites that burrow into the skin, causing intense itching and a pimple-like rash. Lyme Disease is caused by tick bites and can result in a rash, fever, and other flu-like symptoms, while other infestations include bites from insects like fleas, bedbugs, and mosquitoes."
}

# Threshold for detecting "No Skin Condition"
SCORE_THRESHOLD = 0.45  # Adjust this threshold as needed

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Get probabilities
            scores = probabilities[0].tolist()
            
            # Print raw scores for debugging
            print("\nClass scores (raw probabilities):")
            for label, score in zip(class_labels, scores):
                print(f"{label}: {round(score, 4)}")

            predicted_index = torch.argmax(probabilities, dim=1).item()
            predicted_class = class_labels[predicted_index]
            
            # Check if all scores are below the threshold
            if all(score < SCORE_THRESHOLD for score in scores):
                predicted_class = "No Skin Condition"
                description = "The image does not indicate any known skin condition. The skin appears to be in a healthy state."
            else:
                description = class_descriptions[predicted_class]
            
            # Print the predicted class
            print(f"Predicted Class: {predicted_class}\n")

            # Format the scores for each class
            score_dict = {label: round(score, 4) for label, score in zip(class_labels, scores)}

        return jsonify({
            'predicted_class': predicted_class, 
            'description': description, 
            'scores': score_dict
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# Serve frontend files
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
