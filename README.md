# Dermavision: Skin Disease Classification Project
This repository contains the code and resources for a skin disease classification system. The project utilizes a ResNet-34 model fine-tuned to classify skin conditions into four categories based on the DermNet dataset.

## Features
  - **Skin Disease Detection:** Identify and classify skin conditions into the following categories:
    - Acne and Rosacea
    - Atopic Dermatitis
    - Melanoma Skin Cancer Nevi and Moles
    - Scabies, Lyme Disease, and Other Infestations and Bites
  - **Flask API:** A RESTful API to handle image uploads and return classification results.
  - **Frontend Integration:** Easily integrates with a static frontend served from the public folder.

## Setup Instructions
### Prerequisites
Ensure you have the following installed:
  - Python 3.8 or higher
  - Pip
  - Torch and torchvision libraries

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/azza02/dermavision.git
   cd skin-disease-classification
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the trained model file (skin_model_new.pth) and place it in the root directory.

### Running the Application
1. Start the Flask server:
   ```
   python main.py
   ```
2. Access the API at ***http://localhost:8000.***

### API Endpoints
  - POST /predict:
      - Request: Upload an image file under the key image.
      - Response:\
         {\
         "predicted_class": "Acne and Rosacea",\
         "description": "Acne is a common skin condition that causes pimples...",\
         "scores": {\
         "Acne and Rosacea": 0.85,\
         "Atopic Dermatitis": 0.1,\
         "Melanoma Skin Cancer Nevi and Moles": 0.03,\
         "Scabies Lyme Disease and Other Infestations and Bites": 0.02\
           }\
         }
  - Threshold for "No Skin Condition"
    - If all class probabilities are below a threshold (default: 0.45), the API will return "No Skin Condition."
   
### Project Structure
.\
├── main.py                # Flask application and API logic\
├── model_file_final.ipynb # Jupyter notebook for training the ResNet model\
├── index                  # Static files for the frontend\
├── requirements.txt       # Python dependencies\
└── skin_model_new.pth     # Trained ResNet-34 model (download separately\

### Training the Model
The model training script is provided in **model_file_final.ipynb.** It demonstrates:
  - Data preprocessing
  - Model architecture modifications
  - Training and evaluation steps

### Future Enhancements
  - Add more skin conditions for classification.
  - Improve the frontend interface.
  - Deploy the system on a cloud platform.

### License
This project is licensed under the MIT License.
