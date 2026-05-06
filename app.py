from flask import Flask, render_template, request, jsonify
import torch
from model import get_model
from torchvision import transforms
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load model once when app starts
print("Loading model...")
device = torch.device('cpu')
model = get_model(pretrained=False)
model.load_state_dict(torch.load('ham10000_model.pth', map_location=device))
model.eval()
print("Model loaded!")

# Class names and explanations
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
explanations = {
    'akiec': 'Actinic keratosis (pre-cancerous)',
    'bcc': 'Basal cell carcinoma (common skin cancer)',
    'bkl': 'Benign keratosis (non-cancerous)',
    'df': 'Dermatofibroma (benign)',
    'mel': 'Melanoma (serious skin cancer)',
    'nv': 'Melanocytic nevus (mole, usually benign)',
    'vasc': 'Vascular lesion (blood vessel related)'
}

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image):
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    pred_class = class_names[predicted.item()]
    confidence_pct = confidence.item() * 100
    
    # Get all probabilities
    all_probs = {class_names[i]: float(probs[0][i]) * 100 for i in range(len(class_names))}
    all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
    
    return pred_class, confidence_pct, all_probs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Make prediction
        pred_class, confidence, all_probs = predict_image(image)
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'prediction': pred_class,
            'explanation': explanations[pred_class],
            'confidence': f"{confidence:.2f}%",
            'all_probabilities': all_probs,
            'image': img_str,
            'risk': 'HIGH' if pred_class in ['mel', 'bcc'] else 'MEDIUM' if pred_class == 'akiec' else 'LOW'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)