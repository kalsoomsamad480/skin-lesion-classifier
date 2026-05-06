import pandas as pd
import random
from PIL import Image, ImageDraw, ImageFont
import torch
from model import get_model
from torchvision import transforms
import os

# Load model
device = torch.device('cpu')
model = get_model(pretrained=False)
model.load_state_dict(torch.load('ham10000_model.pth', map_location=device))
model.eval()

# Load metadata
df = pd.read_csv('HAM10000_metadata.csv')

class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_and_show(image_id):
    """Show image with prediction vs true label"""
    image_path = f'HAM10000_images/{image_id}.jpg'
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Get true label
    true_label = df[df['image_id'] == image_id]['dx'].values[0]
    
    # Predict
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        pred_label = class_names[predicted.item()]
    
    # Show result
    is_correct = (pred_label == true_label)
    status = "✅ CORRECT" if is_correct else "❌ WRONG"
    
    print(f"\n{status}")
    print(f"Image: {image_id}")
    print(f"True Label:      {true_label}")
    print(f"Predicted:       {pred_label}")
    print(f"Confidence:      {confidence.item()*100:.2f}%")
    
    # Show top 3 predictions
    top3_probs, top3_indices = torch.topk(probs, 3)
    print("Top 3 predictions:")
    for i in range(3):
        cls = class_names[top3_indices[0][i].item()]
        prob = top3_probs[0][i].item() * 100
        marker = " <-- TRUE" if cls == true_label else ""
        print(f"  {i+1}. {cls}: {prob:.2f}%{marker}")
    
    # Open image to view
    image.show()

# Test random images
print("Testing random images...")
for _ in range(5):
    random_id = random.choice(df['image_id'].values)
    predict_and_show(random_id)
    input("\nPress Enter to continue...")