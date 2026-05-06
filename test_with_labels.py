import pandas as pd
import os
from PIL import Image
import torch
from model import get_model
from torchvision import transforms
import random

# Load model
print("Loading model...")
device = torch.device('cpu')
model = get_model(pretrained=False)
model.load_state_dict(torch.load('ham10000_model.pth', map_location=device))
model.eval()

# Load metadata (contains true labels)
df = pd.read_csv('HAM10000_metadata.csv')

# Create label mapping
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
label_to_name = {i: name for i, name in enumerate(class_names)}
name_to_label = {name: i for i, name in enumerate(class_names)}

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    """Predict single image"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    return class_names[predicted.item()], confidence.item()

def test_random_samples(n_samples=10):
    """Test n random images and compare with true labels"""
    print(f"\n{'='*60}")
    print(f"TESTING {n_samples} RANDOM SAMPLES")
    print(f"{'='*60}\n")
    
    # Pick random samples
    samples = df.sample(n=n_samples)[['image_id', 'dx']]
    
    correct = 0
    results = []
    
    for idx, row in samples.iterrows():
        image_id = row['image_id']
        true_label = row['dx']
        image_path = f'HAM10000_images/{image_id}.jpg'
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            continue
        
        # Predict
        pred_label, confidence = predict_image(image_path)
        
        # Compare
        is_correct = (pred_label == true_label)
        if is_correct:
            correct += 1
            status = "✅ CORRECT"
        else:
            status = "❌ WRONG"
        
        # Store result
        results.append({
            'image': image_id,
            'true': true_label,
            'predicted': pred_label,
            'confidence': f"{confidence*100:.1f}%",
            'correct': is_correct
        })
        
        # Print result
        print(f"{status} | Image: {image_id}")
        print(f"      True: {true_label:6s} | Predicted: {pred_label:6s} | Confidence: {confidence*100:.1f}%")
        print()
    
    # Summary
    accuracy = (correct / len(results)) * 100
    print(f"{'='*60}")
    print(f"SUMMARY: {correct}/{len(results)} correct ({accuracy:.1f}%)")
    print(f"{'='*60}\n")
    
    return results

def test_specific_class(class_name, n_samples=5):
    """Test model on specific class"""
    print(f"\n{'='*60}")
    print(f"TESTING CLASS: {class_name} ({n_samples} samples)")
    print(f"{'='*60}\n")
    
    # Filter images of this class
    class_images = df[df['dx'] == class_name]
    
    if len(class_images) == 0:
        print(f"No images found for class {class_name}")
        return
    
    # Sample n images
    samples = class_images.sample(min(n_samples, len(class_images)))
    
    correct = 0
    
    for idx, row in samples.iterrows():
        image_id = row['image_id']
        image_path = f'HAM10000_images/{image_id}.jpg'
        
        if not os.path.exists(image_path):
            continue
        
        pred_label, confidence = predict_image(image_path)
        is_correct = (pred_label == class_name)
        
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} {image_id}: Predicted={pred_label} (conf: {confidence*100:.1f}%)")
    
    accuracy = (correct / len(samples)) * 100
    print(f"\nClass {class_name} Accuracy: {accuracy:.1f}% ({correct}/{len(samples)})")
    print()

def show_class_distribution():
    """Show how many images per class"""
    print(f"\n{'='*60}")
    print("DATASET DISTRIBUTION")
    print(f"{'='*60}\n")
    
    counts = df['dx'].value_counts()
    for class_name in class_names:
        count = counts.get(class_name, 0)
        percentage = (count / len(df)) * 100
        print(f"{class_name:6s}: {count:5d} images ({percentage:5.1f}%)")
    print(f"\nTotal: {len(df)} images")
    print()

# Run tests
if __name__ == "__main__":
    # Show dataset info
    show_class_distribution()
    
    # Test random samples
    test_random_samples(10)
    
    # Test each class specifically
    print("\n" + "="*60)
    print("TESTING EACH CLASS")
    print("="*60)
    for cls in class_names:
        test_specific_class(cls, n_samples=3)