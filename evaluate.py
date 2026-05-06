import torch
from model import get_model
from data_loader import test_loader, class_names
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

device = torch.device('cpu')
model = get_model(pretrained=False)
model.load_state_dict(torch.load('ham10000_model.pth', map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

accuracy = 100 * correct / total
print(f"\n✅ Overall Accuracy: {accuracy:.2f}%")

print("\n📊 Per-Class Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

print("\n🔢 Confusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)

print("\n📈 Per-Class Accuracy:")
for i, class_name in enumerate(class_names):
    class_correct = cm[i][i]
    class_total = cm[i].sum()
    class_acc = 100 * class_correct / class_total
    print(f"  {class_name}: {class_acc:.2f}% ({class_correct}/{class_total})")