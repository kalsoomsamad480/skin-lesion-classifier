import pandas as pd
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

class HAM10000Dataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_id'] + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# Load metadata
df = pd.read_csv('HAM10000_metadata.csv')

# Create numeric labels
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
df['label'] = df['dx'].map({name: i for i, name in enumerate(class_names)})

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)

# Augmented transform for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Clean transform for testing
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = HAM10000Dataset(train_df, 'HAM10000_images', train_transform)
test_dataset  = HAM10000Dataset(test_df,  'HAM10000_images', test_transform)

# Weighted sampler to fix class imbalance
class_counts = np.array([len(train_df[train_df['label'] == i]) for i in range(7)])
class_weights = 1.0 / class_counts
sample_weights = class_weights[train_df['label'].values]
sampler = WeightedRandomSampler(
    weights=torch.FloatTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Class counts: {class_counts}")