import pandas as pd
import os
import shutil

# Load metadata
df = pd.read_csv('HAM10000_metadata.csv')

# Create organized folder
base_folder = 'organized_images'
os.makedirs(base_folder, exist_ok=True)

# Create subfolder for each class
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
for cls in class_names:
    os.makedirs(f'{base_folder}/{cls}', exist_ok=True)

print("Organizing images by class...")

# Copy images to respective folders
for idx, row in df.iterrows():
    image_id = row['image_id']
    true_class = row['dx']
    src_path = f'HAM10000_images/{image_id}.jpg'
    dst_path = f'{base_folder}/{true_class}/{image_id}.jpg'
    
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)

print("Done! Images organized in 'organized_images/' folder")
print("\nStructure:")
for cls in class_names:
    count = len([f for f in os.listdir(f'{base_folder}/{cls}') if f.endswith('.jpg')])
    print(f"  {cls}/: {count} images")