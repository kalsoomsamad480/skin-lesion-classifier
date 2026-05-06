import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from model import get_model
from data_loader import train_loader, test_loader, class_counts

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = get_model().to(device)

# Weighted loss for class imbalance
weights = torch.FloatTensor(1.0 / class_counts).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Scheduler
epochs = 25
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# Early stopping settings
best_acc = 0.0
patience_counter = 0
patience = 7
start_epoch = 0

# Auto-resume from checkpoint if exists
if os.path.exists('checkpoint.pth'):
    print("✅ Resuming from checkpoint...")
    checkpoint = torch.load('checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    patience_counter = checkpoint['patience_counter']
    print(f"Resumed from epoch {start_epoch}, best accuracy so far: {best_acc:.2f}%")
else:
    print("Starting fresh training...")

for epoch in range(start_epoch, epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (i+1) % 10 == 0:
            print(f'  Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc  = 100. * correct / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total
    scheduler.step(epoch_loss)

    print(f'Epoch [{epoch+1}/{epochs}]')
    print(f'  Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%')
    print(f'  Val Acc: {val_acc:.2f}%')

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'ham10000_model.pth')
        print(f'  ✅ Best model saved! Val Acc: {best_acc:.2f}%')
        patience_counter = 0
    else:
        patience_counter += 1
        print(f'  No improvement ({patience_counter}/{patience})')

    # Save checkpoint after every epoch
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_acc': best_acc,
        'patience_counter': patience_counter
    }, 'checkpoint.pth')
    print(f'  💾 Checkpoint saved (epoch {epoch+1}/25)')

    # Early stopping
    if patience_counter >= patience:
        print("Early stopping triggered!")
        break

print(f"\nTraining complete! Best Val Accuracy: {best_acc:.2f}%")
print("Best model saved as 'ham10000_model.pth'")