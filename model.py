import torch.nn as nn
import torchvision.models as models

def get_model(pretrained=True):
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 7)
    )
    return model