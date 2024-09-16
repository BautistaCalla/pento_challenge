import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    model = models.resnet101(pretrained=True)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Ensure the parameters of the new fc layer are set to require gradients
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model    