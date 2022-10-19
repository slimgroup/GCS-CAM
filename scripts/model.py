import timm
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models

# Create vision tranformer model
def create_model():
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    
    n_inputs = model.head.in_features
    # Replace the last classification layer with a new one.
    model.head = nn.Sequential(
        nn.Linear(n_inputs, 2),
        nn.Dropout(0.25),
        nn.Softmax(dim=1))
    
    return model