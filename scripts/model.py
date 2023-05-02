import timm
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models

# Function to create a vision transformer model
def create_model():
    # Instantiate the vision transformer model with the specified configuration and pretrained weights
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)

    # Obtain the number of input features for the last classification layer
    n_inputs = model.head.in_features

    # Replace the last classification layer with a new one consisting of a linear layer, dropout, and softmax activation
    model.head = nn.Sequential(
        nn.Linear(n_inputs, 2),  # Linear layer to reduce features to 2 classes
        nn.Dropout(0.25),  # Dropout layer to prevent overfitting
        nn.Softmax(dim=1))  # Softmax activation to produce class probabilities

    # Return the modified model
    return model