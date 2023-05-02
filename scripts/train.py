# Import necessary libraries
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models
import time
from tqdm import tqdm

# Function to train the model
def train_model(model, trainloader, validationloader, device, num_epochs=100, lr=0.0005, step_size=1, gamma=0.99):
    # Set the loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.head.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    
    # Call the train() function and return the results
    trained_model, train_loss, val_loss, train_acc, val_acc = \
        train(model, trainloader, validationloader, device, criterion, optimizer, scheduler, num_epochs)
    
    return trained_model, train_loss, val_loss, train_acc, val_acc

# Function to train a vision transformer model
def train(model, trainloader, validationloader, device, criterion, optimizer, scheduler, num_epochs):
    # Initialize variables for timing and best accuracy
    since = time.time()
    best_acc = 0.0
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    # Loop through each epoch
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print("-"*10)
        
        # Loop through training and validation phases
        for phase in ['train', 'val']:
            # Set the model to the appropriate mode and choose the data loader
            if phase == 'train':
                model.train()  # Set model to training mode
                loader = trainloader
            else:
                model.eval()   # Set model to evaluate mode
                loader = validationloader

            # Initialize variables for loss and accuracy
            input_size = len(loader.dataset)
            running_loss = 0.0
            running_corrects = 0.0
            
            # Loop through the data in the loader
            for inputs, labels in tqdm(loader):
                # Move inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Compute the outputs and loss
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    loss = criterion(outputs, labels)
                    
                    # If in training phase, update the weights
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Update the running loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == torch.argmax(labels.data, dim=1))
                
            # Update the learning rate scheduler in the training phase
            if phase == 'train':
                scheduler.step()
            
            # Calculate and print the epoch loss and accuracy
            epoch_loss = running_loss / input_size
            epoch_acc = running_corrects.double() / input_size
            
            # Append the results to the appropriate lists
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc.cpu().numpy())
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc.cpu().numpy())
            
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
        print()
    time_elapsed = time.time() - since # slight error
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    # model.load_state_dict(best_model_wts)
    return model, train_loss, val_loss, train_acc, val_acc
