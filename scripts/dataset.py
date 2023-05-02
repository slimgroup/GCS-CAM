import numpy as np
import h5py
import math
import torch
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Function to load the dataset, preprocess it, and create train, validation, and test loaders
def form_dataset(filename, number_of_samples):
    # Read the HDF5 file containing the dataset
    f = h5py.File(filename, "r")

    # Load and preprocess the training data
    train_x = -np.transpose(np.array(f.get('train_x')), (0, 2, 1))
    train_y = np.array(f.get('train_y')).astype(int)

    # Load and preprocess the testing data
    test_x = -np.transpose(np.array(f.get('test_x')), (0, 2, 1))
    test_y = np.array(f.get('test_y')).astype(int)

    # Initialize a numpy array to store all images
    all_images = np.zeros((number_of_samples, 341, 650, 3))
    all_labels = np.concatenate((train_y, test_y), axis=0)

    # Concatenate training and testing data
    all_images[0:1577, :, :, 0] = train_x
    all_images[0:1577, :, :, 1] = train_x
    all_images[0:1577, :, :, 2] = train_x
    all_images[1577:number_of_samples, :, :, 0] = test_x
    all_images[1577:number_of_samples, :, :, 1] = test_x
    all_images[1577:number_of_samples, :, :, 2] = test_x

    # Normalize image data
    all_images = all_images / np.max(abs(all_images))
    all_images = np.transpose(all_images, (0, 3, 1, 2))

    # Convert labels to categorical format
    all_categorical_labels = to_categorical(all_labels)

    # Convert numpy arrays to PyTorch tensors
    all_images = torch.tensor(all_images)
    all_images = F.interpolate(all_images, size=[224, 224]).to(torch.float)
    all_categorical_labels = torch.Tensor(all_categorical_labels).to(torch.float)

    # Create a dataset from image and label tensors
    all_data = TensorDataset(all_images, all_categorical_labels)

    # Create dataloaders for train, validation, and test sets
    trainloader, validationloader, testloader = form_dataloaders(all_data)

    return trainloader, validationloader, testloader

# Function to create dataloaders for train, validation, and test sets
def form_dataloaders(data):
    # Split the data into train and test sets
    train_size = int(0.8 * len(data)) + 1
    test_size = len(data) - train_size
    train_withval_dataset = torch.utils.data.Subset(data, range(train_size))
    test_dataset = torch.utils.data.Subset(data, range(train_size, train_size + test_size))

    # Further split the train set into train and validation sets
    train_withoutval_size = int(0.8 * len(train_withval_dataset)) + 1
    validation_size = len(train_withval_dataset) - train_withoutval_size
    train_dataset, validation_dataset = torch.utils.data.random_split(train_withval_dataset, [train_withoutval_size, validation_size], generator=torch.Generator().manual_seed(42))
    # Create dataloaders for train, validation, and test sets with specified batch sizes
    trainloader = DataLoader(train_dataset, batch_size=8)
    validationloader = DataLoader(validation_dataset, batch_size=8)
    testloader = DataLoader(test_dataset, batch_size=1)

    # Return the dataloaders
    return trainloader, validationloader, testloader

# Function to convert binary labels to categorical one-hot encoded format
def to_categorical(y, num_classes=None, dtype="float32"):
    # Convert labels to integer type
    y = np.array(y, dtype="int")
    input_shape = y.shape

    # Remove the last dimension if it is 1
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    # Flatten the labels
    y = y.ravel()

    # Determine the number of classes if not specified
    if not num_classes:
        num_classes = np.max(y) + 1

    # Initialize a zero-filled categorical array
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)

    # Set the appropriate elements to 1 based on the label indices
    categorical[np.arange(n), y] = 1

    # Reshape the categorical array to match the input shape
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)

    return categorical