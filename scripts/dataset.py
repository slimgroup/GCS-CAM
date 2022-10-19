import numpy as np
import h5py 
import math
import torch
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Load the dataset
def form_dataset(filename, number_of_samples):
#     f = h5py.File("/data/home/agahlot8/TLE_Data/dataset_jrm_1971_seismic_images", "r")
    f = h5py.File(filename, "r")
    train_x = -np.transpose( np.array(f.get('train_x')), (0, 2, 1))
    train_y = np.array(f.get('train_y'))
    train_y = train_y.astype(int)

    test_x = -np.transpose(np.array(f.get('test_x')), (0, 2, 1))
    test_y = np.array(f.get('test_y'))
    test_y = test_y.astype(int)

    # One image has size (341, 650)
    all_images =np.zeros((number_of_samples, 341, 650, 3))
    all_labels = np.concatenate((train_y, test_y), axis=0)

    all_images[0:1577,:,:,0] = train_x
    all_images[0:1577,:,:,1] = train_x
    all_images[0:1577,:,:,2] = train_x

    all_images[1577:number_of_samples,:,:,0] = test_x
    all_images[1577:number_of_samples,:,:,1] = test_x
    all_images[1577:number_of_samples,:,:,2] = test_x

    all_images = all_images/ np.max(abs(all_images))
    all_images = np.transpose(all_images, (0, 3, 1, 2))

    all_categorical_labels = to_categorical(all_labels) 

    all_images =  torch.tensor(all_images)
    all_images = F.interpolate(all_images , size = [224,224]).to(torch.float)

    all_categorical_labels = torch.Tensor(all_categorical_labels).to(torch.float)
    all_data=TensorDataset(all_images,all_categorical_labels )

    trainloader, validationloader, testloader = form_dataloaders(all_data)
    
    return trainloader, validationloader, testloader

# Form the dataloaders for test and train functions
def form_dataloaders(data):
    train_size = int(0.8 * len(data))+1
    test_size = len(data) - train_size
    train_withval_dataset = torch.utils.data.Subset(data, range(train_size))
    test_dataset = torch.utils.data.Subset(data, range(train_size, train_size+test_size))

    train_withoutval_size = int(0.8 * len(train_withval_dataset)) + 1
    validation_size = len(train_withval_dataset) - train_withoutval_size
    train_dataset, validation_dataset = torch.utils.data.random_split(train_withval_dataset, [train_withoutval_size, validation_size], generator=torch.Generator().manual_seed(42))

    trainloader = DataLoader(train_dataset,batch_size=8)
    validationloader = DataLoader(validation_dataset,batch_size=8)
    testloader = DataLoader(test_dataset,batch_size=1) 
    
    return trainloader, validationloader, testloader


# Convert binary labels to categorical
def to_categorical(y, num_classes=None, dtype="float32"):
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical