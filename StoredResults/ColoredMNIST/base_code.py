import numpy as np
import threading
import csv
import warnings
import matplotlib.pyplot as plt
import random
import os
from datetime import datetime
import pandas as pd
from shutil import move
warnings.filterwarnings("ignore")
import cv2
from PIL import Image
import json
import sys

python_file_name = sys.argv[1]
from augumentation import get_augmentation
augmentation_transform = get_augmentation(python_file_name)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, random_split
from torchvision import datasets
from torchvision import models
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

with open('parameters.json', 'r') as f:
    params = json.load(f)

epochs = params['epochs']
batch_size = params['batch_size']
num_runs = params['num_runs']

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_data_loaders(batch_size=batch_size):
    train_dataset = datasets.ImageFolder(root='ColoredMNIST/train', transform=augmentation_transform)
    test_dataset = datasets.ImageFolder(root='ColoredMNIST/test', transform=transform_test)
    
    # Create a subset of 1000 samples per class for the training set
    targets = np.array(train_dataset.targets)
    train_indices = []

    for class_idx in range(10):
        class_indices = np.where(targets == class_idx)[0]
        selected_indices = np.random.choice(class_indices, 1000, replace=False)
        train_indices.extend(selected_indices)

    # Create the training subset
    train_subset = Subset(train_dataset, train_indices)

    # Ensure the test dataset has exactly 2,000 samples

    # Create a subset of 200 samples per class for the test set
    test_targets = np.array(test_dataset.targets)
    test_indices = []

    for class_idx in range(10):
        class_indices = np.where(test_targets == class_idx)[0]
        selected_indices = np.random.choice(class_indices, 200, replace=False)
        test_indices.extend(selected_indices)

    # Create the test subset
    test_subset = Subset(test_dataset, test_indices)
    """if len(test_dataset) > 10000:
        test_subset, _ = random_split(test_dataset, [10000, len(test_dataset) - 10000])
    else:
        test_subset = test_dataset"""

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Training Dataset Size: {len(train_subset)},   Test Dataset Size: {len(test_subset)}")
    
    return train_loader, test_loader

# Define a function for training the model
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

# Define a function to perform a single run
def single_run(run_number, results_writer):
    # Initialize the optimizer and loss function
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define model, loss function, and optimizer
    train_loader, test_loader = get_data_loaders()

    # Training loop
    for epoch in range(1, epochs + 1):
        
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f'Epoch {epoch}/{epochs}, '
            f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, '
            f'Test Accuracy: {test_acc:.4f}')

        results_writer.writerow([run_number, epoch, train_loss, train_acc, test_acc])

# Open a CSV file for writing results
def main():
    with open('resnet_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Run', 'Epoch', 'Train Loss', 'Train Accuracy', 'Test Accuracy'])

        # Start parallel runs
        threads = []
        for i in range(1, num_runs + 1):
            thread = threading.Thread(target=single_run, args=(i, writer))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

    print("All runs completed.")


if __name__ == '__main__':
    main()


"""
unedited base code
import numpy as np
import threading
import csv
import warnings
import matplotlib.pyplot as plt
import random
import os
from datetime import datetime
import pandas as pd
from shutil import move
warnings.filterwarnings("ignore")
import cv2
from PIL import Image
import json
import sys

python_file_name = sys.argv[1]
from augumentation import get_augmentation
augmentation_transform = get_augmentation(python_file_name)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset,DataLoader, random_split
from torchvision import datasets
from torchvision import models
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

with open('parameters.json', 'r') as f:
    params = json.load(f)

epochs = params['epochs']
batch_size = params['batch_size']
num_runs=params['num_runs']

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_data_loaders(batch_size=batch_size):
    train_dataset = datasets.ImageFolder(root='ColoredMNIST/train', transform=augmentation_transform)
    test_dataset = datasets.ImageFolder(root='ColoredMNIST/test', transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

# Define a function for training the model
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

# Define a function to perform a single run
def single_run(run_number, results_writer):
    # Initialize the optimizer and loss function
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()


    # Define model, loss function, and optimizer
    train_loader, test_loader = get_data_loaders()
    

    # Training loop
    for epoch in range(1, epochs + 1):
        
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f'Epoch {epoch}/{epochs}, '
            f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, '
            f'Test Accuracy: {test_acc:.4f}')

        results_writer.writerow([run_number, epoch, train_loss, train_acc, test_acc])

# Open a CSV file for writing results

def main():
    with open('resnet_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Run', 'Epoch', 'Train Loss', 'Train Accuracy', 'Test Accuracy'])

        # Start parallel runs
        threads = []
        for i in range(1, num_runs + 1):
            thread = threading.Thread(target=single_run, args=(i, writer))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

    print("All runs completed.")


if __name__ == '__main__':
    main()"""