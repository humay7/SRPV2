base_code

import numpy as np
import threading
import csv
import warnings
import random
import os
from datetime import datetime
import json
import sys

# Import the AutoAugment policy
from autoaugment import ImageNetPolicy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

with open('parameters.json', 'r') as f:
   params = json.load(f)

epochs = params['epochs']
batch_size = params['batch_size']
images_per_class = params['images_per_class']
num_runs = params['num_runs']
n_samples_add_pool = params['n_samples_add_pool']

# Apply AutoAugment for the labeled set
auto_augment_transform = transforms.Compose([
   transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
   ImageNetPolicy(),  # AutoAugment policy for ImageNet
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3 channels
])

# Rotation transforms for the unlabeled set
rotation_transforms = []
for i in range(4):
   angle = i * 90
   transform = transforms.Compose([
       transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
       transforms.Resize((84, 84)),
       transforms.RandomCrop(84, padding=8),
       transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
       transforms.RandomRotation(angle),  # Apply the rotation
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3 channels
   ])
   rotation_transforms.append(transform)

# Choose a random transform from rotation_transforms for each image in the unlabeled set
def apply_all_rotation_transforms(img):
    transformed_images = []
    for transform in rotation_transforms:
        transformed_images.append(transform(img))  # Apply each transform and store the result
    return transformed_images  # Return a list of all transformed images

# Choose one random rotation transform from rotation_transforms for each image in the unlabeled set
def apply_random_rotation_transform(img):
    transform = random.choice(rotation_transforms)  # Choose one random transformation
    return transform(img)  # Apply the selected transformation

# Define active learning strategy (Uncertainty Sampling)
def uncertainty_sampling(model, unlabeled_loader, n_samples):
    def entropy(p):
        return -torch.sum(p * torch.log2(p), dim=1)

    uncertainties = []
    with torch.no_grad():
        for data in unlabeled_loader:
            images, _ = data
            images = images.cuda()
            outputs = torch.softmax(model(images), dim=1)
            uncertainties.extend(entropy(outputs).tolist())
  
    # Select indices of top uncertain samples
    top_indices = np.argsort(uncertainties)[-n_samples:]
    return top_indices

# Load datasets with appropriate augmentations
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels for testing as well
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

# Print dataset stats
print('Dataset Stats:')
print(f'Train Dataset Size: {len(train_set)}, Test Dataset Size: {len(test_set)}\n')

# Define a function to perform a single run
def single_run(run_number, results_writer):
    # Initialize a new model for each run
    model = torchvision.models.resnet18(pretrained=False)  # ResNet18 expects 3-channel input
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # Modify the last fully connected layer for 10 classes (MNIST)
    model.cuda()  # Move model to GPU

    # Initialize the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Split train set into labeled and unlabeled indices
    labeled_indices = []
    unlabeled_indices = []
    for class_label in range(10):
        class_indices = np.where(np.array(train_set.targets) == class_label)[0]
        np.random.shuffle(class_indices)
        selected_indices = class_indices[:min(images_per_class, len(class_indices))]
        labeled_indices.extend(selected_indices)
    labeled_set = Subset(train_set, labeled_indices)
    unlabeled_indices = list(set(range(len(train_set))) - set(labeled_indices))
    unlabeled_set = Subset(train_set, unlabeled_indices)

    # Apply augmentations
    labeled_set.dataset.transform = auto_augment_transform  # AutoAugment for labeled set
    unlabeled_set.dataset.transform = apply_random_rotation_transform  # Random rotation and other transforms for unlabeled set

    # Create data loaders
    labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_model(model, labeled_loader, optimizer, criterion)
        test_accuracy = test_model(model, test_loader)
        print(f'Run {run_number} -> Epoch [{epoch}/{epochs}], LP:{len(labeled_indices)}, UP:{len(unlabeled_indices)}, Train Acc: {train_accuracy:.2f}%, Loss: {train_loss:.6f}, Test Acc: {test_accuracy:.2f}%')
      
        # Store results
        results_writer.writerow([run_number, epoch, train_loss, train_accuracy, test_accuracy])
      
        if epoch < epochs:  # Avoid running uncertainty sampling during the last iteration
            uncertain_indices = uncertainty_sampling(model, unlabeled_loader, n_samples_add_pool)
            labeled_indices.extend(uncertain_indices)
            unlabeled_indices = list(set(range(len(train_set))) - set(labeled_indices))
            labeled_set = Subset(train_set, labeled_indices)
            unlabeled_set = Subset(train_set, unlabeled_indices)
            labeled_set.dataset.transform = auto_augment_transform
            unlabeled_set.dataset.transform = apply_random_rotation_transform
            labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True)
            unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=True)

# The rest of the code remains unchanged (train_model, test_model functions, etc.)
# Define a function for training the model
def train_model(model, labeled_loader, optimizer, criterion):
    model.train()
    total_train_correct = 0
    total_train_samples = 0
    total_train_loss = 0

    for images, labels in labeled_loader:
        optimizer.zero_grad()
        images = images.cuda()
        labels = labels.cuda() 
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_train_samples += labels.size(0)
        total_train_correct += (predicted == labels).sum().item()
        total_train_loss += loss.item()

    train_accuracy = 100 * total_train_correct / total_train_samples
    train_loss = total_train_loss / len(labeled_loader)
    return train_loss, train_accuracy

# Define a function for testing the model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda() 
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    return test_accuracy

# Open a CSV file for writing results
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
