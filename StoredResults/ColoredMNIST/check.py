from PIL.Image import new
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import random
from torch.utils.data import Subset, DataLoader
import json
import threading
import csv
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import ConcatDataset
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

with open('/content/SRPV2/StoredResults/ColoredMNIST/parameters.json', 'r') as f:
   params = json.load(f)

epochs = params['epochs']
batch_size = params['batch_size']
images_per_class = params['images_per_class']
num_runs = params['num_runs']
n_samples_add_pool = params['n_samples_add_pool']

from autoaugment import ImageNetPolicy  # Import or define ImageNetPolicy

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

auto_augment_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    ImageNetPolicy(),  # AutoAugment policy for ImageNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class AutoAugmentSubset(Subset):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img = auto_augment_transform(img)  # Apply auto augment transform here
        return img, label

# Define rotation transforms for the unlabeled set
rotation_transforms = []
for i in range(4):
    angle = i * 90
    transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),  # Apply cropping with some padding
        transforms.ColorJitter(brightness=0.4, contrast=0.4),  # Adjust brightness and contrast
        transforms.RandomRotation(degrees=angle),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    rotation_transforms.append(transform)

# Function to randomly apply one of the rotation transformations
def apply_random_transform(img):
    transform = random.choice(rotation_transforms)  # Select a random transformation
    return transform(img)

# CustomMNISTDataset for switching between PIL and Tensor formats
class CustomMNISTDataset(torchvision.datasets.ImageFolder):
    def __init__(self, *args, apply_transform=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_transform = apply_transform
        self.transform_to_pil = transforms.Resize((224, 224))  # Transform to PIL
        self.tensor_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # Return PIL image for augmentation
        if self.apply_transform:
            img = self.tensor_transform(img)  # Apply the tensor transformation
        else:
            img = self.transform_to_pil(img)  # Keep the PIL format for augmentation
        return img, target

from torchvision import transforms

# Define a new transformation that converts labels to tensors
class TensorLabelSubset(Subset):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return img, torch.tensor(label)  # Convert label to tensor

# Load datasets
train_set = CustomMNISTDataset(root='/content/SRPV2/StoredResults/ColoredMNIST/ColoredMNIST/train', apply_transform=False)
test_set = datasets.ImageFolder(root='/content/SRPV2/StoredResults/ColoredMNIST/ColoredMNIST/test', transform=transform_test)
    

# Print dataset stats
print('Dataset Stats:')
print(f'Train Dataset Size: {len(train_set)}, Test Dataset Size: {len(test_set)}\n')
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

# Define a function to perform a single run
def single_run(run_number):
    # Initialize a new model for each run
    model = torchvision.models.resnet18(pretrained=False)  # ResNet18 expects 3-channel input
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # Modify the last fully connected layer for 10 classes (MNIST)
    model.cuda()  # Move model to GPU

    # Initialize the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Split train set into labeled and unlabeled indices


# Assuming train_set and test_set are defined elsewhere
# images_per_class: Number of images per class for the labeled set
# batch_size: Your defined batch size for DataLoader
# epochs: Number of training epochs
# run_number: Run identifier
# n_samples_add_pool: Number of samples to add from uncertainty sampling
# model, optimizer, criterion: Your model and optimizer

# Initializing empty lists for labeled and unlabeled indices
    labeled_indices = []
    unlabeled_indices = []

    # Loop over each class and randomly select 'images_per_class' for labeled set
    for class_label in range(10):  # Assuming 10 classes for example
        class_indices = np.where(np.array(train_set.targets) == class_label)[0]
        np.random.shuffle(class_indices)  # Shuffle class-specific indices
        selected_indices = class_indices[:min(images_per_class, len(class_indices))]
        labeled_indices.extend(selected_indices)

    # Create labeled and unlabeled sets
    labeled_set = Subset(train_set, labeled_indices)
    unlabeled_indices = list(set(range(len(train_set))) - set(labeled_indices))
    unlabeled_set = TensorLabelSubset(train_set, unlabeled_indices)  # Assuming you have TensorLabelSubset class

    # Augmentation process for labeled set
    augmented_images = []
    augmented_labels = []
    augmented_indices = []

    # Start index for augmented images (ensure no conflict with existing indices)
    start_idx = len(train_set)  # Start after the end of original train_set

    # Set the train_set to return PIL images for augmentation
    train_set.apply_transform = False  # Ensure it returns PIL images for augmentation

    # Apply augmentation to the labeled set
    for img_idx, (img, label) in enumerate(labeled_set):
        augmented_image = apply_random_transform(img)  # Assuming apply_random_transform function is defined
        augmented_images.append(augmented_image)  # Append transformed image (tensor)
        augmented_labels.append(torch.tensor(label))  # Ensure label is a tensor
        augmented_indices.append(start_idx + len(augmented_images) - 1)  # Assign new unique index

    # Create augmented dataset from augmented images and labels
    augmented_dataset = torch.utils.data.TensorDataset(torch.stack(augmented_images), torch.tensor(augmented_labels))

    # Combine the original unlabeled set with the augmented dataset
    combined_unlabeled_set = torch.utils.data.ConcatDataset([unlabeled_set, augmented_dataset])

    # Update unlabeled indices with the new augmented indices
    unlabeled_indices = list(set(range(len(train_set))) - set(labeled_indices)) + augmented_indices

    # Update DataLoaders
    train_set.apply_transform = True  # Switch back to Tensor format for DataLoader
    unlabeled_loader = DataLoader(combined_unlabeled_set, batch_size=batch_size, shuffle=True)
    labeled_set = AutoAugmentSubset(train_set, labeled_indices)  # Assuming AutoAugmentSubset class exists
    labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    new_train_set = torch.utils.data.ConcatDataset([labeled_set, combined_unlabeled_set])

    print(len(new_train_set))

# Open a CSV file for writing results

    
    
# for i in range(1, num_runs + 1):
single_run(1)
    #     thread = threading.Thread(target=single_run, args=(i, writer))
    #     threads.append(thread)
    #     thread.start()

    # # Wait for all threads to finish
    # for thread in threads:
    #     thread.join()

print("All runs completed.")
