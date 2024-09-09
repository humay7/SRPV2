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
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

with open('/content/SRPV2/parameters.json', 'r') as f:
   params = json.load(f)

epochs = params['epochs']
batch_size = params['batch_size']
images_per_class = params['images_per_class']
num_runs = params['num_runs']
n_samples_add_pool = params['n_samples_add_pool']

from autoaugment import ImageNetPolicy  # Import or define ImageNetPolicy

auto_augment_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    ImageNetPolicy(),  # AutoAugment policy for ImageNet
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3 channels
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
        transforms.Grayscale(num_output_channels=3),  # Ensure the image remains grayscale
        transforms.Resize((28, 28)),
        transforms.RandomCrop(28, padding=4),  # Apply cropping with some padding
        transforms.ColorJitter(brightness=0.4, contrast=0.4),  # Adjust brightness and contrast
        transforms.RandomRotation(degrees=angle),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    rotation_transforms.append(transform)

# Function to randomly apply one of the rotation transformations
def apply_random_transform(img):
    transform = random.choice(rotation_transforms)  # Select a random transformation
    return transform(img)

# CustomMNISTDataset for switching between PIL and Tensor formats
class CustomMNISTDataset(torchvision.datasets.MNIST):
    def __init__(self, *args, apply_transform=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_transform = apply_transform
        self.transform_to_pil = transforms.Grayscale(num_output_channels=3)  # Transform to PIL
        self.tensor_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
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
train_set = CustomMNISTDataset(root='./data', train=True, download=True, apply_transform=False)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels for testing as well
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

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
    for class_label in range(10):  # Loop over each class
        class_indices = np.where(np.array(train_set.targets) == class_label)[0]
        np.random.shuffle(class_indices)  # Shuffle class-specific indices
        selected_indices = class_indices[:min(images_per_class, len(class_indices))]
        labeled_indices.extend(selected_indices)

    labeled_set = Subset(train_set, labeled_indices)
    unlabeled_indices = list(set(range(len(train_set))) - set(labeled_indices))
    # unlabeled_set = Subset(train_set, unlabeled_indices)
    unlabeled_set = TensorLabelSubset(train_set, unlabeled_indices)

    augmented_images = []
    augmented_labels = []
    augmented_indices = []

    # Start index for augmented images (next index in the unlabeled set)
    start_idx = len(unlabeled_set)

    # Set the train_set to return PIL images for augmentation
    train_set.apply_transform = False  # Ensure it returns PIL images

    # Augment the labeled set by applying one random transformation
    for img_idx, (img, label) in enumerate(labeled_set):
      augmented_image = apply_random_transform(img)  # Apply a random augmentation
      augmented_images.append(augmented_image)  # This is already a tensor
      augmented_labels.append(torch.tensor(label))  # Ensure label is a tensor
      augmented_indices.append(start_idx + len(augmented_images) - 1)  # Track the new index

  # Create a dataset with these augmented images and labels
  # Ensure both images and labels are tensors
    augmented_dataset = torch.utils.data.TensorDataset(torch.stack(augmented_images), torch.tensor(augmented_labels))
      # Combine the original unlabeled set with the augmented dataset
    combined_unlabeled_set = torch.utils.data.ConcatDataset([unlabeled_set, augmented_dataset])

    # Update unlabeled indices with augmented indices
    unlabeled_indices.extend(augmented_indices)

    # Update your data loaders
    train_set.apply_transform = True  # Switch to Tensor format for DataLoader
    unlabeled_loader = DataLoader(combined_unlabeled_set, batch_size=batch_size, shuffle=True)
    labeled_set = AutoAugmentSubset(train_set, labeled_indices)
    # Labeled loader remains the same
    labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True)

    # Test loader remains unchanged
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
            labeled_loader = DataLoader(torch.utils.data.Subset(train_set, labeled_indices), batch_size=batch_size, shuffle=True)
            unlabeled_loader = DataLoader(torch.utils.data.Subset(train_set, unlabeled_indices), batch_size=batch_size, shuffle=True)

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
    # threads = []
    for i in range(1, num_runs + 1):
      single_run(i, writer)
    #     thread = threading.Thread(target=single_run, args=(i, writer))
    #     threads.append(thread)
    #     thread.start()

    # # Wait for all threads to finish
    # for thread in threads:
    #     thread.join()

print("All runs completed.")
