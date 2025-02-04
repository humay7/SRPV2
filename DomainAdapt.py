import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import itertools
import os
from datetime import datetime

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset,DataLoader, random_split

# python_file_name = sys.argv[1]
augmentation_techniques = [
    "al_withoutda",
    "blur",
    "brightness",
    "colorjitter",
    "contrast",
    "elastic_transform",
    "histogram_equalization",
    "random_affine",
    "random_autocontrast",
    "random_equalize",
    "random_invert",
    "random_posterize",
    "random_sharpeness",
    "random_solarize",
    "random_transform",
    "rotation",
    "saturation",
    "shear",
    "single_channel"
  ]

from augumentation import get_augmentation
# augmentation_transform = get_augmentation(python_file_name)

# source_transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.RandomInvert(p=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

target_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# source_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=augmentation_transform)
target_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=target_transform)

# source_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=source_transform)
# target_dataset = datasets.ImageFolder(root='./data/MNIST-M/test', transform=target_transform)

# source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=64, shuffle=True)

class DomainAdaptationModel(nn.Module):
    def __init__(self):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        self.classifier = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = DomainAdaptationModel().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    total_loss, total_correct = 0, 0
    
    for source_data, source_labels in source_loader:
        source_data, source_labels = source_data.to(device), source_labels.to(device)
        optimizer.zero_grad()
        source_features = model.feature_extractor(source_data)
        source_outputs = model.classifier(source_features)
        loss = criterion(source_outputs, source_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(source_outputs.data, 1)
        total_correct += (predicted == source_labels).sum().item()
    print(f'Epoch {epoch}, Train Loss: {total_loss/len(source_loader):.4f}, Train Acc: {100*total_correct/len(source_dataset):.2f}%')

def test():
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for data, labels in target_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
    print(f'Test Loss: {total_loss/len(target_loader):.4f}, Test Acc: {100*total_correct/len(target_dataset):.2f}%')

# for epoch in range(1, 11):
#     train(epoch)
#     test()

# os.makedirs("models", exist_ok=True)
# save_path = f"models/PlainModelWithoutAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
# torch.save(model.state_dict(), save_path)
# print(f"Model saved at {save_path}")

def load_model(model_path):
    model = DomainAdaptationModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_loaded_model(model, aug_name):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for data, labels in target_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
    print(f'Loaded Model Test Loss for aug {aug_name}: {total_loss/len(target_loader):.4f}, Test Acc: {100*total_correct/len(target_dataset):.2f}%')

#save_path = 'models/PlainModel_20250130_130831.pth'
# model = load_model(save_path)
# evaluate_loaded_model(model)




results = {}

for aug_name in augmentation_techniques:
    print(f"\nEvaluating augmentation: {aug_name}")

    # Get augmentation transform
    augmentation_transform = get_augmentation(aug_name)

    # Recreate source dataset with new augmentation
    source_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=augmentation_transform)
    source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True)

    # Initialize and train the model
    model = DomainAdaptationModel().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  

    for epoch in range(1, 11):
        train(epoch)
        test()

    

    os.makedirs("models", exist_ok=True)
    save_path = f"models/PlainModelWithoutAL_{aug_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")
    model = load_model(save_path)
    evaluate_loaded_model(model)
# # Print all results
# print("\nFinal Results:")
# for aug, (loss, acc) in results.items():
#     print(f"Augmentation: {aug}, Test Loss: {loss:.4f}, Test Acc: {acc:.2f}%")
